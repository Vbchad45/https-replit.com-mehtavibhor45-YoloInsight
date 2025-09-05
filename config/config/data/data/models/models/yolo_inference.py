

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple
import logging
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

from utils.visualization import Visualizer

logger = logging.getLogger(__name__)

class YOLOInference:
    """YOLOv8 inference class for object detection."""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """Initialize the YOLO inference engine."""
        self.model_path = model_path
        self.device = device
        
        try:
            self.model = YOLO(model_path)
            if device != 'auto':
                self.model.to(device)
            logger.info(f"Loaded model from: {model_path}")
            logger.info(f"Inference device: {self.model.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.visualizer = Visualizer()
    
    def predict(self, source: Union[str, np.ndarray, Image.Image], 
               output_dir: str, **kwargs) -> List[Dict[str, Any]]:
        """Run inference on images/video/directory."""
        logger.info(f"Running inference on: {source}")
        
        try:
            # Set default inference parameters
            inference_params = {
                'conf': kwargs.get('conf', 0.25),
                'iou': kwargs.get('iou', 0.45),
                'max_det': kwargs.get('max_det', 1000),
                'save': kwargs.get('save', True),
                'save_txt': kwargs.get('save_txt', True),
                'save_conf': kwargs.get('save_conf', True),
                'project': output_dir,
                'name': 'inference',
                'exist_ok': True,
                'verbose': False
            }
            
            # Run inference
            results = self.model.predict(source, **inference_params)
            
            # Process results
            processed_results = []
            for i, result in enumerate(results):
                processed_result = self._process_single_result(result, i, output_dir)
                processed_results.append(processed_result)
            
            logger.info(f"Inference completed. Results saved to: {output_dir}")
            return processed_results
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def predict_single(self, image_path: str, output_dir: str, 
                      conf_threshold: float = 0.25) -> Dict[str, Any]:
        """Run inference on a single image."""
        logger.info(f"Running inference on single image: {image_path}")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Run inference
            results = self.model.predict(
                image_path,
                conf=conf_threshold,
                save=False,
                verbose=False
            )
            
            if not results:
                return {'error': 'No results returned'}
            
            result = results[0]
            
            # Process result
            detections = self._extract_detections(result)
            
            # Create visualization
            output_image_path = os.path.join(output_dir, 'inference', 'predicted_image.jpg')
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            
            annotated_image = self._draw_detections(image, detections)
            cv2.imwrite(output_image_path, annotated_image)
            
            return {
                'image_path': image_path,
                'output_image_path': output_image_path,
                'detections': detections,
                'num_detections': len(detections),
                'inference_time_ms': getattr(result, 'speed', {}).get('inference', 0),
                'model_info': {
                    'model_path': self.model_path,
                    'confidence_threshold': conf_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Single image inference failed: {e}")
            return {'error': str(e)}
    
    def predict_batch(self, image_paths: List[str], output_dir: str, 
                     batch_size: int = 16) -> List[Dict[str, Any]]:
        """Run inference on a batch of images."""
        logger.info(f"Running batch inference on {len(image_paths)} images")
        
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            try:
                batch_results = self.model.predict(
                    batch_paths,
                    save=False,
                    verbose=False
                )
                
                for j, result in enumerate(batch_results):
                    image_path = batch_paths[j]
                    detections = self._extract_detections(result)
                    
                    results.append({
                        'image_path': image_path,
                        'detections': detections,
                        'num_detections': len(detections)
                    })
                    
            except Exception as e:
                logger.error(f"Batch inference failed for batch {i//batch_size + 1}: {e}")
                
                # Add error results for this batch
                for path in batch_paths:
                    results.append({
                        'image_path': path,
                        'error': str(e),
                        'detections': [],
                        'num_detections': 0
                    })
        
        return results
    
    def _process_single_result(self, result, index: int, output_dir: str) -> Dict[str, Any]:
        """Process a single inference result."""
        try:
            # Extract detections
            detections = self._extract_detections(result)
            
            # Get image path
            image_path = result.path if hasattr(result, 'path') else f'image_{index}'
            
            # Get inference timing
            speed_info = getattr(result, 'speed', {})
            
            return {
                'image_path': image_path,
                'detections': detections,
                'num_detections': len(detections),
                'inference_time_ms': speed_info.get('inference', 0),
                'postprocess_time_ms': speed_info.get('postprocess', 0),
                'total_time_ms': sum(speed_info.values()) if speed_info else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to process result {index}: {e}")
            return {
                'image_path': f'image_{index}',
                'error': str(e),
                'detections': [],
                'num_detections': 0
            }
    
    def _extract_detections(self, result) -> List[Dict[str, Any]]:
        """Extract detection information from YOLO result."""
        detections = []
        
        try:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                # Get detection data
                if hasattr(boxes, 'xyxy'):
                    bboxes = boxes.xyxy.cpu().numpy()
                else:
                    return detections
                
                confidences = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else np.ones(len(bboxes))
                class_ids = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else np.zeros(len(bboxes))
                
                # Get class names
                class_names = self.model.names if hasattr(self.model, 'names') else {}
                
                for bbox, conf, cls_id in zip(bboxes, confidences, class_ids):
                    x1, y1, x2, y2 = bbox
                    
                    detection = {
                        'bbox': {
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2),
                            'width': float(x2 - x1),
                            'height': float(y2 - y1)
                        },
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': class_names.get(int(cls_id), f'class_{int(cls_id)}')
                    }
                    
                    detections.append(detection)
            
        except Exception as e:
            logger.error(f"Failed to extract detections: {e}")
        
        return detections
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection results on image."""
        annotated_image = image.copy()
        
        # Define colors for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
        ]
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            # Get color
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(
                annotated_image,
                (int(bbox['x1']), int(bbox['y1'])),
                (int(bbox['x2']), int(bbox['y2'])),
                color,
                2
            )
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(
                annotated_image,
                (int(bbox['x1']), int(bbox['y1']) - label_size[1] - 10),
                (int(bbox['x1']) + label_size[0], int(bbox['y1'])),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (int(bbox['x1']), int(bbox['y1']) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated_image
    
    def analyze_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze detection results to provide insights."""
        if not detections:
            return {'total_detections': 0}
        
        # Count detections per class
        class_counts = {}
        confidence_scores = []
        bbox_areas = []
        
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            bbox = detection['bbox']
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_scores.append(confidence)
            bbox_areas.append(bbox['width'] * bbox['height'])
        
        analysis = {
            'total_detections': len(detections),
            'unique_classes': len(class_counts),
            'class_distribution': class_counts,
            'confidence_stats': {
                'mean': np.mean(confidence_scores),
                'std': np.std(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores)
            },
            'bbox_area_stats': {
                'mean': np.mean(bbox_areas),
                'std': np.std(bbox_areas),
                'min': np.min(bbox_areas),
                'max': np.max(bbox_areas)
            }
        }
        
        return analysis
    
    def export_detections(self, detections: List[Dict[str, Any]], 
                         output_path: str, format: str = 'json'):
        """Export detections to file."""
        try:
            if format.lower() == 'json':
                import json
                with open(output_path, 'w') as f:
                    json.dump(detections, f, indent=2)
            
            elif format.lower() == 'csv':
                import pandas as pd
                
                # Flatten detections for CSV
                flattened = []
                for det in detections:
                    flat_det = {
                        'class_name': det['class_name'],
                        'class_id': det['class_id'],
                        'confidence': det['confidence'],
                        'x1': det['bbox']['x1'],
                        'y1': det['bbox']['y1'],
                        'x2': det['bbox']['x2'],
                        'y2': det['bbox']['y2'],
                        'width': det['bbox']['width'],
                        'height': det['bbox']['height']
                    }
                    flattened.append(flat_det)
                
                df = pd.DataFrame(flattened)
                df.to_csv(output_path, index=False)
            
            elif format.lower() == 'txt':
                # YOLO format
                with open(output_path, 'w') as f:
                    for det in detections:
                        bbox = det['bbox']
                        # Convert to YOLO format (assuming image dimensions)
                        # Note: This is a simplified conversion
                        f.write(f"{det['class_id']} {det['confidence']} "
                               f"{bbox['x1']} {bbox['y1']} {bbox['x2']} {bbox['y2']}\n")
            
            logger.info(f"Detections exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export detections: {e}")
            raise
