
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

from config.dataset_config import DatasetConfig
from utils.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

class YOLOEvaluator:
    """YOLOv8 model evaluator."""
    
    def __init__(self, model_path: str, dataset_config: DatasetConfig, output_dir: str):
        """Initialize the YOLO evaluator."""
        self.model_path = model_path
        self.dataset_config = dataset_config
        self.output_dir = output_dir
        
        # Create evaluation output directory
        self.eval_dir = os.path.join(output_dir, 'evaluate')
        Path(self.eval_dir).mkdir(parents=True, exist_ok=True)
        
        # Load model
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded model from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.metrics_calculator = MetricsCalculator(dataset_config.class_names)
    
    def evaluate(self, save_results: bool = True) -> Dict[str, Any]:
        """Evaluate the model on the validation dataset."""
        logger.info("Starting model evaluation...")
        
        try:
            # Create dataset YAML configuration
            yaml_config_path = self.dataset_config.create_yaml_config(self.eval_dir)
            
            # Run validation
            validation_results = self.model.val(
                data=yaml_config_path,
                split='val',
                save_json=True,
                save_hybrid=True,
                plots=True,
                verbose=True
            )
            
            # Extract metrics
            metrics = self._extract_metrics(validation_results)
            
            # Calculate additional metrics
            additional_metrics = self._calculate_additional_metrics()
            metrics.update(additional_metrics)
            
            # Generate evaluation plots
            if save_results:
                self._generate_evaluation_plots(metrics)
                self._save_evaluation_results(metrics)
            
            logger.info("Evaluation completed successfully!")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _extract_metrics(self, validation_results) -> Dict[str, Any]:
        """Extract metrics from validation results."""
        metrics = {}
        
        try:
            # Get metrics from results
            if hasattr(validation_results, 'results_dict'):
                results_dict = validation_results.results_dict
                
                # Extract key metrics
                metrics['mAP50'] = results_dict.get('metrics/mAP50(B)', 0.0)
                metrics['mAP50-95'] = results_dict.get('metrics/mAP50-95(B)', 0.0)
                metrics['precision'] = results_dict.get('metrics/precision(B)', 0.0)
                metrics['recall'] = results_dict.get('metrics/recall(B)', 0.0)
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
            
            # Get per-class metrics if available
            if hasattr(validation_results, 'box'):
                box_metrics = validation_results.box
                if hasattr(box_metrics, 'mp'):  # Mean precision per class
                    metrics['precision_per_class'] = box_metrics.mp.tolist() if hasattr(box_metrics.mp, 'tolist') else box_metrics.mp
                if hasattr(box_metrics, 'mr'):  # Mean recall per class
                    metrics['recall_per_class'] = box_metrics.mr.tolist() if hasattr(box_metrics.mr, 'tolist') else box_metrics.mr
                if hasattr(box_metrics, 'map50'):  # mAP@0.5 per class
                    metrics['map50_per_class'] = box_metrics.map50.tolist() if hasattr(box_metrics.map50, 'tolist') else box_metrics.map50
                if hasattr(box_metrics, 'map'):  # mAP@0.5:0.95 per class
                    metrics['map_per_class'] = box_metrics.map.tolist() if hasattr(box_metrics.map, 'tolist') else box_metrics.map
            
            logger.info(f"Extracted metrics: mAP50={metrics.get('mAP50', 0):.4f}, mAP50-95={metrics.get('mAP50-95', 0):.4f}")
            
        except Exception as e:
            logger.warning(f"Could not extract all metrics: {e}")
        
        return metrics
    
    def _calculate_additional_metrics(self) -> Dict[str, Any]:
        """Calculate additional evaluation metrics."""
        additional_metrics = {}
        
        try:
            # Model size and parameters
            model_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
            additional_metrics['model_size_mb'] = round(model_size, 2)
            
            # Inference speed (approximate)
            additional_metrics['inference_device'] = str(self.model.device)
            
            # Dataset information
            dataset_stats = self.dataset_config.validate_dataset_structure()
            additional_metrics['num_val_images'] = dataset_stats['val_image_count']
            additional_metrics['num_classes'] = self.dataset_config.num_classes
            
        except Exception as e:
            logger.warning(f"Could not calculate additional metrics: {e}")
        
        return additional_metrics
    
    def _generate_evaluation_plots(self, metrics: Dict[str, Any]):
        """Generate evaluation visualization plots."""
        try:
            plots_dir = os.path.join(self.eval_dir, 'plots')
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Overall metrics bar plot
            self._plot_overall_metrics(metrics, plots_dir)
            
            # 2. Per-class metrics plots
            if 'precision_per_class' in metrics:
                self._plot_per_class_metrics(metrics, plots_dir)
            
            # 3. Model performance summary
            self._plot_performance_summary(metrics, plots_dir)
            
            logger.info(f"Evaluation plots saved to: {plots_dir}")
            
        except Exception as e:
            logger.warning(f"Could not generate evaluation plots: {e}")
    
    def _plot_overall_metrics(self, metrics: Dict[str, Any], plots_dir: str):
        """Plot overall performance metrics."""
        metric_names = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1_score']
        metric_values = [metrics.get(name, 0) for name in metric_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Model Evaluation Metrics', fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, 'overall_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_metrics(self, metrics: Dict[str, Any], plots_dir: str):
        """Plot per-class performance metrics."""
        class_names = self.dataset_config.class_names
        
        # Prepare data
        precision_per_class = metrics.get('precision_per_class', [])
        recall_per_class = metrics.get('recall_per_class', [])
        map50_per_class = metrics.get('map50_per_class', [])
        
        if not precision_per_class:
            return
        
        # Limit to first 20 classes for readability
        max_classes = min(20, len(class_names))
        class_names_short = class_names[:max_classes]
        precision_short = precision_per_class[:max_classes]
        recall_short = recall_per_class[:max_classes] if recall_per_class else [0] * max_classes
        map50_short = map50_per_class[:max_classes] if map50_per_class else [0] * max_classes
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Precision and Recall plot
        x = np.arange(len(class_names_short))
        width = 0.35
        
        ax1.bar(x - width/2, precision_short, width, label='Precision', alpha=0.8)
        ax1.bar(x + width/2, recall_short, width, label='Recall', alpha=0.8)
        
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Score')
        ax1.set_title('Per-Class Precision and Recall')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names_short, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # mAP@0.5 plot
        ax2.bar(class_names_short, map50_short, color='lightcoral', alpha=0.8)
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('mAP@0.5')
        ax2.set_title('Per-Class mAP@0.5')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_summary(self, metrics: Dict[str, Any], plots_dir: str):
        """Plot performance summary dashboard."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
        
        # 1. Precision-Recall plot
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        ax1.scatter([recall], [precision], s=100, c='red', alpha=0.7)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision vs Recall')
        ax1.grid(True, alpha=0.3)
        ax1.annotate(f'P: {precision:.3f}\nR: {recall:.3f}', 
                    xy=(recall, precision), xytext=(0.1, 0.9),
                    textcoords='axes fraction', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        # 2. mAP comparison
        map_metrics = ['mAP50', 'mAP50-95']
        map_values = [metrics.get(metric, 0) for metric in map_metrics]
        
        ax2.bar(map_metrics, map_values, color=['skyblue', 'lightgreen'])
        ax2.set_ylabel('mAP Score')
        ax2.set_title('mAP Metrics')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(map_values):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Model info
        model_size = metrics.get('model_size_mb', 0)
        num_classes = metrics.get('num_classes', 0)
        num_images = metrics.get('num_val_images', 0)
        
        info_text = f"""Model Information:
        
Size: {model_size:.1f} MB
Classes: {num_classes}
Val Images: {num_images}
Device: {metrics.get('inference_device', 'Unknown')}"""
        
        ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Model Information')
        
        # 4. Performance gauge
        f1_score = metrics.get('f1_score', 0)
        
        # Create a simple gauge chart
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax4.plot(theta, r, 'k-', linewidth=2)
        ax4.fill_between(theta, 0, r, alpha=0.3, color='lightgray')
        
        # F1 score indicator
        f1_angle = f1_score * np.pi
        ax4.plot([f1_angle, f1_angle], [0, 1], 'r-', linewidth=4)
        ax4.plot(f1_angle, 1, 'ro', markersize=10)
        
        ax4.set_xlim(0, np.pi)
        ax4.set_ylim(0, 1.2)
        ax4.set_title(f'F1 Score: {f1_score:.3f}')
        ax4.set_xticks([0, np.pi/2, np.pi])
        ax4.set_xticklabels(['0', '0.5', '1'])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_evaluation_results(self, metrics: Dict[str, Any]):
        """Save evaluation results to file."""
        try:
            # Save detailed metrics
            results_file = os.path.join(self.eval_dir, 'evaluation_results.json')
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            # Save summary report
            summary_file = os.path.join(self.eval_dir, 'evaluation_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("YOLOv8 Model Evaluation Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Model: {self.model_path}\n")
                f.write(f"Dataset: {self.dataset_config.dataset_name}\n")
                f.write(f"Number of Classes: {self.dataset_config.num_classes}\n\n")
                
                f.write("Overall Performance:\n")
                f.write("-" * 20 + "\n")
                f.write(f"mAP@0.5: {metrics.get('mAP50', 0):.4f}\n")
                f.write(f"mAP@0.5:0.95: {metrics.get('mAP50-95', 0):.4f}\n")
                f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
                f.write(f"Recall: {metrics.get('recall', 0):.4f}\n")
                f.write(f"F1 Score: {metrics.get('f1_score', 0):.4f}\n\n")
                
                f.write("Model Information:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Model Size: {metrics.get('model_size_mb', 0):.1f} MB\n")
                f.write(f"Validation Images: {metrics.get('num_val_images', 0)}\n")
                f.write(f"Inference Device: {metrics.get('inference_device', 'Unknown')}\n")
            
            logger.info(f"Evaluation results saved to: {self.eval_dir}")
            
        except Exception as e:
            logger.warning(f"Could not save evaluation results: {e}")
    
    def benchmark_inference_speed(self, num_images: int = 100) -> Dict[str, float]:
        """Benchmark inference speed."""
        logger.info(f"Benchmarking inference speed with {num_images} images...")
        
        try:
            import time
            from PIL import Image
            
            # Create dummy images
            dummy_images = []
            for _ in range(num_images):
                img = Image.new('RGB', (640, 640), color='red')
                dummy_images.append(img)
            
            # Warm up
            for _ in range(10):
                self.model.predict(dummy_images[0], verbose=False)
            
            # Benchmark
            start_time = time.time()
            for img in dummy_images:
                self.model.predict(img, verbose=False)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_image = total_time / num_images
            fps = 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0
            
            speed_metrics = {
                'total_time_seconds': total_time,
                'avg_time_per_image_ms': avg_time_per_image * 1000,
                'fps': fps,
                'images_benchmarked': num_images
            }
            
            logger.info(f"Benchmark results: {fps:.2f} FPS, {avg_time_per_image*1000:.2f} ms/image")
            return speed_metrics
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {}
