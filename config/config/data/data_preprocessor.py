import cv2
import numpy as np
import random
from typing import Tuple, List, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessor for object detection tasks."""
    
    def __init__(self, img_size: int = 640, augment: bool = True):
        """Initialize the data preprocessor."""
        self.img_size = img_size
        self.augment = augment
        
        # Define augmentation pipeline
        if augment:
            self.train_transform = self._get_train_transforms()
        else:
            self.train_transform = self._get_base_transforms()
        
        self.val_transform = self._get_val_transforms()
    
    def _get_train_transforms(self) -> A.Compose:
        """Get training data augmentation transforms."""
        return A.Compose([
            # Resize while maintaining aspect ratio
            A.LongestMaxSize(max_size=self.img_size, p=1.0),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            
            # Color augmentations
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            ),
            
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def _get_base_transforms(self) -> A.Compose:
        """Get basic transforms without augmentation."""
        return A.Compose([
            A.LongestMaxSize(max_size=self.img_size, p=1.0),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            ),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def _get_val_transforms(self) -> A.Compose:
        """Get validation transforms."""
        return A.Compose([
            A.LongestMaxSize(max_size=self.img_size, p=1.0),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            ),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def preprocess_image(self, image: np.ndarray, mode: str = 'train') -> np.ndarray:
        """Preprocess a single image."""
        if mode == 'train':
            transform = self.train_transform
        else:
            transform = self.val_transform
        
        # Apply transforms
        transformed = transform(image=image)
        return transformed['image']
    
    def preprocess_with_bboxes(self, image: np.ndarray, bboxes: List[List[float]], 
                             class_labels: List[int], mode: str = 'train') -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """Preprocess image with bounding boxes."""
        if mode == 'train':
            transform = self.train_transform
        else:
            transform = self.val_transform
        
        # Apply transforms
        try:
            transformed = transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            return (
                transformed['image'],
                transformed['bboxes'],
                transformed['class_labels']
            )
        except Exception as e:
            logger.warning(f"Transform failed: {e}. Using original image.")
            # Fallback to basic resize
            image_resized = cv2.resize(image, (self.img_size, self.img_size))
            return image_resized, bboxes, class_labels
    
    def mosaic_augmentation(self, images: List[np.ndarray], bboxes_list: List[List[List[float]]], 
                           labels_list: List[List[int]]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """Apply mosaic augmentation (combine 4 images)."""
        if len(images) < 4:
            raise ValueError("Mosaic augmentation requires at least 4 images")
        
        # Select 4 random images
        indices = random.sample(range(len(images)), 4)
        selected_images = [images[i] for i in indices]
        selected_bboxes = [bboxes_list[i] for i in indices]
        selected_labels = [labels_list[i] for i in indices]
        
        # Create mosaic
        mosaic_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_labels = []
        
        # Define quadrants
        half_size = self.img_size // 2
        quadrants = [
            (0, 0, half_size, half_size),           # Top-left
            (half_size, 0, self.img_size, half_size),     # Top-right
            (0, half_size, half_size, self.img_size),     # Bottom-left
            (half_size, half_size, self.img_size, self.img_size)  # Bottom-right
        ]
        
        for i, (img, bboxes, labels) in enumerate(zip(selected_images, selected_bboxes, selected_labels)):
            # Resize image to quadrant size
            img_resized = cv2.resize(img, (half_size, half_size))
            
            # Place image in quadrant
            x1, y1, x2, y2 = quadrants[i]
            mosaic_img[y1:y2, x1:x2] = img_resized
            
            # Adjust bounding boxes for new position and size
            for bbox, label in zip(bboxes, labels):
                # Convert from YOLO format to absolute coordinates
                x_center, y_center, width, height = bbox
                x_center_abs = x_center * half_size + x1
                y_center_abs = y_center * half_size + y1
                width_abs = width * half_size
                height_abs = height * half_size
                
                # Convert back to YOLO format for full image
                x_center_new = x_center_abs / self.img_size
                y_center_new = y_center_abs / self.img_size
                width_new = width_abs / self.img_size
                height_new = height_abs / self.img_size
                
                # Check if bbox is still valid
                if (x_center_new > 0 and x_center_new < 1 and 
                    y_center_new > 0 and y_center_new < 1 and
                    width_new > 0 and height_new > 0):
                    mosaic_bboxes.append([x_center_new, y_center_new, width_new, height_new])
                    mosaic_labels.append(label)
        
        return mosaic_img, mosaic_bboxes, mosaic_labels
    
    def mixup_augmentation(self, img1: np.ndarray, img2: np.ndarray, 
                          bboxes1: List[List[float]], bboxes2: List[List[float]],
                          labels1: List[int], labels2: List[int],
                          alpha: float = 0.5) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """Apply mixup augmentation."""
        # Ensure same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Blend images
        mixed_img = (alpha * img1 + (1 - alpha) * img2).astype(np.uint8)
        
        # Combine bounding boxes and labels
        mixed_bboxes = bboxes1 + bboxes2
        mixed_labels = labels1 + labels2
        
        return mixed_img, mixed_bboxes, mixed_labels
    
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get statistics about augmentation pipeline."""
        return {
            'img_size': self.img_size,
            'augment': self.augment,
            'num_train_transforms': len(self.train_transform.transforms) if self.augment else 0,
            'num_val_transforms': len(self.val_transform.transforms),
            'augmentation_types': [
                'HueSaturationValue',
                'RandomBrightnessContrast',
                'CLAHE',
                'HorizontalFlip',
                'ShiftScaleRotate',
                'GaussNoise/ISONoise',
                'Blur variations',
                'Normalization'
            ] if self.augment else ['Resize', 'Normalize']
        }
