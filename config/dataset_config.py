import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DatasetConfig:
    """Configuration class for datasets."""
    
    dataset_name: str
    data_path: str
    img_size: int = 640
    train_split: float = 0.8
    val_split: float = 0.2
    
    def __post_init__(self):
        """Initialize dataset-specific configurations."""
        self.dataset_name = self.dataset_name.lower()
        
        # Dataset-specific configurations
        self.dataset_configs = {
            'coco': {
                'num_classes': 80,
                'class_names': self._get_coco_classes(),
                'annotation_format': 'coco',
                'train_images': 'train2017',
                'val_images': 'val2017',
                'train_annotations': 'annotations/instances_train2017.json',
                'val_annotations': 'annotations/instances_val2017.json'
            },
            'pascal_voc': {
                'num_classes': 20,
                'class_names': self._get_pascal_voc_classes(),
                'annotation_format': 'xml',
                'train_images': 'JPEGImages',
                'val_images': 'JPEGImages',
                'train_annotations': 'Annotations',
                'val_annotations': 'Annotations'
            }
        }
        
        if self.dataset_name not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        self.config = self.dataset_configs[self.dataset_name]
        self.num_classes = self.config['num_classes']
        self.class_names = self.config['class_names']
    
    def _get_coco_classes(self) -> List[str]:
        """Get COCO dataset class names."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def _get_pascal_voc_classes(self) -> List[str]:
        """Get PASCAL VOC dataset class names."""
        return [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    
    def get_dataset_path(self) -> str:
        """Get the full path to the dataset."""
        return os.path.join(self.data_path, self.dataset_name)
    
    def get_train_images_path(self) -> str:
        """Get path to training images."""
        return os.path.join(self.get_dataset_path(), self.config['train_images'])
    
    def get_val_images_path(self) -> str:
        """Get path to validation images."""
        return os.path.join(self.get_dataset_path(), self.config['val_images'])
    
    def get_train_annotations_path(self) -> str:
        """Get path to training annotations."""
        return os.path.join(self.get_dataset_path(), self.config['train_annotations'])
    
    def get_val_annotations_path(self) -> str:
        """Get path to validation annotations."""
        return os.path.join(self.get_dataset_path(), self.config['val_annotations'])
    
    def create_yaml_config(self, output_path: str) -> str:
        """Create YAML configuration file for YOLOv8."""
        yaml_content = f"""
# YOLOv8 Dataset Configuration for {self.dataset_name.upper()}
path: {self.get_dataset_path()}
train: {self.config['train_images']}
val: {self.config['val_images']}

# Classes
nc: {self.num_classes}
names:
"""
        for i, class_name in enumerate(self.class_names):
            yaml_content += f"  {i}: {class_name}\n"
        
        yaml_file_path = os.path.join(output_path, f'{self.dataset_name}_config.yaml')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(yaml_file_path), exist_ok=True)
        
        with open(yaml_file_path, 'w') as f:
            f.write(yaml_content)
        
        return yaml_file_path
    
    def validate_dataset_structure(self) -> Dict[str, bool]:
        """Validate that the dataset has the required structure."""
        validation_results = {}
        
        # Check if dataset directory exists
        dataset_path = self.get_dataset_path()
        validation_results['dataset_dir_exists'] = os.path.exists(dataset_path)
        
        # Check if required directories exist
        train_images_path = self.get_train_images_path()
        val_images_path = self.get_val_images_path()
        
        validation_results['train_images_exist'] = os.path.exists(train_images_path)
        validation_results['val_images_exist'] = os.path.exists(val_images_path)
        
        # Check annotations
        train_annotations_path = self.get_train_annotations_path()
        val_annotations_path = self.get_val_annotations_path()
        
        validation_results['train_annotations_exist'] = os.path.exists(train_annotations_path)
        validation_results['val_annotations_exist'] = os.path.exists(val_annotations_path)
        
        # Count images if directories exist
        if validation_results['train_images_exist']:
            train_images = list(Path(train_images_path).glob('*.jpg')) + list(Path(train_images_path).glob('*.png'))
            validation_results['train_image_count'] = len(train_images)
        else:
            validation_results['train_image_count'] = 0
            
        if validation_results['val_images_exist']:
            val_images = list(Path(val_images_path).glob('*.jpg')) + list(Path(val_images_path).glob('*.png'))
            validation_results['val_image_count'] = len(val_images)
        else:
            validation_results['val_image_count'] = 0
        
        return validation_results
