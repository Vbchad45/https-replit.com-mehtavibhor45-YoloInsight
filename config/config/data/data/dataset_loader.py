import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import requests
import zipfile
from tqdm import tqdm

from config.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Dataset loader for COCO and PASCAL VOC datasets."""
    
    def __init__(self, config: DatasetConfig):
        """Initialize the dataset loader."""
        self.config = config
        self.dataset_path = config.get_dataset_path()
    
    def prepare_dataset(self) -> Dict[str, any]:
        """Prepare the dataset for training."""
        logger.info(f"Preparing {self.config.dataset_name} dataset...")
        
        # Check if dataset exists, if not, download it
        if not self._dataset_exists():
            logger.info("Dataset not found. Attempting to download...")
            self._download_dataset()
        
        # Validate dataset structure
        validation_results = self.config.validate_dataset_structure()
        
        if not all([
            validation_results['dataset_dir_exists'],
            validation_results['train_images_exist'],
            validation_results['val_images_exist']
        ]):
            raise FileNotFoundError(
                f"Dataset structure is incomplete. Validation results: {validation_results}"
            )
        
        # Convert annotations to YOLO format if needed
        if self.config.dataset_name == 'coco':
            self._prepare_coco_dataset()
        elif self.config.dataset_name == 'pascal_voc':
            self._prepare_pascal_voc_dataset()
        
        dataset_info = {
            'dataset_name': self.config.dataset_name,
            'num_classes': self.config.num_classes,
            'class_names': self.config.class_names,
            'train_images': validation_results['train_image_count'],
            'val_images': validation_results['val_image_count'],
            'dataset_path': self.dataset_path
        }
        
        logger.info(f"Dataset prepared successfully: {dataset_info}")
        return dataset_info
    
    def _dataset_exists(self) -> bool:
        """Check if the dataset exists."""
        validation_results = self.config.validate_dataset_structure()
        return validation_results['dataset_dir_exists'] and \
               validation_results['train_images_exist'] and \
               validation_results['val_images_exist']
    
    def _download_dataset(self):
        """Download the dataset."""
        logger.info(f"Downloading {self.config.dataset_name} dataset...")
        
        if self.config.dataset_name == 'coco':
            self._download_coco()
        elif self.config.dataset_name == 'pascal_voc':
            self._download_pascal_voc()
        else:
            raise ValueError(f"Automatic download not supported for {self.config.dataset_name}")
    
    def _download_coco(self):
        """Download COCO dataset."""
        # COCO dataset URLs
        urls = {
            'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
            'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
            'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        }
        
        os.makedirs(self.dataset_path, exist_ok=True)
        
        for name, url in urls.items():
            logger.info(f"Downloading {name}...")
            self._download_and_extract(url, self.dataset_path)
    
    def _download_pascal_voc(self):
        """Download PASCAL VOC dataset."""
        # Note: PASCAL VOC download URLs might change. This is a placeholder implementation.
        logger.warning(
            "PASCAL VOC automatic download is not implemented. "
            "Please manually download and extract the dataset to the specified path."
        )
        # You would implement the actual download logic here
        pass
    
    def _download_and_extract(self, url: str, extract_path: str):
        """Download and extract a zip file."""
        filename = url.split('/')[-1]
        file_path = os.path.join(extract_path, filename)
        
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        # Extract file
        logger.info(f"Extracting {filename}...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Remove zip file
        os.remove(file_path)
        logger.info(f"Downloaded and extracted {filename}")
    
    def _prepare_coco_dataset(self):
        """Prepare COCO dataset by converting annotations to YOLO format."""
        logger.info("Converting COCO annotations to YOLO format...")
        
        # Load COCO annotations
        train_ann_path = self.config.get_train_annotations_path()
        val_ann_path = self.config.get_val_annotations_path()
        
        if os.path.exists(train_ann_path):
            self._convert_coco_to_yolo(train_ann_path, 'train')
        
        if os.path.exists(val_ann_path):
            self._convert_coco_to_yolo(val_ann_path, 'val')
    
    def _convert_coco_to_yolo(self, annotation_file: str, split: str):
        """Convert COCO annotations to YOLO format."""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Create YOLO labels directory
        labels_dir = os.path.join(self.dataset_path, f'{split}2017_labels')
        os.makedirs(labels_dir, exist_ok=True)
        
        # Create image ID to annotations mapping
        img_to_anns = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Convert annotations
        for img in tqdm(data['images'], desc=f"Converting {split} annotations"):
            img_id = img['id']
            img_width = img['width']
            img_height = img['height']
            
            # Create YOLO label file
            label_file = os.path.join(labels_dir, f"{img['file_name'].split('.')[0]}.txt")
            
            with open(label_file, 'w') as f:
                if img_id in img_to_anns:
                    for ann in img_to_anns[img_id]:
                        # Convert COCO bbox to YOLO format
                        x, y, w, h = ann['bbox']
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        width = w / img_width
                        height = h / img_height
                        
                        # COCO category_id to YOLO class_id (0-indexed)
                        class_id = ann['category_id'] - 1
                        
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    def _prepare_pascal_voc_dataset(self):
        """Prepare PASCAL VOC dataset by converting annotations to YOLO format."""
        logger.info("Converting PASCAL VOC annotations to YOLO format...")
        
        # Create YOLO labels directory
        labels_dir = os.path.join(self.dataset_path, 'labels')
        os.makedirs(labels_dir, exist_ok=True)
        
        # Get all XML annotation files
        annotations_dir = self.config.get_train_annotations_path()
        xml_files = list(Path(annotations_dir).glob('*.xml'))
        
        for xml_file in tqdm(xml_files, desc="Converting VOC annotations"):
            self._convert_voc_xml_to_yolo(xml_file, labels_dir)
    
    def _convert_voc_xml_to_yolo(self, xml_file: Path, labels_dir: str):
        """Convert a single PASCAL VOC XML file to YOLO format."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # Create YOLO label file
        label_file = os.path.join(labels_dir, f"{xml_file.stem}.txt")
        
        with open(label_file, 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                
                # Get class index
                if class_name in self.config.class_names:
                    class_id = self.config.class_names.index(class_name)
                else:
                    continue  # Skip unknown classes
                
                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # Convert to YOLO format
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    def get_dataset_statistics(self) -> Dict[str, any]:
        """Get dataset statistics."""
        validation_results = self.config.validate_dataset_structure()
        
        stats = {
            'dataset_name': self.config.dataset_name,
            'num_classes': self.config.num_classes,
            'class_names': self.config.class_names,
            'train_images': validation_results['train_image_count'],
            'val_images': validation_results['val_image_count'],
            'total_images': validation_results['train_image_count'] + validation_results['val_image_count'],
            'dataset_valid': all([
                validation_results['dataset_dir_exists'],
                validation_results['train_images_exist'],
                validation_results['val_images_exist']
            ])
        }
        
        return stats
