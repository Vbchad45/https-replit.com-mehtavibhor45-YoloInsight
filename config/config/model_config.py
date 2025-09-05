import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """Configuration class for YOLOv8 models."""
    
    model_size: str = 's'  # n, s, m, l, x
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    device: str = 'auto'
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    pose: float = 12.0
    kobj: float = 2.0
    label_smoothing: float = 0.0
    nbs: int = 64
    overlap_mask: bool = True
    mask_ratio: int = 4
    dropout: float = 0.0
    save_period: int = -1
    
    def __post_init__(self):
        """Initialize model-specific configurations."""
        # Set device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model size configurations
        self.model_configs = {
            'n': {'depth_multiple': 0.33, 'width_multiple': 0.25},  # YOLOv8n
            's': {'depth_multiple': 0.33, 'width_multiple': 0.50},  # YOLOv8s
            'm': {'depth_multiple': 0.67, 'width_multiple': 0.75},  # YOLOv8m
            'l': {'depth_multiple': 1.0, 'width_multiple': 1.0},   # YOLOv8l
            'x': {'depth_multiple': 1.0, 'width_multiple': 1.25},  # YOLOv8x
        }
        
        if self.model_size not in self.model_configs:
            raise ValueError(f"Unsupported model size: {self.model_size}. Choose from {list(self.model_configs.keys())}")
        
        # Adjust batch size based on available memory and model size
        if self.device == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            self._adjust_batch_size_for_gpu(gpu_memory)
    
    def _adjust_batch_size_for_gpu(self, gpu_memory_gb: float):
        """Adjust batch size based on GPU memory and model size."""
        # Base memory requirements (approximate)
        memory_requirements = {
            'n': 2,  # GB
            's': 4,
            'm': 6,
            'l': 8,
            'x': 12
        }
        
        required_memory = memory_requirements.get(self.model_size, 4)
        
        # If user didn't specify batch size, calculate optimal
        if hasattr(self, '_batch_size_auto'):
            if gpu_memory_gb < required_memory:
                self.batch_size = max(1, int(self.batch_size * (gpu_memory_gb / required_memory)))
                print(f"Adjusted batch size to {self.batch_size} based on GPU memory")
    
    def get_model_name(self) -> str:
        """Get the YOLOv8 model name."""
        return f'yolov8{self.model_size}.pt'
    
    def get_training_args(self) -> Dict[str, Any]:
        """Get training arguments for YOLOv8."""
        return {
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.img_size,
            'device': self.device,
            'lr0': self.lr0,
            'lrf': self.lrf,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'warmup_momentum': self.warmup_momentum,
            'warmup_bias_lr': self.warmup_bias_lr,
            'box': self.box,
            'cls': self.cls,
            'dfl': self.dfl,
            'pose': self.pose,
            'kobj': self.kobj,
            'label_smoothing': self.label_smoothing,
            'nbs': self.nbs,
            'overlap_mask': self.overlap_mask,
            'mask_ratio': self.mask_ratio,
            'dropout': self.dropout,
            'save_period': self.save_period,
            'save': True,
            'verbose': True,
            'plots': True
        }
    
    def get_validation_args(self) -> Dict[str, Any]:
        """Get validation arguments for YOLOv8."""
        return {
            'imgsz': self.img_size,
            'device': self.device,
            'batch': self.batch_size,
            'verbose': True,
            'plots': True
        }
    
    def get_inference_args(self) -> Dict[str, Any]:
        """Get inference arguments for YOLOv8."""
        return {
            'imgsz': self.img_size,
            'device': self.device,
            'conf': 0.25,  # confidence threshold
            'iou': 0.45,   # IoU threshold for NMS
            'max_det': 1000,  # maximum detections per image
            'verbose': False,
            'save': True,
            'save_txt': True,
            'save_conf': True
        }
    
    def estimate_training_time(self, num_images: int) -> float:
        """Estimate training time in hours."""
        # Base time per image per epoch (in seconds)
        base_time_per_image = {
            'n': 0.001,
            's': 0.002,
            'm': 0.004,
            'l': 0.008,
            'x': 0.016
        }
        
        time_per_image = base_time_per_image.get(self.model_size, 0.002)
        
        # Adjust for device
        if self.device == 'cpu':
            time_per_image *= 10  # CPU is much slower
        
        # Adjust for batch size
        time_per_batch = time_per_image * self.batch_size
        batches_per_epoch = num_images / self.batch_size
        time_per_epoch = time_per_batch * batches_per_epoch
        total_time_hours = (time_per_epoch * self.epochs) / 3600
        
        return total_time_hours
    
    def get_memory_requirements(self) -> Dict[str, float]:
        """Get estimated memory requirements."""
        # Base memory requirements in GB
        base_memory = {
            'n': 1.5,
            's': 3.0,
            'm': 5.0,
            'l': 7.0,
            'x': 10.0
        }
        
        model_memory = base_memory.get(self.model_size, 3.0)
        
        # Batch size scaling
        batch_memory = model_memory * (self.batch_size / 16)
        
        # Image size scaling
        img_memory_scale = (self.img_size / 640) ** 2
        total_memory = batch_memory * img_memory_scale
        
        return {
            'model_memory_gb': model_memory,
            'batch_memory_gb': batch_memory,
            'total_estimated_gb': total_memory,
            'recommended_gpu_memory_gb': total_memory * 1.5  # Add buffer
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate the configuration."""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check device availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            validation_results['errors'].append("CUDA device specified but not available")
            validation_results['valid'] = False
        
        # Check memory requirements
        memory_reqs = self.get_memory_requirements()
        if self.device == 'cuda':
            available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if memory_reqs['total_estimated_gb'] > available_memory:
                validation_results['warnings'].append(
                    f"Estimated memory requirement ({memory_reqs['total_estimated_gb']:.1f}GB) "
                    f"exceeds available GPU memory ({available_memory:.1f}GB)"
                )
        
        # Check batch size
        if self.batch_size < 1:
            validation_results['errors'].append("Batch size must be at least 1")
            validation_results['valid'] = False
        
        # Check image size
        if self.img_size < 32 or self.img_size > 1920:
            validation_results['warnings'].append("Image size should typically be between 32 and 1920")
        
        return validation_results
