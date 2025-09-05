
import os
import argparse
import yaml
from pathlib import Path
import torch
import logging

from config.dataset_config import DatasetConfig
from config.model_config import ModelConfig
from data.dataset_loader import DatasetLoader
from models.yolo_trainer import YOLOTrainer
from models.yolo_evaluator import YOLOEvaluator
from models.yolo_inference import YOLOInference
from utils.visualization import Visualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection Training')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'inference'], 
                       default='train', help='Mode to run the script in')
    parser.add_argument('--dataset', type=str, choices=['coco', 'pascal_voc'], 
                       default='coco', help='Dataset to use for training')
    parser.add_argument('--model-size', type=str, choices=['n', 's', 'm', 'l', 'x'], 
                       default='s', help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--data-path', type=str, default='./datasets', help='Path to dataset')
    parser.add_argument('--model-path', type=str, help='Path to pretrained model for inference/evaluation')
    parser.add_argument('--output-dir', type=str, default='./runs', help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--inference-source', type=str, help='Source for inference (image path or directory)')
    
    return parser.parse_args()

def setup_directories(output_dir):
    """Create necessary directories."""
    directories = [
        output_dir,
        os.path.join(output_dir, 'train'),
        os.path.join(output_dir, 'evaluate'),
        os.path.join(output_dir, 'inference'),
        os.path.join(output_dir, 'models'),
        os.path.join(output_dir, 'logs')
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    return directories

def main():
    """Main function to orchestrate the YOLOv8 training/evaluation/inference pipeline."""
    args = parse_arguments()
    
    # Setup directories
    setup_directories(args.output_dir)
    
    # Initialize configurations
    dataset_config = DatasetConfig(
        dataset_name=args.dataset,
        data_path=args.data_path,
        img_size=args.img_size
    )
    
    model_config = ModelConfig(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    logger.info(f"Starting YOLOv8 {args.mode} mode")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Device: {model_config.device}")
    
    if args.mode == 'train':
        # Training mode
        logger.info("Initializing training pipeline...")
        
        # Load dataset
        dataset_loader = DatasetLoader(dataset_config)
        dataset_info = dataset_loader.prepare_dataset()
        
        # Initialize trainer
        trainer = YOLOTrainer(model_config, dataset_config, args.output_dir)
        
        # Start training
        model_path = trainer.train(resume=args.resume)
        logger.info(f"Training completed. Model saved at: {model_path}")
        
        # Automatic evaluation after training
        logger.info("Starting automatic evaluation...")
        evaluator = YOLOEvaluator(model_path, dataset_config, args.output_dir)
        metrics = evaluator.evaluate()
        logger.info(f"Evaluation completed. Results: {metrics}")
        
    elif args.mode == 'evaluate':
        # Evaluation mode
        if not args.model_path:
            raise ValueError("Model path is required for evaluation mode")
        
        logger.info("Starting evaluation...")
        evaluator = YOLOEvaluator(args.model_path, dataset_config, args.output_dir)
        metrics = evaluator.evaluate()
        
        # Visualize results
        visualizer = Visualizer(args.output_dir)
        visualizer.plot_evaluation_metrics(metrics)
        
        logger.info(f"Evaluation completed. Results: {metrics}")
        
    elif args.mode == 'inference':
        # Inference mode
        if not args.model_path:
            raise ValueError("Model path is required for inference mode")
        if not args.inference_source:
            raise ValueError("Inference source is required for inference mode")
        
        logger.info("Starting inference...")
        inference = YOLOInference(args.model_path, model_config.device)
        results = inference.predict(args.inference_source, args.output_dir)
        
        logger.info(f"Inference completed. Results saved to: {args.output_dir}/inference")
        
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
