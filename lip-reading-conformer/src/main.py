import argparse
from pathlib import Path
import torch
import logging
import os
import gdown


from training.config import TrainingConfig
from training.trainer import LipReadingTrainer
from utils.metrics import LipReadingMetrics
from utils.visualizer import LipReadingVisualizer
from utils.helpers import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Lip Reading Training Script')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                      help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='Dataset/Processed',
                      help='Path to processed dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--vis_dir', type=str, default='visualizations',
                      help='Directory to save visualizations')
    parser.add_argument('--seed', type=int, default=19,
                      help='Random seed for reproducibility')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                      help='Run mode')
    parser.add_argument('--dataset_url', type=str, default=None,
                      help='URL to download dataset if not found locally')
    return parser.parse_args()

def setup_logging(log_dir: str = 'logs'):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def download_dataset(url, output):
    if not os.path.exists(output):
        print(f"Dataset not found at {output}. Downloading...")
        gdown.download_folder(url, output=output, quiet=False)
    else:
        print(f"Dataset found at {output}. Skipping download.")

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Download dataset if URL is provided
    if args.dataset_url:
        download_dataset(args.dataset_url, args.data_dir)
    
    # Load config
    config = TrainingConfig.from_yaml(args.config)
    config.data_dir = args.data_dir
    config.checkpoint_dir = args.checkpoint_dir
    
    # Initialize trainer and visualizer
    trainer = LipReadingTrainer(config)
    visualizer = LipReadingVisualizer(save_dir=args.vis_dir)
    
    if args.mode == 'train':
        logger.info("Starting training...")
        trainer.train()
        
        # Plot final training curves
        visualizer.plot_training_history(trainer.get_metrics())
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.test()
        logger.info(f"Test metrics: {test_metrics}")
        
        # Plot confusion matrix
        confusion_matrix = trainer.get_confusion_matrix()
        visualizer.plot_confusion_matrix(
            confusion_matrix,
            class_names=trainer.get_class_names()
        )
        
    else:  # Test mode
        logger.info("Running in test mode...")
        # Load best checkpoint
        trainer.load_checkpoint(Path(args.checkpoint_dir) / 'best.pt')
        
        # Evaluate on test set
        test_metrics = trainer.test()
        logger.info(f"Test metrics: {test_metrics}")

if __name__ == '__main__':
    main()