from torch.utils.data import DataLoader
import torch
from typing import Tuple, Optional, Dict
import torchvision.transforms as transforms
from pathlib import Path
from .dataset import LRWDataset

def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    frames, labels, words = zip(*batch)
    
    # Stack frames and labels
    frames = torch.stack(frames)
    labels = torch.tensor(labels)
    
    return {
        'frames': frames,          # Shape: (batch_size, 1, max_frames, H, W)
        'labels': labels,          # Shape: (batch_size,)
        'words': words            # Tuple of strings
    }

class LRWDataLoader:
    """Data loader for the LRW dataset."""
    
    def __init__(self,
                 root_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 max_frames: int = 29,
                 image_size: Tuple[int, int] = (88, 88)):
        """
        Initialize the data loader.
        
        Args:
            root_dir: Path to LRW dataset
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            max_frames: Maximum number of frames per video
            image_size: Size to resize frames to (height, width)
        """
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_frames = max_frames
        self.image_size = image_size
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), 
                scale=(0.9, 1.1)
            )
        ])

    def get_loaders(self) -> Dict[str, DataLoader]:
        """Returns train, validation and test dataloaders."""
        
        datasets = {}
        loaders = {}
        
        for split in ['train', 'val', 'test']:
            datasets[split] = LRWDataset(
                root_dir=self.root_dir,
                split=split,
                transform=self.transform if split == 'train' else None,
                max_frames=self.max_frames,
                image_size=self.image_size
            )
            
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=self.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=collate_fn
            )
        
        return loaders

    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        # Create temporary dataset to get vocab size
        temp_dataset = LRWDataset(self.root_dir, 'train')
        return temp_dataset.num_classes