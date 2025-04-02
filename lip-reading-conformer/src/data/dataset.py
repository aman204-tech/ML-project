import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path

class LRWDataset(Dataset):
    """Lip Reading in the Wild (LRW) dataset."""
    
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 transform=None,
                 max_frames: int = 29,
                 image_size: Tuple[int, int] = (88, 88)):
        """
        Initialize LRW dataset.
        
        Args:
            root_dir: Path to LRW dataset
            split: One of ['train', 'val', 'test']
            transform: Optional transforms to apply
            max_frames: Maximum number of frames to use
            image_size: Size to resize frames to (width, height)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.max_frames = max_frames
        self.image_size = image_size
        
        # Get word classes (each directory in root_dir is a word)
        self.words = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.num_classes = len(self.words)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Load samples for the specified split
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Number of unique words: {self.num_classes}")

    def _load_samples(self) -> List[Tuple[Path, int, str]]:
        """Load all video paths and their corresponding labels."""
        samples = []
        # Iterate through each word folder
        for word in self.words:
            split_path = self.root_dir / word / self.split
            if not split_path.exists():
                continue

            # Gather video files with common video extensions
            video_files = list(split_path.glob('*.mp4')) + list(split_path.glob('*.mpg'))
            for video_path in video_files:
                label_idx = self.word_to_idx[word]
                samples.append((video_path, label_idx, word))
        return samples

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame."""
        # Convert to grayscale if image is in BGR format
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to the specified image size
        frame = cv2.resize(frame, self.image_size)
        
        # Normalize pixel values to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        return frame

    def _load_video(self, video_path: Path) -> Optional[np.ndarray]:
        """Load video and extract frames."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = self._preprocess_frame(frame)
            frames.append(frame)
            
        cap.release()
        
        if len(frames) == 0:
            return None
            
        # Pad with the last frame if necessary
        while len(frames) < self.max_frames:
            frames.append(frames[-1])
            
        return np.stack(frames)

    def __len__(self) -> int:
        """Return the number of videos in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Get a video and its label."""
        video_path, label_idx, word = self.samples[idx]
        
        frames = self._load_video(video_path)
        if frames is None:
            raise ValueError(f"Failed to load video: {video_path}")
            
        # Apply transforms if provided
        if self.transform:
            frames = self.transform(frames)
        
        # Convert frames to a tensor and add a channel dimension
        frames = torch.FloatTensor(frames)
        frames = frames.unsqueeze(1)  # (frames, 1, height, width)
        
        return frames, label_idx, word

    def get_vocab(self) -> List[str]:
        """Return the list of all words in the dataset."""
        return self.words