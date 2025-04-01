import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Union
import random

class VideoTransform:
    """Base class for video transformations."""
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms: List[VideoTransform]):
        self.transforms = transforms

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            frames = transform(frames)
        return frames

class RandomHorizontalFlip(VideoTransform):
    """Randomly flip video frames horizontally."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return np.flip(frames, axis=2)
        return frames

class RandomRotation(VideoTransform):
    """Randomly rotate video frames."""
    def __init__(self, degrees: Union[float, Tuple[float, float]], p: float = 0.5):
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.p = p

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            transformed_frames = []
            
            for frame in frames:
                height, width = frame.shape[:2]
                center = (width/2, height/2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
                transformed_frames.append(rotated_frame)
                
            return np.stack(transformed_frames)
        return frames

class Normalize(VideoTransform):
    """Normalize video frames."""
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        return (frames - self.mean) / self.std

class Resize(VideoTransform):
    """Resize video frames."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        transformed_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, self.size)
            transformed_frames.append(resized_frame)
        return np.stack(transformed_frames)

class CenterCrop(VideoTransform):
    """Center crop video frames."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        h, w = frames.shape[1:3]
        th, tw = self.size
        
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        
        return frames[:, i:i+th, j:j+tw]

class RandomCrop(VideoTransform):
    """Random crop video frames."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        h, w = frames.shape[1:3]
        th, tw = self.size
        
        i = random.randint(0, h - th) if h > th else 0
        j = random.randint(0, w - tw) if w > tw else 0
        
        return frames[:, i:i+th, j:j+tw]

class ColorJitter(VideoTransform):
    """Randomly change brightness, contrast, and saturation."""
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, 
                 saturation: float = 0.2, p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            transformed_frames = []
            
            for frame in frames:
                # Convert to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                
                # Adjust brightness (V)
                v = hsv[:,:,2].astype(np.float32)
                v *= random.uniform(1-self.brightness, 1+self.brightness)
                v = np.clip(v, 0, 255).astype(np.uint8)
                hsv[:,:,2] = v
                
                # Adjust saturation (S)
                s = hsv[:,:,1].astype(np.float32)
                s *= random.uniform(1-self.saturation, 1+self.saturation)
                s = np.clip(s, 0, 255).astype(np.uint8)
                hsv[:,:,1] = s
                
                # Convert back to RGB
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
                # Adjust contrast
                frame = frame.astype(np.float32)
                frame *= random.uniform(1-self.contrast, 1+self.contrast)
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                
                transformed_frames.append(frame)
                
            return np.stack(transformed_frames)
        return frames

def get_training_transforms(image_size: Tuple[int, int]) -> Compose:
    """Get transforms for training."""
    return Compose([
        Resize(image_size),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=10, p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_validation_transforms(image_size: Tuple[int, int]) -> Compose:
    """Get transforms for validation/testing."""
    return Compose([
        Resize(image_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])