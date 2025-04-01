


from dataclasses import dataclass
from typing import Tuple, Optional
import yaml
from pathlib import Path

@dataclass
class TrainingConfig:
    # Model parameters
    num_classes: int = 500
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    
    # Data parameters
    batch_size: int = 32
    num_workers: int = 4
    max_frames: int = 29
    image_size: Tuple[int, int] = (88, 88)
    
    # Training parameters
    epochs: int = 100
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    warmup_steps: int = 4000
    grad_clip: float = 5.0
    
    # Paths
    data_dir: str = "Dataset/Processed"
    checkpoint_dir: str = "checkpoints"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def save_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f)