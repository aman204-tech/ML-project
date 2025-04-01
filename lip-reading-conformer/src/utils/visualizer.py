import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from datetime import datetime

class LipReadingVisualizer:
    """Class for visualizing training progress and results."""
    
    def __init__(self, save_dir: str = 'visualizations'):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualization outputs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def plot_training_history(self, 
                            metrics: Dict[str, List[float]], 
                            save_name: Optional[str] = None):
        """
        Plot training metrics history.
        
        Args:
            metrics: Dictionary containing lists of metrics
            save_name: Optional filename for saving the plot
        """
        plt.style.use('seaborn')
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot loss
        axes[0].plot(metrics['train_loss'], label='Train', color='blue')
        axes[0].plot(metrics['val_loss'], label='Validation', color='red')
        axes[0].set_title('Loss History')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(metrics['train_acc'], label='Train', color='blue')
        axes[1].plot(metrics['val_acc'], label='Validation', color='red')
        axes[1].set_title('Accuracy History')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_name is None:
            save_name = f'training_history_{self.timestamp}.png'
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self,
                            confusion_matrix: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            save_name: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Optional list of class names
            save_name: Optional filename for saving the plot
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_matrix, 
                   annot=True, 
                   fmt='d',
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_name is None:
            save_name = f'confusion_matrix_{self.timestamp}.png'
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_attention(self,
                          attention_weights: torch.Tensor,
                          save_name: Optional[str] = None):
        """
        Visualize attention weights.
        
        Args:
            attention_weights: Attention weights tensor
            save_name: Optional filename for saving the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights.cpu().numpy(),
                   cmap='viridis',
                   center=0)
        plt.title('Attention Weights')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        if save_name is None:
            save_name = f'attention_weights_{self.timestamp}.png'
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_lip_frames(self,
                       frames: np.ndarray,
                       predictions: Optional[List[str]] = None,
                       save_name: Optional[str] = None):
        """
        Visualize lip frames with optional predictions.
        
        Args:
            frames: Array of video frames
            predictions: Optional list of predicted words
            save_name: Optional filename for saving the plot
        """
        num_frames = min(len(frames), 8)  # Show up to 8 frames
        fig, axes = plt.subplots(1, num_frames, figsize=(20, 4))
        
        for i in range(num_frames):
            axes[i].imshow(frames[i], cmap='gray')
            axes[i].axis('off')
            if predictions:
                axes[i].set_title(f'Frame {i}\n{predictions[i]}')
        
        plt.tight_layout()
        
        if save_name is None:
            save_name = f'lip_frames_{self.timestamp}.png'
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()