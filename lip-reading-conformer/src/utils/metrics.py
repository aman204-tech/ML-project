import torch
import numpy as np
from typing import List, Tuple, Dict
from jiwer import wer

class LipReadingMetrics:
    """Class for computing various metrics for lip reading model evaluation."""
    
    @staticmethod
    def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute classification accuracy.
        
        Args:
            outputs: Model predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Classification accuracy as percentage
        """
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
        return 100.0 * correct / total
    
    @staticmethod
    def compute_wer(predictions: List[str], targets: List[str]) -> float:
        """
        Compute Word Error Rate between predicted and target sequences.
        
        Args:
            predictions: List of predicted word sequences
            targets: List of target word sequences
            
        Returns:
            Word Error Rate score
        """
        return wer(targets, predictions)
    
    @staticmethod
    def compute_confusion_matrix(outputs: torch.Tensor, 
                                 targets: torch.Tensor, 
                                 num_classes: int) -> np.ndarray:
        """
        Compute confusion matrix for classification results.
        
        Args:
            outputs: Model predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            num_classes: Number of classes
            
        Returns:
            Confusion matrix as numpy array of shape (num_classes, num_classes)
        """
        _, predicted = outputs.max(1)
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for t, p in zip(targets.view(-1), predicted.view(-1)):
            conf_matrix[t.long(), p.long()] += 1
        return conf_matrix.numpy()
    
    @staticmethod
    def compute_per_class_accuracy(confusion_matrix: np.ndarray) -> np.ndarray:
        """
        Compute per-class accuracy from confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix of shape (num_classes, num_classes)
            
        Returns:
            Array of per-class accuracies
        """
        return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    
    @staticmethod
    def compute_top_k_accuracy(outputs: torch.Tensor, 
                               targets: torch.Tensor, 
                               k: int = 5) -> float:
        """
        Compute top-k accuracy.
        
        Args:
            outputs: Model predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy as percentage
        """
        _, pred = outputs.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / targets.size(0)).item()