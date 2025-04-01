import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Optional

from models.conformer import LipReadingConformer
from data.loader import LRWDataLoader
from training.config import TrainingConfig
from utils.metrics import LipReadingMetrics
from torch.optim.lr_scheduler import LambdaLR

class LipReadingTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model
        self.model = LipReadingConformer(
            num_classes=config.num_classes,
            image_size=config.image_size,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout
        ).to(self.device)
        
        # Setup data
        self.setup_data()
        
        # Setup training
        self.setup_training()
        
        # Initialize best metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_data(self):
        """Setup data loaders."""
        data_loader = LRWDataLoader(
            root_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            max_frames=self.config.max_frames
        )
        self.train_loader, self.val_loader, self.test_loader = data_loader.get_loaders()

    def setup_training(self):
        """Setup optimizer and scheduler."""
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self.get_scheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps
        )

    @staticmethod
    def get_scheduler(optimizer, warmup_steps: int):
        """Get learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        return LambdaLR(optimizer, lr_lambda)

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pt')

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (frames, labels, _) in enumerate(pbar):
            frames, labels = frames.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': total_loss/(batch_idx+1), 'acc': 100.*correct/total})

        return total_loss/len(self.train_loader), 100.*correct/total

    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for frames, labels, _ in self.val_loader:
            frames, labels = frames.to(self.device), labels.to(self.device)
            
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return total_loss/len(self.val_loader), 100.*correct/total

    def train(self):
        """Main training loop."""
        for epoch in range(self.config.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            
            self.logger.info(
                f'Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, '
                f'val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%'
            )
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, metrics, is_best)
            
        self.logger.info(f'Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}')

    @torch.no_grad()
    def test(self):
        """Test the model."""
        self.model.eval()
        correct = 0
        total = 0
        
        for frames, labels, _ in self.test_loader:
            frames, labels = frames.to(self.device), labels.to(self.device)
            
            outputs = self.model(frames)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        accuracy = 100.*correct/total
        self.logger.info(f'Test accuracy: {accuracy:.2f}%')
        return accuracy