import torch
import torch.nn as nn
from .layers import ConformerBlock, PositionalEncoding

class ConformerEncoder(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 input_dim: int = 88*88,  # Size of each frame
                 d_model: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 d_ff: int = 1024,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        # Initial processing of input frames
        self.frame_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, d_ff, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            mask: Optional mask tensor
        """
        # Embed frames
        x = self.frame_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply Conformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        # Global average pooling over sequence length
        x = x.mean(dim=1)
        
        # Classification
        x = self.norm(x)
        x = self.fc(x)
        
        return x

class LipReadingConformer(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 image_size: tuple = (88, 88),
                 d_model: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 d_ff: int = 1024,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        # CNN for initial frame processing
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, image_size[0], image_size[1])
            cnn_output = self.cnn(dummy_input)
            cnn_output_dim = cnn_output.view(1, -1).shape[1]
        
        # Conformer encoder
        self.conformer = ConformerEncoder(
            num_classes=num_classes,
            input_dim=cnn_output_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            kernel_size=kernel_size,
            dropout=dropout
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length, height, width)
        """
        batch_size, channels, seq_len, height, width = x.shape
        
        # Process each frame through CNN
        x = x.view(-1, channels, height, width)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)
        
        # Process through Conformer
        x = self.conformer(x)
        
        return x