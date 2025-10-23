"""
evacuation_cnn.py
pyTorch CNN class
"""
import torch
import torch.nn as nn

class EvacuationCNN(nn.Module):
    """CNN model for predicting next evacuation frame"""
    def __init__(self, input_frames=5):
        super(EvacuationCNN, self).__init__()
        
        # Process temporal sequence
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(input_frames, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Spatial processing
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, 1, H, W)
        batch_size, seq_len, _, H, W = x.shape
        
        # Reshape to process all frames together
        x = x.view(batch_size, seq_len, H, W)
        
        # Temporal convolution
        x = self.temporal_conv(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)
        
        return x
