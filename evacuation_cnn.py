"""
evacuation_cnn.py
pyTorch CNN class
"""
import torch
import torch.nn as nn


class GradualMovementCNN(nn.Module):
    """
    CNN focused on learning small, incremental movements
    """
    def __init__(self, input_frames=2):
        super(GradualMovementCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(input_frames, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Third block
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Fourth block
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Output
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        batch_size, seq_len, _, H, W = x.shape
        x = x.view(batch_size, seq_len, H, W)
        return self.conv_layers(x)


class MovementConstrainedLoss(nn.Module):
    """
    Loss that penalizes large movements between frames
    """
    def __init__(self):
        super(MovementConstrainedLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, prev_frame):
        # Standard MSE
        mse_loss = self.mse(pred, target)
        
        # Calculate movement between previous and predicted
        # People in previous frame
        prev_people_mask = (prev_frame > -0.5) & (prev_frame < 0.5)
        pred_people_mask = (pred > -0.5) & (pred < 0.5)
        
        # Penalty for large jumps in people count
        prev_count = torch.sum(prev_people_mask, dim=[1,2,3]).float()
        pred_count = torch.sum(pred_people_mask, dim=[1,2,3]).float()
        count_diff = torch.abs(prev_count - pred_count)
        
        # Allow small decreases (people leaving), but penalize large changes
        count_penalty = torch.mean(torch.relu(count_diff - 3)) * 0.1
        
        # Encourage discrete values
        non_wall = target > -0.5
        if non_wall.any():
            discrete_penalty = torch.mean(torch.abs(pred[non_wall] - torch.round(pred[non_wall]))) * 0.2
        else:
            discrete_penalty = 0
        
        return mse_loss + count_penalty + discrete_penalty
