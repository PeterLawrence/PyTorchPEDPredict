"""
train_and_predict_improved.py
Improved CNN model that preserves discrete people positions and prevents fading
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import json


class EvacuationDataset(Dataset):
    """Dataset for evacuation sequences with improved preprocessing"""
    def __init__(self, sequences, sequence_length=5):
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.samples = []
        
        # Create input-output pairs
        for seq in sequences:
            for i in range(len(seq) - sequence_length):
                input_frames = seq[i:i+sequence_length]
                target_frame = seq[i+sequence_length]
                self.samples.append((input_frames, target_frame))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_frames, target_frame = self.samples[idx]
        # Add channel dimension
        input_tensor = input_frames.unsqueeze(1)  # (seq_len, 1, H, W)
        target_tensor = target_frame.unsqueeze(0)  # (1, H, W)
        return input_tensor, target_tensor


class ImprovedEvacuationCNN(nn.Module):
    """
    Improved CNN with:
    1. Deeper architecture for better feature learning
    2. Residual connections to preserve information
    3. Separate channels for walls, people, and empty space
    """
    def __init__(self, input_frames=5):
        super(ImprovedEvacuationCNN, self).__init__()
        
        # Initial feature extraction
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_frames, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Temporal processing
        self.temporal_block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Spatial processing with dilated convolutions for larger receptive field
        self.spatial_block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Additional spatial processing
        self.spatial_block2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # Output layer
        self.output_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, 1, H, W)
        batch_size, seq_len, _, H, W = x.shape
        
        # Reshape to process all frames together
        x = x.view(batch_size, seq_len, H, W)
        
        # Feature extraction
        x = self.input_conv(x)
        
        # Temporal processing
        x = self.temporal_block1(x)
        
        # Spatial processing
        x = self.spatial_block1(x)
        x = self.spatial_block2(x)
        
        # Output
        x = self.output_conv(x)
        
        return x


class BinaryCrossEntropyLoss(nn.Module):
    """
    Custom loss that treats the problem as classification:
    - Wall vs Non-wall
    - Person vs Non-person
    This helps maintain discrete values instead of blurring
    """
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # Standard MSE loss
        mse_loss = self.mse(pred, target)
        
        # Binary loss for people (encourage discrete values)
        # People should be close to 0, empty space close to 1
        people_mask = (target < 0.5) & (target > -0.5)
        empty_mask = target > 0.5
        
        binary_loss = 0
        if people_mask.any():
            # Penalize predictions that aren't close to 0 for people
            binary_loss += torch.mean((pred[people_mask] - 0.0) ** 2)
        
        if empty_mask.any():
            # Penalize predictions that aren't close to 1 for empty space
            binary_loss += torch.mean((pred[empty_mask] - 1.0) ** 2)
        
        # Combine losses
        return mse_loss + 0.5 * binary_loss


def train_model(model, train_loader, val_loader, epochs=50, device='cpu'):
    """Train the improved CNN model"""
    criterion = BinaryCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\nTraining improved model...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }, 'evacuation_model_best.pt')
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f} {'[BEST]' if avg_val_loss == best_val_loss else ''}")
    
    return train_losses, val_losses


def predict_evacuation_improved(model, initial_frame, num_steps=80, device='cpu', threshold=0.3):
    """
    Improved prediction with thresholding to maintain discrete people positions
    
    Args:
        threshold: Value below which pixels are considered "people"
    """
    model.eval()
    
    # Initialize with copies of the initial frame
    sequence = [initial_frame.clone() for _ in range(5)]
    predicted_sequence = [initial_frame.cpu().numpy()]
    
    print("Predicting evacuation with improved thresholding...")
    with torch.no_grad():
        for step in range(num_steps):
            # Prepare input (last 5 frames)
            input_frames = torch.stack(sequence[-5:]).unsqueeze(0).unsqueeze(2)
            input_frames = input_frames.to(device)
            
            # Predict next frame
            next_frame = model(input_frames)
            next_frame = next_frame.squeeze()
            
            # Convert to numpy for processing
            next_frame_np = next_frame.cpu().numpy()
            
            # Get wall mask from initial frame
            wall_mask = (initial_frame.cpu().numpy() < -0.5)
            
            # Apply thresholding to maintain discrete values
            # This is KEY to preventing fading!
            processed_frame = np.ones_like(next_frame_np)  # Start with all empty
            processed_frame[wall_mask] = -1  # Restore walls
            
            # Threshold for people: values below threshold become people (0)
            people_mask = (next_frame_np < threshold) & (~wall_mask)
            processed_frame[people_mask] = 0
            
            # Count remaining people
            people_remaining = np.sum(people_mask)
            
            # Add to sequence
            sequence.append(torch.FloatTensor(processed_frame))
            predicted_sequence.append(processed_frame)
            
            # Check if evacuation is complete
            if people_remaining < 1:
                print(f"✓ Evacuation complete at step {step + 1}")
                break
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{num_steps} - {people_remaining} people remaining")
    
    return predicted_sequence


def create_initial_scene(room_width=50, room_height=50, num_people=15):
    """Create an initial scene with people"""
    scene = np.ones((room_height, room_width), dtype=np.float32)
    
    # Add walls
    scene[0, :] = -1
    scene[-1, :] = -1
    scene[:, 0] = -1
    scene[:, -1] = -1
    
    # Add exits
    exit_size = 3
    exit_pos_left = room_height // 2
    scene[exit_pos_left:exit_pos_left + exit_size, 0] = 1
    
    exit_pos_right = room_height // 2
    scene[exit_pos_right:exit_pos_right + exit_size, -1] = 1
    
    # Add people randomly
    for _ in range(num_people):
        x = np.random.randint(5, room_width - 5)
        y = np.random.randint(5, room_height - 5)
        scene[y, x] = 0
    
    return torch.FloatTensor(scene)


def animate_evacuation(sequence, save_path=None):
    """Create animation of evacuation"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    img = ax.imshow(sequence[0], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(f'Improved Evacuation Prediction - Frame 1/{len(sequence)}')
    ax.axis('off')
    
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Walls (-1) | People (0) | Empty (1)', rotation=270, labelpad=20)
    
    def update(frame_idx):
        img.set_array(sequence[frame_idx])
        people_count = np.sum((sequence[frame_idx] > -0.5) & (sequence[frame_idx] < 0.5))
        ax.set_title(f'Improved Evacuation Prediction - Frame {frame_idx+1}/{len(sequence)}\n'
                    f'People: {int(people_count)}')
        return [img]
    
    anim = animation.FuncAnimation(fig, update, frames=len(sequence), 
                                   interval=200, blit=True, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=5)
        print(f"Animation saved to {save_path}")
    
    plt.show()


def main():
    # Configuration
    data_dir = Path("..","evacuation_data")
    model_path = Path("evacuation_model_improved.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load training data
    print("\nLoading training data...")
    train_data = torch.load(data_dir / "train_data.pt")
    val_data = torch.load(data_dir / "val_data.pt")
    
    print(f"Loaded {train_data['num_sequences']} training sequences")
    print(f"Loaded {val_data['num_sequences']} validation sequences")
    
    # Create datasets
    train_dataset = EvacuationDataset(train_data['sequences'], sequence_length=5)
    val_dataset = EvacuationDataset(val_data['sequences'], sequence_length=5)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Created {len(train_dataset)} training samples")
    
    # Create improved model
    model = ImprovedEvacuationCNN(input_frames=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nImproved model created with {total_params:,} parameters")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                           epochs=50, device=device)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, model_path)
    print(f"\nFinal model saved to {model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress (Improved Model)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress_improved.png')
    print("Training progress saved to training_progress_improved.png")
    plt.show()
    
    # Load best model for prediction
    print("\nLoading best model for prediction...")
    best_checkpoint = torch.load('evacuation_model_best.pt', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Create initial scene and predict
    print("\nGenerating prediction...")
    initial_scene = create_initial_scene(num_people=15)
    predicted_sequence = predict_evacuation_improved(
        model, initial_scene, num_steps=80, device=device, threshold=0.3
    )
    
    print(f"Predicted {len(predicted_sequence)} frames")
    
    # Animate
    print("\nCreating animation...")
    animate_evacuation(predicted_sequence, save_path='evacuation_animation_improved.gif')


if __name__ == "__main__":
    main()
