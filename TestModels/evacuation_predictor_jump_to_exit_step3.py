"""
train_and_predict_fixed.py
Fixed CNN model that preserves individual people and prevents clustering/merging
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
from scipy.ndimage import label as connected_components


class EvacuationDataset(Dataset):
    """Dataset for evacuation sequences"""
    def __init__(self, sequences, sequence_length=3):
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.samples = []
        
        # Create input-output pairs with stride
        for seq in sequences:
            # Use stride of 2 to reduce correlation between samples
            for i in range(0, len(seq) - sequence_length, 2):
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


class FlowBasedEvacuationCNN(nn.Module):
    """
    CNN that predicts movement flow field instead of next frame directly.
    This helps preserve individual people and prevent merging.
    """
    def __init__(self, input_frames=3):
        super(FlowBasedEvacuationCNN, self).__init__()
        
        # Encoder - extract features from input sequence
        self.encoder = nn.Sequential(
            nn.Conv2d(input_frames, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 50x50 -> 25x25
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Decoder - upsample back to original size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 25x25 -> 50x50
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, 1, H, W)
        batch_size, seq_len, _, H, W = x.shape
        
        # Reshape to process all frames together
        x = x.view(batch_size, seq_len, H, W)
        
        # Encode
        features = self.encoder(x)
        
        # Decode
        output = self.decoder(features)
        
        return output


class PeoplePenaltyLoss(nn.Module):
    """
    Loss function that:
    1. Standard MSE for overall frame similarity
    2. Penalty for wrong number of people (prevents merging/splitting)
    3. Penalty for clustering (encourages maintaining separation)
    """
    def __init__(self):
        super(PeoplePenaltyLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # Basic MSE loss
        mse_loss = self.mse(pred, target)
        
        # Count people in prediction and target
        pred_people = torch.sum((pred > -0.5) & (pred < 0.5), dim=[1,2,3])
        target_people = torch.sum((target > -0.5) & (target < 0.5), dim=[1,2,3])
        
        # Penalty for wrong number of people
        people_count_loss = torch.mean((pred_people - target_people) ** 2.0) / 100.0
        
        # Penalty for values that are neither 0 (person) nor 1 (empty)
        # This encourages discrete values
        discreteness_loss = 0
        non_wall = target > -0.5
        if non_wall.any():
            pred_non_wall = pred[non_wall]
            # Penalize values between 0 and 1 (should be either/or)
            middle_values = (pred_non_wall > 0.2) & (pred_non_wall < 0.8)
            if middle_values.any():
                discreteness_loss = torch.mean(torch.abs(pred_non_wall[middle_values] - 0.5)) * 0.5
        
        return mse_loss + people_count_loss + discreteness_loss


def train_model(model, train_loader, val_loader, epochs=40, device='cpu'):
    """Train the model with early stopping"""
    criterion = PeoplePenaltyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=3)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 8
    
    print("\nTraining model with early stopping...")
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
            
            # Gradient clipping
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
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }, 'evacuation_model_best.pt')
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f} [BEST - SAVED]")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f} "
                  f"(No improvement for {patience_counter} epochs)")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    return train_losses, val_losses


def predict_evacuation_fixed(model, initial_frame, num_steps=100, device='cpu'):
    """
    Improved prediction that maintains individual people using morphological operations
    """
    model.eval()
    
    # Initialize sequence
    sequence = [initial_frame.clone() for _ in range(3)]
    predicted_sequence = [initial_frame.cpu().numpy()]
    
    # Store initial wall mask
    initial_np = initial_frame.cpu().numpy()
    wall_mask = initial_np < -0.5
    
    # Count initial people
    initial_people = np.sum((initial_np > -0.5) & (initial_np < 0.5))
    print(f"Starting evacuation with {int(initial_people)} people")
    
    with torch.no_grad():
        for step in range(num_steps):
            # Prepare input
            input_frames = torch.stack(sequence[-3:]).unsqueeze(0).unsqueeze(2)
            input_frames = input_frames.to(device)
            
            # Predict
            next_frame = model(input_frames).squeeze().cpu().numpy()
            
            # Post-process to maintain discrete people
            processed_frame = post_process_frame(next_frame, wall_mask, initial_people)
            
            # Count remaining people
            people_remaining = np.sum((processed_frame > -0.5) & (processed_frame < 0.5))
            
            # Add to sequence
            sequence.append(torch.FloatTensor(processed_frame))
            predicted_sequence.append(processed_frame)
            
            if people_remaining < 1:
                print(f"✓ Evacuation complete at step {step + 1}")
                break
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{num_steps} - {int(people_remaining)} people remaining")
    
    return predicted_sequence


def post_process_frame(raw_prediction, wall_mask, expected_people_count):
    """
    Post-process prediction to maintain discrete individual people.
    Uses connected components and morphology to prevent merging.
    """
    # Start with empty room
    processed = np.ones_like(raw_prediction)
    processed[wall_mask] = -1
    
    # Identify potential people locations (below threshold)
    threshold = 0.4
    potential_people = (raw_prediction < threshold) & (~wall_mask)
    
    if not potential_people.any():
        return processed
    
    # Use connected components to identify clusters
    labeled_array, num_features = connected_components(potential_people)
    
    # For each cluster, place a single person at the center of mass
    people_placed = 0
    max_people = int(expected_people_count * 1.2)  # Allow some variation
    
    for region_id in range(1, num_features + 1):
        if people_placed >= max_people:
            break
        
        region_mask = labeled_array == region_id
        region_size = np.sum(region_mask)
        
        # Skip very large regions (likely artifacts)
        if region_size > 15:
            continue
        
        # Find center of mass of this region
        y_coords, x_coords = np.where(region_mask)
        if len(y_coords) > 0:
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
            
            # Place person at center
            processed[center_y, center_x] = 0
            people_placed += 1
    
    return processed


def create_initial_scene(room_width=50, room_height=50, num_people=12):
    """Create an initial scene with people"""
    scene = np.ones((room_height, room_width), dtype=np.float32)
    
    # Add walls
    scene[0, :] = -1
    scene[-1, :] = -1
    scene[:, 0] = -1
    scene[:, -1] = -1
    
    # Add exits (larger for easier evacuation)
    exit_size = 4
    exit_pos_left = room_height // 2 - 1
    scene[exit_pos_left:exit_pos_left + exit_size, 0] = 1
    
    exit_pos_right = room_height // 2 - 1
    scene[exit_pos_right:exit_pos_right + exit_size, -1] = 1
    
    # Add people with minimum spacing to prevent initial overlap
    people_added = 0
    min_distance = 3
    people_positions = []
    
    attempts = 0
    while people_added < num_people and attempts < num_people * 100:
        x = np.random.randint(5, room_width - 5)
        y = np.random.randint(5, room_height - 5)
        
        # Check minimum distance from other people
        too_close = False
        for px, py in people_positions:
            if np.sqrt((x - px)**2 + (y - py)**2) < min_distance:
                too_close = True
                break
        
        if not too_close and scene[y, x] == 1:
            scene[y, x] = 0
            people_positions.append((x, y))
            people_added += 1
        
        attempts += 1
    
    print(f"Created initial scene with {people_added} people")
    return torch.FloatTensor(scene)


def animate_evacuation(sequence, save_path=None):
    """Create animation of evacuation"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    img = ax.imshow(sequence[0], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(f'Fixed Evacuation Prediction - Frame 1/{len(sequence)}')
    ax.axis('off')
    
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Walls (-1) | People (0) | Empty (1)', rotation=270, labelpad=20)
    
    def update(frame_idx):
        img.set_array(sequence[frame_idx])
        people_count = np.sum((sequence[frame_idx] > -0.5) & (sequence[frame_idx] < 0.5))
        ax.set_title(f'Fixed Evacuation Prediction - Frame {frame_idx+1}/{len(sequence)}\n'
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
    model_path = Path("evacuation_model_fixed.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load training data
    print("\nLoading training data...")
    train_data = torch.load(data_dir / "train_data.pt")
    val_data = torch.load(data_dir / "val_data.pt")
    
    print(f"Loaded {train_data['num_sequences']} training sequences")
    print(f"Loaded {val_data['num_sequences']} validation sequences")
    
    # Create datasets (with shorter sequence length)
    train_dataset = EvacuationDataset(train_data['sequences'], sequence_length=3)
    val_dataset = EvacuationDataset(val_data['sequences'], sequence_length=3)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Created {len(train_dataset)} training samples")
    
    # Create model
    model = FlowBasedEvacuationCNN(input_frames=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {total_params:,} parameters")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                           epochs=40, device=device)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, model_path)
    print(f"\nFinal model saved to {model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress (Fixed Model)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_progress_fixed.png', dpi=150, bbox_inches='tight')
    print("Training progress saved to training_progress_fixed.png")
    plt.show()
    
    # Load best model for prediction
    print("\nLoading best model for prediction...")
    best_checkpoint = torch.load('evacuation_model_best.pt', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"Best model from epoch {best_checkpoint['epoch']+1} with val loss {best_checkpoint['best_val_loss']:.6f}")
    
    # Create initial scene and predict
    print("\nGenerating prediction...")
    initial_scene = create_initial_scene(num_people=12)
    predicted_sequence = predict_evacuation_fixed(
        model, initial_scene, num_steps=100, device=device
    )
    
    print(f"\nPredicted {len(predicted_sequence)} frames")
    
    # Animate
    print("\nCreating animation...")
    animate_evacuation(predicted_sequence, save_path='evacuation_animation_fixed.gif')


if __name__ == "__main__":
    main()
