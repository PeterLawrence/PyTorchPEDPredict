"""
train_exit_aware.py
CNN that explicitly learns to move people toward exits using distance fields
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from scipy.ndimage import label as connected_components, distance_transform_edt
from scipy.spatial.distance import cdist


def create_exit_distance_field(frame):
    """
    Create a distance field showing distance to nearest exit.
    This helps the model understand spatial goals.
    """
    wall_mask = frame < -0.5
    exit_mask = (frame > -0.5) & (wall_mask == False)
    
    # Find exits (non-wall pixels at boundaries)
    exits = np.zeros_like(frame, dtype=bool)
    # Left and right boundaries
    exits[:, 0] = (frame[:, 0] > -0.5)
    exits[:, -1] = (frame[:, -1] > -0.5)
    # Top and bottom boundaries  
    exits[0, :] = (frame[0, :] > -0.5)
    exits[-1, :] = (frame[-1, :] > -0.5)
    
    # Distance transform from exits
    distance = distance_transform_edt(~exits)
    
    # Normalize to [0, 1] range
    if distance.max() > 0:
        distance = distance / distance.max()
    
    return distance.astype(np.float32)


class ExitAwareDataset(Dataset):
    """Dataset that includes exit distance information"""
    def __init__(self, sequences, sequence_length=2):
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.samples = []
        
        for seq in sequences:
            # Create distance field once per sequence (exits don't change)
            distance_field = create_exit_distance_field(seq[0].numpy())
            distance_field_tensor = torch.FloatTensor(distance_field)
            
            for i in range(len(seq) - sequence_length):
                input_frames = seq[i:i+sequence_length]
                target_frame = seq[i+sequence_length]
                self.samples.append((input_frames, target_frame, distance_field_tensor))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_frames, target_frame, distance_field = self.samples[idx]
        
        # Stack: [frame_t-1, frame_t, distance_field]
        # This gives the model 3 channels of information
        input_with_distance = torch.cat([
            input_frames,
            distance_field.unsqueeze(0)
        ], dim=0)
        
        input_tensor = input_with_distance.unsqueeze(1)  # Add channel dim
        target_tensor = target_frame.unsqueeze(0)
        
        return input_tensor, target_tensor


class ExitAwareCNN(nn.Module):
    """
    CNN that processes both current state and exit distance field
    """
    def __init__(self, input_frames=3):  # 2 frames + 1 distance field
        super(ExitAwareCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            # Initial processing
            nn.Conv2d(input_frames, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Deep feature extraction
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Refinement
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Output
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, 1, H, W)
        batch_size, seq_len, _, H, W = x.shape
        x = x.view(batch_size, seq_len, H, W)
        return self.encoder(x)


class DirectionalLoss(nn.Module):
    """
    Loss that encourages movement toward exits
    """
    def __init__(self):
        super(DirectionalLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, prev_frame):
        # Standard MSE
        mse_loss = self.mse(pred, target)
        
        # People count consistency
        non_wall = target > -0.5
        target_people = torch.sum((target > -0.5) & (target < 0.5)).float()
        pred_people = torch.sum((pred > -0.5) & (pred < 0.5)).float()
        
        people_loss = torch.abs(pred_people - target_people) / (target_people + 1.0)
        
        # Encourage discrete values
        if non_wall.any():
            discrete_loss = torch.mean(torch.abs(pred[non_wall] - torch.round(pred[non_wall]))) * 0.3
        else:
            discrete_loss = 0
        
        return mse_loss + people_loss + discrete_loss


def train_model(model, train_loader, val_loader, epochs=40, device='cpu'):
    """Train exit-aware model"""
    criterion = DirectionalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 0
    
    print("\nTraining exit-aware model...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Get previous frame (second to last in input sequence)
            prev_frame = inputs[:, 1, :, :, :]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, prev_frame)
            loss.backward()
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
                prev_frame = inputs[:, 1, :, :, :]
                outputs = model(inputs)
                loss = criterion(outputs, targets, prev_frame)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, 'evacuation_exit_aware_best.pt')
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f} [BEST]")
        else:
            patience += 1
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
            
            if patience >= 8:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    return train_losses, val_losses


def find_people_positions(frame):
    """Extract people positions"""
    people_mask = (frame > -0.5) & (frame < 0.5)
    if not people_mask.any():
        return []
    
    labeled, num_features = connected_components(people_mask)
    positions = []
    for i in range(1, num_features + 1):
        region = labeled == i
        y_coords, x_coords = np.where(region)
        if len(y_coords) > 0:
            positions.append((int(np.mean(x_coords)), int(np.mean(y_coords))))
    
    return positions


def move_toward_exits(current_frame, distance_field, max_movement=1.5):
    """
    Move people along gradient descent of distance field (toward exits)
    """
    current_positions = find_people_positions(current_frame)
    
    if len(current_positions) == 0:
        return current_frame
    
    wall_mask = current_frame < -0.5
    new_frame = np.ones_like(current_frame)
    new_frame[wall_mask] = -1
    
    # Calculate gradient of distance field
    gy, gx = np.gradient(distance_field)
    
    new_positions = []
    for x, y in current_positions:
        # Get gradient direction (negative = toward exits)
        dx = -gx[y, x]
        dy = -gy[y, x]
        
        # Normalize and scale
        distance = np.sqrt(dx**2 + dy**2)
        if distance > 0:
            dx = (dx / distance) * max_movement
            dy = (dy / distance) * max_movement
        
        # Calculate new position
        new_x = int(round(x + dx))
        new_y = int(round(y + dy))
        
        # Bounds check
        new_x = np.clip(new_x, 1, 48)
        new_y = np.clip(new_y, 1, 48)
        
        # Check if reached exit
        at_exit = False
        if new_x <= 1 or new_x >= 48 or new_y <= 1 or new_y >= 48:
            if new_frame[new_y, new_x] != -1:
                at_exit = True
        
        if not at_exit:
            new_positions.append((new_x, new_y))
    
    # Place people
    for x, y in new_positions:
        if 0 < x < 49 and 0 < y < 49:
            new_frame[y, x] = 0
    
    return new_frame


def predict_evacuation(model, initial_frame, num_steps=100, device='cpu'):
    """Predict with exit-aware model"""
    model.eval()
    
    # Create distance field
    distance_field = create_exit_distance_field(initial_frame.cpu().numpy())
    distance_tensor = torch.FloatTensor(distance_field)
    
    current_frame = initial_frame.clone()
    sequence = [current_frame.clone() for _ in range(2)]
    predicted_sequence = [initial_frame.cpu().numpy()]
    
    wall_mask = initial_frame.cpu().numpy() < -0.5
    initial_people = len(find_people_positions(initial_frame.cpu().numpy()))
    print(f"Starting evacuation with {initial_people} people")
    
    with torch.no_grad():
        for step in range(num_steps):
            # Prepare input with distance field
            input_frames = torch.cat([
                torch.stack(sequence[-2:]),
                distance_tensor.unsqueeze(0)
            ], dim=0).unsqueeze(0).unsqueeze(2)
            
            input_frames = input_frames.to(device)
            
            # Get prediction
            prediction = model(input_frames).squeeze().cpu().numpy()
            
            # Use distance field to guide movement
            current_np = current_frame.cpu().numpy()
            new_frame = move_toward_exits(current_np, distance_field, max_movement=1.5)
            
            people_remaining = len(find_people_positions(new_frame))
            
            current_frame = torch.FloatTensor(new_frame)
            sequence.append(current_frame.clone())
            predicted_sequence.append(new_frame)
            
            if people_remaining == 0:
                print(f"✓ Evacuation complete at step {step + 1}")
                break
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{num_steps} - {people_remaining} people remaining")
    
    return predicted_sequence


def create_initial_scene(num_people=12):
    """Create initial scene"""
    scene = np.ones((50, 50), dtype=np.float32)
    
    scene[0, :] = -1
    scene[-1, :] = -1
    scene[:, 0] = -1
    scene[:, -1] = -1
    
    # Exits
    exit_size = 5
    scene[23:28, 0] = 1
    scene[23:28, -1] = 1
    
    # Place people
    people_positions = []
    while len(people_positions) < num_people:
        x = np.random.randint(10, 40)
        y = np.random.randint(10, 40)
        
        too_close = any(np.sqrt((x - px)**2 + (y - py)**2) < 4 
                       for px, py in people_positions)
        
        if not too_close:
            scene[y, x] = 0
            people_positions.append((x, y))
    
    return torch.FloatTensor(scene)


def animate_evacuation(sequence, save_path='evacuation_exit_aware.gif'):
    """Create animation"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    img = ax.imshow(sequence[0], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(f'Exit-Aware Prediction - Frame 1/{len(sequence)}')
    ax.axis('off')
    
    def update(frame_idx):
        img.set_array(sequence[frame_idx])
        people = np.sum((sequence[frame_idx] > -0.5) & (sequence[frame_idx] < 0.5))
        ax.set_title(f'Exit-Aware Prediction - Frame {frame_idx+1}/{len(sequence)}\nPeople: {int(people)}')
        return [img]
    
    anim = animation.FuncAnimation(fig, update, frames=len(sequence), 
                                   interval=200, blit=True, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=5)
        print(f"Animation saved to {save_path}")
    
    plt.show()


def main():
    data_dir = Path("..", "evacuation_data")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nLoading training data...")
    train_data = torch.load(data_dir / "train_data.pt")
    val_data = torch.load(data_dir / "val_data.pt")
    
    train_dataset = ExitAwareDataset(train_data['sequences'])
    val_dataset = ExitAwareDataset(val_data['sequences'])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    
    model = ExitAwareCNN(input_frames=3).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=40, device=device)
    
    torch.save({'model_state_dict': model.state_dict()}, 'evacuation_exit_aware_final.pt')
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress (Exit-Aware)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_exit_aware.png', dpi=150)
    plt.show()
    
    print("\nLoading best model...")
    checkpoint = torch.load('evacuation_exit_aware_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nGenerating prediction...")
    initial_scene = create_initial_scene(num_people=12)
    predicted_sequence = predict_evacuation(model, initial_scene, num_steps=100, device=device)
    
    print(f"\nGenerated {len(predicted_sequence)} frames")
    animate_evacuation(predicted_sequence, save_path='evacuation_exit_aware.gif')


if __name__ == "__main__":
    main()
