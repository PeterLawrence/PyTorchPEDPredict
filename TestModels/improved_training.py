"""
train_improved_v2.py
Improved training with better loss function and data augmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from scipy.ndimage import label as connected_components
from scipy.spatial.distance import cdist


class EvacuationDataset(Dataset):
    """Dataset with improved sampling"""
    def __init__(self, sequences, sequence_length=2):
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.samples = []
        
        for seq in sequences:
            for i in range(len(seq) - sequence_length):
                input_frames = seq[i:i+sequence_length]
                target_frame = seq[i+sequence_length]
                self.samples.append((input_frames, target_frame))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_frames, target_frame = self.samples[idx]
        input_tensor = input_frames.unsqueeze(1)
        target_tensor = target_frame.unsqueeze(0)
        return input_tensor, target_tensor


class ImprovedCNN(nn.Module):
    """Improved CNN with better architecture"""
    def __init__(self, input_frames=2):
        super(ImprovedCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(input_frames, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Block 3
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Block 4
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Output
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, x):
        batch_size, seq_len, _, H, W = x.shape
        x = x.view(batch_size, seq_len, H, W)
        return self.conv_layers(x)


class ImprovedLoss(nn.Module):
    """
    Loss that strongly penalizes wrong people count
    """
    def __init__(self):
        super(ImprovedLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, prev_frame):
        # MSE loss
        mse_loss = self.mse(pred, target)
        
        # Get non-wall regions
        non_wall = target > -0.5
        
        # Count people (values near 0)
        target_people = torch.sum((target > -0.5) & (target < 0.5))
        pred_people = torch.sum((pred > -0.5) & (pred < 0.5))
        prev_people = torch.sum((prev_frame > -0.5) & (prev_frame < 0.5))
        
        # Strong penalty for wrong people count
        # People should decrease or stay same, not increase
        people_diff = torch.abs(pred_people - target_people).float()
        people_loss = (people_diff / (target_people + 1)) * 2.0
        
        # Binary loss - encourage sharp transitions
        if non_wall.any():
            pred_non_wall = pred[non_wall]
            target_non_wall = target[non_wall]
            
            # For people (target=0), pred should be near 0
            # For empty (target=1), pred should be near 1
            binary_loss = torch.mean((pred_non_wall - target_non_wall) ** 2) * 0.5
        else:
            binary_loss = 0
        
        # Focal loss component - focus on hard examples
        focal_weight = torch.abs(pred - target) ** 2
        focal_loss = torch.mean(focal_weight * (pred - target) ** 2) * 0.3
        
        total_loss = mse_loss + people_loss + binary_loss + focal_loss
        
        return total_loss


def train_model(model, train_loader, val_loader, epochs=50, device='cpu'):
    """Train with improved loss"""
    criterion = ImprovedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nTraining improved model...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            prev_frame = inputs[:, -1, :, :, :]
            
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
                prev_frame = inputs[:, -1, :, :, :]
                outputs = model(inputs)
                loss = criterion(outputs, targets, prev_frame)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }, 'evacuation_model_best_v2.pt')
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f} [BEST]")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
        
        if patience_counter >= 10:
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
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
            positions.append((center_x, center_y))
    
    return positions


def move_people_gradually(current_frame, predicted_frame, wall_mask, max_movement=2.0):
    """Move people gradually"""
    current_positions = find_people_positions(current_frame)
    
    if len(current_positions) == 0:
        return current_frame
    
    predicted_positions = find_people_positions(predicted_frame)
    
    new_frame = np.ones_like(current_frame)
    new_frame[wall_mask] = -1
    
    if len(predicted_positions) == 0:
        # Keep people where they are
        for x, y in current_positions:
            new_frame[y, x] = 0
        return new_frame
    
    current_array = np.array(current_positions)
    predicted_array = np.array(predicted_positions)
    distances = cdist(current_array, predicted_array)
    
    used_predictions = set()
    new_positions = []
    
    for i, (curr_x, curr_y) in enumerate(current_positions):
        sorted_indices = np.argsort(distances[i])
        
        target_found = False
        for pred_idx in sorted_indices:
            if pred_idx not in used_predictions:
                pred_x, pred_y = predicted_positions[pred_idx]
                used_predictions.add(pred_idx)
                target_found = True
                break
        
        if not target_found:
            # Move toward nearest exit
            if curr_x < 25:
                pred_x, pred_y = max(0, curr_x - 1), curr_y
            else:
                pred_x, pred_y = min(49, curr_x + 1), curr_y
        
        dx = pred_x - curr_x
        dy = pred_y - curr_y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > max_movement:
            dx = dx / distance * max_movement
            dy = dy / distance * max_movement
        
        new_x = int(round(curr_x + dx))
        new_y = int(round(curr_y + dy))
        
        new_x = np.clip(new_x, 1, 48)
        new_y = np.clip(new_y, 1, 48)
        
        # Check exit
        at_exit = False
        if new_x <= 1 or new_x >= 48 or new_y <= 1 or new_y >= 48:
            if new_frame[new_y, new_x] != -1:
                at_exit = True
        
        if not at_exit:
            new_positions.append((new_x, new_y))
    
    for x, y in new_positions:
        if 0 < x < 49 and 0 < y < 49:
            new_frame[y, x] = 0
    
    return new_frame


def predict_evacuation(model, initial_frame, num_steps=100, device='cpu'):
    """Predict with improved model"""
    model.eval()
    
    current_frame = initial_frame.clone()
    sequence = [current_frame.clone() for _ in range(2)]
    predicted_sequence = [initial_frame.cpu().numpy()]
    
    wall_mask = initial_frame.cpu().numpy() < -0.5
    
    initial_people = len(find_people_positions(initial_frame.cpu().numpy()))
    print(f"Starting evacuation with {initial_people} people")
    
    with torch.no_grad():
        for step in range(num_steps):
            input_frames = torch.stack(sequence[-2:]).unsqueeze(0).unsqueeze(2)
            input_frames = input_frames.to(device)
            
            prediction = model(input_frames).squeeze().cpu().numpy()
            
            current_np = current_frame.cpu().numpy()
            new_frame = move_people_gradually(current_np, prediction, wall_mask, max_movement=2.0)
            
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


def create_initial_scene(num_people=10):
    """Create initial scene"""
    scene = np.ones((50, 50), dtype=np.float32)
    
    scene[0, :] = -1
    scene[-1, :] = -1
    scene[:, 0] = -1
    scene[:, -1] = -1
    
    exit_size = 5
    exit_left = 23
    scene[exit_left:exit_left + exit_size, 0] = 1
    exit_right = 23
    scene[exit_right:exit_right + exit_size, -1] = 1
    
    people_positions = []
    while len(people_positions) < num_people:
        x = np.random.randint(8, 42)
        y = np.random.randint(8, 42)
        
        too_close = any(np.sqrt((x - px)**2 + (y - py)**2) < 4 
                       for px, py in people_positions)
        
        if not too_close:
            scene[y, x] = 0
            people_positions.append((x, y))
    
    return torch.FloatTensor(scene)


def animate_evacuation(sequence, save_path='evacuation_v2.gif'):
    """Create animation"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    img = ax.imshow(sequence[0], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(f'Improved Prediction - Frame 1/{len(sequence)}')
    ax.axis('off')
    
    def update(frame_idx):
        img.set_array(sequence[frame_idx])
        people = np.sum((sequence[frame_idx] > -0.5) & (sequence[frame_idx] < 0.5))
        ax.set_title(f'Improved Prediction - Frame {frame_idx+1}/{len(sequence)}\nPeople: {int(people)}')
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
    
    train_dataset = EvacuationDataset(train_data['sequences'])
    val_dataset = EvacuationDataset(val_data['sequences'])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    
    model = ImprovedCNN(input_frames=2).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, device=device)
    
    torch.save({'model_state_dict': model.state_dict()}, 'evacuation_model_final_v2.pt')
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress (Improved V2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_v2.png', dpi=150)
    plt.show()
    
    print("\nLoading best model...")
    checkpoint = torch.load('evacuation_model_best_v2.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nGenerating prediction...")
    initial_scene = create_initial_scene(num_people=10)
    predicted_sequence = predict_evacuation(model, initial_scene, num_steps=100, device=device)
    
    print(f"\nGenerated {len(predicted_sequence)} frames")
    animate_evacuation(predicted_sequence, save_path='evacuation_improved_v2.gif')


if __name__ == "__main__":
    main()
