"""
train_and_predict.py
Loads training data, trains a CNN model, and predicts evacuation animations
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
from evacuation_cnn import EvacuationCNN

class EvacuationDataset(Dataset):
    """Dataset for evacuation sequences"""
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


def train_model(model, train_loader, val_loader, epochs=20, device='cpu'):
    """Train the CNN model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    print("\nTraining model...")
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
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    return train_losses, val_losses


def predict_evacuation(model, initial_frame, num_steps=50, device='cpu'):
    """Predict evacuation sequence from initial frame"""
    model.eval()
    
    # Initialize with copies of the initial frame
    sequence = [initial_frame.clone() for _ in range(5)]
    predicted_sequence = [initial_frame.cpu().numpy()]
    
    with torch.no_grad():
        for step in range(num_steps):
            # Prepare input (last 5 frames)
            input_frames = torch.stack(sequence[-5:]).unsqueeze(0).unsqueeze(2)  # (1, 5, 1, H, W)
            input_frames = input_frames.to(device)
            
            # Predict next frame
            next_frame = model(input_frames)
            next_frame = next_frame.squeeze()
            
            # Threshold to keep walls, people, and empty space distinct
            next_frame_np = next_frame.cpu().numpy()
            
            # Preserve walls (-1), threshold people/empty
            wall_mask = (sequence[-1].cpu().numpy() < -0.5)
            next_frame_np[wall_mask] = -1
            
            # Check if evacuation is complete (no more people)
            people_remaining = np.sum((next_frame_np > -0.5) & (next_frame_np < 0.5))
            
            sequence.append(torch.FloatTensor(next_frame_np))
            predicted_sequence.append(next_frame_np)
            
            if people_remaining < 1:
                print(f"Evacuation complete at step {step}")
                break
    
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
    
    # Display first frame
    img = ax.imshow(sequence[0], cmap='gray', vmin=-1, vmax=1)
    ax.set_title('Evacuation Simulation')
    ax.axis('off')
    
    def update(frame_idx):
        img.set_array(sequence[frame_idx])
        ax.set_title(f'Evacuation Simulation - Frame {frame_idx+1}/{len(sequence)}')
        return [img]
    
    anim = animation.FuncAnimation(fig, update, frames=len(sequence), 
                                   interval=200, blit=True, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=5)
        print(f"Animation saved to {save_path}")
    
    plt.show()


def main():
    # Configuration
    data_dir = Path("evacuation_data")
    model_dir = Path("model_data")
    model_path = Path(model_dir,"evacuation_model.pt")
    visualizations_path = Path('visualizations','evacuation_animation.gif')
    training_progres_png = Path(model_dir,"training_progress.png")
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
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"Created {len(train_dataset)} training samples")
    
    # Create model
    model = EvacuationCNN(input_frames=5).to(device)
    print(f"\nModel architecture:\n{model}")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                           epochs=50, device=device)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(training_progres_png)
    print("Training progress saved to training_progress.png")
    plt.show()
    
    # Create initial scene and predict
    print("\nGenerating prediction...")
    initial_scene = create_initial_scene(num_people=15)
    predicted_sequence = predict_evacuation(model, initial_scene, num_steps=60, device=device)
    
    print(f"Predicted {len(predicted_sequence)} frames")
    
    # Animate
    print("\nCreating animation...")
    animate_evacuation(predicted_sequence, save_path=visualizations_path)


if __name__ == "__main__":
    main()
