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
from pathlib import Path
from scipy.ndimage import label as connected_components
from evacuation_cnn import GradualMovementCNN, MovementConstrainedLoss

class EvacuationDataset(Dataset):
    """Dataset that emphasizes gradual movement patterns"""
    def __init__(self, sequences, sequence_length=2):
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.samples = []
        
        # Create consecutive frame pairs (t -> t+1)
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


def train_model(model, model_path, train_loader, val_loader, epochs=30, device='cpu'):
    """Train with movement-constrained loss"""
    criterion = MovementConstrainedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=4)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nTraining gradual movement model...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            prev_frame = inputs[:, -1, :, :, :]  # Last input frame
            
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
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }, model_path)
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f} [BEST]")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
        
        if patience_counter >= 7:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    return train_losses, val_losses


def main():
    # Configuration
    data_dir = Path("evacuation_data")
    model_dir = Path("model_data")
    model_path = Path(model_dir,"evacuation_model.pt")
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
    train_dataset = EvacuationDataset(train_data['sequences'], sequence_length=2)
    val_dataset = EvacuationDataset(val_data['sequences'], sequence_length=2)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"Created {len(train_dataset)} training samples")
    
    model = GradualMovementCNN(input_frames=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_losses, val_losses = train_model(model, model_path, train_loader,
                                           val_loader, epochs=30, device=device)
    
    torch.save({'model_state_dict': model.state_dict()}, model_path)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(training_progres_png, dpi=150)
    plt.show()
    
    
if __name__ == "__main__":
    main()
