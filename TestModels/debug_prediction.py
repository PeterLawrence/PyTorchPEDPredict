"""
debug_prediction.py
Debug script to visualize what the model is predicting at each step
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import label as connected_components
from scipy.spatial.distance import cdist


class GradualMovementCNN(nn.Module):
    """Same model architecture"""
    def __init__(self, input_frames=2):
        super(GradualMovementCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_frames, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        batch_size, seq_len, _, H, W = x.shape
        x = x.view(batch_size, seq_len, H, W)
        return self.conv_layers(x)


def find_people_positions(frame):
    """Extract individual people positions"""
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


def move_people_gradually(current_frame, predicted_frame, wall_mask, max_movement=2.5):
    """Move people gradually with detailed logging"""
    current_positions = find_people_positions(current_frame)
    
    if len(current_positions) == 0:
        return current_frame, []
    
    predicted_positions = find_people_positions(predicted_frame)
    
    new_frame = np.ones_like(current_frame)
    new_frame[wall_mask] = -1
    
    if len(predicted_positions) == 0:
        print("    WARNING: Model predicted no people!")
        for x, y in current_positions:
            new_frame[y, x] = 0
        return new_frame, current_positions
    
    movements = []
    
    if len(predicted_positions) > 0:
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
            
            # Check if at exit
            at_exit = False
            if new_x <= 1 or new_x >= 48 or new_y <= 1 or new_y >= 48:
                if new_frame[new_y, new_x] != -1:
                    at_exit = True
            
            if not at_exit:
                new_positions.append((new_x, new_y))
                movements.append({
                    'from': (curr_x, curr_y),
                    'to': (new_x, new_y),
                    'distance': np.sqrt((new_x - curr_x)**2 + (new_y - curr_y)**2)
                })
        
        for x, y in new_positions:
            if 0 < x < 49 and 0 < y < 49:
                new_frame[y, x] = 0
    
    return new_frame, movements


def debug_prediction_step(model, initial_frame, num_steps=20, device='cpu'):
    """Run prediction with detailed debugging output"""
    model.eval()
    
    current_frame = initial_frame.clone()
    sequence = [current_frame.clone() for _ in range(2)]
    
    wall_mask = initial_frame.cpu().numpy() < -0.5
    
    initial_people = len(find_people_positions(initial_frame.cpu().numpy()))
    print(f"\n{'='*60}")
    print(f"Starting evacuation with {initial_people} people")
    print(f"{'='*60}\n")
    
    with torch.no_grad():
        for step in range(num_steps):
            print(f"Step {step + 1}:")
            
            # Current state
            current_np = current_frame.cpu().numpy()
            current_people = find_people_positions(current_np)
            print(f"  Current people: {len(current_people)}")
            print(f"  Positions: {current_people[:5]}{'...' if len(current_people) > 5 else ''}")
            
            # Prepare input
            input_frames = torch.stack(sequence[-2:]).unsqueeze(0).unsqueeze(2)
            input_frames = input_frames.to(device)
            
            # Get raw prediction
            raw_prediction = model(input_frames).squeeze().cpu().numpy()
            
            # Analyze raw prediction
            pred_min = raw_prediction[~wall_mask].min()
            pred_max = raw_prediction[~wall_mask].max()
            pred_mean = raw_prediction[~wall_mask].mean()
            print(f"  Raw prediction range: [{pred_min:.3f}, {pred_max:.3f}], mean: {pred_mean:.3f}")
            
            # Find predicted people
            predicted_people = find_people_positions(raw_prediction)
            print(f"  Predicted people: {len(predicted_people)}")
            
            # Apply movement
            new_frame, movements = move_people_gradually(current_np, raw_prediction, wall_mask, max_movement=2.5)
            
            # Report movements
            if movements:
                avg_movement = np.mean([m['distance'] for m in movements])
                max_movement = max([m['distance'] for m in movements])
                print(f"  Movement: avg={avg_movement:.2f}, max={max_movement:.2f}")
                
                # Show a few example movements
                for i, m in enumerate(movements[:3]):
                    print(f"    Person {i+1}: {m['from']} -> {m['to']} (dist={m['distance']:.2f})")
            else:
                print(f"  No movements recorded")
            
            new_people = len(find_people_positions(new_frame))
            print(f"  Resulting people: {new_people}")
            print()
            
            # Update state
            current_frame = torch.FloatTensor(new_frame)
            sequence.append(current_frame.clone())
            
            if new_people == 0:
                print("✓ All people evacuated!")
                break


def create_initial_scene(room_width=50, room_height=50, num_people=10):
    """Create initial scene"""
    scene = np.ones((room_height, room_width), dtype=np.float32)
    
    scene[0, :] = -1
    scene[-1, :] = -1
    scene[:, 0] = -1
    scene[:, -1] = -1
    
    exit_size = 5
    exit_left = room_height // 2 - 2
    scene[exit_left:exit_left + exit_size, 0] = 1
    
    exit_right = room_height // 2 - 2
    scene[exit_right:exit_right + exit_size, -1] = 1
    
    people_positions = []
    min_distance = 4
    attempts = 0
    
    while len(people_positions) < num_people and attempts < num_people * 200:
        x = np.random.randint(8, room_width - 8)
        y = np.random.randint(8, room_height - 8)
        
        too_close = any(np.sqrt((x - px)**2 + (y - py)**2) < min_distance 
                       for px, py in people_positions)
        
        if not too_close:
            scene[y, x] = 0
            people_positions.append((x, y))
        
        attempts += 1
    
    print(f"Created scene with {len(people_positions)} people")
    return torch.FloatTensor(scene)


def visualize_single_step(current_frame, predicted_frame, save_path='debug_step.png'):
    """Visualize current frame vs raw prediction"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(current_frame, cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
    ax1.set_title('Current Frame')
    ax1.axis('off')
    
    ax2.imshow(predicted_frame, cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
    ax2.set_title('Raw Prediction')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved debug visualization to {save_path}")
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = Path('evacuation_model_best.pt')
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found!")
        print("Please train the model first.")
        return
    
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = GradualMovementCNN(input_frames=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    
    # Create test scene
    print("\nCreating initial scene...")
    initial_scene = create_initial_scene(num_people=8)
    
    # Run debug prediction
    print("\nRunning debug prediction...")
    debug_prediction_step(model, initial_scene, num_steps=20, device=device)


if __name__ == "__main__":
    main()
