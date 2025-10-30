"""
predict_evacuation.py
Loads a trained model and generates evacuation predictions from random initial scenes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from scipy.ndimage import label as connected_components
from scipy.spatial.distance import cdist
from evacuation_cnn import GradualMovementCNN


def move_people_gradually(current_frame, predicted_frame, wall_mask, max_movement=2.5):
    """
    Move people from current positions toward predicted positions, 
    but constrain maximum movement distance
    """
    # Find current people positions
    current_positions = find_people_positions(current_frame)
    
    if len(current_positions) == 0:
        return current_frame
    
    # Find predicted people positions
    predicted_positions = find_people_positions(predicted_frame)
    
    # Start with empty frame
    new_frame = np.ones_like(current_frame)
    new_frame[wall_mask] = -1
    
    if len(predicted_positions) == 0:
        # If no predictions, keep people where they are
        for x, y in current_positions:
            new_frame[y, x] = 0
        return new_frame
    
    # Match each current person to nearest predicted position
    if len(predicted_positions) > 0:
        current_array = np.array(current_positions)
        predicted_array = np.array(predicted_positions)
        
        # Calculate distances
        distances = cdist(current_array, predicted_array)
        
        used_predictions = set()
        new_positions = []
        
        for i, (curr_x, curr_y) in enumerate(current_positions):
            # Find nearest predicted position that hasn't been used
            sorted_indices = np.argsort(distances[i])
            
            target_found = False
            for pred_idx in sorted_indices:
                if pred_idx not in used_predictions:
                    pred_x, pred_y = predicted_positions[pred_idx]
                    used_predictions.add(pred_idx)
                    target_found = True
                    break
            
            if not target_found:
                # No available target, try to move toward nearest exit
                if curr_x < 25:
                    pred_x, pred_y = max(0, curr_x - 1), curr_y
                else:
                    pred_x, pred_y = min(49, curr_x + 1), curr_y
            
            # Calculate movement vector
            dx = pred_x - curr_x
            dy = pred_y - curr_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Constrain movement
            if distance > max_movement:
                dx = dx / distance * max_movement
                dy = dy / distance * max_movement
            
            # Calculate new position
            new_x = int(round(curr_x + dx))
            new_y = int(round(curr_y + dy))
            
            # Keep in bounds and not on walls
            new_x = np.clip(new_x, 1, 48)
            new_y = np.clip(new_y, 1, 48)
            
            # Check if person reached exit (at boundary)
            if new_x <= 1 or new_x >= 48 or new_y <= 1 or new_y >= 48:
                # Check if there's an exit here
                if new_frame[new_y, new_x] != -1:
                    # Person exits - don't place them
                    continue
            
            new_positions.append((new_x, new_y))
        
        # Place people at new positions
        for x, y in new_positions:
            if 0 < x < 49 and 0 < y < 49:
                new_frame[y, x] = 0
    
    return new_frame


def find_people_positions(frame):
    """Extract individual people positions from frame"""
    people_mask = (frame > -0.5) & (frame < 0.5)
    
    if not people_mask.any():
        return []
    
    # Use connected components
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


def predict_evacuation_gradual(model, initial_frame, num_steps=100, device='cpu'):
    """
    Predict evacuation with gradual, constrained movement
    """
    model.eval()
    
    # Initialize
    current_frame = initial_frame.clone()
    sequence = [current_frame.clone() for _ in range(2)]
    predicted_sequence = [initial_frame.cpu().numpy()]
    
    wall_mask = initial_frame.cpu().numpy() < -0.5
    
    initial_people = len(find_people_positions(initial_frame.cpu().numpy()))
    print(f"Starting evacuation with {initial_people} people")
    
    consecutive_no_change = 0
    prev_people_count = initial_people
    
    with torch.no_grad():
        for step in range(num_steps):
            # Prepare input
            input_frames = torch.stack(sequence[-2:]).unsqueeze(0).unsqueeze(2)
            input_frames = input_frames.to(device)
            
            # Predict
            prediction = model(input_frames).squeeze().cpu().numpy()
            
            # Apply gradual movement constraint
            current_np = current_frame.cpu().numpy()
            new_frame = move_people_gradually(current_np, prediction, wall_mask, max_movement=2.5)
            
            # Count people
            people_remaining = len(find_people_positions(new_frame))
            
            # Check for stagnation
            if people_remaining == prev_people_count:
                consecutive_no_change += 1
            else:
                consecutive_no_change = 0
            
            prev_people_count = people_remaining
            
            # Add to sequence
            current_frame = torch.FloatTensor(new_frame)
            sequence.append(current_frame.clone())
            predicted_sequence.append(new_frame)
            
            if people_remaining == 0:
                print(f"✓ Evacuation complete at step {step + 1}")
                break
            
            # Detect if stuck
            if consecutive_no_change > 15:
                print(f"⚠ Evacuation appears stuck at step {step + 1} with {people_remaining} people remaining")
                break
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{num_steps} - {people_remaining} people remaining")
    
    return predicted_sequence


def create_initial_scene(room_width=50, room_height=50, num_people=10):
    """Create initial scene with well-spaced people"""
    scene = np.ones((room_height, room_width), dtype=np.float32)
    
    # Walls
    scene[0, :] = -1
    scene[-1, :] = -1
    scene[:, 0] = -1
    scene[:, -1] = -1
    
    # Exits
    exit_size = 5
    exit_left = room_height // 2 - 2
    scene[exit_left:exit_left + exit_size, 0] = 1
    
    exit_right = room_height // 2 - 2
    scene[exit_right:exit_right + exit_size, -1] = 1
    
    # Place people with spacing
    people_positions = []
    min_distance = 4
    attempts = 0
    
    while len(people_positions) < num_people and attempts < num_people * 200:
        x = np.random.randint(8, room_width - 8)
        y = np.random.randint(8, room_height - 8)
        
        # Check spacing
        too_close = any(np.sqrt((x - px)**2 + (y - py)**2) < min_distance 
                       for px, py in people_positions)
        
        if not too_close:
            scene[y, x] = 0
            people_positions.append((x, y))
        
        attempts += 1
    
    print(f"Created scene with {len(people_positions)} people")
    return torch.FloatTensor(scene)


def animate_evacuation(sequence, save_path=None):
    """Create animation"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    img = ax.imshow(sequence[0], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(f'Gradual Movement Prediction - Frame 1/{len(sequence)}')
    ax.axis('off')
    
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Walls (-1) | People (0) | Empty (1)', rotation=270, labelpad=20)
    
    def update(frame_idx):
        img.set_array(sequence[frame_idx])
        people_count = np.sum((sequence[frame_idx] > -0.5) & (sequence[frame_idx] < 0.5))
        ax.set_title(f'Gradual Movement Prediction - Frame {frame_idx+1}/{len(sequence)}\n'
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
    data_dir = Path("evacuation_data")
    model_dir = Path("model_data")
    model_path = Path(model_dir,"evacuation_model.pt")
    visualizations_path = Path('visualizations','evacuation_animation.gif')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = GradualMovementCNN(input_frames=2).to(device)
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nGenerating prediction...")
    initial_scene = create_initial_scene(num_people=10)
    predicted_sequence = predict_evacuation_gradual(model, initial_scene, 
                                                    num_steps=100, device=device)
    
    print(f"\nGenerated {len(predicted_sequence)} frames")
    animate_evacuation(predicted_sequence, save_path=visualizations_path)


if __name__ == "__main__":
    main()
