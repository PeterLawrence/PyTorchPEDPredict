"""
predict_evacuation.py
Loads a trained model and generates evacuation predictions from random initial scenes
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse
import os


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


def create_random_initial_scene(room_width=50, room_height=50, num_people=None,
                                min_people=5, max_people=20):
    """
    Create a random initial scene with people positioned randomly
    
    Args:
        room_width: Width of the room
        room_height: Height of the room
        num_people: Number of people (if None, random between min and max)
        min_people: Minimum number of people if num_people is None
        max_people: Maximum number of people if num_people is None
    """
    scene = np.ones((room_height, room_width), dtype=np.float32)
    
    # Add walls (boundaries)
    scene[0, :] = -1  # Top wall
    scene[-1, :] = -1  # Bottom wall
    scene[:, 0] = -1  # Left wall
    scene[:, -1] = -1  # Right wall
    
    # Add exits (gaps in walls)
    exit_size = 3
    
    # Left exit
    exit_pos_left = np.random.randint(10, room_height - 10)
    scene[exit_pos_left:exit_pos_left + exit_size, 0] = 1
    
    # Right exit
    exit_pos_right = np.random.randint(10, room_height - 10)
    scene[exit_pos_right:exit_pos_right + exit_size, -1] = 1
    
    # Optionally add top/bottom exits
    if np.random.random() > 0.5:
        exit_pos_top = np.random.randint(10, room_width - 10)
        scene[0, exit_pos_top:exit_pos_top + exit_size] = 1
    
    if np.random.random() > 0.5:
        exit_pos_bottom = np.random.randint(10, room_width - 10)
        scene[-1, exit_pos_bottom:exit_pos_bottom + exit_size] = 1
    
    # Determine number of people
    if num_people is None:
        num_people = np.random.randint(min_people, max_people + 1)
    
    # Add people at random positions
    people_added = 0
    attempts = 0
    max_attempts = num_people * 10
    
    while people_added < num_people and attempts < max_attempts:
        x = np.random.randint(3, room_width - 3)
        y = np.random.randint(3, room_height - 3)
        
        # Check if position is free (not a wall or another person)
        if scene[y, x] == 1:
            scene[y, x] = 0  # Place person
            people_added += 1
        
        attempts += 1
    
    print(f"Created initial scene with {people_added} people")
    return torch.FloatTensor(scene), people_added


def predict_evacuation(model, initial_frame, num_steps=60, device='cpu'):
    """
    Predict evacuation sequence from initial frame
    
    Args:
        model: Trained EvacuationCNN model
        initial_frame: Initial room configuration (torch.Tensor)
        num_steps: Maximum number of prediction steps
        device: Device to run predictions on
    
    Returns:
        List of predicted frames as numpy arrays
    """
    model.eval()
    
    # Initialize with copies of the initial frame
    sequence = [initial_frame.clone() for _ in range(5)]
    predicted_sequence = [initial_frame.cpu().numpy()]
    
    print("Predicting evacuation...")
    with torch.no_grad():
        for step in range(num_steps):
            # Prepare input (last 5 frames)
            input_frames = torch.stack(sequence[-5:]).unsqueeze(0).unsqueeze(2)  # (1, 5, 1, H, W)
            input_frames = input_frames.to(device)
            
            # Predict next frame
            next_frame = model(input_frames)
            next_frame = next_frame.squeeze()
            
            # Convert to numpy for processing
            next_frame_np = next_frame.cpu().numpy()
            
            # Preserve walls (-1)
            wall_mask = (sequence[-1].cpu().numpy() < -0.5)
            next_frame_np[wall_mask] = -1
            
            # Count remaining people (values close to 0)
            people_remaining = np.sum((next_frame_np > -0.5) & (next_frame_np < 0.5))
            
            # Add to sequence
            sequence.append(torch.FloatTensor(next_frame_np))
            predicted_sequence.append(next_frame_np)
            
            # Check if evacuation is complete
            if people_remaining < 1:
                print(f"✓ Evacuation complete at step {step + 1}")
                break
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{num_steps} - {people_remaining:.0f} people remaining")
    
    return predicted_sequence


def animate_evacuation(sequence, save_path='evacuation_prediction.gif', fps=5, dpi=100):
    """
    Create and save animation of evacuation
    
    Args:
        sequence: List of frames (numpy arrays)
        save_path: Path to save the GIF
        fps: Frames per second
        dpi: DPI for the output
    """
    print(f"\nCreating animation with {len(sequence)} frames...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display first frame
    img = ax.imshow(sequence[0], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(f'Evacuation Prediction - Frame 1/{len(sequence)}')
    ax.axis('off')
    
    # Add colorbar legend
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Walls (-1) | People (0) | Empty (1)', rotation=270, labelpad=20)
    
    def update(frame_idx):
        img.set_array(sequence[frame_idx])
        ax.set_title(f'Evacuation Prediction - Frame {frame_idx+1}/{len(sequence)}')
        return [img]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(sequence), 
                                   interval=1000//fps, blit=True, repeat=True)
    
    # Save as GIF
    print(f"Saving animation to {save_path}...")
    anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
    print(f"✓ Animation saved successfully!")
    
    plt.close()


def visualize_initial_and_final(initial_frame, final_frame, save_path='comparison.png'):
    """Create a side-by-side comparison of initial and final frames"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(initial_frame, cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
    ax1.set_title('Initial Scene')
    ax1.axis('off')
    
    ax2.imshow(final_frame, cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
    ax2.set_title('Final Scene (Evacuated)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison image saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Predict evacuation from a trained model')
    parser.add_argument('--model-dir', type=str, default='model_data',
                        help='Directory model data')
    parser.add_argument('--model', type=str, default='evacuation_model.pt',
                        help='Path to trained model file')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--output', type=str, default='evacuation_prediction.gif',
                        help='Output GIF filename')
    parser.add_argument('--num-people', type=int, default=None,
                        help='Number of people (random if not specified)')
    parser.add_argument('--min-people', type=int, default=8,
                        help='Minimum number of people if random')
    parser.add_argument('--max-people', type=int, default=25,
                        help='Maximum number of people if random')
    parser.add_argument('--steps', type=int, default=80,
                        help='Maximum prediction steps')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second for output GIF')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Random seed set to {args.seed}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model_path = Path(args.model_dir,args.model)
    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first using train_and_predict.py")
        return

    output_results = Path(args.output_dir,args.output)
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = EvacuationCNN(input_frames=5).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded successfully\n")
    
    # Create random initial scene
    print("Generating random initial scene...")
    initial_scene, num_people = create_random_initial_scene(
        num_people=args.num_people,
        min_people=args.min_people,
        max_people=args.max_people
    )
    print()
    
    # Predict evacuation
    predicted_sequence = predict_evacuation(
        model, 
        initial_scene, 
        num_steps=args.steps, 
        device=device
    )
    
    # Create and save animation
    animate_evacuation(
        predicted_sequence, 
        save_path=output_results,
        fps=args.fps
    )
    
    # Create comparison image
    comparison_path = args.output.replace('.gif', '_comparison.png')
    comparison_path = Path(args.output_dir,comparison_path)
    visualize_initial_and_final(
        predicted_sequence[0],
        predicted_sequence[-1],
        save_path=comparison_path
    )
    
    # Summary
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    print(f"Initial people: {num_people}")
    print(f"Total frames: {len(predicted_sequence)}")
    print(f"Animation saved: {args.output}")
    print(f"Comparison saved: {comparison_path}")
    print("="*50)


if __name__ == "__main__":
    main()
