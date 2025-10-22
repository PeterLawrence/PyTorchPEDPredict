"""
visualize_training_data.py
Loads training data and metadata, visualizes evacuation sequences as animated GIFs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import json
import argparse


def load_training_data(data_dir='evacuation_data'):
    """
    Load training data and metadata
    
    Args:
        data_dir: Directory containing training data files
    
    Returns:
        Tuple of (training data dict, metadata dict)
    """
    data_path = Path(data_dir)
    
    # Load training data
    train_path = data_path / 'train_data.pt'
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")
    
    print(f"Loading training data from {train_path}...")
    train_data = torch.load(train_path)
    
    # Load metadata
    metadata_path = data_path / 'metadata.json'
    metadata = {}
    if metadata_path.exists():
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        print("Warning: metadata.json not found")
    
    return train_data, metadata


def print_data_summary(train_data, metadata):
    """Print summary of training data"""
    print("\n" + "="*60)
    print("TRAINING DATA SUMMARY")
    print("="*60)
    
    if metadata:
        print(f"Number of sequences: {metadata.get('num_train_sequences', 'N/A')}")
        print(f"Room dimensions: {metadata.get('room_width', 'N/A')} x {metadata.get('room_height', 'N/A')}")
        print(f"Description: {metadata.get('description', 'N/A')}")
    
    print(f"\nActual sequences loaded: {len(train_data['sequences'])}")
    
    # Analyze sequences
    frame_counts = [len(seq) for seq in train_data['sequences']]
    people_counts = []
    
    for seq in train_data['sequences']:
        # Count people in first frame (black pixels, value = 0)
        first_frame = seq[0].numpy()
        num_people = np.sum((first_frame > -0.5) & (first_frame < 0.5))
        people_counts.append(num_people)
    
    print(f"\nSequence statistics:")
    print(f"  Average frames per sequence: {np.mean(frame_counts):.1f}")
    print(f"  Min frames: {np.min(frame_counts)}")
    print(f"  Max frames: {np.max(frame_counts)}")
    print(f"\nPeople statistics:")
    print(f"  Average people per sequence: {np.mean(people_counts):.1f}")
    print(f"  Min people: {np.min(people_counts)}")
    print(f"  Max people: {np.max(people_counts)}")
    print("="*60 + "\n")


def visualize_single_sequence(sequence, sequence_idx, save_path=None, fps=5, dpi=100):
    """
    Create animation for a single evacuation sequence
    
    Args:
        sequence: Torch tensor of shape (num_frames, H, W)
        sequence_idx: Index of the sequence (for labeling)
        save_path: Path to save the GIF
        fps: Frames per second
        dpi: DPI for output
    """
    # Convert to numpy
    if torch.is_tensor(sequence):
        sequence = sequence.numpy()
    
    print(f"Creating animation for sequence {sequence_idx}...")
    print(f"  Frames: {len(sequence)}")
    
    # Count initial people
    first_frame = sequence[0]
    num_people = np.sum((first_frame > -0.5) & (first_frame < 0.5))
    print(f"  Initial people: {int(num_people)}")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display first frame
    img = ax.imshow(sequence[0], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(f'Training Sequence {sequence_idx} - Frame 1/{len(sequence)}\nPeople: {int(num_people)}')
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Walls (-1) | People (0) | Empty (1)', rotation=270, labelpad=20)
    
    def update(frame_idx):
        img.set_array(sequence[frame_idx])
        # Count remaining people
        current_people = np.sum((sequence[frame_idx] > -0.5) & (sequence[frame_idx] < 0.5))
        ax.set_title(f'Training Sequence {sequence_idx} - Frame {frame_idx+1}/{len(sequence)}\n'
                    f'People remaining: {int(current_people)}')
        return [img]
    
    anim = animation.FuncAnimation(fig, update, frames=len(sequence), 
                                   interval=1000//fps, blit=True, repeat=True)
    
    if save_path:
        print(f"  Saving to {save_path}...")
        anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
        print(f"  ✓ Saved successfully!")
    
    plt.close()
    return anim


def create_sequence_grid(sequences, num_sequences=6, save_path='training_grid.png'):
    """
    Create a grid showing first and last frames of multiple sequences
    
    Args:
        sequences: List of sequence tensors
        num_sequences: Number of sequences to show
        save_path: Path to save the grid image
    """
    num_sequences = min(num_sequences, len(sequences))
    
    fig, axes = plt.subplots(num_sequences, 2, figsize=(8, 4*num_sequences))
    if num_sequences == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Training Data: Initial and Final Frames', fontsize=16, y=0.995)
    
    for i in range(num_sequences):
        seq = sequences[i].numpy()
        
        # First frame
        axes[i, 0].imshow(seq[0], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
        num_people = np.sum((seq[0] > -0.5) & (seq[0] < 0.5))
        axes[i, 0].set_title(f'Sequence {i} - Initial ({int(num_people)} people)')
        axes[i, 0].axis('off')
        
        # Last frame
        axes[i, 1].imshow(seq[-1], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
        remaining = np.sum((seq[-1] > -0.5) & (seq[-1] < 0.5))
        axes[i, 1].set_title(f'Sequence {i} - Final ({int(remaining)} remaining)')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Grid visualization saved to {save_path}")
    plt.close()


def create_statistics_plot(train_data, save_path='training_statistics.png'):
    """Create plots showing statistics about the training data"""
    sequences = train_data['sequences']
    
    # Collect statistics
    frame_counts = []
    initial_people = []
    final_people = []
    
    for seq in sequences:
        seq_np = seq.numpy()
        frame_counts.append(len(seq))
        
        # Count people in first and last frames
        initial = np.sum((seq_np[0] > -0.5) & (seq_np[0] < 0.5))
        final = np.sum((seq_np[-1] > -0.5) & (seq_np[-1] < 0.5))
        
        initial_people.append(initial)
        final_people.append(final)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Frame counts histogram
    axes[0, 0].hist(frame_counts, bins=20, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Number of Frames')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Sequence Lengths')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Initial people histogram
    axes[0, 1].hist(initial_people, bins=15, color='forestgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Number of People')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Initial People Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Evacuation success rate
    evacuation_success = [1 if final == 0 else 0 for final in final_people]
    success_rate = np.mean(evacuation_success) * 100
    labels = ['Complete\nEvacuation', 'Partial\nEvacuation']
    sizes = [sum(evacuation_success), len(evacuation_success) - sum(evacuation_success)]
    colors = ['#66b3ff', '#ff9999']
    
    axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 12})
    axes[1, 0].set_title(f'Evacuation Completion Rate\n({success_rate:.1f}% complete)')
    
    # Plot 4: Scatter plot - initial people vs frames
    axes[1, 1].scatter(initial_people, frame_counts, alpha=0.6, color='coral', s=50)
    axes[1, 1].set_xlabel('Initial Number of People')
    axes[1, 1].set_ylabel('Number of Frames')
    axes[1, 1].set_title('Evacuation Duration vs Initial People')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(initial_people, frame_counts, 1)
    p = np.poly1d(z)
    axes[1, 1].plot(sorted(initial_people), p(sorted(initial_people)), 
                    "r--", alpha=0.8, linewidth=2, label='Trend')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Statistics plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize evacuation training data as animated GIFs'
    )
    parser.add_argument('--data-dir', type=str, default='evacuation_data',
                        help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--sequences', type=str, default='0,1,2',
                        help='Comma-separated list of sequence indices to animate (e.g., "0,1,2")')
    parser.add_argument('--all', action='store_true',
                        help='Animate all sequences')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second for animations')
    parser.add_argument('--dpi', type=int, default=100,
                        help='DPI for output images')
    parser.add_argument('--grid', action='store_true',
                        help='Create grid visualization of multiple sequences')
    parser.add_argument('--stats', action='store_true',
                        help='Create statistics plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    try:
        train_data, metadata = load_training_data(args.data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run generate_training_data.py first to create training data.")
        return
    
    # Print summary
    print_data_summary(train_data, metadata)
    
    sequences = train_data['sequences']
    
    # Create grid visualization
    if args.grid:
        print("\nCreating grid visualization...")
        create_sequence_grid(sequences, num_sequences=6, 
                           save_path=output_dir / 'training_grid.png')
    
    # Create statistics plots
    if args.stats:
        print("\nCreating statistics plots...")
        create_statistics_plot(train_data, 
                              save_path=output_dir / 'training_statistics.png')
    
    # Determine which sequences to animate
    if args.all:
        sequence_indices = range(len(sequences))
        print(f"\nAnimating all {len(sequences)} sequences...")
    else:
        try:
            sequence_indices = [int(idx.strip()) for idx in args.sequences.split(',')]
            # Filter valid indices
            sequence_indices = [idx for idx in sequence_indices if 0 <= idx < len(sequences)]
            print(f"\nAnimating sequences: {sequence_indices}")
        except ValueError:
            print("Error: Invalid sequence indices. Using default: [0, 1, 2]")
            sequence_indices = [0, 1, 2]
    
    # Create animations
    print()
    for idx in sequence_indices:
        if idx >= len(sequences):
            print(f"Warning: Sequence {idx} does not exist (only {len(sequences)} sequences available)")
            continue
        
        output_path = output_dir / f'training_sequence_{idx}.gif'
        visualize_single_sequence(
            sequences[idx], 
            idx, 
            save_path=output_path,
            fps=args.fps,
            dpi=args.dpi
        )
        print()
    
    # Summary
    print("="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Animated sequences: {len(sequence_indices)}")
    if args.grid:
        print(f"Grid visualization: training_grid.png")
    if args.stats:
        print(f"Statistics plots: training_statistics.png")
    print("="*60)


if __name__ == "__main__":
    main()
