"""
generate_training_data.py
Generates synthetic evacuation training data and saves as PyTorch tensors
"""

import torch
import numpy as np
from pathlib import Path
import json

class EvacuationDataGenerator:
    def __init__(self, room_width=50, room_height=50):
        self.room_width = room_width
        self.room_height = room_height
        
    def create_room_layout(self):
        """Create a room with walls and exits"""
        room = np.ones((self.room_height, self.room_width), dtype=np.float32)
        
        # Walls are at boundaries (kept as 1 = white/empty)
        # We'll use 0 for people and -1 for walls to distinguish
        room[0, :] = -1  # Top wall
        room[-1, :] = -1  # Bottom wall
        room[:, 0] = -1  # Left wall
        room[:, -1] = -1  # Right wall
        
        # Create exits (gaps in walls)
        exit_size = 3
        # Exit on left wall
        exit_pos_left = np.random.randint(10, self.room_height - 10)
        room[exit_pos_left:exit_pos_left + exit_size, 0] = 1
        
        # Exit on right wall
        exit_pos_right = np.random.randint(10, self.room_height - 10)
        room[exit_pos_right:exit_pos_right + exit_size, -1] = 1
        
        return room, [(0, exit_pos_left + exit_size//2), 
                      (self.room_width - 1, exit_pos_right + exit_size//2)]
    
    def initialize_people(self, num_people, exits):
        """Randomly place people in the room"""
        people = []
        for _ in range(num_people):
            x = np.random.randint(5, self.room_width - 5)
            y = np.random.randint(5, self.room_height - 5)
            
            # Assign to nearest exit
            exit_distances = [np.sqrt((x - ex)**2 + (y - ey)**2) for ex, ey in exits]
            nearest_exit = exits[np.argmin(exit_distances)]
            
            people.append({
                'x': float(x),
                'y': float(y),
                'target_exit': nearest_exit,
                'active': True
            })
        return people
    
    def move_people(self, people, room):
        """Move people one step toward their target exit"""
        for person in people:
            if not person['active']:
                continue
                
            tx, ty = person['target_exit']
            x, y = person['x'], person['y']
            
            # Calculate direction
            dx = tx - x
            dy = ty - y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < 1.5:  # Reached exit
                person['active'] = False
                continue
            
            # Normalize direction and add some randomness
            step_size = 0.8 + np.random.random() * 0.4
            dx = (dx / distance) * step_size
            dy = (dy / distance) * step_size
            
            # Add small random deviation
            dx += (np.random.random() - 0.5) * 0.3
            dy += (np.random.random() - 0.5) * 0.3
            
            # Update position
            new_x = x + dx
            new_y = y + dy
            
            # Keep within bounds
            new_x = np.clip(new_x, 1, self.room_width - 2)
            new_y = np.clip(new_y, 1, self.room_height - 2)
            
            person['x'] = new_x
            person['y'] = new_y
    
    def render_frame(self, room_layout, people):
        """Render current state as an image"""
        frame = room_layout.copy()
        
        for person in people:
            if person['active']:
                x = int(round(person['x']))
                y = int(round(person['y']))
                if 0 <= y < self.room_height and 0 <= x < self.room_width:
                    frame[y, x] = 0  # Black pixel for person
        
        return frame
    
    def generate_sequence(self, num_people, max_steps=100):
        """Generate one complete evacuation sequence"""
        room_layout, exits = self.create_room_layout()
        people = self.initialize_people(num_people, exits)
        
        sequence = []
        for step in range(max_steps):
            frame = self.render_frame(room_layout, people)
            sequence.append(frame)
            
            # Move people
            self.move_people(people, room_layout)
            
            # Check if all people have evacuated
            if not any(p['active'] for p in people):
                break
        
        return np.array(sequence)
    
    def generate_dataset(self, num_sequences, min_people=5, max_people=20):
        """Generate multiple evacuation sequences"""
        sequences = []
        
        for i in range(num_sequences):
            num_people = np.random.randint(min_people, max_people + 1)
            sequence = self.generate_sequence(num_people)
            sequences.append(sequence)
            print(f"Generated sequence {i+1}/{num_sequences} with {num_people} people, {len(sequence)} frames")
        
        return sequences


def main():
    # Configuration
    num_train_sequences = 50
    num_val_sequences = 10
    output_dir = Path("evacuation_data")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating evacuation training data...")
    generator = EvacuationDataGenerator(room_width=50, room_height=50)
    
    # Generate training data
    print("\nGenerating training sequences...")
    train_sequences = generator.generate_dataset(num_train_sequences)
    
    # Generate validation data
    print("\nGenerating validation sequences...")
    val_sequences = generator.generate_dataset(num_val_sequences)
    
    # Convert to tensors and save
    print("\nSaving data...")
    
    # Save training data
    train_data = {
        'sequences': [torch.FloatTensor(seq) for seq in train_sequences],
        'num_sequences': len(train_sequences),
        'room_size': (50, 50)
    }
    torch.save(train_data, output_dir / "train_data.pt")
    
    # Save validation data
    val_data = {
        'sequences': [torch.FloatTensor(seq) for seq in val_sequences],
        'num_sequences': len(val_sequences),
        'room_size': (50, 50)
    }
    torch.save(val_data, output_dir / "val_data.pt")
    
    # Save metadata
    metadata = {
        'num_train_sequences': num_train_sequences,
        'num_val_sequences': num_val_sequences,
        'room_width': 50,
        'room_height': 50,
        'description': 'Evacuation simulation data. Values: -1=wall, 0=person, 1=empty space'
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Data saved to {output_dir}/")
    print(f"  - train_data.pt: {num_train_sequences} sequences")
    print(f"  - val_data.pt: {num_val_sequences} sequences")
    print(f"  - metadata.json: dataset information")


if __name__ == "__main__":
    main()
