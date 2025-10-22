# PyTorch PED Predict

Experimental code related to predicting bitmap outline evacuations 

### generate_training_data.py

- Generates synthetic evacuation sequences with realistic movement patterns
- People move toward nearest exits with some randomness
- Creates 50 training sequences and 10 validation sequences
- Saves data as PyTorch .pt files in evacuation_data/ directory
- Encoding: -1 = wall, 0 = person (black pixel), 1 = empty space (white)

### train_and_predict.py

- Loads the training data from .pt files
- Creates a CNN with temporal and spatial convolutions using conv2d
- Trains the model to predict the next frame given the previous 5 frames
- After training, generates a prediction from an initial scene
- Creates an animated GIF showing the evacuation

### predict_evacuation.py

- Loads trained model from evacuation_model.pt
- Generates random initial scenes with configurable number of people
- Predicts complete evacuation sequence using the CNN
- Saves animated GIF of the evacuation
- Creates comparison image showing initial vs final state

### visualize_training_data.py
- Loads training data from train_data.pt and metadata.json
- Prints detailed statistics about the dataset
- Creates animated GIFs of individual evacuation sequences
- Generates grid visualization showing initial/final frames
- Creates statistical plots analyzing the training data