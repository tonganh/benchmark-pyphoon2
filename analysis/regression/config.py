import torch

# Training Hyperparameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 4
NUM_WORKERS = 4
MAX_EPOCHS = 50

# Dataset parameters
WEIGHTS = "DEFAULT"
LABELS = 'wind'  # Overwritten if labels argument is given
SPLIT_BY = 'sequence'
LOAD_DATA = 'images'
DATASET_SPLIT = (0.8, 0.2, 0)
STANDARDIZE_RANGE = (170, 300)
DOWNSAMPLE_SIZE = 224  # Overwritten if size argument is given
NUM_CLASSES = 1

# Computation parameters
ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
DEVICES = 1  # Overwritten if device argument is given
DATA_DIR = '/dataset/0/wnp'
