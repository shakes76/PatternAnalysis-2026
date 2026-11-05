DATA_PATH = "./archive"
CSV_NAME = "cleaned_metadata.csv"
IMG_DIR = "combined_images"

MODEL_PATH = "./models"
IMAGE_PATH = "./img"

# Splitting data and loading
SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1 # train/validation/testing sets
USE_GROUP_SPLIT = True # Split by patient 

BATCH_TRIPLET = 64                 # Batch size for Siamese training (triplet loss)
BATCH_TEST = 64                    # Batch size for testing
NUM_WORKERS = 8                    # Number of worker threads for data loading

# Image Preprocessing
MEAN = [0.5, 0.5, 0.5]             
STD  = [0.5, 0.5, 0.5]             
IMAGE_SIZE = 256                   # Image resize dimension
ROT_DEG = 15                       # Max rotation degree for data augmentation
FLIP_PROB = 0.5                    # Probability of horizontal/vertical flip
COLOR_JITTER = dict(               #  color jitter parameters
    brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02
)

# Model Settings 
OUT_DIM = 512                      # Output embedding dimension of Siamese encoder
HIDDEN_DIMS = (256, 64)            # Hidden layer dimensions for classifier MLP
RESNET_OUT = 2048                  # Output features of ResNet subnetwork

# Training Hyperparameters
TRIPLET_MARGIN = 1.0               # Margin for triplet loss
EPOCHS_SIAMESE = 25                # Epochs for Siamese encoder training
LR_SIAMESE = 0.001                # Learning rate for Siamese encoder

# Early Stop / Scheduler
PATIENCE = 5                       # Early stopping patience in epochs
MIN_DELTA = 0.001                  # Minimum improvement threshold for validation loss
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 3

# Output Filenames
SAVE_SAMPLE_NAME = "input_sample.png"   # Example image filename
SIAMESE_LOSS_NAME = "siamese_loss.png"  # Siamese loss plot filename
CLS_LOSS_NAME = "classifier_loss.png"   # Classifier loss plot filename
CM_NAME = "confusion_matrix.png"        # Confusion matrix filename