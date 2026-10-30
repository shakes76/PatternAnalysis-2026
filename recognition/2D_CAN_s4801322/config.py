
from pathlib import Path
BASE_DIRECETORY = Path(
    r"C:\Users\Riley\Desktop\Uni\COMP3710\PatternAnalysis-2026"
)
DATA_DIRECTORY = Path(
    r"C:\Users\Riley\Desktop\Uni\COMP3710\keras_slices_data"
)

TRAIN_DIRECTORY = DATA_DIRECTORY / "keras_slices_train"
VALIDATION_DIRECTORY = DATA_DIRECTORY / "keras_slices_validate"
TEST_DIRECTORY = DATA_DIRECTORY / "keras_slices_test"

TRAIN_SEG_DIRECTORY = DATA_DIRECTORY / "keras_slices_seg_train"
VALIDATION_SEG_DIRECTORY = DATA_DIRECTORY / "keras_slices_seg_validate"
TEST_SEG_DIRECTORY = DATA_DIRECTORY / "keras_slices_seg_test"

PAD_SIZE = (256, 144)

# Model
IN_CHANNELS = 1
NUM_CLASSES = 6
BASE_CHANNELS = 32
HIDDEN_CHANNELS = 32

# Training
BATCH_SIZE = 16
SEED = 4801322
EPOCHS = 20
LEARNING_RATE = 0.0001
NUM_WORKERS = 0

# Output directories
MODEL_PATH = BASE_DIRECETORY/"recognition"/"2D_CAN_s4801322"/"outputs"/"model.pth" # path for saved model
LOSS_PLOT_PATH = BASE_DIRECETORY/"recognition"/"2D_CAN_s4801322"/"outputs"/"loss_model.png" # path for training loss plot
WORST_CASE_PATH = BASE_DIRECETORY/"recognition"/"2D_CAN_s4801322"/"outputs"/"worst_seg_predicitons"/"model" # path for worst test segmentations