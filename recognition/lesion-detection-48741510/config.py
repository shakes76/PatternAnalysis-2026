"""
Due to the big use of CONSTANTS and PATHS, this file acts as an anchor for all other files for this project.
It contains all of the CONSTANTS utilised throughout, ensuring the code is robust and easy to modify.

Paths are relevant to this file, and paths I personally used for data, sorted data, weights and output are provided. 
If wanting to use different paths, simply replace paths with location of relevant files.
"""
# NOTE: Structuring and formatting aided by Claude Sonnet 5.0 (Medium), particularly use of pathlib --> Path objects
import os
from pathlib import Path # standard Python library, for building paths from a core anchor

# ---------- paths (relative to this file, so they work on any machine) ----------
PROJECT_DIR = Path(__file__).resolve().parent # parent directory for project/download

# Creation of main directories using above-defined function/functionality
RAW_DIR = Path(PROJECT_DIR / "data" / "ISIC2017")  # raw downloaded ISIC2017 folders
DATA_DIR = Path(PROJECT_DIR / "data" / "yolo")    # dataset.py output - raw ISIC data now sorted into splits
WEIGHTS = Path(PROJECT_DIR / "weights" / "best.pt") # weights of currently best trained model - taken following Rangpur training
OUT_DIR = Path(PROJECT_DIR / "outputs") # saved output folder

# ISIC2017 raw data split names - provided by dataset itself
SPLITS = ["train", "val", "test"]
_ISIC_NAMES = {"train": "Training", "val": "Validation", "test": "Test_v2"}
SPLIT_DIRS = {
s: {
        "imgs":  RAW_DIR / f"ISIC-2017_{n}_Data",
        "masks": RAW_DIR / f"ISIC-2017_{n}_Part1_GroundTruth",
        "csv":   RAW_DIR / f"ISIC-2017_{n}_Part3_GroundTruth.csv", # general file names of ISIC2017 download
    }
    for s, n in _ISIC_NAMES.items()
}
# Classes as part of ISIC2017 dataset
CLASS_NAMES = ["melanoma", "seborrheic keratosis", "nevus"]

# For dataset.py - allows for ignoring of small resolution, treated instead as simply noise
MIN_AREA = 50

# For modules.py and train.py - training parameters
PROJECT_NAME = "yolov8" # NOTE: change this name depending on what you want output saved as
BASE_MODEL = "yolov8n.pt" # default pre-trained YOLO model, imported from ultralytics
IMGSZ = 1024
EPOCHS = 50
BATCH = 16
WORKERS = 1 # Rangpur recommendation
SEED = 0 # For repeatability
BOX = 15
DFL = 2.0
CLS = 0.3

# For evaluation.py and predict.py
CONF_EVAL = 0.001   # low, so scores are kept for AUC (ie. how well model distinguises between classes)
CONF_DEMO = 0.25    # display threshold for predict.py
IOU_TARGET = 0.8    # task target: test IoU >= 0.8 for this application
IMG_EXTS = (".jpg", ".jpeg", ".png") # image extensions, ensures other non-relevant files ignored
NO_DET = -1         # no (lesion) detection

# evaluation settings (edit these, then run: python evaluate.py) 
EVAL_SPLIT = "test"      # "val" or "test": choose settings on val, report test once
EVAL_DEVICE = "cpu"      # "cpu", or a GPU index such as 0
RUN_MAP = True           # False skips the slow Ultralytics mAP step # NOTE Claude Sonnet 5.0 (Medium)
EVAL_OUT = OUT_DIR / f"{EVAL_SPLIT}_final"   # follows the split, so val and test outputs can't mix

# NOTE: written by Claude SOnnet 5.0 (Medium)
def write_data_yaml(data_dir=None, path=None):
    """Writes a YOLO data yaml for data_dir (default DATA_DIR), so it can never go stale.
    Written to `path` (default: data_dir/data.yaml). Returns the yaml path as a string."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    path = Path(path) if path else data_dir / "data.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    path.write_text(
        f"path: {data_dir.resolve().as_posix()}\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n{names}\n")
    return str(path)
