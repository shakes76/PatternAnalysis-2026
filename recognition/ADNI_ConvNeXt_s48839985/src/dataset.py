from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import yaml
import numpy as np
import torch

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

#######################
# DATA CONFIGURATIONS
#######################

YAML_DATA_PATH = Path("/home/Student/s4883998/ADNI_ConvNeXt2.0/config/data.yaml")

with open(YAML_DATA_PATH, "r") as stream:
    data_config = yaml.safe_load(stream)

    ADNI_DIR = Path(data_config["adni-dir"])
    IMAGE_EXTENSION = data_config["image-extension"]

    NUM_CLASSES = data_config["num-classifications"]
    NC_LABEL = data_config["nc-label"]  # Normal Control
    AD_LABEL = data_config["ad-label"]  # Alzheimer's Disease
    NC_NAME = data_config["nc-name"]
    AD_NAME = data_config["ad-name"]

    VAL_SPLIT = data_config["validation-split"]
    RANDOM_STATE = data_config["random-state"]

# Basic slice entry object
# Metadata for single MRI image path
@dataclass(slots=True)
class ADNISliceEntry:
    patient_id: int
    label: int
    img_path: Path


######################
# ADNI DATASET CLASS
######################


class ADNIDataset(Dataset):
    """PyTorch dataset for ADNI MRI slices."""

    def __init__(
        self, entries: list[ADNISliceEntry], transform: Callable | None = None
    ) -> None:
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]

        with Image.open(entry.img_path) as image:
            # Convert to grayscale image (1 channel) in PyTorch usable format
            image = image.convert("L")
            image_tensor = (
                torch.from_numpy(
                    np.array(image, dtype=np.float32, copy=True)
                ).unsqueeze(0)
                / 255.0
            )

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return image_tensor, entry.label


###########################
# DATA HANDLING UTILITIES
###########################


def _patient_id_from_path(image_path: Path) -> int:
    """
    Extract patient ID from MRI image filename / path
    e.g. ~/.../.../396531_78.jpeg -> 396531
    """

    return int(image_path.stem.split("_")[0])


def slice_entry_idx_from_path(image_path: Path) -> int:
    """
    Extract the slice index of ADNI slice entry
    e.g. ~/.../.../396531_78.jpeg -> 78
    """
    return int(image_path.stem.split("_")[1])


def _collect_ADNI_slice_entries(root: Path, split: str) -> list[ADNISliceEntry]:
    """
    Collect ADNI slice entries from the AD and NC directories

    Args:
        root  (Path) : Root directory of the dataset
        split (str)  : Dataset split ("train" or "test")

    Returns:
        List of ADNISliceEntry objects for the specified split
    """

    entries: list[ADNISliceEntry] = []

    # Directories
    ad_path = root / split / AD_NAME
    nc_path = root / split / NC_NAME

    for label, path in ((AD_LABEL, ad_path), (NC_LABEL, nc_path)):
        if not path.is_dir():
            raise ValueError(f"Missing directory: {path}")

        # Register all images found in directory
        for img_path in path.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() != IMAGE_EXTENSION:
                continue

            patient_id = _patient_id_from_path(img_path)
            entry = ADNISliceEntry(patient_id, label, img_path)
            entries.append(entry)

    if not entries:
        raise ValueError(f"No entries found in {root / split}")

    return entries


def get_patient_labels(entries: list[ADNISliceEntry]) -> dict[int, int]:
    """
    Map patients to their diagnosis, checking for conflicting diagnoses

    Args:
        entries (list[ADNISliceEntry]) : List of slice entries to check

    Returns:
        Mapping between patient ids to their corresponding label
    """

    patient_labels: dict[int, int] = {}

    for entry in entries:
        patient_id = entry.patient_id
        patient_label = entry.label

        prev = patient_labels.setdefault(patient_id, patient_label)
        if prev != patient_label:
            raise ValueError(f"Patient {patient_id} has conflicting diagnoses")

    return patient_labels


##########################
# DATASETS / DATALOADERS
##########################


def generate_datasets(
    root: Path,
    validation_split: float,
    random_state: int,
    train_transforms: Callable | None = None,
    eval_transforms: Callable | None = None,
) -> tuple[ADNIDataset, ADNIDataset | None, ADNIDataset]:
    """
    Generate PyTorch ADNIDataset objects for train, validation (if needed),
    and test data

    Args:
        root : Root directory of the dataset
        validation_split : %'age of train data to use in validation set
        random_state : seed for random generators
        train_transforms : transformers to run on train datasets
        eval_transforms : transforms to run on evaluation / test datasets

    Returns a tuple of train, validation (NULL if 0.0 split), and test datasets
    """

    if not 0 <= validation_split < 1:
        raise ValueError("validation_split must be in [0, 1).")

    train_entries = _collect_ADNI_slice_entries(root, "train")
    test_entries = _collect_ADNI_slice_entries(root, "test")

    # Map patient id's to their diagnosis
    train_labels = get_patient_labels(train_entries)
    test_labels = get_patient_labels(test_entries)

    # Check overlap between train and test patients
    overlap = train_labels.keys() & test_labels.keys()
    if overlap:
        raise ValueError(f"Patient overlap found between train and test: {overlap}")

    # Patient info veriefed!!
    # Can now send datasets (Hip Hip Hooray)

    if validation_split == 0.0:
        return (
            ADNIDataset(train_entries, train_transforms),
            None,
            ADNIDataset(test_entries, eval_transforms),
        )

    # Partition train data into train and validation sets
    patient_ids = np.array(sorted(train_labels))
    labels = np.array([train_labels[pid] for pid in patient_ids])

    # Splitting up list of unique patient IDs
    train_ids, val_ids = train_test_split(
        patient_ids,
        test_size=validation_split,
        random_state=random_state,
        stratify=labels,
    )

    train_ids = set(train_ids)
    val_ids = set(val_ids)

    # Assign existing slice entries according to their patient's split
    train_slice_entries = [e for e in train_entries if e.patient_id in train_ids]
    val_slice_entries = [e for e in train_entries if e.patient_id in val_ids]

    return (
        ADNIDataset(train_slice_entries, train_transforms),
        ADNIDataset(val_slice_entries, eval_transforms),
        ADNIDataset(test_entries, eval_transforms),
    )


def generate_dataloaders(train_ds, val_ds, test_ds, batch_size):
    """Create training and validation dataloaders"""

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    if val_ds is None:
        return train_loader, None, test_loader

    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader
