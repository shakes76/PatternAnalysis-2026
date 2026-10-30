"""Utilities for loading brain-slice image datasets."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 240
BATCH_SIZE = 16
DATA_DIR = Path("/home/groups/comp3710/ADNI/AD_NC")
CLASS_TO_LABEL = {"AD": 0, "NC": 1}


class BrainSliceDataset(Dataset):
    """Index, load, and normalize labeled brain-slice images."""

    def __init__(
        self, split, image_size=(IMAGE_WIDTH, IMAGE_HEIGHT), data_dir=DATA_DIR
    ):
        self.image_size = image_size
        self.data_dir = Path(data_dir)
        split_dir = self.data_dir / split
        self.samples = sorted(
            (path, label)
            for class_name, label in CLASS_TO_LABEL.items()
            for path in (split_dir / class_name).glob("*.jpeg")
        )
        if not self.samples:
            raise FileNotFoundError(f"No JPEG images found under {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")
        if image.size != self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        return image_tensor, label


def load_dataset(split, batch_size=BATCH_SIZE, data_dir=DATA_DIR):
    """Create a data loader for the requested dataset split."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    dataset = BrainSliceDataset(
        split, image_size=(IMAGE_WIDTH, IMAGE_HEIGHT), data_dir=data_dir
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


def load_train_validation(
    batch_size=BATCH_SIZE,
    validation_fraction=0.1,
    data_dir=DATA_DIR,
    seed=42,
):
    """Split training data by subject into training and validation loaders."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    dataset = BrainSliceDataset("train", data_dir=data_dir)
    subject_indices = {}
    for index, (image_path, _) in enumerate(dataset.samples):
        subject_id = image_path.stem.split("_", 1)[0]
        subject_indices.setdefault(subject_id, []).append(index)

    if len(subject_indices) < 2:
        raise ValueError(
            "At least two subjects are required to split the training dataset"
        )

    validation_size = round(len(dataset) * validation_fraction)
    validation_size = max(1, min(validation_size, len(dataset) - 1))
    subject_ids = list(subject_indices)
    generator = torch.Generator().manual_seed(seed)
    shuffled_subject_ids = [
        subject_ids[index]
        for index in torch.randperm(len(subject_ids), generator=generator).tolist()
    ]
    validation_subjects = []
    validation_count = 0
    for subject_id in shuffled_subject_ids[:-1]:
        subject_size = len(subject_indices[subject_id])
        if abs(validation_count + subject_size - validation_size) < abs(
            validation_count - validation_size
        ):
            validation_subjects.append(subject_id)
            validation_count += subject_size

    if not validation_subjects:
        validation_subjects = [
            min(
                shuffled_subject_ids[:-1],
                key=lambda subject_id: abs(
                    len(subject_indices[subject_id]) - validation_size
                ),
            )
        ]

    validation_subject_set = set(validation_subjects)
    validation_indices = [
        index
        for subject_id in validation_subjects
        for index in subject_indices[subject_id]
    ]
    train_indices = [
        index
        for subject_id in subject_ids
        if subject_id not in validation_subject_set
        for index in subject_indices[subject_id]
    ]

    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), batch_size=batch_size, shuffle=False
    )
    return train_loader, validation_loader
