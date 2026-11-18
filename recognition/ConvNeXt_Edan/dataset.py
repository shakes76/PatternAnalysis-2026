"""
ADNI MRI dataset loading and patient-independent splitting.

Classes:
    NC = 0
    AD = 1

The original dataset's train/test directories are combined because
an audit identified patient overlap between the supplied splits.

All images belonging to the same patient are assigned to exactly
one of the new train, validation, or test partitions.
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DATA_ROOT = Path("/home/groups/comp3710/ADNI")
CLASS_TO_IDX = {"NC": 0, "AD": 1}
SEED = 42


def extract_patient_id(record):
    """Extract an ADNI participant identifier from metadata."""

    raw_path = record["raw"]

    match = re.search(
        r"ADNI_(\d{3}_S_\d{4})",
        raw_path,
    )

    if match is None:
        raise ValueError(
            f"Unable to extract patient ID from {raw_path}"
        )

    return match.group(1)


def collect_images(data_root=DATA_ROOT):
    """
    Collect all JPEG images from the supplied train/test folders.

    Returns a list of records containing:
        path, patient_id, scan_id, label
    """

    data_root = Path(data_root)

    metadata_path = data_root / "meta_data_with_label.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    records = []

    for original_split in ("train", "test"):
        for class_name, label in CLASS_TO_IDX.items():

            image_dir = (
                data_root / "AD_NC" / original_split / class_name
            )

            for image_path in sorted(image_dir.glob("*.jpeg")):

                scan_id = image_path.stem.split("_")[0]

                if scan_id not in metadata:
                    raise KeyError(
                        f"Missing metadata for scan {scan_id}"
                    )

                record = metadata[scan_id]
                patient_id = extract_patient_id(record)

                # Verify that folder labels match metadata labels.
                expected_metadata_label = (
                    2 if class_name == "AD" else 0
                )

                if record["label"] != expected_metadata_label:
                    raise ValueError(
                        f"Label mismatch for {image_path}"
                    )

                records.append({
                    "path": str(image_path),
                    "patient_id": patient_id,
                    "scan_id": scan_id,
                    "label": label,
                })

    return records


def create_patient_splits(records, seed=SEED):
    """
    Create reproducible, stratified patient-level splits.

    Target proportions:
        train: 70%
        val:   15%
        test:  15%
    """

    patient_labels = {}

    for record in records:
        patient_id = record["patient_id"]
        label = record["label"]

        if (
            patient_id in patient_labels
            and patient_labels[patient_id] != label
        ):
            raise ValueError(
                f"Conflicting labels for patient {patient_id}"
            )

        patient_labels[patient_id] = label

    patients = sorted(patient_labels)
    labels = [patient_labels[p] for p in patients]

    # First: 70% train, 30% temporary.
    train_patients, temp_patients = train_test_split(
        patients,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )

    # Second: divide temporary patients equally.
    temp_labels = [patient_labels[p] for p in temp_patients]

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels,
    )

    patient_sets = {
        "train": set(train_patients),
        "val": set(val_patients),
        "test": set(test_patients),
    }

    # Verify zero patient overlap.
    assert not (
        patient_sets["train"] & patient_sets["val"]
    )
    assert not (
        patient_sets["train"] & patient_sets["test"]
    )
    assert not (
        patient_sets["val"] & patient_sets["test"]
    )

    splits = {
        split: [
            record
            for record in records
            if record["patient_id"] in patient_ids
        ]
        for split, patient_ids in patient_sets.items()
    }

    assert sum(len(s) for s in splits.values()) == len(records)

    return splits


def save_splits(splits, output_path="splits.json"):
    """Save a reproducible split manifest."""

    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(input_path="splits.json"):
    """Load a previously generated split manifest."""

    with open(input_path) as f:
        return json.load(f)


def get_transforms(image_size=224, training=False):
    """
    Define preprocessing for grayscale MRI images.

    Images are converted to RGB for compatibility with
    pretrained ConvNeXt architectures.
    """

    operations = [
        transforms.Resize((image_size, image_size)),
    ]

    if training:
        operations.extend([
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
        ])

    operations.extend([
        transforms.ToTensor(),

        # Repeat grayscale images across three RGB channels.
        transforms.Lambda(
            lambda image: image.repeat(3, 1, 1)
            if image.shape[0] == 1
            else image
        ),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(operations)


class ADNIDataset(Dataset):
    """PyTorch dataset for ADNI MRI JPEG images."""

    def __init__(
        self,
        records,
        image_size=224,
        training=False,
    ):
        self.records = records

        self.transform = get_transforms(
            image_size=image_size,
            training=training,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        with Image.open(record["path"]) as image:
            image = image.convert("L")
            image = self.transform(image)

        label = torch.tensor(
            record["label"],
            dtype=torch.long,
        )

        return image, label


def create_dataloaders(
    splits,
    batch_size=32,
    image_size=224,
    num_workers=4,
):
    """Create training, validation and testing DataLoaders."""

    loaders = {}

    for split_name in ("train", "val", "test"):

        dataset = ADNIDataset(
            records=splits[split_name],
            image_size=image_size,
            training=(split_name == "train"),
        )

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


def print_split_summary(splits):
    """Display image, patient and class counts."""

    print("\n=== DATASET SPLIT SUMMARY ===")

    for split_name, records in splits.items():

        patients = {
            record["patient_id"]
            for record in records
        }

        labels = Counter(
            record["label"]
            for record in records
        )

        print(f"\n{split_name.upper()}")
        print("Images:", len(records))
        print("Patients:", len(patients))
        print("NC images:", labels[0])
        print("AD images:", labels[1])


if __name__ == "__main__":

    print("Collecting ADNI images...")

    records = collect_images()

    print("Total images:", len(records))
    print(
        "Unique patients:",
        len({r["patient_id"] for r in records}),
    )

    splits = create_patient_splits(records)

    print_split_summary(splits)

    save_splits(splits)

    print("\nSaved patient-independent splits to splits.json")

    # Smoke-test one batch without requiring a GPU.
    loaders = create_dataloaders(
        splits,
        batch_size=4,
        num_workers=0,
    )

    images, labels = next(iter(loaders["train"]))

    print("\n=== DATALOADER TEST ===")
    print("Image batch shape:", images.shape)
    print("Labels:", labels.tolist())
