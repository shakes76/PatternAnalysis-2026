import csv
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Configuration
# Override the dataset location with: export ADNI_ROOT=/path/to/ADNI
ROOT = Path(os.environ.get("ADNI_ROOT", "/home/groups/comp3710/ADNI"))
DATA_DIR = ROOT / "AD_NC"

SEED = 42

# splits.csv is saved next to this file
SPLITS_CSV = Path(__file__).resolve().parent / "splits.csv"

LABEL_MAP = {
    "NC": 0,
    "AD": 1,
}


def generate_splits():
    """
    Generate reproducible train/validation/test splits.

    Splitting is performed at the patient level, so MRI slices
    from the same patient never appear in more than one split.

    Ratios:
        Train: 70%
        Validation: 15%
        Test: 15%

    Image paths are stored relative to DATA_DIR, so splits.csv
    is portable and does not depend on where the dataset lives.
    """

    with open(ROOT / "meta_data_with_label.json") as f:
        metadata = json.load(f)

    records = []
    patient_labels = {}

    # Collect images from both original directories.
    # The original train/test separation is not used
    # because it contains overlapping patients.
    for original_split in ("train", "test"):
        for label in ("AD", "NC"):
            folder = DATA_DIR / original_split / label

            for path in sorted(folder.glob("*.jpeg")):
                # Example filename: 932015_96.jpeg -> scan ID 932015
                scan_id = path.stem.rsplit("_", 1)[0]

                info = metadata.get(scan_id)
                if not isinstance(info, dict):
                    raise ValueError(f"Missing metadata for scan: {scan_id}")

                # Extract the ADNI patient ID from the metadata.
                # Example: ADNI_068_S_0473 -> 068_S_0473
                text = " ".join(str(v) for v in info.values())
                match = re.search(r"ADNI_(\d{3}_S_\d{4})", text)
                if not match:
                    raise ValueError(f"Missing patient ID for scan: {scan_id}")

                patient_id = match.group(1)

                # A patient must have the same label everywhere
                if patient_labels.get(patient_id, label) != label:
                    raise ValueError(f"Conflicting labels for patient: {patient_id}")
                patient_labels[patient_id] = label

                records.append({
                    "path": str(path.relative_to(DATA_DIR)),
                    "patient_id": patient_id,
                    "scan_id": scan_id,
                    "label": label,
                })

    # Patient-level stratified split: 70% train, 30% temporary
    patients = sorted(patient_labels)
    labels = [patient_labels[p] for p in patients]

    train_ids, temp_ids = train_test_split(
        patients,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )

    # Split the temporary set in half: 15% validation, 15% test
    temp_labels = [patient_labels[p] for p in temp_ids]

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels,
    )

    splits = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }

    # Verify patient independence
    assert splits["train"].isdisjoint(splits["val"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["val"].isdisjoint(splits["test"])
    assert sum(len(ids) for ids in splits.values()) == len(patients)

    patient_to_split = {
        patient_id: split_name
        for split_name, patient_ids in splits.items()
        for patient_id in patient_ids
    }

    for record in records:
        record["split"] = patient_to_split[record["patient_id"]]

    # Save the CSV manifest
    fieldnames = ["path", "patient_id", "scan_id", "label", "split"]
    with open(SPLITS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Print dataset statistics
    print(f"Total images: {len(records)}")
    print(f"Unique patients: {len(patients)}")

    for split_name, patient_ids in splits.items():
        subset = [r for r in records if r["split"] == split_name]
        ad_count = sum(r["label"] == "AD" for r in subset)
        nc_count = len(subset) - ad_count
        print(
            f"{split_name}: {len(patient_ids)} patients, "
            f"{len(subset)} images (AD={ad_count}, NC={nc_count})"
        )

    print("Patient overlap: 0")
    print(f"Saved: {SPLITS_CSV}")


class ADNIDataset(Dataset):
    """
    PyTorch Dataset for ADNI brain MRI slice classification.

    Labels:
        NC = 0 (Cognitively Normal)
        AD = 1 (Alzheimer's Disease)
    """

    def __init__(self, csv_path, split, transform=None, data_dir=DATA_DIR):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")

        df = pd.read_csv(csv_path, dtype={"scan_id": str, "patient_id": str})
        self.data = df[df["split"] == split].reset_index(drop=True)

        if self.data.empty:
            raise ValueError(f"No images found for split: {split}")

        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load the grayscale MRI as 3-channel RGB so it matches
        # the input expected by ImageNet-pretrained ConvNeXt.
        with Image.open(self.data_dir / row["path"]) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, LABEL_MAP[row["label"]]


def get_transforms(image_size=224):
    """
    Create training and evaluation transforms.

    Training uses a small random rotation as augmentation.
    Validation and test only resize and normalise.
    Normalisation uses ImageNet statistics to match the
    pretrained ConvNeXt weights.
    """

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


def get_dataloaders(
    csv_path=SPLITS_CSV,
    data_dir=DATA_DIR,
    batch_size=32,
    num_workers=4,
    image_size=224,
):
    """
    Create train, validation and test DataLoaders.

    Returns:
        dict with keys "train", "val" and "test".
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {csv_path}\n"
            "Run 'python dataset.py' to generate it."
        )

    train_transform, eval_transform = get_transforms(image_size=image_size)
    split_transforms = {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }

    loaders = {}
    for split_name, transform in split_transforms.items():
        dataset = ADNIDataset(
            csv_path,
            split=split_name,
            transform=transform,
            data_dir=data_dir,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    return loaders


if __name__ == "__main__":
    generate_splits()
