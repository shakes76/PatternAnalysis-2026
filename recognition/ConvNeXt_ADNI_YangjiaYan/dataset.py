
"""ADNI dataset loading and patient-independent data splitting."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


DATA_ROOT = Path("/home/groups/comp3710/ADNI")
LABEL_TO_INDEX = {"NC": 0, "AD": 1}
PATIENT_PATTERN = re.compile(r"ADNI_(\d{3}_S_\d{4})")


def load_records(root=DATA_ROOT):
    """Index slices and extract subject identities from metadata."""
    root = Path(root)

    with open(root / "meta_data_with_label.json") as f:
        metadata = json.load(f)

    records = []
    patient_labels = {}
    missing = []

    for split in ("train", "test"):
        for class_name in ("AD", "NC"):
            folder = root / "AD_NC" / split / class_name

            for image_path in sorted(folder.glob("*.jpeg")):
                scan_id = image_path.stem.rsplit("_", 1)[0]
                info = metadata.get(scan_id)

                if info is None:
                    missing.append(scan_id)
                    continue

                source = info.get("raw") or info.get("c1") or ""
                match = PATIENT_PATTERN.search(source)

                if match is None:
                    missing.append(scan_id)
                    continue

                patient_id = match.group(1)
                label = LABEL_TO_INDEX[class_name]

                if patient_id in patient_labels:
                    if patient_labels[patient_id] != label:
                        raise ValueError(
                            f"Conflicting labels for patient {patient_id}"
                        )

                patient_labels[patient_id] = label

                records.append({
                    "path": str(image_path),
                    "patient_id": patient_id,
                    "scan_id": scan_id,
                    "label": label,
                    "original_split": split,
                })

    if missing:
        raise ValueError(
            f"Missing metadata/patient IDs for {len(missing)} images"
        )

    return records, patient_labels


def create_patient_splits(patient_labels, seed=42):
    """Produce stratified, patient-independent 70/15/15 splits."""
    patients = sorted(patient_labels)
    labels = [patient_labels[p] for p in patients]

    train_ids, remaining_ids = train_test_split(
        patients,
        test_size=0.30,
        stratify=labels,
        random_state=seed,
    )

    remaining_labels = [
        patient_labels[p] for p in remaining_ids
    ]

    val_ids, test_ids = train_test_split(
        remaining_ids,
        test_size=0.50,
        stratify=remaining_labels,
        random_state=seed,
    )

    splits = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }

    assert not (splits["train"] & splits["val"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["val"] & splits["test"])

    assert set(patients) == set().union(*splits.values())

    return splits


def assign_splits(records, splits):
    """Assign every image using its patient's split."""
    patient_to_split = {
        patient: name
        for name, patients in splits.items()
        for patient in patients
    }

    result = []

    for record in records:
        updated = dict(record)
        updated["split"] = patient_to_split[record["patient_id"]]
        result.append(updated)

    return result


def save_manifest(records, output_path):
    """Save the fixed image-to-split mapping for reproducibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "path", "patient_id", "scan_id",
        "label", "original_split", "split",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def build_transform(training=False, image_size=224):
    """Adapt grayscale MRI images to three-channel ConvNeXt input."""
    operations = [
        transforms.Resize((image_size, image_size)),
    ]

    if training:
        operations.append(
            transforms.RandomRotation(degrees=5)
        )

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(operations)


class ADNIDataset(Dataset):
    """PyTorch dataset for indexed ADNI JPEG images."""

    def __init__(self, records, split, image_size=224):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")

        self.records = [
            r for r in records if r["split"] == split
        ]

        self.transform = build_transform(
            training=(split == "train"),
            image_size=image_size,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, record["label"]


def print_summary(records, splits):
    print("\n===== Patient-independent Split Audit =====")

    all_patients = set().union(*splits.values())
    print("Total patients:", len(all_patients))
    print("Total images:", len(records))

    for name in ("train", "val", "test"):
        rows = [r for r in records if r["split"] == name]

        image_counts = Counter(r["label"] for r in rows)
        patient_counts = Counter()

        for patient in splits[name]:
            label = next(
                r["label"] for r in rows
                if r["patient_id"] == patient
            )
            patient_counts[label] += 1

        print(
            f"{name}: patients={len(splits[name])}, "
            f"images={len(rows)}, "
            f"AD_patients={patient_counts[1]}, "
            f"NC_patients={patient_counts[0]}, "
            f"AD_images={image_counts[1]}, "
            f"NC_images={image_counts[0]}"
        )

    print("Patient overlap: 0 (verified)")
    print("All patients assigned exactly once: YES")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=DATA_ROOT
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "data" /
                "comp3710_a3_adni_split.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records, patient_labels = load_records(args.root)

    splits = create_patient_splits(
        patient_labels, seed=args.seed
    )

    records = assign_splits(records, splits)

    print_summary(records, splits)

    # Test loading one actual image without using a GPU.
    train_dataset = ADNIDataset(records, "train")
    image, label = train_dataset[0]

    print("\nSample image shape:", tuple(image.shape))
    print("Sample label:", label)

    save_manifest(records, args.manifest)
    print("Manifest saved to:", args.manifest)


if __name__ == "__main__":
    main()

