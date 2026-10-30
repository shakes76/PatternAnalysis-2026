"""Data loading, subject-level splits and training-only augmentation for ADNI.

Subject IDs are read from the course metadata JSON (the filename prefix is a
scan ID, not a subject ID). load_datasets() returns (train, validation, test):
validation is always split by subject; split_mode "strict" also removes every
training subject that appears in the official test folder. input_mode
"context3" stacks the previous/current/next slice of the same scan as channels.
"""
import json
import os
import random
import re
from pathlib import Path
from collections import defaultdict, Counter

from torchvision import datasets, transforms
from torchvision.transforms import RandomAffine, InterpolationMode
from torch.utils.data import Dataset
import torch
from torch.utils.data import Subset
from PIL import Image


# Environment overrides exist only for local smoke tests; Rangpur defaults are unchanged.
DATA_ROOT = Path(os.environ.get(
    "ADNI_DATA_ROOT", "/home/groups/comp3710/ADNI/AD_NC"))
METADATA_PATH = Path(os.environ.get(
    "ADNI_METADATA_PATH",
    "/home/groups/comp3710/ADNI/meta_data_with_label.json"))

# "strict":   remove from training every ADNI subject that also has a scan in
#             the official test folder (strict subject-level separation).
# "official": keep the course-provided train/test folders exactly as given.
#             Course staff state these folders are leak-free, but they are
#             separated by scan (Image ID), not by subject: some subjects have
#             different visits in both. Validation is still split by subject.
SPLIT_MODES = ("strict", "official")

# "single":   one slice, grey value copied into 3 channels (original design).
# "context3": 2.5D input; channels = previous, current and next slice of the
#             SAME scan (filename suffix = slice number, confirmed by staff).
#             Edge slices repeat themselves. Every sample is still one slice
#             with its own label, so evaluation remains per slice.
INPUT_MODES = ("single", "context3")

SEED = 42
VAL_RATIO = 0.2

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# Context slices already form a 3-channel image, so no Grayscale step.
context_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def get_transform(input_mode="single"):
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}")
    return transform if input_mode == "single" else context_transform


def split_slice_name(path):
    """'392586_92.jpeg' -> ('392586', 92)."""
    scan_id, _, number = Path(path).stem.rpartition("_")
    if not scan_id or not number.isdigit():
        raise ValueError(f"Unexpected slice filename: {path}")
    return scan_id, int(number)


_SCAN_SLICES = {}


def scan_slice_numbers(directory):
    """Map scan id -> sorted slice numbers for one class folder (cached)."""
    directory = str(directory)
    if directory not in _SCAN_SLICES:
        groups = defaultdict(list)
        for name in os.listdir(directory):
            if Path(name).suffix.lower() not in (".jpeg", ".jpg", ".png"):
                continue
            scan_id, number = split_slice_name(name)
            groups[scan_id].append(number)
        _SCAN_SLICES[directory] = {
            scan_id: sorted(numbers) for scan_id, numbers in groups.items()
        }
    return _SCAN_SLICES[directory]


def context_slice_loader(path):
    """Load [previous, current, next] slices of one scan as an RGB image."""
    path = Path(path)
    scan_id, number = split_slice_name(path)
    numbers = scan_slice_numbers(path.parent)[scan_id]
    position = numbers.index(number)
    neighbours = (
        numbers[max(position - 1, 0)],
        number,
        numbers[min(position + 1, len(numbers) - 1)],
    )
    channels = []
    for neighbour in neighbours:
        with Image.open(path.with_name(f"{scan_id}_{neighbour}{path.suffix}")) as image:
            channels.append(image.convert("L"))
    return Image.merge("RGB", channels)


def audit_slice_numbering(root=None):
    """Report whether each scan's slice numbers are consecutive integers."""
    root = Path(root or DATA_ROOT)
    total, consecutive, sizes = 0, 0, Counter()
    for split in ("train", "test"):
        for class_dir in sorted((root / split).iterdir()):
            for numbers in scan_slice_numbers(class_dir).values():
                total += 1
                sizes[len(numbers)] += 1
                if numbers == list(range(numbers[0], numbers[0] + len(numbers))):
                    consecutive += 1
    print(f"Scans: {total}; consecutive slice numbers: {consecutive}; "
          f"slices per scan: {dict(sizes)}")
    return total, consecutive


def load_patient_map():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    patient_map = {}

    for image_id, record in metadata.items():
        raw_path = str(record.get("raw", ""))
        match = re.search(r"\d{3}_S_\d{4}", raw_path)

        if match:
            patient_map[str(image_id)] = match.group(0)

    return patient_map


def get_image_id(path):
    return Path(path).stem.split("_")[0]


def load_datasets(split_mode="strict", input_mode="single"):
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}")

    folder_options = {"transform": get_transform(input_mode)}
    if input_mode == "context3":
        folder_options["loader"] = context_slice_loader

    full_train = datasets.ImageFolder(
        DATA_ROOT / "train", **folder_options
    )

    test_dataset = datasets.ImageFolder(
        DATA_ROOT / "test", **folder_options
    )

    assert full_train.class_to_idx == test_dataset.class_to_idx

    patient_map = load_patient_map()

    def get_patient(path):
        image_id = get_image_id(path)

        if image_id not in patient_map:
            raise ValueError(
                f"Missing patient ID for image {image_id}"
            )

        return patient_map[image_id]

    # Identify all patients in the official test set.
    test_patients = {
        get_patient(path)
        for path, _ in test_dataset.samples
    }

    # strict: exclude official test subjects from training/validation.
    # official: keep every course training scan.
    patient_groups = defaultdict(list)
    excluded_images = 0
    excluded_patients = set()

    for index, (path, label) in enumerate(full_train.samples):
        patient_id = get_patient(path)

        if split_mode == "strict" and patient_id in test_patients:
            excluded_images += 1
            excluded_patients.add(patient_id)
            continue

        patient_groups[patient_id].append(index)

    # Stratify patient groups by their available class labels.
    strata = defaultdict(list)

    for patient_id, indices in patient_groups.items():
        labels = tuple(sorted({
            full_train.targets[i]
            for i in indices
        }))
        strata[labels].append(patient_id)

    rng = random.Random(SEED)
    train_patients = set()
    val_patients = set()

    for label_group, patients in sorted(strata.items()):
        patients = sorted(patients)
        rng.shuffle(patients)

        if len(patients) > 1:
            n_val = max(
                1,
                min(
                    len(patients) - 1,
                    round(len(patients) * VAL_RATIO)
                )
            )
        else:
            n_val = 0

        val_patients.update(patients[:n_val])
        train_patients.update(patients[n_val:])

    train_indices = []
    val_indices = []

    for patient_id, indices in patient_groups.items():
        if patient_id in val_patients:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    # Train and validation never share a subject, in either mode.
    assert not (train_patients & val_patients)

    if split_mode == "strict":
        assert not (train_patients & test_patients)
        assert not (val_patients & test_patients)

    # Overlap is reported explicitly so the official-split result is never
    # presented as subject-independent.
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients

    train_labels = Counter(
        full_train.targets[i] for i in train_indices
    )
    val_labels = Counter(
        full_train.targets[i] for i in val_indices
    )

    expected_classes = set(range(len(full_train.classes)))

    if set(train_labels) != expected_classes:
        raise ValueError("Training set is missing a class")

    if set(val_labels) != expected_classes:
        raise ValueError("Validation set is missing a class")

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_train, val_indices)

    print("Split mode:", split_mode)
    print("Excluded overlapping patients:",
          len(excluded_patients))
    print("Excluded training images:", excluded_images)
    print("Train subjects also in official test:",
          len(train_test_overlap))
    print("Validation subjects also in official test:",
          len(val_test_overlap))
    if split_mode == "official" and (train_test_overlap or val_test_overlap):
        print("WARNING: official split shares ADNI subjects (different "
              "visits) with the test folder; test accuracy is NOT "
              "subject-independent. Report it separately from strict mode.")
    print("Training patients:", len(train_patients))
    print("Validation patients:", len(val_patients))
    print("Testing patients:", len(test_patients))
    print("Train class counts:", dict(train_labels))
    print("Val class counts:", dict(val_labels))
    print("Train/validation subject separation verified.")

    return train_dataset, val_dataset, test_dataset



class MRIAugmentedTrainingSet(Dataset):
    """Mild per-slice augmentation applied to TRAIN only.

    Uses inputs already normalised to [-1, 1] by dataset.py; fill=-1
    corresponds to black outside the MRI when normalisation is (x-.5)/.5.
    Avoid left-right flips: they can distort clinically meaningful anatomy.
    """
    def __init__(self, original):
        self.original = original
        self.affine = RandomAffine(
            degrees=6,
            translate=(0.03, 0.03),
            scale=(0.97, 1.03),
            interpolation=InterpolationMode.BILINEAR,
            fill=-1.0,
        )

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        image, label = self.original[index]
        image = self.affine(image)
        image = (image + torch.randn_like(image) * 0.01).clamp(-1.0, 1.0)
        return image, label


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "strict"
    if mode == "audit-slices":
        audit_slice_numbering()
        raise SystemExit
    train_dataset, val_dataset, test_dataset = load_datasets(mode)

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Testing images:", len(test_dataset))
