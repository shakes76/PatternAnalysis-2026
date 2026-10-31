"""Contains the data loader for loading and preprocessing the data, 
plus additional data manipulations
"""

import copy
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import v2


CLASS_TO_IDX = {"AD": 0, "NC": 1}
IMAGE_SIZE = (240, 256)  # (height, width)

def make_transforms(augment: bool) -> transforms.Compose:
    """
    Create the transformation composition used for the dataset.
    Scales pixel values between [0,1], and optionally applies augmentation.

    Returns:
        transforms.Compose: The transformations
    """
    ops: list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
    if augment:
        # small affine augmentation for variations in positioning.
        ops.append(transforms.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
        ))
        # small color augmentation for variations in brightness and contrast.
        ops.append(v2.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.0, hue=0.0
        ))
        
    return transforms.Compose(ops)


def validation_split(dataset, val_fraction: float, seed: int | None = None) -> Tuple[list, list]:
    """
    Split indices within each class, preserving class representation and eforcing patient level separation

    Args:
        dataset (_type_): The dataset to split for a given class
        val_fraction (float): The fraction of data to use for validation (appox since we maintain patient level separation)
        seed (int | None): Seed for reproducibility. If None, a random seed is used.

    Returns:
        Tuple[list, list]: (train_indices, val_indices) in that order
    """
    generator = torch.Generator().manual_seed(seed if seed is not None else torch.seed())
    patient_to_indices: dict[str, list[int]] = {}
    patient_to_class: dict[str, int] = {}

    # map each patient number to the indices of their images in the dataset,
    # and also map each patient to their class
    samples = (
        [dataset.dataset.samples[i] for i in dataset.indices] # type: ignore
        if isinstance(dataset, Subset)
        else dataset.samples
    )

    for index, (path, class_idx) in enumerate(samples):
        patient_id = _patient_id(path)

        # explicitly ensure the same patient does not appear in multiple class folders
        # useful for the dataset I manually downloaded
        previous_class = patient_to_class.setdefault(patient_id, class_idx)
        if previous_class != class_idx:
            raise ValueError(
                f"Patient {patient_id} appears in multiple class folders"
            )

        patient_to_indices.setdefault(patient_id, []).append(index)


    class_to_patients: dict[int, list[str]] = {0: [], 1: []}
    for patient_id, class_idx in patient_to_class.items():
        class_to_patients[class_idx].append(patient_id)

    # now we have patients in each class, we can shuffle them and split into train/val
    train_indices, val_indices = [], []
    for class_idx in sorted(class_to_patients):
        patients = class_to_patients[class_idx]
        permutation = torch.randperm(len(patients), generator=generator).tolist()
        shuffled_patients = [patients[i] for i in permutation]
        n_val = int(round(len(shuffled_patients) * val_fraction))

        val_patients = shuffled_patients[:n_val]
        train_patients = shuffled_patients[n_val:]
        for patient_id in train_patients:
            train_indices.extend(patient_to_indices[patient_id])
        for patient_id in val_patients:
            val_indices.extend(patient_to_indices[patient_id])

    train_indices = [train_indices[i] for i in torch.randperm(len(train_indices), generator=generator).tolist()]
    val_indices = [val_indices[i] for i in torch.randperm(len(val_indices), generator=generator).tolist()]
    return train_indices, val_indices


def _patient_id(path: str | Path) -> str:
    """Extract the patient identifier from a 'patient_slice.ext' filename."""
    stem = Path(path).stem
    patient_id, separator, slice_id = stem.partition("_")
    if not separator or not patient_id or not slice_id:
        raise ValueError(
            f"Image filename does not match '<patient number>_<slice number>.<ext>': {path}"
        )
    return patient_id


def report_dataset(dataset, name):
    """
    Helper function to report on the pooled dataset, including the number of images, unique patients, counts per class, and min/max images per patient.
    """
    patient_images = {}
    patient_classes = {}

    for path, class_idx in dataset.samples:
        patient_id = _patient_id(path)

        patient_images[patient_id] = (
            patient_images.get(patient_id, 0) + 1
        )
        patient_classes[patient_id] = class_idx

    image_counts = {
        class_name: sum(
            label == class_idx
            for _, label in dataset.samples
        )
        for class_name, class_idx in CLASS_TO_IDX.items()
    }

    patient_counts = {
        class_name: sum(
            label == class_idx
            for label in patient_classes.values()
        )
        for class_name, class_idx in CLASS_TO_IDX.items()
    }

    print(f"\n{name}")
    print(f"Images: {len(dataset.samples)}")
    print(f"Unique patients: {len(patient_classes)}")
    print(f"Images per class: {image_counts}")
    print(f"Patients per class: {patient_counts}")
    print(
        "Images per patient — min/mean/max:",
        min(patient_images.values()),
        max(patient_images.values()),
    )


def report_subset(subset, name):
    """
    Helper function to report on the train, test, or validation subsets, including the number of images, unique patients, and counts per class.

    Args:
        subset (_type_): _description_
        name (_type_): _description_
    """
    print(f"\n{name}")

    counts = {"AD": 0, "NC": 0}
    patients = set()

    for index in subset.indices:
        path, class_idx = subset.dataset.samples[index]
        counts[list(CLASS_TO_IDX.keys())[class_idx]] += 1
        patients.add(_patient_id(path))

    print(f"Images: {len(subset)}")
    print(f"Unique patients: {len(patients)}")
    print(f"Images per class: {counts}")


def create_pooled_dataset(root: Path, transform: transforms.Compose) -> datasets.ImageFolder:
    """
    Create the pooled dataset from all the images in the /test and /train folders together

    Args:
        root (str | Path): path to dataset root folder, containing /train and /test subfolders
        transform (transforms.Compose): transformations to apply

    Returns:
        datasets.ImageFolder: pooled dataset
    """
    train_dir = root / "train"
    test_dir = root / "test"

    train_dataset = datasets.ImageFolder(
        train_dir, transform=transform
    )
    test_dataset = datasets.ImageFolder(
        test_dir, transform=transform
    )

    assert train_dataset.class_to_idx == CLASS_TO_IDX
    assert test_dataset.class_to_idx == CLASS_TO_IDX

    train_dataset.samples.extend(test_dataset.samples)
    train_dataset.imgs = train_dataset.samples
    train_dataset.targets = [
        label for _, label in train_dataset.samples
    ]

    return train_dataset


def get_patients(subset: Subset) -> set:
    """
    Get the set of unique patient identifiers in a subset.

    Args:
        subset (Subset): The subset to analyze.

    Returns:
        set: A set of unique patient identifiers.
    """
    return {
        _patient_id(subset.dataset.samples[index][0]) # type: ignore
        for index in subset.indices
    }


def create_dataloaders(
    data_dir: str | Path = "datasets",
    batch_size: int = 32,
    val_fraction: float = 0.1,
    test_fraction: float = 0.3,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> Dict[str, DataLoader]:
    """Create train, validation, and test DataLoaders.

    data_dir should contain train/AD, train/NC, test/AD, and test/NC.

    Returns loaders keyed by 'train', 'val', and 'test'. Images have shape (B, 1, 240, 256), are grayscale, and have values in [0, 1].
    Note: If val_fraction == 0, the 'val' key will not be present in the returned dictionary and similarly for test_fraction.
    """
    if val_fraction < 0 or val_fraction >= 1:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if test_fraction < 0 or test_fraction >= 1:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    if test_fraction + val_fraction >= 1:
        raise ValueError(
            f"val_fraction + test_fraction must be < 1, got {val_fraction + test_fraction}"
        )

    root = Path(data_dir)
    train_dir, test_dir = root / "train", root / "test"

    # Pool all images from the original training and testing folders.
    pooled_dataset = create_pooled_dataset(
        root,
        transform=make_transforms(augment=True),
    )

    # Create separate datasets so validation and testing do not use augmentation.
    eval_dataset = create_pooled_dataset(
        root,
        transform=make_transforms(augment=False),
    )

    report_dataset(pooled_dataset, "Original pooled dataset")

    # First split: training pool vs temporary evaluation pool.
    train_fraction = 1.0 - val_fraction - test_fraction

    train_indices, eval_indices = validation_split(
        pooled_dataset,
        1.0 - train_fraction,
        seed,
    )

    # Create subsets using the appropriate transformations.
    train_dataset = Subset(pooled_dataset, train_indices)
    eval_subset = Subset(eval_dataset, eval_indices)

    # Second split: temporary evaluation pool into validation and test.
    if val_fraction + test_fraction > 0:
        adjusted_val_fraction = (
            val_fraction / (val_fraction + test_fraction)
        )

        val_relative_indices, test_relative_indices = validation_split(
            eval_subset,
            1.0 - adjusted_val_fraction,
            seed + 1,
        )

        val_indices = [eval_indices[i] for i in val_relative_indices]
        test_indices = [eval_indices[i] for i in test_relative_indices]

        val_dataset = Subset(eval_dataset, val_indices)
        test_dataset = Subset(eval_dataset, test_indices)
    else:
        val_dataset = None
        test_dataset = None

    # Explicitly verify strict patient-level separation between training and validation/testing datasets.
    train_patients = get_patients(train_dataset)
    val_patients = get_patients(val_dataset) if val_dataset is not None else set()
    test_patients = get_patients(test_dataset) if test_dataset is not None else set()

    assert train_patients.isdisjoint(val_patients), (
        "Patient overlap detected between training and validation datasets"
    )
    assert train_patients.isdisjoint(test_patients), (
        "Patient overlap detected between training and testing datasets"
    )
    assert val_patients.isdisjoint(test_patients), (
        "Patient overlap detected between validation and testing datasets"
    )

    report_subset(train_dataset, "Training")
    if val_dataset is not None:
        report_subset(val_dataset, "Validation")
    if test_dataset is not None:
        report_subset(test_dataset, "Testing")

    # Enable page-locked memory if we have a GPU
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_generator = torch.Generator().manual_seed(seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(train_dataset, shuffle=True, generator=loader_generator, **common)
    }

    if val_dataset is not None:
        loaders["val"] = DataLoader(val_dataset, shuffle=False, **common)

    if test_dataset is not None:
        loaders["test"] = DataLoader(test_dataset, shuffle=False, **common)

    print(f"Class mapping: {pooled_dataset.class_to_idx}")
    for split, loader in loaders.items():
        print(f"{split}: {len(loader.dataset)} images, {len(loader)} batches") # type: ignore
    return loaders
