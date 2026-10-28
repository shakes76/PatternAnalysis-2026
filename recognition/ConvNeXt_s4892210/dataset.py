"""
Author: Lawrence Tiang
Student ID: s4892210

Description:
    Load and prepare the ADNI brain MRI image dataset. Apply data
    preprocessing and augmentation to training images. Also splits
    the training dataset at the patient level to prevent images from
    the same patient appearing in both the training and validation sets.
    Provide DataLoaders and dataset summaries containing image counts
    and unique patient counts for each class.

Usage:
    The dataset functions are imported into train.py and predict.py
    to load, preprocess, and prepare datasets for training and evaluation.
"""

import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Dataset configuration and normalisation settings for the ADNI images.
DEFAULT_ROOT = "/home/groups/comp3710/ADNI/AD_NC"
ADNI_MEAN = 0.115540   # Normalisation statistics computed from training dataset in util.py
ADNI_STD = 0.225449
IMAGE_SIZE = 224


def build_transform(is_train):
    """
    Build the data preprocessing and image transformations
    for training, validation or testing datasets.

    Args:
        is_train (bool): If True, apply data augmentation for training.
            If False, apply deterministic preprocessing only.

    Returns:
        transforms.Compose: A sequence of transformations
        used to prepare MRI images for model input.
    """

    if is_train:
        # Apply random transformations to increase training data diversity.
        return transforms.Compose([
            # Randomly crop and resize the image to the required input size
            transforms.RandomResizedCrop(
                size=(IMAGE_SIZE, IMAGE_SIZE),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),

            # Convert grayscale MRI images to three channels for model compatibility.
            transforms.Grayscale(num_output_channels=3),
            
            # Apply geometric and intensity-based data augmentation
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandAugment(5, 5),

            # Converts the image to a tensor then normalises each channel
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[ADNI_MEAN] * 3,
                std=[ADNI_STD] * 3
            ),

            # Randomly erase a region of the image
            transforms.RandomErasing(p=0.25),
        ])
    else:
        # Apply deterministic preprocessing for validation and testing data
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[ADNI_MEAN] * 3,
                std=[ADNI_STD] * 3
            )
        ])


def build_dataloader(
    is_train,
    root_dir = DEFAULT_ROOT,
    batch_size = 128,
    num_workers = 2,
    validation_ratio = 0.1
):
    """
    Build DataLoaders for training, validation and testing datasets.

    Args:
        is_train (bool): If True, create training and validation
            DataLoaders. If False, create a test DataLoader.
        root_dir (str): Root directory containing the train and test
            dataset folders.
        batch_size (int): Number of images processed in each batch.
        num_workers (int): Number of subprocesses used to load images.
        validation_ratio (float): Percentage of unique training patients
            assigned to the validation set.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation
            DataLoaders when is_train is True.
        DataLoader: Test DataLoader when is_train is False.
    """

    # Select the appropriate directory based on the requested dataset.
    if is_train:
        data_dir = os.path.join(root_dir, "train")
    else:
        data_dir = os.path.join(root_dir, "test")

    # Load image paths and class labels
    base_dataset = datasets.ImageFolder(root=data_dir)

    if is_train:
        # Extract image paths and identify the patient IDs
        # ("218987_48.jpg" -> "218987")
        all_paths = []
        patient_ids = []

        for sample in base_dataset.samples:
            path = sample[0]
            all_paths.append(path)

        for path in all_paths:
            patient_id = os.path.basename(path).split("_")[0]
            patient_ids.append(patient_id)

        # Group image indices by patient
        id_to_indices = {}
        for index, pid in enumerate(patient_ids):
            id_to_indices.setdefault(pid, []).append(index)

        # Obtain the unique patient IDs for splitting
        unique_patients = list(id_to_indices.keys())

        # Split patients reproducibly using a fixed random seed
        train_patients, val_patients = train_test_split(
            unique_patients,
            test_size=validation_ratio,
            shuffle=True,
            random_state=42
        )

        # Convert the patient-level split back into image indices
        train_indices = []
        validation_indices = []

        for pid in train_patients:
            for i in id_to_indices[pid]:
                train_indices.append(i)

        for pid in val_patients:
            for i in id_to_indices[pid]:
                validation_indices.append(i)

        # Create separate dataset instances for training
        # and validation with different transforms
        train_dataset = Subset(
            type(base_dataset)(
                root=base_dataset.root,
                transform=build_transform(is_train=True)
            ),
            train_indices
        )
        val_dataset = Subset(
            type(base_dataset)(
                root=base_dataset.root,
                transform=build_transform(is_train=False)
            ),
            validation_indices
        )

        # Create data loader for training and validation dataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_loader

    else:
        # Apply deterministic preprocessing to test images
        transform = build_transform(is_train=False)
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)

        # Create data loader for test dataset
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return test_loader


def get_class_distribution(dataset):
    """
    Count images and unique patients for each diagnostic class.

    Args:
        dataset (Dataset or Subset): An ImageFolder dataset or a subset
            of one, such as the training or validation dataset.

    Returns:
        tuple[dict, dict]: A pair containing image counts by class and
        sets of unique patient IDs by class.
    """
    # Initialise image counts and patient ID sets for both classes
    image_counts = {"AD": 0, "NC": 0}
    patients_by_class = {"AD": set(), "NC": set()}

    # Handle subsets by using their original dataset and selected indices
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = range(len(base_dataset.samples))

    # Count images and collect unique patient IDs for each class
    for index in indices:
        image_path, label = base_dataset.samples[index]
        class_name = base_dataset.classes[label]
        patient_id = os.path.basename(image_path).split("_")[0]
        image_counts[class_name] += 1
        patients_by_class[class_name].add(patient_id)

    return image_counts, patients_by_class


def print_dataset_summary(train_loader, val_loader, test_loader):
    """
    Print image counts, unique patient counts and class distributions
    for the training, validation and test datasets.

    Args:
        train_loader (DataLoader): DataLoader for training images.
        val_loader (DataLoader): DataLoader for validation images.
        test_loader (DataLoader): DataLoader for test images.

    Returns:
        None: Prints a summary of the dataset
    """
    # Calculate image counts and unique patient IDs for each split
    train_images, train_patients = get_class_distribution(train_loader.dataset)
    val_images, val_patients = get_class_distribution(val_loader.dataset)
    test_images, test_patients = get_class_distribution(test_loader.dataset)

    # Access the underlying training dataset to retrieve class mappings
    train_base_dataset = train_loader.dataset.dataset

    # Print the overall number of unique patients and images per split
    print("\n=== ADNI dataset ===")
    print(f"Train subjects      : {len(train_patients['AD'] | train_patients['NC']):,}")
    print(f"Validation subjects : {len(val_patients['AD'] | val_patients['NC']):,}")
    print(f"Test subjects       : {len(test_patients['AD'] | test_patients['NC']):,}")
    print(f"Train images        : {sum(train_images.values()):,}")
    print(f"Validation images   : {sum(val_images.values()):,}")
    print(f"Test images         : {sum(test_images.values()):,}")
    print(f"Classes             : {train_base_dataset.class_to_idx}")

    # Display class-specific image and patient counts for training
    print("\n=== Training Class Distribution ===")
    print(f"AD: {train_images['AD']:,} images, {len(train_patients['AD']):,} subjects")
    print(f"NC: {train_images['NC']:,} images, {len(train_patients['NC']):,} subjects")

    # Display class-specific image and patient counts for validation
    print("\n=== Validation Class Distribution ===")
    print(f"AD: {val_images['AD']:,} images, {len(val_patients['AD']):,} subjects")
    print(f"NC: {val_images['NC']:,} images, {len(val_patients['NC']):,} subjects")

    # Display class-specific image and patient counts for testing
    print("\n=== Test Class Distribution ===")
    print(f"AD: {test_images['AD']:,} images, {len(test_patients['AD']):,} subjects")
    print(f"NC: {test_images['NC']:,} images, {len(test_patients['NC']):,} subjects")