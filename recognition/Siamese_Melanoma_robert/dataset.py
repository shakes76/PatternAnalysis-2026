import os
import time
import random
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, precision_recall_curve, 
                            average_precision_score, RocCurveDisplay,
                            PrecisionRecallDisplay)
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

import matplotlib.pyplot as plt
from tqdm import tqdm

# Hyper Parameters
from params import (
    DATA_PATH, CSV_NAME, IMG_DIR, SEED, MEAN, STD, ROT_DEG, FLIP_PROB, COLOR_JITTER, USE_GROUP_SPLIT
)

class TripletDataGenerator(torch.utils.data.Dataset):
    """
    PyTorch Dataset that yields triplets (anchor, positive, negative) for training with Triplet Loss
    - In training mode, each __getitem__ returns three transformed images and the anchor label
    - In eval mode, it returns only the transformed anchor image plus empty tensors for positive/negative and the label
    """

    def __init__(self, images, labels=None, train=True, transform=None):
        """
        Args:
            images: list/array of image file paths for anchors, positives, and negatives
            labels: list/array of class labels aligned with images
            train: if True, generate triplets; if False, return anchor-only samples
            transform: transform applied to each image
        """
        self.is_train = train
        self.transform = transform

        self.images = images
        self.labels = labels

    def __len__(self):
        """
        Return the number of images available for sampling
        """
        return len(self.images)
    
    def _read_pil(self, path):
        """
        Open an image from disk using PIL and return it in RGB mode
        Args: path: string path to the image file
        Returns: PIL.Image in RGB mode
        """
        return Image.open(path).convert("RGB")

    def _sample_positive(self, anchor_index: int, anchor_label: int) -> int:
        """
        Pick an index of a positive sample for the given anchor
        Args:
            anchor_index: index of the anchor image in self.images
            anchor_label: class label of the anchor
        Returns:
            Index of a different image with the same label as the anchor
            If no other sample exists for this class, returns anchor_index as a fallback
        """

        # choose a different index with the same label as anchor
        positive_list = [idx for idx, label in enumerate(self.labels) if label == anchor_label and idx != anchor_index]
        if len(positive_list) == 0:
            return anchor_index  # fallback when class has a single sample
        return random.choice(positive_list)
    
    def _sample_negative(self, anchor_index: int, anchor_label: int) -> int:
        """
        Pick an index of a negative sample for the given anchor
        Args:
            anchor_index: index of the anchor image in self.images
            anchor_label: class label of the anchor
        Returns:
            Index of an image whose label is different from the anchor label and not the anchor itself
        """
        negative_list = [idx for idx, label in enumerate(self.labels) if label != anchor_label and idx != anchor_index]

        return random.choice(negative_list)

    def __getitem__(self, anchor_index) -> tuple[torch.tensor, torch.tensor, torch.tensor, int]:
        """
        Get a triplet (anchor, positive, negative) and the anchor label for training
        or a single anchor image with dummy tensors when not training
        
        Args: anchor_index (int): index of the anchor image
        
        Returns: anchor_img, positive_img, negative_img, anchor_label
        """
        # Read anchor image
        anchor_img = self._read_pil(self.images[anchor_index])
        anchor_label = self.labels[anchor_index]

        if self.is_train:
            positive_index = self._sample_positive(anchor_index, anchor_label)
            # Read positive image
            positive_img = self._read_pil(self.images[positive_index])

            negative_index = self._sample_negative(anchor_index, anchor_label)
            # Read negative image
            negative_img = self._read_pil(self.images[negative_index])

            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return anchor_img, positive_img, negative_img, anchor_label

        else:
            if self.transform:
                anchor_img = self.transform(anchor_img)
            return anchor_img, torch.empty(1), torch.empty(1), anchor_label

def get_data_path(root:str=DATA_PATH, metadata_path:str=CSV_NAME, image_dir:str=os.path.join(DATA_PATH, IMG_DIR)) -> tuple[list]:
    """
    Load ISIC 2020 image paths and labels from metadata and an image folder
    Args:
        metadata_path: path to cleaned_metadata.csv containing isic_id and target columns
        image_dir: directory containing image files named <isic_id>.jpg
    Returns:
        images: List of absolute image paths
        labels: List of integer targets
        patient_ids: List of corresponding patient IDs
    """
    root = Path(root)
    metadata = pd.read_csv(root / metadata_path)

    # Build expected filename for each image in metadata
    metadata['image_file'] = metadata['isic_id'] + '.jpg'

    # Keep only images that exist in the image directory
    available_images = set(os.listdir(image_dir))
    metadata = metadata[metadata['image_file'].isin(available_images)]

    # Build aligned lists
    image_paths = [
        os.path.join(image_dir, img)
        for img in metadata['image_file']
    ]
    labels = metadata['target'].astype(int).tolist()
    patient_ids = metadata['patient_id'].tolist()

    return image_paths, labels, patient_ids

def split_train_val_test(images: list, labels: list, patient_ids: list):
    """
    Split the dataset into approximately:
        80% training
        10% validation
        10% testing

    Groups samples by patient_id and approximately
    preserves label distribution.

    Returns:
        train_images, val_images, test_images,
        train_labels, val_labels, test_labels
    """

    images = np.array(images)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)

    # First split: 80% train, 20% remaining
    sgkf1 = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    train_idx, remaining_idx = next(
        sgkf1.split(images, labels, groups=patient_ids)
    )

    # Extract remaining data
    remaining_images = images[remaining_idx]
    remaining_labels = labels[remaining_idx]
    remaining_patient_ids = patient_ids[remaining_idx]

    # Second split: approximately 50% validation, 50% test
    sgkf2 = StratifiedGroupKFold(
        n_splits=2,
        shuffle=True,
        random_state=SEED
    )

    val_idx, test_idx = next(
        sgkf2.split(
            remaining_images,
            remaining_labels,
            groups=remaining_patient_ids
        )
    )

    # Construct final splits
    train_images = images[train_idx]
    val_images = remaining_images[val_idx]
    test_images = remaining_images[test_idx]

    train_labels = labels[train_idx]
    val_labels = remaining_labels[val_idx]
    test_labels = remaining_labels[test_idx]

    return (
        train_images, val_images, test_images,
        train_labels, val_labels, test_labels
    )

def oversample_training(train_images, train_labels):
    """
    Oversampling will be performed on the training set to ensure equal number of
    samples for each class
    Returns: balanced_train_images, balanced_train_labels
    """
    # Preform oversampling of training data
    # Split the images into class 0 and class 1
    class_0_images = train_images[train_labels == 0]
    class_1_images = train_images[train_labels == 1]
    class_0_labels = train_labels[train_labels == 0]
    class_1_labels = train_labels[train_labels == 1]

    # Number of samples to match the larger class
    num_class_0 = len(class_0_images)
    num_class_1 = len(class_1_images)

    np.random.seed(seed=SEED)

    # If class 1 (the minority class) has fewer samples, oversample it
    if num_class_1 < num_class_0:
        # Randomly choose from class_1_images to oversample it
        oversample_indices = np.random.choice(
            np.arange(num_class_1), size=num_class_0 - num_class_1, replace=True
        )
        oversampled_class_1_images = class_1_images[oversample_indices]
        oversampled_class_1_labels = class_1_labels[oversample_indices]

        # Combine original class 1 images with the oversampled ones
        class_1_images = np.concatenate([class_1_images, oversampled_class_1_images], axis=0)
        class_1_labels = np.concatenate([class_1_labels, oversampled_class_1_labels], axis=0)
    elif num_class_1 > num_class_0:
        # oversample class 0
        # Randomly choose from class_0_images to oversample it
        oversample_indices = np.random.choice(
            np.arange(num_class_0), size=num_class_1 - num_class_0, replace=True
        )
        oversampled_class_0_images = class_0_images[oversample_indices]
        oversampled_class_0_labels = class_0_labels[oversample_indices]

        # Combine original class 0 images with the oversampled ones
        class_0_images = np.concatenate([class_0_images, oversampled_class_0_images], axis=0)
        class_0_labels = np.concatenate([class_0_labels, oversampled_class_0_labels], axis=0)

    # Concatenate class 0 and the class 1 images to get the final balanced dataset
    balanced_train_images = np.concatenate([class_0_images, class_1_images], axis=0)
    balanced_train_labels = np.concatenate([class_0_labels, class_1_labels], axis=0)

    return balanced_train_images, balanced_train_labels

def get_data_loaders(train_batch_size=32, test_val_batch_size=64):
    """
    Build train, validation, and test DataLoaders for ISIC 2020 from image paths and labels
    All splits are converted to tensors and normalized, with data augmentation applied only to the training split
    Args:
        images: file paths to images
        labels: class labels aligned with images
        train_batch_size: batch size for the training loader
        test_val_batch_size: batch size for validation and test loaders
    Returns: tuple: (train_loader, val_loader, test_loader)
    """

    # Extract the data from the given locations
    images, labels = get_data_path(
        root=DATA_PATH,
        metadata_path=CSV_NAME,
        image_dir=os.path.join(DATA_PATH, IMG_DIR)
    )

    # Data split
    train_images, val_images, test_images, train_labels, val_labels, test_labels = split_train_val_test(images, labels)

    # Oversampling for train dataset
    train_images, train_labels = oversample_training(train_images, train_labels)

    # Train dataset transform
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(FLIP_PROB),
        transforms.RandomVerticalFlip(FLIP_PROB),
        transforms.RandomRotation(ROT_DEG),
        transforms.ColorJitter(COLOR_JITTER['brightness'], COLOR_JITTER['contrast'], COLOR_JITTER['saturation'], COLOR_JITTER['hue']),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,
                            STD),
    ])

    # Validate dataset transform
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            MEAN,
            STD
        ),
    ])

    # Test dataset transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            MEAN,
            STD
        )
    ])

    # Train dataset
    train_ds = TripletDataGenerator(
        images=train_images,
        labels=train_labels,
        train=True,
        transform=train_transform 
    )

    # Validation dataset
    val_ds = TripletDataGenerator(
        images=val_images,
        labels=val_labels,
        train=True,
        transform=val_transform
    )

    # Testing dataset
    test_ds = TripletDataGenerator(
        images=test_images,
        labels=test_labels,
        train=False,
        transform=test_transform

    )

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=test_val_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=test_val_batch_size, shuffle=True, num_workers=4)
    return train_loader, val_loader, test_loader