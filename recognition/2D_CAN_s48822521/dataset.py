from torch.utils.data import Dataset, DataLoader

import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random
from scipy import ndimage as ndi

ROOT = r"Dataset\keras_slices_data"
IMAGE_SHAPE = (256, 128) # Got it from looking inside the data
FLIP_CHANCE = 0.4



def extract_key(name: str):
    """Extract patient, week, and slice numbers from a slice filename.

    Args:
        name: Filename formatted like "case_004_week_0_slice_3.nii.gz".

    Returns:
        Tuple of integer (patient, week, slice) identifiers.
    """
    _, patient_num, _, week_num, _, slice_part = name.split("_")
    slice_num, _, _ = slice_part.split(".")

    return (int(patient_num), int(week_num), int(slice_num))

def build_img_seg_pairs(img_dir, seg_dir):
    """Pair image and segmentation files by patient, week, and slice.

    Files whose image or segmentation shape differs from "IMAGE_SHAPE" are
    excluded. Image and segmentation directories must contain the same number
    of files, and corresponding filenames must have matching identifiers.

    Args:
        img_dir: Directory containing image files.
        seg_dir: Directory containing segmentation files.

    Returns:
        Dictionary mapping each (patient, week, slice) key to its image and
        segmentation file paths.
    """

    img_files = sorted(os.listdir(img_dir), key=extract_key)
    seg_files = sorted(os.listdir(seg_dir), key=extract_key)

    # Should be same, as we're relying on indices to pair them up
    assert len(img_files) == len(seg_files), "not same file count between images and masks" 

    img_seg_pairs = {}
    for i in range(len(img_files)):
        full_img_path = os.path.join(img_dir, img_files[i])
        full_seg_path = os.path.join(seg_dir, seg_files[i])

        # Should match based off of file names
        assert extract_key(img_files[i]) == extract_key(seg_files[i]), "Mismatched key between images and masks" 

        # There was this like couple of datafiles that had a different shape, (256, 144), so decided to remove them from training set (didn't appear in validation nor test sets so we chill)
        if (nib.load(full_img_path).shape != IMAGE_SHAPE 
            or nib.load(full_seg_path).shape != IMAGE_SHAPE):
            # print(f"Removed: {extract_key(img_files[i])}")
            continue

        img_seg_pairs[extract_key(img_files[i])] =  (full_img_path, full_seg_path)

    return img_seg_pairs

def get_seg_classes(seg_dir):
    """Count the distinct label values present in a segmentation directory.

    Gets the num_classes essentially.

    Args:
        seg_dir: Directory containing segmentation files.

    Returns:
        Number of distinct segmentation labels found.
    """

    labelIDs = set()

    for filename in os.listdir(seg_dir):
        full_path = os.path.join(seg_dir, filename)
        seg = nib.load(full_path).get_fdata(caching='unchanged') # A grid of numbers, with entries as the labels IDs to segment
        arr = np.rint(seg).astype(int) # Same grid, but with clean integers

        labelIDs.update(np.unique(arr).tolist()) # Add all unique labels to set

    return len(labelIDs)

def enforce_patient_level_splitting(train_pairs, validate_pairs, test_pairs):
    """Check that patient IDs do not overlap across dataset splits.

    Args:
        train_pairs: Image-segmentation pairs for training.
        validate_pairs: Image-segmentation pairs for validation.
        test_pairs: Image-segmentation pairs for testing.

    Raises:
        ValueError: If any patient appears in more than one split.
    """
    train_set = set(key[0] for key in train_pairs.keys())
    validate_set = set(key[0] for key in validate_pairs.keys())
    test_set = set(key[0] for key in test_pairs.keys())
    
    if (train_set & validate_set != set()) or (validate_set & test_set != set()) or (train_set & test_set != set()):
        raise ValueError("Patient-level leakage detected between training/validate/test splitting")
    else:
        print("No patient-level data leakage!")






class HipMRIDataset2D(Dataset):
    """PyTorch dataset for 2D hip MRI slices and segmentation masks.

    Loads and normalizes images, optionally augments the image and mask with
    matching flips and rotations, and converts masks to one-hot tensors.

    Args:
        pairs: Mapping from (patient, week, slice) keys to image and
            segmentation file paths.
        num_classes: Number of segmentation classes for one-hot encoding.
        augmentationEnabled: Whether to apply random augmentation.
    """

    def __init__(self, pairs, num_classes, augmentationEnabled = False):
        self.pairs = pairs
        self.keys = list(pairs.keys()) # Already in order/sorted thanks to dictionary's order
        self.num_classes = num_classes
        self.augmentationEnabled = augmentationEnabled

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, seg_path = self.pairs[self.keys[i]]

        # Load image and segmentation label
        img = nib.load(img_path).get_fdata(caching='unchanged') # (256, 128)
        img = img.astype(np.float32) # PyTorch expects float32, not float64
        seg = nib.load(seg_path).get_fdata(caching='unchanged') # (256, 128)
        seg = seg.astype(np.int64)

        # z-score normalisation for the image
        img_mean = img.mean()
        img_std = img.std() + 1e-8 # "+ 1e-8" To avoid divide by 0 error
        img = (img - img_mean) / img_std

        # Data augmentation
        if (self.augmentationEnabled):
            # Random flips (FLIP_CHANCE of doing it)
            if random.random() < FLIP_CHANCE:
                # Flip with respect to the horizontal line
                img = np.flip(img, axis=0).copy()
                seg = np.flip(seg, axis=0).copy()

            # Random slight rotations (-15 to 15 degrees)
            random_angle = np.random.uniform(-15, 15)
            img = ndi.rotate(img, random_angle, order=1, reshape=False, mode="constant", cval=0.0, prefilter=False)
            seg = ndi.rotate(seg, random_angle, order=0, reshape=False, mode="constant", cval=0.0, prefilter=False)

        # Convert image to tensor (add channel/class axis)
        img_tensor = torch.from_numpy(img).unsqueeze(0) # (256, 128) -> (1, 256, 128)

        # Convert segmentation label to tensor
        seg_tensor = torch.from_numpy(seg)

        # One-hot encoding on the segmentation mask
        seg_onehot_tensor = F.one_hot(seg_tensor, num_classes=self.num_classes) # (256, 128, 6)
        seg_onehot_tensor = seg_onehot_tensor.permute(2, 0, 1).float() # (256, 128, 6) -> (6, 256, 128)

        # Key, if needed while in the loops of the dataset
        key_tensor = torch.tensor(self.keys[i])


        return img_tensor, seg_onehot_tensor, key_tensor 


def get_dataloaders(root, batch_size, num_workers=0):
    """Create training, validation, and test data loaders.

    Builds image-segmentation pairs and datasets from the expected subdirectories
    under root, checks patient-level split separation, and enables augmentation
    for the training dataset only.

    Args:
        root: Directory containing the image and segmentation split folders.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes used to load data.

    Returns:
        Tuple of training, validation, and test data loaders, followed by the
        number of segmentation classes.
    """

    # Get all paths
    train_img = os.path.join(root, "keras_slices_train")
    train_seg = os.path.join(root, "keras_slices_seg_train")
    validate_img = os.path.join(root, "keras_slices_validate")
    validate_seg = os.path.join(root, "keras_slices_seg_validate")
    test_img = os.path.join(root, "keras_slices_test")
    test_seg = os.path.join(root, "keras_slices_seg_test")

    # Get all img-seg pairs
    train_pairs = build_img_seg_pairs(train_img, train_seg)
    validate_pairs = build_img_seg_pairs(validate_img, validate_seg)
    test_pairs = build_img_seg_pairs(test_img, test_seg)

    # Ensure there is patient-level splitting amongst the 3 levels
    enforce_patient_level_splitting(train_pairs, validate_pairs, test_pairs)

    # Find number of classes
    num_classes = get_seg_classes(train_seg) 

    # Build datasets
    train_dataset = HipMRIDataset2D(train_pairs, num_classes, augmentationEnabled=True)
    validate_dataset = HipMRIDataset2D(validate_pairs, num_classes, augmentationEnabled=False)
    test_dataset = HipMRIDataset2D(test_pairs, num_classes, augmentationEnabled=False)

    # Build the data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, validate_dataloader, test_dataloader, num_classes