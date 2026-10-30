"""
==============================================================================
File:        dataset.py
Author:      Jonathan Lu
Course:      COMP3710 / COMP7710 - Pattern Recognition
Date:        October 2026
Description: Data loader and preprocessing module for 2D Hip MRI slice data.
             Handles reading NIfTI files, intensity normalisation, dimension
             filtering/padding, and PyTorch Dataset wrapping with data augmentations.
==============================================================================
"""

import numpy as np
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import os
import glob
import torch
from torchvision import transforms


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Convert a single-channel label mask into a one-hot multi-channel array.

    Args:
        arr (np.ndarray): Input array of integer class labels.
        dtype: Output data type for the resulting mask (default: np.uint8).

    Returns:
        np.ndarray: One-hot encoded multi-channel array with channels on the last axis.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res


def load_data_2D(imageNames, shape, normImage, dtype, early_stop, pad):
    """
    Load 2D medical NIfTI image slices into a pre-allocated 4D array.

    Args:
        imageNames (list): List of file paths to .nii or .nii.gz slice files.
        shape (tuple): Target (rows, cols) dimensions, e.g. (256, 128).
        normImage (bool): Whether to min-max normalise pixel values to [0.0, 1.0].
        dtype: Data type for the loaded NumPy array (e.g. np.float32).
        early_stop (bool): If True, stops loading after 20 slices for smoke-testing.
        pad (bool): If True, zero-pads smaller slices to target shape; otherwise discards mismatched slices.

    Returns:
        np.ndarray: Array of shape (N, 1, target_rows, target_cols) containing loaded slices.
    """
    affines = []
    # Pre-allocate 4D array (N, 1, H, W) to avoid dynamic memory reallocations
    num = len(imageNames)
    target_rows = shape[0]
    target_cols = shape[1]
    images = np.zeros((num, 1, target_rows, target_cols), dtype=dtype)
    write_idx = 0
    
    for inName in tqdm(imageNames):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        
        # Squeeze singleton depth dimension if present in raw slices
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]
        inImage = inImage.astype(dtype)
        
        # Min-max intensity normalisation to [0.0, 1.0]
        if normImage:
            img_min = inImage.min()
            img_max = inImage.max()
            inImage = (inImage - img_min) / (img_max - img_min + 1e-8)
            
        rows, cols = inImage.shape
        
        # Handle slice dimensions (pad with zeroes or filter mismatched sizes)
        if pad:
            row_pad = max(0, target_rows - rows)
            top_pad = row_pad // 2
            bot_pad = row_pad - top_pad
            
            col_pad = max(0, target_cols - cols)
            left_pad = col_pad // 2
            right_pad = col_pad - left_pad
            
            if row_pad > 0 or col_pad > 0:
                inImage = np.pad(inImage, ((top_pad, bot_pad), (left_pad, right_pad)), 'constant', constant_values=0)
        else:
            # Discard slices that do not match the expected target resolution
            if rows != target_rows or cols != target_cols:
                continue
            
        images[write_idx, 0, :, :] = inImage
        write_idx += 1
        
        # Early exit for smoke testing
        if write_idx > 20 and early_stop:
            break
            
    return images[:write_idx]


class HipMRIDataset(Dataset):
    """
    PyTorch Dataset wrapper for 2D Hip MRI slices.

    Loads preprocessed NIfTI slices from a specified directory and applies optional
    on-the-fly data augmentations during training.
    """
    def __init__(self, dir, shape=(256, 128), normImage=True, dtype=np.float32, early_stop=False, pad=False, augment=False):
        """
        Initialise the HipMRIDataset.

        Args:
            dir (str): Directory path containing .nii slice files.
            shape (tuple): Target (height, width) dimensions. Defaults to (256, 128).
            normImage (bool): Whether to normalise slice intensities to [0, 1]. Defaults to True.
            dtype: NumPy data type for stored images. Defaults to np.float32.
            early_stop (bool): If True, limits loading to 20 images for quick debugging. Defaults to False.
            pad (bool): If True, zero-pads mismatched slices; if False, drops them. Defaults to False.
            augment (bool): If True, applies random horizontal flips and mild affine jitter. Defaults to False.
        """
        self.img_names = sorted(glob.glob(os.path.join(dir, '*.nii*')))
        self.imgs = load_data_2D(self.img_names, shape, normImage, dtype, early_stop, pad)
        
        if augment:
            # Training data augmentation: random horizontal flips and mild spatial affine jitter
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=5, translate=(0.02, 0.02))
            ])
        else:
            self.transform = None

    def __len__(self):
        """Return the total number of loaded slices in the dataset."""
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Retrieve a single MRI slice by index.

        Args:
            idx (int): Sample index.

        Returns:
            torch.Tensor: Single-channel image tensor of shape (1, H, W).
        """
        img = torch.from_numpy(self.imgs[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img
