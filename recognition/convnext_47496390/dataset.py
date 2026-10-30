"""
ADNI 2D MRI Slice Data Pipeline and Leakage-Free Splitting.
Contains:
    - ADNISliceDataset: PyTorch Dataset for loading single-channel 2D MRI slices.
    - get_transforms: Domain-appropriate augmentation pipeline (rotation, affine, jitter, erasing).
    - build_dataloaders: Patient-level stratified train/val/test dataloader builder ensuring zero-leakage cross-patient separation
     and uniform validation distribution coverage across volumetric pixel-level intensity quintiles..
"""

import os
import glob
import json
import random
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# https://share.gemini.google/3vBDdHud5qyP
class ADNISliceDataset(Dataset):
    """
    Custom Dataset for ADNI 2D MRI slices.
    Loads single-channel grayscale brain MRI images and returns (image_tensor, label).
    Attributes:
        file_paths (list[str]): List of absolute or relative image file paths.
        labels (list[int]): Class integer labels (0 for NC, 1 for AD).
        transform (callable, optional): PyTorch transformation pipeline.
        patient_ids (list[str], optional): Corresponding patient ID for each slice
    """
    def __init__(self, file_paths, labels, transform=None, patient_ids=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load image as grayscale ('L')
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), self.patient_ids[idx]


def get_transforms(in_channels=1):
    """
    Returns training and validation/test transformation pipelines.
    """
    train_ops = [
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1,contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, value = 0),
    ]

    val_ops = [
        transforms.ToTensor(),
    ]

    # Standard grayscale normalization (mean=0.5, std=0.5)
    train_ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    val_ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    train_transform = transforms.Compose(train_ops)
    val_transform = transforms.Compose(val_ops)

    return train_transform, val_transform

def compute_patient_mean_intensities(patient_slices, cache_path=None):
    """
    Computes volumetric mean pixel intensity per patient across representative slices.
    Caches computed values to JSON to provide instantaneous O(1) startup on subsequent runs.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as fp:
                cached = json.load(fp)
                if len(cached) == len(patient_slices):
                    return cached
        except Exception:
            pass
    patient_intensities = {}
    for pid, flist in patient_slices.items():
        # Sample 5 evenly distributed slices across the 20-slice volume for speed and precision
        sample_slices = flist[:: max(1, len(flist) // 5)]
        slice_means = []
        for f in sample_slices:
            try:
                img = Image.open(f).convert('L')
                t = transforms.functional.to_tensor(img)  # Tensor in [0.0, 1.0]
                slice_means.append(t.mean().item() * 255.0)  # Scale [0, 255]
            except Exception:
                continue
        patient_intensities[pid] = float(sum(slice_means) / len(slice_means)) if slice_means else 0.0
    if cache_path:
        try:
            with open(cache_path, "w") as fp:
                json.dump(patient_intensities, fp)
        except Exception:
            pass
    return patient_intensities


def build_dataloaders(
    in_channels=1,
    data_root="AD_NC",
    val_split=0.2,
    batch_size=16,
    num_workers=4,
    seed=42
):
    """
    Constructs PyTorch DataLoaders with strict patient-level separation.
    All 2D slices from any given patient are placed exclusively into either
    the training, validation, or test partition, mathematically guaranteeing zero
    cross-patient data leakage:
    """
    classes = {'NC': 0, 'AD': 1}
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')
    
    num_bins = 5

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Testing directory not found at: {test_dir}")

    # ---------------------------------------------------------
    # Group training slices by Patient ID to prevent leakage
    # ---------------------------------------------------------
    patient_slices = defaultdict(list)
    patient_label = {}

    for cls_name, cls_idx in classes.items():
        cls_folder = os.path.join(train_dir, cls_name)
        pattern = os.path.join(cls_folder, "*.jpeg")
        for file_path in glob.glob(pattern):
            # Filename: <patient_id>_<slice_idx>.jpeg
            filename = os.path.basename(file_path)
            patient_id = filename.split('_')[0]

            patient_slices[patient_id].append(file_path)
            patient_label[patient_id] = cls_idx

    # Pixel-Level Volumetric Intensity Computation
    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patient_intensity_cache.json")
    if not os.path.exists(cache_file) and os.path.exists("patient_intensity_cache.json"):
        cache_file = "patient_intensity_cache.json"
    
    patient_intensities = compute_patient_mean_intensities(patient_slices, cache_path=cache_file)
    
    # Binned Stratified Split Across Intensity Quintiles
    rng = random.Random(seed)
    train_patients, val_patients = [], []

    print(f"\n--- Binned Validation Partitioning ({num_bins} Bins) ---")

    for cls_name, cls_idx in classes.items():
        cls_pids = [pid for pid, l in patient_label.items() if l == cls_idx]
        # Sort by volumetric mean pixel intensity [0 - 255]
        cls_pids = sorted(cls_pids, key=lambda p: patient_intensities.get(p, 0.0))
        
        n_pids = len(cls_pids)

        effective_bins = max(1, min(num_bins, n_pids))
        bin_size = n_pids / effective_bins

        print(f"Class {cls_name} ({n_pids} patients) partitioned into {effective_bins} bins:")
        
        for b in range(effective_bins):
            start_idx = int(round(b * bin_size))
            end_idx = int(round((b + 1) * bin_size))
            bin_pids = cls_pids[start_idx:end_idx]
            
            # Deterministically shuffle bin before sampling
            shuffled_bin = list(bin_pids)
            rng.shuffle(shuffled_bin)
            
            # Draw exactly val_split fraction from this bin
            n_val = int(round(len(shuffled_bin) * val_split))
            val_p = shuffled_bin[:n_val]
            train_p = shuffled_bin[n_val:]
            
            val_patients.extend(val_p)
            train_patients.extend(train_p)

            b_vals = [patient_intensities.get(p, 0.0) for p in bin_pids]
            print(f"  Bin {b}: Intensity [{min(b_vals):.2f}, {max(b_vals):.2f}] (mean={sum(b_vals)/len(b_vals):.2f}) -> Train: {len(train_p)}, Val: {len(val_p)}")

    # Verify mathematical separation: strictly zero leakage
    leakage = set(train_patients).intersection(set(val_patients))
    if leakage:
        raise ValueError(f"CRITICAL ERROR: Patient leakage detected in train/val split: {leakage}")
    
    # Flatten patient slices back into slice lists with patient ID tracking
    train_paths, train_labels, train_pids = [], [], []
    for pid in train_patients:
        for path in patient_slices[pid]:
            train_paths.append(path)
            train_labels.append(patient_label[pid])
            train_pids.append(pid)

    val_paths, val_labels, val_pids = [], [], []
    for pid in val_patients:
        for path in patient_slices[pid]:
            val_paths.append(path)
            val_labels.append(patient_label[pid])
            val_pids.append(pid)

    # Collect test slices
    test_paths, test_labels, test_pids = [], [], []
    test_patient_set = set()

    for cls_name, cls_idx in classes.items():
        cls_folder = os.path.join(test_dir, cls_name)
        for file_path in glob.glob(os.path.join(cls_folder, "*.jpeg")):
            filename = os.path.basename(file_path)
            patient_id = filename.split('_')[0]
            test_paths.append(file_path)
            test_labels.append(cls_idx)
            test_pids.append(patient_id)
            test_patient_set.add(patient_id)

    # Check for train/test leakage
    all_train_val_patients = set(train_patients).union(set(val_patients))
    test_leakage = all_train_val_patients.intersection(test_patient_set)
    if test_leakage:
        raise ValueError(f"CRITICAL ERROR: Patient leakage detected between train/val and test: {test_leakage}")
    
    print(f"\nADNI Dataset Summary (Strict Patient-Level Partitioning):")
    print(f"  Train: {len(train_paths):>6} slices from {len(train_patients):>4} unique patients")
    print(f"  Val:   {len(val_paths):>6} slices from {len(val_patients):>4} unique patients")
    print(f"  Test:  {len(test_paths):>6} slices from {len(test_patient_set):>4} unique patients\n")

    # Create datasets and loaders
    train_tf, val_tf = get_transforms(in_channels=in_channels)
    
    train_dataset = ADNISliceDataset(train_paths, train_labels, transform=train_tf, patient_ids=train_pids)
    val_dataset = ADNISliceDataset(val_paths, val_labels, transform=val_tf, patient_ids=val_pids)
    test_dataset = ADNISliceDataset(test_paths, test_labels, transform=val_tf, patient_ids=test_pids)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders(
        in_channels=1,
        data_root="/home/groups/comp3710/ADNI/AD_NC",
        val_split=0.2,
        batch_size=16
    )

    # Verification batch (unpacking images, labels, patient_ids)
    images, labels, pids = next(iter(train_loader))
    print("\nVerification Batch Tensor Shapes:")

    print(f"  Images shape: {images.shape} (Expected: [16, 1, 240, 256])")
    print(f"  Labels shape: {labels.shape} (Expected: [16])")
    print(f"  Sample Patient IDs: {pids[:3]}")

