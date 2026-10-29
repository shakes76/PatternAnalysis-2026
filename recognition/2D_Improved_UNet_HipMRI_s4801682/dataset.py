from pathlib import Path
import re

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# Location of the HipMRI 2D dataset on Rangpur
DATA_DIR = Path(
    "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
)

# Regex pattern matching case_XXX_week_Y_slice_Z.nii.gz and seg_XXX_week_y_slice_z.nii.gz
IMAGE_PATTERN = re.compile(
    r"^case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$",
    re.IGNORECASE
)

MASK_PATTERN = re.compile(
    r"^seg_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$",
    re.IGNORECASE
)

SPLIT_FOLDERS = { 
    "train": (
        "keras_slices_train", 
        "keras_slices_seg_train"
    ), 
    "val": (
        "keras_slices_validate", 
        "keras_slices_seg_validate"
    ), 
    "test": (
        "keras_slices_test", 
        "keras_slices_seg_test"
    )
}

# Label mapping based on full 12,660-slice audit 
NUM_CLASSES = 6 
TARGET_PROSTATE_LABEL = 3

class HipMRIDataset(Dataset): 
    """Custom PyTorch Dataset for 2D Pelvic 
       MRI slices and multi-class segmentation masks.
    """
    def __init__(self, file_pairs, is_train=False):
        """
            Arguments:
            file_pairs (list of tuple): list of (image_path, mask_path) tuples.
            is_train (bool): If True, applies random augmentations (flips, rotations).
        """
        self.file_pairs = file_pairs
        self.is_train = is_train

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.file_pairs[idx]

        # Load the MRI image and its segmentation mask
        img_nii = nib.load(str(img_path))
        mask_nii = nib.load(str(mask_path))

        img_data = img_nii.get_fdata(caching="unchanged")
        mask_data = mask_nii.get_fdata(caching="unchanged")

        # Ensure the image and mask are 2D spatial dimensions (256, 128)
        if img_data.ndim == 3 and img_data.shape[-1] == 1:
            img_data = img_data[:, :, 0]

        if mask_data.ndim == 3 and mask_data.shape[-1] == 1:
            mask_data = mask_data[:, :, 0]

        if img_data.ndim != 2 or mask_data.ndim != 2:
            raise ValueError("Expected 2D MRI images and masks.")

        # Check that the image and mask are spatially compatible
        if img_data.shape != mask_data.shape:
            raise ValueError("Image and mask shapes do not match.")

        # Validate image and mask values
        if not np.all(np.isfinite(img_data)):
            raise ValueError("MRI image contains non-finite values.")

        if not np.all(np.isfinite(mask_data)):
            raise ValueError("Mask contains non-finite values.")

        if not np.all(mask_data == np.floor(mask_data)):
            raise ValueError("Mask contains non-integer labels.")

        mask_data = mask_data.astype(np.int64)

        if np.any(mask_data < 0) or np.any(mask_data >= NUM_CLASSES):
            raise ValueError("Mask contains unexpected label values.")

        # Intensity Normalisation (0 mean, unit veriance)
        img_data = img_data.astype(np.float32)
        img_mean = img_data.mean()
        img_std = img_data.std()
        img_data = (img_data - img_mean) / (img_std + 1e-8)

        # Convert the image and mask to PyTorch Tensors
        # Image shape: (C, H, W) = (1, 256, 128)
        # C = channels. The num. of color or modality feature maps. 
        # MRI scan is single-modulity greyscale slice, so it only has C=1 channel.
        # H = Height. The vertical spatial resolution (num. of rows of pixels). In dataset, H = 256
        # W = width. The horizontal spatial resolution (num. of columns of pixels). In dataset, W = 128
        img_tensor = torch.from_numpy(img_data).unsqueeze(0)
        # Mask shape: (H, W), with integer class labels
        mask_tensor = torch.from_numpy(mask_data.copy()).long()

        # Standardise spatial dimensions to (256, 128) 
        # MRI images use bilinear interpolation. 
        # Masks use nearest-neighbour interpolation to preserve labels.
        target_size = (256, 128)

        if img_tensor.shape[-2:] != target_size:
            img_tensor = TF.resize(
                img_tensor,
                list(target_size),
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True
            )

            mask_tensor = TF.resize(
                mask_tensor.unsqueeze(0),
                list(target_size),
                interpolation=TF.InterpolationMode.NEAREST
            ).squeeze(0).long()
        
        # Synchronised Data Augmentation (applied only during Training)
        if self.is_train:
            # Random Horizontal flip (p=0.5)
            if torch.rand(1).item() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                mask_tensor = TF.hflip(mask_tensor)

            # Random Subtle Rotation (-10 to +10 degrees)
            if torch.rand(1).item() > 0.5:
                angle = float(torch.empty(1).uniform_(-10, 10).item())
                img_tensor = TF.rotate(
                    img_tensor, 
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=0
                )
                # Use NEAREST interpolation on masks to preserve discrete label integers
                mask_tensor = TF.rotate(
                    mask_tensor.unsqueeze(0),
                    angle, 
                    interpolation=TF.InterpolationMode.NEAREST,
                    fill=0
                ).squeeze(0).long()

        # Final shape checks:
        if img_tensor.shape != (1, 256, 128):
            raise ValueError(
                f"Unexpected image tensor shape {tuple(img_tensor.shape)}"
                f"for {img_path.name}"
            )

        if mask_tensor.shape != (256, 128):
            raise ValueError(
                f"Unexpected mask tensor shape {tuple(mask_tensor.shape)}"
                f"for {mask_path.name}"
            )

        # Return 1 image and its corresponding mask
        return img_tensor, mask_tensor

def load_split_pairs(data_dir=DATA_DIR, split="train"):
    """
        Pairs existing image and mask files directly 
        from their allocated split folders.
    """
    img_folder, mask_folder = SPLIT_FOLDERS[split]
    img_dir = Path(data_dir) / img_folder
    mask_dir = Path(data_dir) / mask_folder

    # Index masks by (case, week , slice)
    mask_lookup = {}

    for mp in mask_dir.glob("seg_*.nii.gz"):
        match = MASK_PATTERN.match(mp.name)
        if match:
            mask_lookup[match.groups()] = mp

    pairs = []

    for ip in sorted(img_dir.glob("case_*.nii.gz")):
        match = IMAGE_PATTERN.match(ip.name)
        if match and match.groups() in mask_lookup:
            pairs.append((ip, mask_lookup[match.groups()]))

    return pairs

def get_dataloaders(data_dir=DATA_DIR, batch_size=16, num_workers=2):
    """
        Builds training, validation and testing dataloaders from existing folder splits
        Arguments:
            data_dir (Path): Path to dataset folder
            batch_size (int): Batch size for gradient steps and evaluation
            num_workers (int): parallelworker processses for data loading.
        Returns:
            tuple: (train_loader, val_loader, test_loader)
    """
    train_pairs = load_split_pairs(data_dir, split="train")
    val_pairs = load_split_pairs(data_dir, split="val")
    test_pairs = load_split_pairs(data_dir, split="test")

    train_dataset = HipMRIDataset(train_pairs, is_train=True) 
    val_dataset = HipMRIDataset(val_pairs, is_train=False) 
    test_dataset = HipMRIDataset(test_pairs, is_train=False)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader( 
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader( 
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=use_pin_memory
    ) 
    
    test_loader = DataLoader( 
        test_dataset,
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=use_pin_memory
    ) 
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Standalone sanity test for dataset.py
    print("=" * 60)
    print("RUNNING DATASET.PY PIPELINE SANITY TEST")
    print("=" * 60)

    # Test pair discovery across splits
    train_pairs = load_split_pairs(DATA_DIR, split="train")
    val_pairs = load_split_pairs(DATA_DIR, split="val")
    test_pairs = load_split_pairs(DATA_DIR, split="test")

    print(f"Discovered Slice Pairs:")
    print(f"    - Training slices: {len(train_pairs):,}")
    print(f"    - Validation slices: {len(val_pairs):,}")
    print(f"    - Testing slices: {len(test_pairs):,}")

    # Test single-sample loading and transformations
    print(f"\nTesting Single-Sample Loading (HipMRIDataset)...")
    train_dataset = HipMRIDataset(train_pairs, is_train=True)
    sample_img, sample_mask = train_dataset[0]

    print(
        f"    - Image Shape: {tuple(sample_img.shape)} | Dtype:"
        f" {sample_img.dtype}"
    )
    print(
        f"    - Image Stats: Mean = {sample_img.mean():.4f}, Std ="
        f" {sample_img.std():.4f}"
    )
    print(
        f"    - Mask Shape: {tuple(sample_mask.shape)} | Dtype:"
        f" {sample_mask.dtype}"
    )
    print(
        "   - Mask Unique Labels:"
        f" {torch.unique(sample_mask).tolist()} (Should be integers in 0..5)"
    )

    # Validate shapes
    assert sample_img.shape == (
        1, 
        256,
        128,
    ), f"Unexpected image shape: {sample_img.shape}"
    assert sample_mask.shape == (
        256, 
        128,
    ), f"Unexpected mask shape: {sample_mask.shape}"
    assert sample_mask.dtype == torch.int64, "Mask must be torch.int64"

    # test DataLoader batching and worker pipeline
    print(f"\n Testing DataLoader Batch Assembly...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=DATA_DIR, batch_size=8, num_workers=2
    )

    batch_imgs, batch_masks = next(iter(train_loader))

    print(f"    - Batch Images Shape: {tuple(batch_imgs.shape)} (Expected: [8, 1, 256, 128])")
    print(f"    - Batch Masks Shape: {tuple(batch_masks.shape)} (Expected: [8, 256, 128])")

    assert batch_imgs.shape == (8, 1, 256, 128), "batch image dimensions mismatch!"
    assert batch_masks.shape == (8, 256, 128), "Batch mask dimensions mismatch!"

    print("\n" + "=" * 65)
    print("SUCCESS: dataset.py pipeline verified and ready for training!")
    print("=" * 65)