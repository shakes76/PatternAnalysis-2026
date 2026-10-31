import os
import re
import glob
 
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


HIPMRI_ROOT = "C:/Users/lucas/Desktop/keras_slices_data"

SPLITS = {
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "validate": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}
IMG_SHAPE = (256, 128) 
NUM_CLASSES = 6

def to_channels(arr, num_classes=NUM_CLASSES, dtype=np.uint8):
    """One-hot encode a label map (H, W) -> (H, W, num_classes)."""
    res = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    for c in range(num_classes):
        res[..., c][arr == c] = 1
    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, early_stop=False, shape=IMG_SHAPE):
    """
    Load 2D NIfTI slices into one pre-allocated array.
 
    normImage:   z-score normalise each slice (mean 0, std 1)
    categorical: one-hot encode label maps -> (N, H, W, NUM_CLASSES)
    """
    num = len(imageNames)
    if categorical:
        images = np.zeros((num, *shape, NUM_CLASSES), dtype=dtype)
    else:
        images = np.zeros((num, *shape), dtype=dtype)
    affines = []
 
    for i, inName in enumerate(tqdm(imageNames, desc="Loading")):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")   
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]                        
        w0 = (inImage.shape[1] - shape[1]) // 2
        inImage = inImage[:shape[0], w0:w0 + shape[1]].astype(np.float32)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            images[i] = to_channels(inImage.round(), dtype=dtype)
        else:
            images[i] = inImage.astype(dtype)
        affines.append(niftiImage.affine)
        if early_stop and i > 20:
            images = images[:i + 1]
            break
    if getAffines:
        return (images, affines) 
    else: 
        return images


def id(path):
    """Patient number from a file"""
    match = re.search(r"(\d+)", os.path.basename(path))
    if match:
        return match.group(1)
    return None

def get_file_pairs(root_dir, split):
    """Sorted (image paths, mask paths) for a split"""
    img_folder, seg_folder = SPLITS[split]
    img_paths = sorted(glob.glob(os.path.join(root_dir, img_folder, "*.nii*")))
    seg_paths = sorted(glob.glob(os.path.join(root_dir, seg_folder, "*.nii*")))
 
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No NIfTI files found")
    if len(img_paths) != len(seg_paths):
        raise ValueError(f"mask issue")
    for img_p, seg_p in zip(img_paths, seg_paths):
        if os.path.basename(img_p).split("_", 1)[1] != os.path.basename(seg_p).split("_", 1)[1]:
            raise ValueError(f"Image/mask mismatch: {img_p} vs {seg_p}")
    return img_paths, seg_paths

class HipMRIDataset(Dataset):
    """
    Loads a whole split into memory with load_data_2D, then serves
    (image, mask) pairs: image (1, H, W), mask (H, W)  labels 0-5.
    """
    def __init__(self, root_dir=HIPMRI_ROOT, split="train", augment=False, early_stop=False):
        img_paths, seg_paths = get_file_pairs(root_dir, split)
        self.images = load_data_2D(img_paths, normImage=True, early_stop=early_stop)
        self.masks = load_data_2D(seg_paths, dtype=np.uint8, early_stop=early_stop)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        mask = torch.from_numpy(self.masks[idx]).long()
        if self.augment:
            image, mask = self._augment(image, mask)
        return image.unsqueeze(0), mask

    def _augment(self, image, mask):
        if torch.rand(1) < 0.5:                    
            image, mask = image.flip(-1), mask.flip(-1)
        if torch.rand(1) < 0.5:                       
            image = image * (0.9 + 0.2 * torch.rand(1)) + 0.1 * torch.randn(1)
        return image, mask

def get_dataloaders(root_dir=HIPMRI_ROOT, batch_size=16, num_workers=0, early_stop=False):
    """Build train/validate/test DataLoaders. Augmentation is used for training only."""
    train_ds = HipMRIDataset(root_dir, "train", augment=True, early_stop=early_stop)
    val_ds = HipMRIDataset(root_dir, "validate", early_stop=early_stop)
    test_ds = HipMRIDataset(root_dir, "test", early_stop=early_stop)
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    for split in SPLITS:
        img_paths, _ = get_file_pairs(HIPMRI_ROOT, split)
        print(f"{split}: {len(img_paths)} slices from {len({id(p) for p in img_paths})} patients")
 
    train = HipMRIDataset(split="train")
    masks = torch.from_numpy(train.masks)
    counts = torch.bincount(masks.flatten().long(), minlength=NUM_CLASSES)
    print("Pixel share per label (train):")
    for label, n in enumerate(counts.tolist()):
        print(f"  {label}: {100 * n / counts.sum().item():.2f}%")
    has_prostate = (masks == 5).flatten(1).any(dim=1).float().mean().item()
    print(f"Slices containing prostate: {100 * has_prostate:.1f}%")
 