import os
import torch
import numpy as np
from tqdm import tqdm
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

def to_channels(arr: np.ndarray, num_classes=6, dtype=np.uint8) -> np.ndarray:
    """Fixed version that enforces exactly 6 classes regardless of slice content."""
    res = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    for c in range(num_classes):
        res[..., c][arr == c] = 1
    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    """Load medical image data from names list into a padded, uniform pre-allocated 4D array."""
    affines = []
    num = len(imageNames)

    # Standardize dimensions to 256x256 (ideal for U-Net pooling)
    TARGET_ROWS = 256
    TARGET_COLS = 256

    if categorical:
        # Peek at the first case to determine the number of one-hot channels
        first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
        if len(first_case.shape) == 3:
            first_case = first_case[:, :, 0]
        first_case = to_channels(first_case, dtype=dtype)
        channels = first_case.shape[-1]

        images = np.zeros((num, TARGET_ROWS, TARGET_COLS, channels), dtype=dtype)
    else:
        images = np.zeros((num, TARGET_ROWS, TARGET_COLS), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)

        # Safely pad the images into the top-left corner of the 256x256 array
        r = min(inImage.shape[0], TARGET_ROWS)
        c = min(inImage.shape[1], TARGET_COLS)

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :r, :c, :] = inImage[:r, :c, :]
        else:
            images[i, :r, :c] = inImage[:r, :c]

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    return images


class HipMRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        # Pre-load all images and masks directly into system RAM
        self.images = load_data_2D(image_paths, normImage=True, early_stop=False)
        self.masks = load_data_2D(mask_paths, categorical=True, dtype=np.uint8, early_stop=False)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        # PyTorch expects (Channels, Height, Width)
        # Images are (256, 256), add channel dim -> (1, 256, 256)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()

        # Masks are (256, 256, 6), permute to (6, 256, 256)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float()

        return img_tensor, mask_tensor


def get_dataloaders(base_dir, batch_size=16):
    """Loads pre-split data directly from the respective directories."""

    def get_paths(img_folder, seg_folder):
        imgs = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.nii.gz')])
        masks = sorted([os.path.join(seg_folder, f) for f in os.listdir(seg_folder) if f.endswith('.nii.gz')])
        return imgs, masks

    # Define paths based on your keras_slices_data folder structure
    train_imgs, train_masks = get_paths(os.path.join(base_dir, 'keras_slices_train'),
                                        os.path.join(base_dir, 'keras_slices_seg_train'))

    val_imgs, val_masks = get_paths(os.path.join(base_dir, 'keras_slices_validate'),
                                    os.path.join(base_dir, 'keras_slices_seg_validate'))

    test_imgs, test_masks = get_paths(os.path.join(base_dir, 'keras_slices_test'),
                                      os.path.join(base_dir, 'keras_slices_seg_test'))

    # Create DataLoaders
    train_loader = DataLoader(HipMRIDataset(train_imgs, train_masks), batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(HipMRIDataset(val_imgs, val_masks), batch_size=batch_size, shuffle=False, num_workers=0,
                            pin_memory=False)
    test_loader = DataLoader(HipMRIDataset(test_imgs, test_masks), batch_size=batch_size, shuffle=False, num_workers=0,
                             pin_memory=False)

    return train_loader, val_loader, test_loader


def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, orient=False,
                 early_stop=False):
    """
    Load medical image data from names, cases list provided into a list for each.
    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.
    """
    affines = []
    interp = 'linear'
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    first_case = niftiImage.get_fdata(caching='unchanged')

    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine

        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data

        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :] = inImage
        else:
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage

        affines.append(affine)

        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

class HipMRI3DDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        # Pre-load all 3D volumes and masks directly into system RAM
        self.images = load_data_3D(image_paths, normImage=True, early_stop=False)
        self.masks = load_data_3D(mask_paths, categorical=True, dtype=np.uint8, early_stop=False)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        # PyTorch Conv3d expects (Channels, Depth, Height, Width)
        # Images from loader are (H, W, D), permute to (D, H, W) and add channel dim -> (1, D, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

        # Masks from loader are (H, W, D, C), permute to (C, D, H, W)
        mask_tensor = torch.from_numpy(mask).permute(3, 2, 0, 1).float()

        return img_tensor, mask_tensor