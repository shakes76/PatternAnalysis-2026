import os
import glob
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import nibabel as nib
import numpy as np

class HipMRIDataset(Dataset):
    """
    Custom PyTorch Dataset for COMP3710 HipMRI 2D slices (.nii.gz format).
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing .nii.gz slices 
                            (e.g., '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train')
            transform (callable, optional): Optional transform/augmentation to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Collect all .nii.gz files in the target directory
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .nii.gz files found in {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Load NIfTI file using Nibabel
        nii_img = nib.load(file_path)
        img_data = nii_img.get_fdata(dtype=np.float32)
        
        # Remove singleton dimensions if any exist (e.g., shape (H, W, 1) -> (H, W))
        img_data = np.squeeze(img_data)
        
        # Min-Max Normalization to scale pixel intensities between [0.0, 1.0]
        min_val, max_val = np.min(img_data), np.max(img_data)
        if max_val - min_val > 0:
            img_data = (img_data - min_val) / (max_val - min_val)
        else:
            img_data = np.zeros_like(img_data)
        # Shift from [0, 1] to [-1, 1] for WGAN compatibility
        img_data = img_data * 2.0 - 1.0

        # Convert to PyTorch Tensor and add Channel dimension -> shape: [1, H, W]
        tensor_img = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)

        # Resize to uniform dimensions
        # F.interpolate expects 4D tensor [B, C, H, W], so we temporarily add a batch dim
        tensor_img = F.interpolate(
            tensor_img.unsqueeze(0), 
            size=(256, 256),  # Target size (H, W)
            mode='bilinear', 
            align_corners=False
        ).squeeze(0) # Remove batch dim back to [1, H, W]

        if self.transform:
            tensor_img = self.transform(tensor_img)

        return tensor_img