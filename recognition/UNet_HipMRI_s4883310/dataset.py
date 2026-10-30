import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import re

# DATA PREPARATION

# Local computer:
data_dir = r"C:\Users\User\Documents\HipMRI_Data\keras_slices_data"

batch_size = 4

# Directories for training, validation and test sets
train_dir = os.path.join(data_dir, "keras_slices_train")
train_seg_dir = os.path.join(data_dir, "keras_slices_seg_train")

validation_dir = os.path.join(data_dir, "keras_slices_validate")
validation_seg_dir = os.path.join(data_dir, "keras_slices_seg_validate")

test_dir = os.path.join(data_dir, "keras_slices_test")
test_seg_dir = os.path.join(data_dir, "keras_slices_seg_test")

def get_patient_ids(image_folder):
    """Extract patient IDs from MRI filenames."""
    patient_ids = set()

    # Find all NIfTI MRI images in the specified folder.
    image_paths = glob.glob(
        os.path.join(image_folder, "case_*.nii.gz")
    )

    for image_path in image_paths:
        filename = os.path.basename(image_path)

        # Extract the patient ID from the filename
        match = re.match(r"case_(\d+)_", filename)

        if match is None:
            raise ValueError(f"Cannot extract patient ID: {filename}")

        patient_ids.add(match.group(1))

    return patient_ids

def validate_patient_splits():
    """Ensure patients do not appear in multiple dataset splits."""


    # Extract the unique patient IDs from each dataset split.
    train_patients = get_patient_ids(train_dir)
    validation_patients = get_patient_ids(validation_dir)
    test_patients = get_patient_ids(test_dir)

    # Checks for overlap of patient IDs in sets.
    if train_patients & validation_patients:
        raise ValueError("Patient leakage between training and validation")

    if train_patients & test_patients:
        raise ValueError("Patient leakage between training and testing")

    if validation_patients & test_patients:
        raise ValueError("Patient leakage between validation and testing")

    print("Patient-level splitting verified: no overlap.")


# DATA PREPROCESSING

class MRIDataset(Dataset):
    """
    Loads and preprocesses 2D HipMRI images and segmentation masks.
    """

    # Find MRI images and their corresponding segmentation masks
    def __init__(
        self,
        image_folder,
        mask_folder,
        one_hot=False,
        augment=False
    ):
        """
        Initialises the MRI dataset.
        """

        self.one_hot = one_hot
        self.augment = augment

        # Find all MRI files and sort them for consistent ordering.
        self.image_paths = sorted(
            glob.glob(os.path.join(image_folder, "case_*.nii.gz"))
        )

        # Store the segmentation mask corresponding to each MRI image.
        self.mask_paths = []

        for image_path in self.image_paths:

            # Construct the mask filename from the MRI filename
            mask_name = os.path.basename(image_path).replace(
                "case_", "seg_", 1
            )

            # Construct the full path to the segmentation mask
            mask_path = os.path.join(mask_folder, mask_name)

            # Ensure every MRI image has a corresponding mask
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Missing mask: {mask_path}")

            self.mask_paths.append(mask_path)

        # Prevent creating an empty dataset
        if len(self.image_paths) == 0:
            raise ValueError(f"No MRI images found in {image_folder}")

    def __len__(self):
        """ 
        Returns the total number of MRI slices in the dataset. 
        """

        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Loads and preprocesses one MRI image and its mask.
        """

        # Load the MRI image and segmentation mask as NIfTI objects.
        image_nifti = nib.load(self.image_paths[index])
        mask_nifti = nib.load(self.mask_paths[index])

        # Verify spatial alignment
        if not np.allclose(
            image_nifti.affine,
            mask_nifti.affine,
            atol=1e-5
        ):
            raise ValueError(
                f"MRI and mask affine mismatch: {self.image_paths[index]}"
            )

        # Convert the NIfTI images into NumPy arrays.
        image = image_nifti.get_fdata(dtype=np.float32)
        mask = mask_nifti.get_fdata()

        # Check for invalid MRI intensity values
        if not np.isfinite(image).all():
            raise ValueError("MRI contains NaN or infinite values")

        # Check for invalid mask values
        if not np.isfinite(mask).all():
            raise ValueError("Mask contains NaN or infinite values")

        # Check that segmentation labels are integers
        if not np.all(mask == np.round(mask)):
            raise ValueError("Mask contains non-integer labels")

        # Check that segmentation labels are within the expected range of 0 to 5
        if mask.min() < 0 or mask.max() > 5:
            raise ValueError("Mask contains invalid class labels")

        # Remove unnecessary third dimension if its size is 1.
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]

        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]

        # Check that images and masks represent 2D slices.
        if image.ndim != 2 or mask.ndim != 2:
            raise ValueError("Expected 2D MRI image and mask")

        # MRI images and masks must have identical dimensions
        if image.shape != mask.shape:
            raise ValueError("Image and mask dimensions do not match")

        # Check dimensions before padding
        if image.shape[0] != 256 or image.shape[1] > 144:
            raise ValueError(f"Unexpected image dimensions: {image.shape}")

        # Normalise MRI intensity values
        image = (image - image.mean()) / (image.std() + 1e-6)

        # Pad images and masks to 256 x 144
        padding = 144 - image.shape[1]

        image = np.pad(
            image,
            ((0, 0), (0, padding)),
            mode="constant"
        )

        mask = np.pad(
            mask,
            ((0, 0), (0, padding)),
            mode="constant"
        )

        # Convert to PyTorch tensors
        image = torch.from_numpy(image.copy()).unsqueeze(0).float()
        mask = torch.from_numpy(mask.copy()).long()

        # Apply random augmentation only when enabled
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        # Optionally convert integer segmentation labels into
        # six separate binary channels using one-hot encoding.
        if self.one_hot:
            mask = F.one_hot(mask, num_classes=6)
            mask = mask.permute(2, 0, 1).float()

        return image, mask
    
    def apply_augmentation(self, image, mask):
        """
        Applies random augmentation to an MRI image and mask.
        """

        # Random rotation and translation
        if torch.rand(1).item() < 0.5:

            angle = np.deg2rad(np.random.uniform(-10, 10))

            tx = np.random.uniform(-0.05, 0.05)
            ty = np.random.uniform(-0.05, 0.05)

            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            theta = torch.tensor(
                [
                    [cos_a, -sin_a, tx],
                    [sin_a, cos_a, ty]
                ],
                dtype=torch.float32
            ).unsqueeze(0)

            grid = F.affine_grid(
                theta,
                image.unsqueeze(0).shape,
                align_corners=False
            )

            # Bilinear interpolation for MRI intensity values
            image = F.grid_sample(
                image.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False
            ).squeeze(0)

            # Nearest-neighbour interpolation preserves mask labels
            mask = F.grid_sample(
                mask.unsqueeze(0).unsqueeze(0).float(),
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False
            ).squeeze(0).squeeze(0).long()

        # Random intensity scaling
        if torch.rand(1).item() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            image = image * scale

        return image, mask

def analyse_class_imbalance(dataset):
    """
    Calculates the pixel count and percentage of each segmentation class.
    """

    num_classes = 6
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for index in range(len(dataset)):
        _, mask = dataset[index]

        counts = torch.bincount(
            mask.flatten(),
            minlength=num_classes
        )

        class_counts += counts.numpy()

    total_pixels = class_counts.sum()

    print("\nClass Distribution:")

    for class_id, count in enumerate(class_counts):
        percentage = (count / total_pixels) * 100

        print(f"Class {class_id}: {count:,} pixels ({percentage:.2f}%)")

    return class_counts

# DATA INITIALISATION

# Verify that no patient appears in multiple dataset splits
validate_patient_splits()

# Create datasets
train_dataset = MRIDataset(
    train_dir,
    train_seg_dir,
    augment=True
)

validation_dataset = MRIDataset(
    validation_dir,
    validation_seg_dir,
    augment=False
)

test_dataset = MRIDataset(
    test_dir,
    test_seg_dir,
    augment=False
)


# DATALOADERS

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

