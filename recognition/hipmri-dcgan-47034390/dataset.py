from pathlib import Path
import re

import nibabel as nib
import torch
import numpy as np
from torch.utils.data import Dataset

# Hip MRI dataset loader.
valid_extensions = ('.nii', '.nii.gz', '.png', '.jpg', '.jpeg', '.npy', '.npz')
CROP_WIDTH = 128
CROP_START = 8
IMAGE_SHAPE = (256, 128)

def normalize_image(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1, 99])
    image = np.clip(image, low, high)

    if high > low:
        image = 2.0 * (image - low) / (high - low) - 1.0
    else:
        image = np.zeros_like(image)

    return image

def preprocess_image(
    image: np.ndarray,
    target_shape: tuple[int, int] = IMAGE_SHAPE,
    ) -> np.ndarray:
    """Preprocess the image by cropping it to the target shape.

    Args:
        image (np.ndarray): The input image to preprocess.
        target_shape (tuple[int, int], optional): The desired output shape. Defaults to IMAGE_SHAPE.

    Raises:
        ValueError: If the input image is smaller than the target shape.
        ValueError: If the cropped image does not match the target shape.

    Returns:
        np.ndarray: The cropped image.
    """
    height, width = target_shape
    image_height, image_width = image.shape

    if image_height < height or image_width < width:
        raise ValueError(
            f"Image shape {image.shape} is smaller than target {target_shape}"
        )

    top = (image_height - height) // 2
    left = (image_width - width) // 2

    image = image[top:top + height, left:left + width]

    if image.shape != target_shape:
        raise ValueError(
            f"Expected image shape {target_shape}, got {image.shape}"
        )

    return image

def get_slice_count(path: Path) -> int:
    """Get the number of slices in a NIfTI file.

    Args:
        path (Path): Path to the NIfTI file.

    Returns:
        int: Number of slices in the NIfTI file.
    """
    shape = nib.load(str(path)).shape #type:ignore

    if len(shape) == 2:
        return 1
    if len(shape) == 3:
        return shape[2]

    raise ValueError(f"Unexpected NIfTI shape {shape}: {path}")

class HipMRIDataset(Dataset):
    # Hip MRI dataset loader.
    def __init__(
        self,
        data_dir,
        max_images=None,
        norm_image=True,
        dtype=np.float32,
        week=None,
        cases=None,
    ):
        """Initialize the HipMRIDataset. A data directory must be supplied.
        Returns a HipMRIDataset instance, normalized and preprocessed to the size IMAGE_SHAPE.

        Args:
            data_dir (str or Path): Path to the dataset directory.
            max_images (int, optional): Maximum number of images to load.
            norm_image (bool, optional): Whether to normalize images.
            dtype (type, optional): Data type for the images.
            week (int, optional): Week number to filter images.
            cases (list, optional): List of case IDs to filter images.
        """
        
        self.norm_image = norm_image
        self.dtype = dtype
        self.image_paths = []
        
        for path in sorted(Path(data_dir).rglob("*.nii.gz")):
            match = re.search(
                r"case_(\d+)_week_(\d+)_slice_(\d+)",
                path.name,
            )

            if match is None:
                continue

            case_id, week_id = map(int, match.groups()[:2])

            if week is not None and week_id != week:
                continue
            if cases is not None and case_id not in cases:
                continue
            self.image_paths.append(path)

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        if not self.image_paths:
            raise ValueError(
                "No matching NIfTI files found. Check data_dir, week and cases."
            )

        print(f"NIfTI files found: {len(self.image_paths)}")

    def __len__(self):
        """Get the number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get a preprocessed image by index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        path = self.image_paths[idx]
        img = nib.load(str(path))

        if len(img.header.get_data_shape()) != 2: #type:ignore
            raise ValueError(
                f"Expected a 2D image, got {img.header.get_data_shape()}: {path}" #type:ignore
            )

        image = np.asarray(img.dataobj, dtype=self.dtype) #type:ignore
        image = preprocess_image(image)
        if self.norm_image:
            image = normalize_image(image)
        image = np.ascontiguousarray(image[None, :, :])
        return torch.from_numpy(image)

def main():
    import yaml
    with open("config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    settings = {
        **config["defaults"],
        **config["hipmri_dcgan"],
    }

    dataset = HipMRIDataset(
        data_dir=settings["data_dir"],
        week=settings["week"],
        cases=settings.get("cases"),
        norm_image=settings.get("norm_image", True),
    )

    sample = dataset[0]

    print(f"Dataset length: {len(dataset)}")
    print(f"Sample shape: {tuple(sample.shape)}")
    print(f"Sample range: {sample.min().item():.3f} "
          f"to {sample.max().item():.3f}")
    print(f"Sample mean: {sample.mean().item():.3f}")
    print(f"Sample std: {sample.std().item():.3f}")

if __name__ == "__main__":
    main()