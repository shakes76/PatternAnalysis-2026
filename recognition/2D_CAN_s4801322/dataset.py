from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import config
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

# load medical image functions
def load_data_2D(
        imageNames, 
        normImage = False, 
        categorical = False, 
        dtype = np.float32, 
        getAffines = False, 
        early_stop = False,
        target_size = config.PAD_SIZE
        ):
    """
    Load medical image data from names , cases list ``provided`` into a list for each.

    This function pre - allocates 3D arrays for conv2d to avoid excessive
    memory usage.

    early_stop : Stop loading pre - maturely , allocates and loads only the first 22 images, for
    quick loading and testing scripts.
    """
    affines = []

    # get fixed size
    num = min(len(imageNames), 22) if early_stop else len(imageNames)
    rows, cols = target_size
    images = np.zeros((num, rows, cols), dtype=dtype)
    if categorical:
        raise ValueError("Use categorical=False for fixed-label segmentation; one-hot encode after loading")

    for i, inName in enumerate(tqdm(imageNames[:num], desc="Loading NIfTI")):
        niftiImage = nib.load(str(inName))
        inImage = niftiImage.get_fdata(caching="unchanged")
        affine = niftiImage.affine

        if inImage.ndim == 3 and inImage.shape[-1] == 1:
            inImage = inImage[:, :, 0]
        if inImage.ndim != 2:
            raise ValueError(f"Expected a 2D slice or [H,W,1], got {inImage.shape}: {inName}")

        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        height, width = inImage.shape
        if height > rows or width > cols:
            raise ValueError(
                f"{inName} has shape {inImage.shape}, larger than "
                f"config.PAD_SIZE={target_size}; increase PAD_SIZE"
            )
        top = (rows - height) // 2
        left = (cols - width) // 2
        images[i, top:top + height, left:left + width] = inImage
        affines.append(affine)

    return (images, affines) if getAffines else images

class HipMRI_Dataset(Dataset):
    """Load paired HipMRI images and segmentation masks.

    Returns:
        image: float32 tensor [1, H, W], standardized per slice.
        mask: int64 tensor [H, W], containing class indices.
    """

    def __init__(self, image_directory, segmentation_directory, early_stop=False):
        self.image_directory = Path(image_directory)
        self.segmentation_directory = Path(segmentation_directory)

        for folder in (self.image_directory, self.segmentation_directory):
            if not folder.is_dir():
                raise FileNotFoundError(f"Directory not found: {folder.resolve()}")

        # Match by the path relative to each root, not list position.
        image_files = sorted(image_directory.rglob("case_*.nii.gz"))
        if not image_files:
            raise RuntimeError(f"No case_*.nii.gz files in {image_directory}")
        self.pairs = []
        for image_path in image_files:
            relative_path = image_path.relative_to(image_directory)

            if not relative_path.name.startswith("case_"):
                raise ValueError(f"Unexpected MRI filename: {relative_path.name}")

            mask_name = "seg_" + relative_path.name[len("case_"):]
            mask_path = self.segmentation_directory / relative_path.parent / mask_name

            if mask_path.is_file():
                self.pairs.append((image_path, mask_path))
            else:
                raise FileNotFoundError(f"No segmentation matching {image_path}: {mask_path}")
        
        if early_stop:
            self.pairs = self.pairs[:22]

        print(f"Found {len(self.pairs)} paired HipMRI slices")

        self.images = load_data_2D(
            [p for p, _ in self.pairs], normImage=True, dtype=np.float32
        )
        raw_masks = load_data_2D(
            [p for _, p in self.pairs], normImage=False, categorical=False, dtype=np.float32
        )
        if self.images.shape != raw_masks.shape:
            raise ValueError("MRI images and masks have incompatible dimensions")
        
        label_values = getattr(config, "LABEL_VALUES", list(range(config.NUM_CLASSES)))
        if len(label_values) != config.NUM_CLASSES or len(set(label_values)) != len(label_values):
            raise ValueError("config.LABEL_VALUES must contain NUM_CLASSES unique labels")
        self.masks = np.full(raw_masks.shape, -1, dtype=np.int64)
        for class_index, raw_value in enumerate(label_values):
            self.masks[raw_masks == raw_value] = class_index
        if np.any(self.masks < 0):
            unknown = np.unique(raw_masks[self.masks < 0]).tolist()
            raise ValueError(f"Unknown segmentation label values: {unknown}; update config.LABEL_VALUES")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].copy()).unsqueeze(0).float()
        mask = torch.from_numpy(self.masks[index].copy()).long()
        return image, mask


def main():
    train_dataset = HipMRI_Dataset(
    image_directory=config.TRAIN_DIRECTORY,
    segmentation_directory=config.TRAIN_SEG_DIRECTORY,
    )
    image, mask = train_dataset[40]
    print(f"Training pairs: {len(train_dataset)}")
    print(f"MRI shape: {tuple(image.shape)}, dtype: {image.dtype}")
    print(f"Mask shape: {tuple(mask.shape)}, dtype: {mask.dtype}")
    print(f"Mask labels in first image: {torch.unique(mask).tolist()}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # MRI image
    axes[0].imshow(image.squeeze().numpy(), cmap="gray")
    axes[0].set_title("Original HipMRI Image")
    axes[0].axis("off")

    # Ground truth segmentation
    axes[1].imshow(mask.numpy(), cmap="viridis", interpolation="nearest")
    axes[1].set_title("Ground Truth Segmentation")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
