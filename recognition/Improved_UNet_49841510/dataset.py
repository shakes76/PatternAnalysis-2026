from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class HipMRIDataset(Dataset):

    def __init__(self, root, split, target_size=(256, 144)):
        self.root = Path(root)
        self.split = split
        self.target_size = target_size
        # Select MRI and segmentation directories
        self.image_dir = self.root / f"keras_slices_{split}"
        self.mask_dir = self.root / f"keras_slices_seg_{split}"
        self.image_files = sorted(
            self.image_dir.glob("*.nii.gz")
        )

        self.pairs = []

        for image_path in self.image_files:

            mask_name = image_path.name.replace(
                "case_", "seg_", 1
            )

            mask_path = self.mask_dir / mask_name

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing mask for {image_path.name}"
                )

            self.pairs.append((image_path, mask_path))

        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]

        # 1. Load MRI and mask
        image = nib.load(str(image_path)).get_fdata()
        mask = nib.load(str(mask_path)).get_fdata()

        # 2. Validate original spatial alignment
        if image.shape != mask.shape:
            raise ValueError(
                f"Image-mask shape mismatch: {image_path.name}"
            )

        # 3. Normalize MRI
        image = image.astype(np.float32)

        mean = image.mean()
        std = image.std()

        image = (image - mean) / (std + 1e-8)

        # 4. Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0)

        # 5. Pad image and mask to target size
        height, width = image.shape[-2:]

        target_height, target_width = self.target_size

        # Calculate required padding
        pad_height = target_height - height
        pad_width = target_width - width

        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                f"Image {image_path.name} is larger than target size: "
                f"{(height, width)} vs {self.target_size}"
            )

        # Split padding equally between opposite sides
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padding = (
            pad_left,
            pad_right,
            pad_top,
            pad_bottom
        )

        # Pad MRI with normalized intensity 0
        image = F.pad(
            image,
            padding,
            mode="constant",
            value=0
        )

        # Pad segmentation mask with ignore label
        mask = F.pad(
            mask,
            padding,
            mode="constant",
            value=-100
        )

        # Remove mask channel dimension
        mask = mask.squeeze(0)

        return image, mask

if __name__ == "__main__":

    DATA_ROOT = Path(
        r"D:\academic\Semester 2 2026\COMP3710"
        r"\data_for_report(private)\keras_slices_data"
    )

    train_dataset = HipMRIDataset(DATA_ROOT, "train")
    validation_dataset = HipMRIDataset(DATA_ROOT, "validate")
    test_dataset = HipMRIDataset(DATA_ROOT, "test")

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))
    print("Test samples:", len(test_dataset))

    print("First pair:", train_dataset.pairs[0])
    image, mask = train_dataset[0]

    print("===== First Sample =====")
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)

    print("Mask shape:", mask.shape)
    print("Mask dtype:", mask.dtype)

    print("Image mean:", image.mean().item())
    print("Image std:", image.std().item())

    print("Mask labels:", torch.unique(mask))

    # Check a sample with original width 144
    special_index = None

    for i, (image_path, _) in enumerate(train_dataset.pairs):
        if nib.load(str(image_path)).shape == (256, 144):
            special_index = i
            break

    if special_index is not None:
        special_image, special_mask = train_dataset[special_index]

        print("===== Special Sample =====")
        print("Image shape:", special_image.shape)
        print("Mask shape:", special_mask.shape)
        print("Mask labels:", torch.unique(special_mask))

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    images, masks = next(iter(train_loader))

    print("===== First Batch =====")
    print("Images:", images.shape)
    print("Masks:", masks.shape)
    print("Image dtype:", images.dtype)
    print("Mask dtype:", masks.dtype)

    all_labels = set()

    for i in range(len(train_dataset)):
        _, mask = train_dataset[i]

        all_labels.update(torch.unique(mask).tolist())

    print("Training set labels:", sorted(all_labels))