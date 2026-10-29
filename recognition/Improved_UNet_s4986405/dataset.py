"""Minimal NIfTI file pairing and PyTorch Dataset for HipMRI 2D slices."""

from pathlib import Path

import nibabel as nib
import torch
from torch.utils.data import Dataset


class HipMRI2DDataset(Dataset):
    """Pair case_*.nii.gz images with seg_*.nii.gz masks by filename."""

    def __init__(
        self, image_dir, mask_dir, normalize=False, one_hot=False,
        training=False, augment=False,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.normalize = normalize
        self.one_hot = one_hot
        if augment and not training:
            raise ValueError("Augmentation requires training=True")
        self.training = training
        self.augment = augment

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.pairs = []
        for image_path in sorted(self.image_dir.glob("case_*.nii.gz")):
            mask_name = "seg_" + image_path.name[len("case_"):]
            mask_path = self.mask_dir / mask_name
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
            self.pairs.append((image_path, mask_path))

        if not self.pairs:
            raise ValueError(f"No case_*.nii.gz images found in {self.image_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = torch.tensor(nib.load(str(image_path)).get_fdata(), dtype=torch.float32)
        if not torch.isfinite(image).all():
            raise ValueError(f"Image contains NaN or Inf: {image_path}")

        mask = torch.tensor(
            nib.load(str(mask_path)).get_fdata(), dtype=torch.float64
        )

        if not torch.isfinite(mask).all():
            raise ValueError(f"Mask contains NaN or Inf: {mask_path}")
        if not (mask == mask.round()).all():
            raise ValueError(f"Mask contains non-integer labels: {mask_path}")
        if not ((mask >= 0) & (mask <= 5)).all():
            raise ValueError(f"Mask labels must be in range 0-5: {mask_path}")

        mask = mask.long()

        if image.ndim != 2:
            raise ValueError(
                f"Expected a 2D image, got shape {tuple(image.shape)}: {image_path}"
            )
        if mask.ndim != 2:
            raise ValueError(
                f"Expected a 2D mask, got shape {tuple(mask.shape)}: {mask_path}"
            )
        if image.shape != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch: "
                f"{tuple(image.shape)} vs {tuple(mask.shape)}; "
                f"image={image_path}, mask={mask_path}"
            )

        # Apply the same spatial operation before normalization and encoding.
        if self.augment and torch.rand(()).item() < 0.5:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        if self.normalize:
            values = image.to(torch.float64)
            std = values.std(unbiased=False)
            if std == 0:
                image = torch.zeros_like(image)
            else:
                image = ((values - values.mean()) / std).to(torch.float32)

        if self.one_hot:
            mask = torch.nn.functional.one_hot(mask, num_classes=6)
            mask = mask.permute(2, 0, 1).to(torch.float32).contiguous()

        return image.unsqueeze(0), mask
