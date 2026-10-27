"""Probe the fixed reference in the full training path list; run from repo root."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from dataset import (
    IGNORE_INDEX,
    HipMRIPairDataset,
    collate_padded,
    collect_pairs,
    crop_to_original,
)

data_root = Path(
    "../../Final Report/HipMRI_2D_Improved_UNet_学习/data/keras_slices_data"
)
image_path = data_root / "keras_slices_train" / "case_004_week_0_slice_12.nii.gz"
mask_path = data_root / "keras_slices_seg_train" / "seg_004_week_0_slice_12.nii.gz"

image = nib.load(image_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

print("MRI shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask labels:", np.unique(mask))

training_pairs = collect_pairs(
    data_root / "keras_slices_train", data_root / "keras_slices_seg_train"
)
reference_index = training_pairs.index((image_path, mask_path))
training_dataset = HipMRIPairDataset(training_pairs)
print("Dataset samples:", len(training_dataset))
print("Reference list index:", reference_index)
reference_loader = DataLoader(
    training_dataset, batch_size=1, sampler=[reference_index], num_workers=0,
    collate_fn=collate_padded,
)
print("Reference loader batches:", len(reference_loader))
batch = next(iter(reference_loader))
image_batch, mask_batch = batch["image"], batch["target"]
valid_batch = batch["valid_mask"]
original_size = batch["original_sizes"][0]
normalized_image = crop_to_original(image_batch[0], original_size)
mask_tensor = crop_to_original(mask_batch[0], original_size)

print("MRI range before:", image.min().item(), image.max().item())
print("MRI range after:", normalized_image.min().item(), normalized_image.max().item())
print("Normalized MRI shape:", tuple(normalized_image.shape))
print("Normalized MRI dtype:", normalized_image.dtype)
print("Mask tensor shape:", tuple(mask_tensor.shape))
print("Mask tensor dtype:", mask_tensor.dtype)
print("Mask labels after MRI normalization:", torch.unique(mask_tensor).tolist())

print("MRI batch shape:", tuple(image_batch.shape))
print("Mask batch shape:", tuple(mask_batch.shape))
print("Batch size:", image_batch.shape[0])
print("Original size:", original_size)
print("Valid pixels:", valid_batch.sum().item())
print("Padding pixels:", valid_batch.numel() - valid_batch.sum().item())
print("Padding target ignore index:", IGNORE_INDEX)

# These four display colors cover the labels observed in this fixed slice.
mask_colors = ListedColormap(["black", "gold", "deepskyblue", "magenta"])
fig, axes = plt.subplots(1, 2, figsize=(8, 6), layout="constrained")
axes[0].imshow(image, cmap="gray", origin="upper", interpolation="nearest")
axes[0].set_title("Training MRI")
mask_plot = axes[1].imshow(
    mask, cmap=mask_colors, vmin=-0.5, vmax=3.5,
    origin="upper", interpolation="nearest"
)
axes[1].set_title("Reference mask")
fig.colorbar(mask_plot, ax=axes[1], ticks=[0, 1, 2, 3], label="Class ID")
fig.suptitle("Training pair: case 004 | week 0 | slice 12")

output_path = Path("outputs/training_pair_case_004_week_0_slice_12.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
plt.close(fig)
print("Preview saved:", output_path.resolve())
