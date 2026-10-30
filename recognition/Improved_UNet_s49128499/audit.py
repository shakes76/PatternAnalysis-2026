import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import torch
from dataset import build_dataloaders
import glob
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "keras_slices_data")

# check labels
# img = nib.load(os.path.join(DATA_DIR, "keras_slices_train/case_004_week_7_slice_44.nii.gz")).get_fdata()
# lbl = nib.load(os.path.join(DATA_DIR, "keras_slices_seg_train/seg_004_week_7_slice_44.nii.gz")).get_fdata()
# if lbl.ndim == 3: 
#     lbl = lbl[:, :, 0]

# fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# axes[0].imshow(img.T, cmap="gray", origin="lower"); axes[0].set_title("MRI")

# cmap = ListedColormap(["black", "grey", "green", "blue", "purple", "red"])
# # observed_labels = ['Background', 'Body', 'Bone', 'Bladder', 'Rectum', 'Prostate']
# axes[1].imshow(lbl.T, cmap=cmap, origin="lower"); axes[1].set_title("All labels")
# plt.show()

# class balance audit
train_loader, val_loader, test_loader = build_dataloaders(base_dir=DATA_DIR, batch_size=4, num_workers=0)

# xb, yb = next(iter(val_loader))
# print("Batch shape:", xb.shape, yb.shape)

# total_pixels = torch.zeros(6)
# n_slices_with_class = torch.zeros(6)
# n_slices = 0

# for images, masks in val_loader:
#     total_pixels += masks.sum(dim=(0, 2, 3))
#     # How many slices in this batch contain each class?
#     per_slice = masks.sum(dim=(2, 3)) > 0    # (B, 6) → True if class present
#     n_slices_with_class += per_slice.sum(dim=0)
#     n_slices += masks.shape[0]

# print(f"\nAcross {n_slices} training slices:")
# for c in range(6):
#     pct = 100 * total_pixels[c] / total_pixels.sum()
#     print(f"  class {c}: {int(total_pixels[c]):>12,} px ({pct:5.2f}%)  "
#           f"present in {int(n_slices_with_class[c])}/{n_slices} slices")

# leakage
def patient_ids(label_dir):
    files = glob.glob(os.path.join(label_dir, "*.nii.gz"))
    ids = set()
    for f in files:
        name = os.path.basename(f)
        # "seg_004_week_0_slice_0" → "004"
        parts = name.split("_")
        if len(parts) >= 2:
            ids.add(parts[1])
    return ids

train_ids = patient_ids(os.path.join(DATA_DIR, "keras_slices_seg_train"))
val_ids   = patient_ids(os.path.join(DATA_DIR, "keras_slices_seg_validate"))
test_ids  = patient_ids(os.path.join(DATA_DIR, "keras_slices_seg_test"))

print(f"\nPatients — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
print("train ∩ val :", train_ids & val_ids)
print("train ∩ test:", train_ids & test_ids)
print("val   ∩ test:", val_ids & test_ids)