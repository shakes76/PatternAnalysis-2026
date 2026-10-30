"""
data_stats.py - Quick facts about the HipMRI data, for the feasibility review / README.

Prints, for the training split:
  - how much of the data each class takes up (class imbalance)
  - the 3D volume shapes and voxel sizes
Run:  python data_stats.py [data_root] [split]     (defaults: ~/Datasets/HipMRI train)
"""
import glob
import os
import sys
from collections import Counter

import nibabel as nib
import numpy as np

NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Datasets/HipMRI")
split = sys.argv[2] if len(sys.argv) > 2 else "train"

# ---- 2D slices ----
counts2d = np.zeros(6)
segs2d = sorted(glob.glob(f"{root}/keras_slices_data/keras_slices_seg_{split}/*.nii*"))
slices_with_prostate = 0
for f in segs2d:
    m = np.rint(nib.load(f).get_fdata()).astype(int).ravel()
    c = np.bincount(m, minlength=6)[:6]
    counts2d += c
    slices_with_prostate += c[5] > 0
print(f"2D {split}: {len(segs2d)} mask slices, {slices_with_prostate} contain prostate "
      f"({100 * slices_with_prostate / max(len(segs2d), 1):.1f}%)")
for i, n in enumerate(NAMES):
    print(f"  {n:<11}{100 * counts2d[i] / max(counts2d.sum(), 1):7.3f}% of pixels")

# ---- 3D volumes ----
import re
cases = {"train": range(4, 36), "validate": range(36, 40), "test": range(40, 43)}[split]
segs3d = [f for f in sorted(glob.glob(f"{root}/semantic_labels_only/*.nii*"))
          if int(re.search(r"(\d{3})", os.path.basename(f)).group(1)) in cases]
counts3d, shapes, spacings = np.zeros(6), Counter(), Counter()
for f in segs3d:
    img = nib.load(f)
    m = np.rint(img.get_fdata()).astype(int).ravel()
    counts3d += np.bincount(m, minlength=6)[:6]
    shapes[img.shape[:3]] += 1
    spacings[tuple(round(float(z), 2) for z in img.header.get_zooms()[:3])] += 1
print(f"\n3D {split}: {len(segs3d)} volumes")
for i, n in enumerate(NAMES):
    print(f"  {n:<11}{100 * counts3d[i] / max(counts3d.sum(), 1):7.3f}% of voxels")
print("  shapes:   ", dict(shapes))
print("  voxel mm: ", dict(spacings))