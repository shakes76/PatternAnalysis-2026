"""
audit.py - One-off check (don't commit): which 3D slice does each 2D slice come from?

For case 040 week 0, compares several 2D slices against every slice of the 3D volume
(along axis 0 and axis 1) and reports the best-matching 3D index and its correlation.
A correlation near 1.000 means the 2D slice is an exact cut of the 3D volume.
"""

import os

import nibabel as nib
import numpy as np

r = os.path.expanduser("~/Datasets/HipMRI")
vol = nib.load(f"{r}/semantic_MRs/B040_Week0_LFOV.nii.gz").get_fdata()
print("3D volume shape:", vol.shape)


def load(p):
    """Load a 2D slice, dropping the extra trailing dimension if present."""
    a = nib.load(p).get_fdata()
    return a[..., 0] if a.ndim == 3 else a


def zrows(M):
    """Standardise each row (mean 0, std 1); blank rows become all zeros."""
    s = M.std(axis=1, keepdims=True)
    s[s == 0] = np.inf
    return (M - M.mean(axis=1, keepdims=True)) / s


for k in [5, 15, 25, 35, 45, 55]:
    sl = load(f"{r}/keras_slices_data/keras_slices_test/case_040_week_0_slice_{k}.nii.gz")
    if sl.std() == 0:
        print(f"2D slice {k}: blank")
        continue
    s = (sl.ravel() - sl.mean()) / sl.std()
    out = []
    for ax in (0, 1):
        # every 3D slice along this axis, flattened into one row each
        V = zrows(np.moveaxis(vol, ax, 0).reshape(vol.shape[ax], -1))
        c = V @ s / s.size  # correlation of the 2D slice with each 3D slice
        out.append(f"axis {ax}: best 3D index {c.argmax():3d} corr {c.max():.3f}")
    print(f"2D slice {k:2d} -> " + " | ".join(out))