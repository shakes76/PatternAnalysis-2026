"""
audit_resampled.py - Third check: are the 2D slices axial slices of the 3D volume,
squashed from 256x256 to 256x128 (resampled rather than cropped)?

Tries every axial slice in all 8 rotations/flips, halves its width by averaging
column pairs, and reports the best correlation with each 2D slice.
"""
import os
import sys

import nibabel as nib
import numpy as np

r = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Datasets/HipMRI")
vol = nib.load(f"{r}/semantic_MRs/B040_Week0_LFOV.nii.gz").get_fdata()


def load(p):
    a = nib.load(p).get_fdata()
    return a[..., 0] if a.ndim == 3 else a


def orientations(a):
    """All 8 rotations/flips of a square 2D array."""
    for t in (a, a.T):
        for name, f in (("", t), ("flipUD", t[::-1]), ("flipLR", t[:, ::-1]), ("flipBoth", t[::-1, ::-1])):
            yield ("T " if t is not a else "") + name, f


def z(a):
    return (a - a.mean()) / (a.std() + 1e-12)


for k in [10, 30, 50]:
    sl = load(f"{r}/keras_slices_data/keras_slices_test/case_040_week_0_slice_{k}.nii.gz")
    if sl.std() == 0:
        print(f"2D slice {k}: blank"); continue
    s = z(sl)
    best = (-2, None)
    for zi in range(vol.shape[2]):
        for name, a in orientations(vol[:, :, zi]):
            small = a.reshape(a.shape[0], a.shape[1] // 2, 2).mean(axis=2)  # 256x256 -> 256x128
            if small.std() == 0:
                continue
            c = float((z(small) * s).mean())
            if c > best[0]:
                best = (c, zi, name)
    print(f"2D slice {k:2d} -> axial z={best[1]:3d} [{best[2] or 'as-is'}] corr {best[0]:.4f}")