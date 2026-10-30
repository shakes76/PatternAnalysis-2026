"""
audit_axial.py - Second check: are the 2D slices axial slices of the 3D volume, cropped to 128 wide?

The first audit only compared against cuts along axes 0 and 1. The 2D images look axial
(bladder in the centre, femoral heads either side), so this checks axis-2 slices of the
256x256x128 volume, cropped to every possible 128-wide window, as-is and transposed/flipped.
"""
import os
import sys

import nibabel as nib
import numpy as np

r = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Datasets/HipMRI")
vol = nib.load(f"{r}/semantic_MRs/B040_Week0_LFOV.nii.gz").get_fdata()
print("3D volume shape:", vol.shape)


def load(p):
    a = nib.load(p).get_fdata()
    return a[..., 0] if a.ndim == 3 else a


for k in [10, 30, 50]:
    sl = load(f"{r}/keras_slices_data/keras_slices_test/case_040_week_0_slice_{k}.nii.gz")
    if sl.std() == 0:
        print(f"2D slice {k}: blank"); continue
    s = (sl - sl.mean()) / sl.std()
    best = (-2, None)
    for z in range(vol.shape[2]):
        ax = vol[:, :, z]
        # candidate 256x128 windows: crop columns of the axial slice, or rows of its transpose
        for name, base in (("as-is", ax), ("transposed", ax.T)):
            for flip in (False, True):
                b = base[::-1] if flip else base
                if b.shape[0] != sl.shape[0]:
                    continue
                for o in range(0, b.shape[1] - sl.shape[1] + 1):
                    c = b[:, o:o + sl.shape[1]]
                    sd = c.std()
                    if sd == 0:
                        continue
                    corr = float(((c - c.mean()) / sd * s).mean())
                    if corr > best[0]:
                        best = (corr, (z, name, "flipped" if flip else "", o, float(np.abs(c - sl).max())))
    corr, (z, name, flip, o, maxdiff) = best
    print(f"2D slice {k:2d} -> axial z={z:3d} {name} {flip} cols {o}:{o + sl.shape[1]} "
          f"corr {corr:.4f} max abs diff {maxdiff:.4f}")