"""
check_crop.py - Verify that centre-cropping the odd-shaped slices to width 128
removes only background.

Control: each original (uncropped) slice is shown with the strips that the crop
removes shaded red, next to the cropped result. Quantitative checks: brightness
of the removed strips vs the whole slice, and the number of labelled pixels
(from the matching segmentation mask) inside the removed strips.

Usage (from the project folder):
    python eda/check_crop.py --data-dir "<path>/keras_slices_data"
"""
import argparse
import glob
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Rectangle

TARGET_W = 128


def slice_num(path):
    return int(re.search(r"slice_(\d+)", path).group(1))


def strip_widths(w):
    """Columns removed on the left and right, matching dataset.py's centre crop."""
    excess = max(w - TARGET_W, 0)
    left = excess // 2
    return left, excess - left


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="images/crop_check.png")
    parser.add_argument("--show", action="store_true", help="also open the figure")
    args = parser.parse_args()

    img_dir = os.path.join(args.data_dir, f"keras_slices_{args.split}")
    seg_dir = os.path.join(args.data_dir, f"keras_slices_seg_{args.split}")

    # 1. Find odd-shaped slices from the file headers (fast: no pixel loading)
    files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    odd = sorted((f for f in files if nib.load(f).shape[1] != TARGET_W),
                 key=lambda f: (os.path.basename(f).split("_slice_")[0], slice_num(f)))
    if not odd:
        print("No odd-shaped slices found.")
        return
    scans = Counter(os.path.basename(f).split("_slice_")[0] for f in odd)
    print(f"Odd-shaped slices: {len(odd)}, in scans: {dict(scans)}")

    # 2. Measure what the crop removes, slice by slice
    worst_ratio, labelled_removed, missing_masks = 0.0, 0, 0
    for f in odd:
        img = nib.load(f).get_fdata(dtype=np.float32)
        left, right = strip_widths(img.shape[1])
        strips = np.concatenate([img[:, :left], img[:, img.shape[1] - right:]], axis=1)
        if img.max() > 0:
            worst_ratio = max(worst_ratio, strips.max() / img.max())

        seg_path = os.path.join(seg_dir, os.path.basename(f).replace("case_", "seg_", 1))
        if os.path.exists(seg_path):
            seg = nib.load(seg_path).get_fdata()
            seg_strips = np.concatenate([seg[:, :left], seg[:, seg.shape[1] - right:]], axis=1)
            labelled_removed += int((seg_strips > 0).sum())
        else:
            missing_masks += 1

    print(f"Columns removed per slice: left {left}, right {right}")
    print(f"Brightest removed pixel, as a fraction of its slice's max: {worst_ratio:.3f}")
    print(f"Labelled (non-background) mask pixels inside removed strips: {labelled_removed}")
    if missing_masks:
        print(f"Warning: {missing_masks} slices had no matching mask file")

    # 3. Visual control: original with removed strips shaded, next to the cropped result
    picks = [odd[0], odd[len(odd) // 2], odd[-1]]
    fig, axes = plt.subplots(len(picks), 2, figsize=(7, 4 * len(picks)))
    for row, f in enumerate(picks):
        img = nib.load(f).get_fdata(dtype=np.float32)
        h, w = img.shape
        left, right = strip_widths(w)
        cropped = img[:, left:w - right]

        axes[row, 0].imshow(img, cmap="gray")
        for x0, width in ((0, left), (w - right, right)):
            axes[row, 0].add_patch(Rectangle((x0 - 0.5, -0.5), width, h,
                                             color="red", alpha=0.35))
        axes[row, 0].set_title(f"original {h}x{w}, slice {slice_num(f)}")
        axes[row, 1].imshow(cropped, cmap="gray")
        axes[row, 1].set_title(f"cropped {cropped.shape[0]}x{cropped.shape[1]}")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f"Saved figure to {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()