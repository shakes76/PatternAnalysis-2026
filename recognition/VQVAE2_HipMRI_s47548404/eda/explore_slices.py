"""
explore_slices.py - First look at the HipMRI 2D slices (exploratory data analysis).

For one scan, prints the slice shape, stored dtype and intensity range, and saves:
  - eda_slices.png       5 slices spread through the scan
  - eda_masks.png        MRI | segmentation mask | overlay, for 3 slices
  - eda_slice_means.png  mean intensity of every slice, in order (edge-slice check)

Usage (from the project folder):
    python eda/explore_slices.py --data-dir "<path>/keras_slices_data"
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def slice_num(path):
    return int(re.search(r"slice_(\d+)", path).group(1))


def scan_files(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))
    return sorted(files, key=slice_num)


def load(path):
    return nib.load(path).get_fdata(dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--scan", default="004_week_0", help="e.g. 004_week_0")
    parser.add_argument("--out-dir", default="images")
    parser.add_argument("--show", action="store_true", help="also open the figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    imgs = scan_files(os.path.join(args.data_dir, "keras_slices_train"),
                      f"case_{args.scan}_slice_*.nii.gz")
    segs = scan_files(os.path.join(args.data_dir, "keras_slices_seg_train"),
                      f"seg_{args.scan}_slice_*.nii.gz")
    if not imgs:
        raise FileNotFoundError(f"No slices found for scan {args.scan}")
    print(f"Scan {args.scan}: {len(imgs)} slices, {len(segs)} masks")

    volume = np.stack([load(f) for f in imgs])
    print(f"Slice shape {volume.shape[1:]}, stored dtype "
          f"{nib.load(imgs[0]).get_data_dtype()}, "
          f"intensity range {volume.min():.1f} to {volume.max():.1f}")
    means = volume.mean(axis=(1, 2))

    # 1. Five slices spread through the scan
    picks = np.linspace(0, len(imgs) - 1, 5).astype(int)
    fig, axes = plt.subplots(1, 5, figsize=(12, 5))
    for ax, i in zip(axes, picks):
        ax.imshow(volume[i], cmap="gray")
        ax.set_title(f"slice {i}")
        ax.axis("off")
    fig.suptitle(f"HipMRI scan {args.scan}: slices across the volume")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "eda_slices.png"), dpi=120)

    # 2. MRI | mask | overlay
    if len(segs) == len(imgs):
        print("Label values in middle mask:", np.unique(load(segs[len(segs) // 2])))
        rows = picks[1:4]
        fig, axes = plt.subplots(len(rows), 3, figsize=(9, 4 * len(rows)))
        for r, i in enumerate(rows):
            msk = load(segs[i])
            axes[r, 0].imshow(volume[i], cmap="gray")
            axes[r, 0].set_title(f"slice {i}: MRI")
            axes[r, 1].imshow(msk, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
            axes[r, 1].set_title("mask")
            axes[r, 2].imshow(volume[i], cmap="gray")
            axes[r, 2].imshow(np.ma.masked_where(msk == 0, msk), cmap="tab10",
                              vmin=0, vmax=9, alpha=0.5, interpolation="nearest")
            axes[r, 2].set_title("overlay")
        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "eda_masks.png"), dpi=120)
    else:
        print("Mask count does not match slice count; skipping mask figure")

    # 3. Mean intensity per slice
    plt.figure(figsize=(7, 3.5))
    plt.plot(means, marker=".")
    plt.xlabel("slice index")
    plt.ylabel("mean raw intensity")
    plt.title(f"Mean intensity per slice, scan {args.scan}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "eda_slice_means.png"), dpi=120)
    print(f"Mean intensity per slice ranges {means.min():.1f} to {means.max():.1f}")
    print(f"Saved figures to {args.out_dir}/")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()