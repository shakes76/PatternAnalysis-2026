"""Inspect paired training MRI slices and segmentation masks."""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import nibabel as nib
import numpy as np


LABEL_NAMES = ["Background", "Body", "Bones", "Bladder", "Rectum", "Prostate"]
COLORS = ["black", "lightgray", "gold", "deepskyblue", "limegreen", "magenta"]
PATTERN = re.compile(r"case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz")


def load_pair(image_path, mask_path):
    """Load and validate a single image-mask pair."""
    image = nib.load(image_path)
    mask = nib.load(mask_path)
    x = image.get_fdata(dtype=np.float32)
    y = mask.get_fdata(dtype=np.float32)

    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError(f"Invalid shapes: {image_path.name}")
    if not np.allclose(image.affine, mask.affine, rtol=0, atol=1e-5):
        raise ValueError(f"Affine mismatch: {image_path.name}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite values: {image_path.name}")
    if not np.equal(y, np.round(y)).all():
        raise ValueError(f"Non-integer labels: {mask_path.name}")
    if not np.isin(y, np.arange(6)).all():
        raise ValueError(f"Unexpected labels: {mask_path.name}")

    return x, y.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
        ),
    )
    parser.add_argument("--case", type=int, default=4)
    parser.add_argument("--week", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("figures/training_data_preview.png")
    )
    args = parser.parse_args()

    records = []
    for image_path in (args.data_root / "keras_slices_train").glob("*.nii.gz"):
        match = PATTERN.fullmatch(image_path.name)
        if match is None:
            continue
        case, week, section = map(int, match.groups())
        if case == args.case and week == args.week:
            mask_path = (
                args.data_root / "keras_slices_seg_train"
                / image_path.name.replace("case_", "seg_", 1)
            )
            records.append((section, image_path, mask_path))

    if not records:
        raise RuntimeError("No matching training slices found.")

    positive = []
    for section, image_path, mask_path in sorted(records):
        x, y = load_pair(image_path, mask_path)
        if np.any(y == 5):
            positive.append((section, x, y))

    if not positive:
        raise RuntimeError("No prostate-positive slices found.")

    count = min(5, len(positive))
    indices = np.unique(
        np.linspace(0, len(positive) - 1, count).round().astype(int)
    )
    selected = [positive[index] for index in indices]

    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
    fig, axes = plt.subplots(
        len(selected), 3,
        figsize=(10, 3 * len(selected)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, (section, x, y) in enumerate(selected):
        # Use percentiles for display only; training data remain unchanged.
        low, high = np.percentile(x, [1, 99])
        if high <= low:
            high = low + 1

        axes[row, 0].imshow(
            x, cmap="gray", vmin=low, vmax=high, interpolation="nearest"
        )
        mask_plot = axes[row, 1].imshow(
            y, cmap=cmap, norm=norm, interpolation="nearest"
        )
        axes[row, 2].imshow(
            x, cmap="gray", vmin=low, vmax=high, interpolation="nearest"
        )
        # Hide background and body to make organ overlays easier to inspect.
        organs = np.ma.masked_where(y < 2, y)
        axes[row, 2].imshow(
            organs, cmap=cmap, norm=norm,
            alpha=0.45, interpolation="nearest",
        )

        titles = [
            f"Slice {section}: MRI",
            "Ground-truth labels",
            "Organ overlay",
        ]
        for axis, title in zip(axes[row], titles):
            axis.set_title(title)
            axis.set_axis_off()

    colorbar = fig.colorbar(
        mask_plot, ax=axes[:, 1:].ravel().tolist(),
        ticks=np.arange(6), shrink=0.7, pad=0.02,
    )
    colorbar.ax.set_yticklabels(LABEL_NAMES)
    fig.suptitle(
        f"Training case {args.case:03d}, week {args.week}\n"
        "Stored array orientation; no anatomical direction assigned"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    print("Matched slices:", len(records))
    print("Prostate-positive slices:", len(positive))
    print("Selected slice IDs:", [item[0] for item in selected])
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()
