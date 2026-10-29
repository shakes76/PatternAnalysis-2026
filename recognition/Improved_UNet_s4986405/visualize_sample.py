"""Save a side-by-side view of one HipMRI image and mask."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from dataset import HipMRI2DDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = HipMRI2DDataset(args.image_dir, args.mask_dir)
    if not 0 <= args.index < len(dataset):
        parser.error(f"--index must be between 0 and {len(dataset) - 1}")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    image, mask = dataset[args.index]
    image_path, mask_path = dataset.pairs[args.index]

    colors = ListedColormap([
        "black", "gray", "gold", "deepskyblue", "magenta", "limegreen"
    ])
    norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], colors.N
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    axes[0].imshow(
        image[0].numpy(), cmap="gray", interpolation="nearest", origin="lower"
    )
    axes[0].set_title("MRI")
    plotted_mask = axes[1].imshow(
        mask.numpy(), cmap=colors, norm=norm,
        interpolation="nearest", origin="lower"
    )
    axes[1].set_title("Segmentation labels")
    for axis in axes:
        axis.set_axis_off()

    fig.colorbar(plotted_mask, ax=axes[1], ticks=range(6), label="Label ID")
    fig.suptitle(image_path.name)
    fig.subplots_adjust(top=0.88, wspace=0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("xb") as handle:
            fig.savefig(handle, format="png", dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    print("Image:", image_path)
    print("Mask:", mask_path)
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()
