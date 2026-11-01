"""Save a grid of 8 train images (4 malignant, 4 benign) titled with label and real ID."""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import dataset as d

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    """Save a 2x4 grid of the first 4 malignant and first 4 benign train images.

    Images go through the train transform (so they are augmented) and are
    un-normalised with the train mean/std for display.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder containing cleaned_metadata.csv")
    parser.add_argument("--image_dir", help="default: <data_dir>/combined_images")
    parser.add_argument("--out", default=os.path.join(HERE, "outputs", "grid.png"))
    args = parser.parse_args()
    image_dir = args.image_dir or os.path.join(args.data_dir, "combined_images")

    splits = d.split_by_id(d.load_clean_metadata(args.data_dir))
    train_loader, _, _ = d.get_dataloaders(splits, image_dir)
    train_set = train_loader.dataset
    # shape mean/std as (3, 1, 1) so they broadcast over a (3, H, W) image
    mean = torch.tensor(train_set.mean).view(3, 1, 1)
    std = torch.tensor(train_set.std).view(3, 1, 1)

    targets = train_set.df["target"]
    # row indices of the first 4 malignant and the first 4 benign train images
    picks = (targets[targets == 1].index[:4].tolist()
             + targets[targets == 0].index[:4].tolist())

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for ax, idx in zip(axes.flat, picks):
        image, label = train_set[idx]
        # undo the normalisation (x * std + mean) and reorder to H x W x C for imshow
        shown = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(shown)
        ax.set_title(f"{'malignant' if label == 1 else 'benign'}\n"
                     f"{train_set.df.loc[idx, 'real_id']}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print("saved", args.out)


if __name__ == "__main__":
    main()
