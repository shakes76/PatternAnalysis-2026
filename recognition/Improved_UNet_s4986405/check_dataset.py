"""Inspect one batch from the HipMRI 2D Dataset."""

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import HipMRI2DDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--normalize", action="store_true",
                        help="Apply per-slice MRI z-score normalization")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    dataset = HipMRI2DDataset(
        args.image_dir, args.mask_dir, normalize=args.normalize
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    images, masks = next(iter(loader))

    print("MRI normalization:", "z-score" if args.normalize else "none")
    print("Dataset pairs:", len(dataset))
    print("Image batch shape:", tuple(images.shape))
    print("Image dtype:", images.dtype)
    print("Mask batch shape:", tuple(masks.shape))
    print("Mask dtype:", masks.dtype)
    print("Batch mask labels:", torch.unique(masks).tolist())
    print("Batch loading: PASS")


if __name__ == "__main__":
    main()
