"""Smoke-check Dataset, model, and metrics together on CPU."""

import argparse
from itertools import islice

import torch
from torch.utils.data import DataLoader

from dataset import HipMRI2DDataset
from metrics import confusion_matrix, dice_iou_from_confusion
from modules import UNet2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=4)
    args = parser.parse_args()

    if min(args.batch_size, args.max_batches, args.base_channels) < 1:
        parser.error("Batch size, max batches, and base channels must be positive")

    torch.manual_seed(3710)
    torch.set_num_threads(1)

    dataset = HipMRI2DDataset(args.image_dir, args.mask_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = UNet2D(base_channels=args.base_channels).cpu()
    model.eval()

    total_confusion = torch.zeros(6, 6, dtype=torch.long)
    samples = 0
    pixels = 0
    batches = 0

    print("Randomly initialized model; smoke check only.")
    with torch.no_grad():
        for images, masks in islice(loader, args.max_batches):
            logits = model(images)
            expected = (images.shape[0], 6, *masks.shape[-2:])
            if tuple(logits.shape) != expected:
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
            if not torch.isfinite(logits).all():
                raise RuntimeError("Model output contains NaN or Inf")

            prediction = logits.argmax(dim=1)
            total_confusion += confusion_matrix(prediction, masks)
            samples += images.shape[0]
            pixels += masks.numel()
            batches += 1

    if total_confusion.sum().item() != pixels:
        raise RuntimeError("Confusion matrix pixel count mismatch")

    scores = dice_iou_from_confusion(total_confusion)
    print("Dataset pairs:", len(dataset))
    print("Processed batches:", batches)
    print("Processed samples:", samples)
    print("Counted pixels:", pixels)
    for label in range(6):
        dice = scores["dice"][label].item()
        iou = scores["iou"][label].item()
        print(f"Label {label}: Dice={dice:.4f}, IoU={iou:.4f}")

    print("Pipeline smoke check: PASS")


if __name__ == "__main__":
    main()
