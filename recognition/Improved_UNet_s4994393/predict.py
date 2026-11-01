"""Memory-bounded sliding-window prediction with overlap averaging."""

import argparse
import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from modules import ImprovedUNet3D


@torch.inference_mode()
def sliding_window_predict(
    model, image, patch_size, num_classes, device, overlap=0.5, window_batch_size=1
):
    """Predict a full normalized MRI without using a ground-truth mask.

    The input is a (D, H, W) array. Overlapping windows contribute equally
    to averaged class probabilities, then argmax produces integer labels.
    Padding is removed before returning an array with the original shape.
    window_batch_size controls inference memory, not the training batch size.
    """
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be in [0, 1)")
    if window_batch_size < 1:
        raise ValueError("window_batch_size must be positive")
    model.eval()
    original = image.shape
    padding = [(0, max(0, p - s)) for s, p in zip(original, patch_size)]
    if any(after for _, after in padding):
        image = np.pad(image, padding)
    starts = []
    for size, patch in zip(image.shape, patch_size):
        axis = list(range(0, size - patch + 1, max(1, int(patch * (1 - overlap)))))
        # Include the far edge even when the stride does not divide the dimension.
        if axis[-1] != size - patch:
            axis.append(size - patch)
        starts.append(axis)
    # Accumulate on CPU so GPU memory depends on patch size, not volume size.
    scores = torch.zeros((num_classes, *image.shape), dtype=torch.float32)
    counts = torch.zeros(image.shape, dtype=torch.float32)
    windows = iter(itertools.product(*starts))
    while True:
        batch = list(itertools.islice(windows, window_batch_size))
        if not batch:
            break
        regions = [tuple(slice(s, s + p) for s, p in zip(start, patch_size)) for start in batch]
        x = torch.from_numpy(np.stack([image[region] for region in regions]))[:, None].to(device)
        logits, _ = model(x)
        probabilities = logits.float().softmax(1).cpu()
        for region, probability in zip(regions, probabilities):
            scores[(slice(None), *region)] += probability
            counts[region] += 1
    scores /= counts.clamp_min(1)[None]
    crop = tuple(slice(0, s) for s in original)
    return scores.argmax(0)[crop].numpy()


def export_prediction_overlay(image, prediction, affine, path, num_classes):
    """Choose slices using predictions only; no ground truth is needed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis = next(i for i, code in enumerate(nib.aff2axcodes(affine)) if code in ("S", "I"))
    occupied = np.flatnonzero(np.any(prediction == 5, axis=tuple(i for i in range(3) if i != axis)))
    indices = (
        [int(occupied[0]), int(occupied[len(occupied) // 2]), int(occupied[-1])]
        if occupied.size
        else [0, image.shape[axis] // 2, image.shape[axis] - 1]
    )
    figure, axes = plt.subplots(3, 2, figsize=(8, 10))
    for row, index in enumerate(indices):
        scan, labels = np.take(image, index, axis).T, np.take(prediction, index, axis).T
        for col in range(2):
            axes[row, col].imshow(scan, cmap="gray", origin="lower")
            axes[row, col].axis("off")
        axes[row, 1].imshow(
            np.ma.masked_where(labels == 0, labels),
            cmap="tab10",
            vmin=0,
            vmax=num_classes - 1,
            alpha=0.6,
            origin="lower",
        )
        axes[row, 0].set_title(f"MRI slice {index}")
        axes[row, 1].set_title("Predicted classes")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main():
    """Load a checkpoint, save an aligned segmentation and export visual checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mask", type=Path, help="Optional aligned ground truth for comparison only"
    )
    parser.add_argument("--window-batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = ImprovedUNet3D(
        config["num_classes"], config["base_channels"], config["architecture"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    nii = nib.load(args.image)
    image = nii.get_fdata(dtype=np.float32)
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 3D MRI")
    nonzero = image != 0
    values = image[nonzero]
    if not values.size:
        raise ValueError("MRI contains no nonzero voxels")
    # Match training normalization without requiring or consulting a label mask.
    image[nonzero] = (values - values.mean()) / max(float(values.std()), 1e-6)
    prediction = sliding_window_predict(
        model,
        image,
        config["patch_size"],
        config["num_classes"],
        device,
        window_batch_size=args.window_batch_size,
    )
    header = nii.header.copy()
    header.set_data_dtype(np.int16)
    result = nib.Nifti1Image(prediction.astype(np.int16), nii.affine, header)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, args.output)
    print(f"Saved segmentation: {args.output}")
    for label in range(config["num_classes"]):
        print(f"Class {label}: {np.count_nonzero(prediction == label)} voxels")
    stem = args.output.name.removesuffix(".gz").removesuffix(".nii")
    overlay_path = args.output.with_name(stem + "_overlay.png")
    export_prediction_overlay(image, prediction, nii.affine, overlay_path, config["num_classes"])
    print(f"Saved prediction overlay: {overlay_path}")
    if args.mask:
        # Load optional labels only after inference and mask-free output are complete.
        from dataset import load_volume
        from evaluate_3d import export_panels, overlap_metrics

        _, target = load_volume((args.image, args.mask), config["num_classes"])
        dice, iou, _, _ = overlap_metrics(prediction, target, config["num_classes"])
        print("Per-class Dice:", dice.tolist())
        print("Per-class IoU:", iou.tolist())
        comparison_path = args.output.with_name(stem + "_comparison.png")
        export_panels(image, target, prediction, nii.affine, comparison_path)
        print(f"Saved ground-truth comparison: {comparison_path}")


if __name__ == "__main__":
    main()
