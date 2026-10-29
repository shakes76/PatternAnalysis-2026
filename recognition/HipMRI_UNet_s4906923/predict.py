"""Run saved model weights on one MRI case and export the result.

The MRI alone determines the prediction. The paired mask is used afterwards
for metrics and illustration. Exported labels use the downsampled spatial grid
and its affine; they must not be treated as original-resolution predictions.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from dataset import discover, load_case, downsample, zscore_nonzero
from utils import (load_model, predict_volume, confusion, overlap, spatial_diagnostics,
                   atomic_json, sha256_file, CLASS_NAMES)


def make_panel(volume, prediction, output):
    """Draw all five foreground labels on GT-largest-area and largest-error slices.

    Red marks predicted-only voxels and blue marks missed GT. Slice selection
    uses GT for illustration after inference; it does not constrain prediction.
    The arrays and NIfTI export remain on the same downsampled XYZ grid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(5, 4, figsize=(14, 16))
    for k in range(1, 6):
        truth = volume.mask == k; predicted = prediction == k
        # Show both the largest labelled cross-section and the slice with most errors.
        areas = truth.sum(axis=(0, 1)); errors = (truth != predicted).sum(axis=(0, 1))
        zs = (int(areas.argmax()), int(errors.argmax()))
        for j, z in enumerate(zs):
            ax = axes[k-1, 2*j]
            ax.imshow(volume.image[:, :, z].T, cmap="gray", origin="lower")
            gt, pr = truth[:, :, z].T, predicted[:, :, z].T
            if gt.any(): ax.contour(gt, [.5], colors=["cyan"], linewidths=.8)
            if pr.any(): ax.contour(pr, [.5], colors=["orange"], linewidths=.8)
            ax.set_title(f"{CLASS_NAMES[k]}: {'GT centre' if j==0 else 'max error'}, z={z}")
            code = np.zeros(gt.shape, dtype=np.uint8)
            code[pr & ~gt] = 1; code[gt & ~pr] = 2
            error_ax = axes[k-1, 2*j+1]
            error_ax.imshow(volume.image[:, :, z].T, cmap="gray", origin="lower")
            error_ax.imshow(np.ma.masked_where(code==0, code),
                            cmap=ListedColormap(["black", "red", "blue"]), vmin=0, vmax=2,
                            origin="lower", alpha=.7)
            error_ax.set_title("Red FP / blue FN")
    for ax in axes.flat: ax.set_axis_off()
    fig.suptitle(f"{volume.case.identifier}: cyan GT / orange prediction; downsampled XYZ voxel views")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def main():
    """Select one frozen-split case, time fixed-weight inference and export three files.

    metrics.json records weights/input hashes and six-class/physical measurements;
    overlay.png is the diagnostic view; prediction_downsampled.nii.gz contains
    integer labels on the resampled affine. Existing output directories fail.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("standard", "v3"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists(): parser.error("Use a new output directory")
    choices = [c for c in discover(args.dataset_root)[args.split] if c.identifier==args.case]
    if len(choices)!=1: parser.error("Case not present in the requested frozen split")
    volume = zscore_nonzero(downsample(load_case(choices[0])))
    model = load_model(args.model, args.checkpoint, args.device)
    for _ in range(3): predict_volume(model, volume, mode="3d", device=args.device)
    times = []
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        if args.device.startswith("cuda"): torch.cuda.synchronize()
        start = perf_counter()
        prediction = predict_volume(model, volume, mode="3d", device=args.device)
        if args.device.startswith("cuda"): torch.cuda.synchronize()
        times.append(perf_counter()-start)
    result = {"case": args.case, "model": args.model, "split": args.split,
              "checkpoint_sha256": sha256_file(args.checkpoint),
              "mri_sha256": sha256_file(choices[0].mri), "mask_sha256": sha256_file(choices[0].mask),
              "grid": "2x downsampled XYZ physical grid, high-end model padding removed",
              "affine": volume.affine.tolist(), "shape_xyz": list(volume.mask.shape),
              "confusion": confusion(volume.mask, prediction).tolist(),
              "overlap": overlap(confusion(volume.mask, prediction)),
              **spatial_diagnostics(volume, prediction),
              "benchmark": {"warmup": 3, "repeats": 5, "seconds": times,
                  "median_seconds": float(np.median(times)),
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else None,
                  "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
                  "scope": "Resident whole-volume inference, transfer and argmax; excludes preprocessing, IO and metrics"}}
    args.output_dir.mkdir(parents=True)
    import nibabel as nib
    image = nib.Nifti1Image(prediction.astype(np.uint8), volume.affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, str(args.output_dir / "prediction_downsampled.nii.gz"))
    atomic_json(args.output_dir / "metrics.json", result)
    make_panel(volume, prediction, args.output_dir / "overlay.png")
    print("Classes:", CLASS_NAMES)
    print("Dice:", result["overlap"]["dice"])
    print("Saved:", args.output_dir)


if __name__ == "__main__": main()
