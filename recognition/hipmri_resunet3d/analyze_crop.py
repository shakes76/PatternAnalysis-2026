"""Audit a fixed crop and label-resampling ceiling on train/validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dataset import central_xy_slices, load_native_anatomy, load_split, read_manifest


def crop_contains_target(target: np.ndarray, crop_xy: int) -> bool:
    """Check that every target voxel is inside the central X/Y window."""
    x_range, y_range = central_xy_slices((target.shape[2], target.shape[1]), crop_xy)
    kept = target[:, y_range, x_range].sum()
    return int(kept) == int(target.sum())


def roundtrip_dice(
    target: np.ndarray, size: tuple[int, int, int], crop_xy: int | None
) -> tuple[float, int]:
    """Downsample an oracle mask and restore it; no model is involved."""
    x_range, y_range = central_xy_slices((target.shape[2], target.shape[1]), crop_xy)
    cropped = target[:, y_range, x_range]
    tensor = torch.from_numpy(cropped.astype(np.float32))[None, None]
    reduced = F.interpolate(tensor, size=size, mode="nearest")
    restored = F.interpolate(reduced, size=cropped.shape, mode="nearest")
    native = np.zeros_like(target, dtype=bool)
    native[:, y_range, x_range] = restored[0, 0].numpy().astype(bool)
    intersection = int(np.logical_and(native, target).sum())
    denominator = int(native.sum()) + int(target.sum())
    return (2 * intersection / denominator if denominator else 1.0,
            int(reduced.sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--prostate-label", type=int, required=True)
    parser.add_argument("--size", nargs=3, type=int, default=[32, 64, 64])
    parser.add_argument("--crop-xy", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    split = load_split(read_manifest(args.manifest), args.split)
    torch.set_num_threads(4)
    coverage = {"train": 0, "val": 0}
    scores = {"full_field": [], "crop": []}
    foreground = {"full_field": [], "crop": []}
    for name in ("train", "val"):
        for case in split[name]:
            anatomy, _ = load_native_anatomy(case)
            target = anatomy == args.prostate_label
            coverage[name] += crop_contains_target(target, args.crop_xy)
            if name == "val":
                for design, crop in (("full_field", None), ("crop", args.crop_xy)):
                    score, voxels = roundtrip_dice(target, tuple(args.size), crop)
                    scores[design].append(score)
                    foreground[design].append(voxels)

    report = {
        "scope": "this crop audit reads train and validation masks only",
        "network_size_dhw": args.size,
        "crop_xy": args.crop_xy,
        "coverage": {
            name: {"fully_contained": coverage[name], "cases": len(split[name])}
            for name in ("train", "val")
        },
        "validation_label_roundtrip": {
            name: {
                "mean_dice": float(np.mean(scores[name])),
                "fifth_percentile_dice": float(np.percentile(scores[name], 5)),
                "median_network_foreground_voxels": float(np.median(foreground[name])),
            }
            for name in scores
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
