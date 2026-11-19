"""CPU-only class-presence analysis of seeded 3D training patches (no model, no GPU).

For each patch size, draws seeded patch plans from the TRAINING split only and
reports, per class: the share of patches that contain it, and the mean and
median share of patch voxels it occupies when present. Uniform sampling is
always measured; class-balanced sampling is measured for each ``--fg-prob``.

Run from the project directory:

    python -m analysis.class_presence3d --metadata /path/to/metadata.csv \
        --out class_presence3d.json

The validation and test splits are never loaded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dataset import (
    EXPECTED_PATIENTS_PER_SPLIT,
    EXPECTED_SPLIT_SHA256,
    NUM_CLASSES,
    PatchSampler,
    VolumeSet,
    as_patch_size,
    extract_patch,
    load_manifest,
    split_sha256,
)
from metrics import CLASS_NAMES

ANALYSIS_EPOCH = 0  # plan index used for the analysis; training plans use their own epochs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Output JSON file")
    parser.add_argument("--patches", type=int, default=5000, help="Patches per size and mode")
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[64, 96])
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument(
        "--fg-prob",
        type=float,
        nargs="*",
        default=[],
        help="Also measure class_balanced sampling at these probabilities",
    )
    return parser.parse_args(argv)


def summarise_plan(volumes: VolumeSet, sampler: PatchSampler, count: int) -> dict:
    """Class presence and voxel shares over one seeded plan."""
    counts = np.zeros((count, NUM_CLASSES), dtype=np.int64)
    for i, (v, *start) in enumerate(sampler.plan(ANALYSIS_EPOCH, count)):
        _, mask = extract_patch(volumes.images[v], volumes.masks[v], start, sampler.size)
        counts[i] = np.bincount(mask.numpy().ravel(), minlength=NUM_CLASSES)
    shares = counts / counts.sum(axis=1, keepdims=True)
    present = counts > 0
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        when_present = shares[present[:, c], c]
        per_class[name] = {
            "patches_containing": round(float(present[:, c].mean()), 4),
            "voxel_share_when_present_mean": (
                round(float(when_present.mean()), 4) if when_present.size else None
            ),
            "voxel_share_when_present_median": (
                round(float(np.median(when_present)), 4) if when_present.size else None
            ),
        }
    return {
        "patches": count,
        "patches_without_foreground": round(float((~present[:, 1:].any(axis=1)).mean()), 4),
        "patches_with_all_foreground_classes": round(float(present[:, 1:].all(axis=1).mean()), 4),
        "per_class": per_class,
    }


def run_analysis(
    args: argparse.Namespace,
    expected_patients: dict[str, int] | None = EXPECTED_PATIENTS_PER_SPLIT,
    expected_sha256: str | None = EXPECTED_SPLIT_SHA256,
) -> dict:
    frame = load_manifest(args.metadata, expected_patients, expected_sha256)
    start = time.perf_counter()
    # Intensities are not analysed, so normalisation is skipped.
    volumes = VolumeSet.from_manifest(frame, "train", normalise=False, verbose=True)
    voxels = sum(np.bincount(m.ravel(), minlength=NUM_CLASSES) for m in volumes.masks)

    report = {
        "split": "train",
        "split_sha256": split_sha256(frame),
        "volumes": len(volumes),
        "seed": args.seed,
        "split_voxel_share": {
            name: round(float(n / voxels.sum()), 4)
            for name, n in zip(CLASS_NAMES, voxels, strict=True)
        },
        "results": {},
    }
    for size in args.patch_sizes:
        modes = [("uniform", None)] + [("class_balanced", p) for p in args.fg_prob]
        for mode, fg_prob in modes:
            sampler = PatchSampler(volumes, size, mode, args.seed, fg_prob)
            key = f"{size}^3 {mode}" + (f" fg_prob={fg_prob}" if fg_prob is not None else "")
            report["results"][key] = summarise_plan(volumes, sampler, args.patches)
            report["results"][key]["patch_size_dhw"] = list(as_patch_size(size))
            print_summary(key, report["results"][key])
    report["seconds"] = round(time.perf_counter() - start, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[done] wrote {args.out}")
    return report


def print_summary(key: str, result: dict) -> None:
    print(f"\n[{key}] {result['patches']} patches")
    print(f"  no foreground: {result['patches_without_foreground']:.3f}")
    for name, stats in result["per_class"].items():
        median = stats["voxel_share_when_present_median"]
        median_text = "-" if median is None else f"{median:.4f}"
        print(
            f"  {name:<10} in {stats['patches_containing']:.3f} of patches, "
            f"median share when present {median_text}"
        )


def main(argv: list[str] | None = None) -> None:
    run_analysis(parse_args(argv))


if __name__ == "__main__":
    main()
