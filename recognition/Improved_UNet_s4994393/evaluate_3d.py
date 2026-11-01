"""Evaluate a frozen checkpoint on its held-out patients, with cost and error exports."""

import argparse
import csv
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from dataset import load_volume
from predict import sliding_window_predict
from modules import ImprovedUNet3D


def overlap_metrics(prediction, target, classes):
    """Return per-class Dice, IoU, intersections and prediction-plus-target counts.

    A class absent from both arrays is undefined (NaN), not a perfect score.
    Raw counts allow pooled metrics without averaging differently sized scans.
    """
    intersection = np.array(
        [np.count_nonzero((prediction == c) & (target == c)) for c in range(classes)]
    )
    total = np.array(
        [np.count_nonzero(prediction == c) + np.count_nonzero(target == c) for c in range(classes)]
    )
    dice = np.divide(2 * intersection, total, out=np.full(classes, np.nan), where=total > 0)
    union = total - intersection
    iou = np.divide(intersection, union, out=np.full(classes, np.nan), where=union > 0)
    return dice, iou, intersection, total


def export_panels(image, target, prediction, affine, path):
    """Compare MRI, truth, prediction and prostate errors at three axial slices.

    Ground truth selects slices for retrospective error analysis only; it is
    never used to crop or guide inference. Outermost labelled slices are
    base/apex proxies rather than independently verified anatomical landmarks.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Select the physical superior/inferior voxel axis, not an assumed array axis.
    codes = nib.aff2axcodes(affine)
    axis = next(i for i, code in enumerate(codes) if code in ("S", "I"))
    occupied = np.flatnonzero(np.any(target == 5, axis=tuple(i for i in range(3) if i != axis)))
    if not occupied.size:
        indices = [0, image.shape[axis] // 2, image.shape[axis] - 1]
    else:
        indices = [int(occupied[0]), int(occupied[len(occupied) // 2]), int(occupied[-1])]
    names = ["apex", "middle", "base"] if codes[axis] == "S" else ["base", "middle", "apex"]
    figure, axes = plt.subplots(3, 4, figsize=(12, 9))
    for row, (index, name) in enumerate(zip(indices, names)):
        scan = np.take(image, index, axis).T
        truth = np.take(target, index, axis).T
        pred = np.take(prediction, index, axis).T
        error = np.zeros((*truth.shape, 4))
        error[(pred == 5) & (truth != 5)] = [1, 0, 0, 0.8]
        error[(pred != 5) & (truth == 5)] = [0, 0.5, 1, 0.8]
        for col in range(4):
            axes[row, col].imshow(scan, cmap="gray", origin="lower")
            axes[row, col].axis("off")
        axes[row, 1].imshow(
            np.ma.masked_where(truth == 0, truth),
            vmin=0,
            vmax=5,
            cmap="tab10",
            alpha=0.6,
            origin="lower",
        )
        axes[row, 2].imshow(
            np.ma.masked_where(pred == 0, pred),
            vmin=0,
            vmax=5,
            cmap="tab10",
            alpha=0.6,
            origin="lower",
        )
        axes[row, 3].imshow(error, origin="lower")
        for col, title in enumerate(
            ["MRI", "Ground truth", "Prediction", "Prostate FP red / FN blue"]
        ):
            axes[row, col].set_title(f"{name} slice {index}: {title}", fontsize=8)
    figure.suptitle(path.stem + " (outermost prostate slices are base/apex proxies)")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main(argv=None):
    """Evaluate saved split membership and export metrics, timing and worst cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Relocate the dataset while retaining checkpoint patient assignments",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--failure-cases", type=int, default=5)
    parser.add_argument("--window-batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    if not 0 <= args.failure_cases <= 5:
        parser.error("failure-cases must be between 0 and 5")
    if args.window_batch_size < 1:
        parser.error("window-batch-size must be positive")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    model = ImprovedUNet3D(config["num_classes"], config["base_channels"], config["architecture"])
    model.load_state_dict(state["model"])
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    model.to(device).eval()
    # Checkpoint membership is authoritative; relocation changes paths only.
    pairs = state["split"][args.split]
    if args.data_root:
        # Handle checkpoints moved between Windows and Linux.
        basename = lambda p: p.replace("\\", "/").rsplit("/", 1)[-1]
        pairs = [
            (
                str(args.data_root / "semantic_MRs" / basename(p[0])),
                str(args.data_root / "semantic_labels_only" / basename(p[1])),
            )
            for p in pairs
        ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, ranked = [], []
    intersections = np.zeros(config["num_classes"])
    totals = np.zeros(config["num_classes"])
    # Warm up outside the timed measurements; no label information is used.
    with torch.inference_mode():
        model(torch.zeros(1, 1, *config["patch_size"], device=device))
    for pair in pairs:
        image, target = load_volume(pair, config["num_classes"])
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        prediction = sliding_window_predict(
            model,
            image,
            config["patch_size"],
            config["num_classes"],
            device,
            window_batch_size=args.window_batch_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
        dice, iou, intersection, total = overlap_metrics(prediction, target, config["num_classes"])
        intersections += intersection
        totals += total
        row = {"case": Path(pair[0]).name, "inference_seconds": seconds, "peak_gpu_gib": peak}
        row.update({f"dice_{c}": dice[c] for c in range(config["num_classes"])})
        row.update({f"iou_{c}": iou[c] for c in range(config["num_classes"])})
        rows.append(row)
        ranked.append((float(dice[5]) if np.isfinite(dice[5]) else 1.0, pair))
        print(f'{row["case"]}: prostate Dice={dice[5]:.4f}, time={seconds:.2f}s', flush=True)
    with (args.output_dir / "per_case.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Keep pooled voxel scores distinct from the equally weighted scan mean.
    summary = {
        "architecture": config["architecture"],
        "split": args.split,
        "window_batch_size": args.window_batch_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "cases": len(rows),
        "mean_inference_seconds": float(np.mean([r["inference_seconds"] for r in rows])),
        "peak_gpu_gib": max(r["peak_gpu_gib"] for r in rows) if device.type == "cuda" else None,
        "timing_scope": "warmed full-volume inference including CPU overlap aggregation; excludes NIfTI loading and plotting",
        "pooled_dice": [float(2 * i / t) if t else None for i, t in zip(intersections, totals)],
        "pooled_iou": [
            float(i / (t - i)) if t - i else None for i, t in zip(intersections, totals)
        ],
        "mean_case_dice": [
            (
                float(np.nanmean([r[f"dice_{c}"] for r in rows]))
                if any(np.isfinite(r[f"dice_{c}"]) for r in rows)
                else None
            )
            for c in range(config["num_classes"])
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    # Generate panels after timing; plotting and extra inference are not benchmarked.
    for _, pair in sorted(ranked, key=lambda item: item[0])[: args.failure_cases]:
        image, target = load_volume(pair, config["num_classes"])
        prediction = sliding_window_predict(
            model,
            image,
            config["patch_size"],
            config["num_classes"],
            device,
            window_batch_size=args.window_batch_size,
        )
        export_panels(
            image,
            target,
            prediction,
            nib.load(pair[0]).affine,
            args.output_dir / (Path(pair[0]).name.replace(".nii.gz", "") + ".png"),
        )
    print(
        f"Evaluation saved in {args.output_dir}; inspect worst-case panels before drawing conclusions."
    )


if __name__ == "__main__":
    main()
