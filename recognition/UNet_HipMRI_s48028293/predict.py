"""
predict.py runs a trained HipMRI checkpoint on the test split, saves per-volume and per-slice metrics, 
and plots the prostate Dice by slice position and the worst failure cases.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (DATA_ROOT_2D, DATA_ROOT_3D, DOWNSAMPLE, PROSTATE, HipMRI2D, HipMRI3D,get_splits, get_volume_splits)
from modules import build_model
from utils import evaluate, format_results

DEFAULT_DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


def parse_args():
    # Parse command-line arguments and fill in defaults based on the model type.
    p = argparse.ArgumentParser(description="Run a trained HipMRI model on the test split")
    p.add_argument("--model", default="unet2d", help="which run folder to use")
    p.add_argument("--checkpoint", default=None, help="default: ../runs/<model>/best_model.pt")
    p.add_argument("--data-root", default=None, help="default depends on the model")
    p.add_argument("--out-dir", default=None, help="default: ../runs/<model>")
    p.add_argument("--batch-size", type=int, default=None, help="default 32 (2D) / 1 (3D)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-failures", type=int, default=5, help="worst slices to visualise")
    p.add_argument("--limit", type=int, default=None, help="first N test slices only")
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(RUNS_DIR, args.model)
    if args.checkpoint is None:
        args.checkpoint = os.path.join(RUNS_DIR, args.model, "best_model.pt")
    return args

def add_positions(rows):
    # Add a "position" field to each row, indicating the relative position of the slice along the 
    # prostate volume (0 = first slice, 1 = last slice).
    volumes = {}
    for r in rows:
        if r["target_px"] > 0:
            volumes.setdefault((r["case"], r["week"]), []).append(r)
    for vol_rows in volumes.values():
        vol_rows.sort(key=lambda r: r["slice"])
        for i, r in enumerate(vol_rows):
            r["position"] = i / (len(vol_rows) - 1) if len(vol_rows) > 1 else 0.5

def plot_dice_by_position(rows, path, n_bins=10):
    # Plot the mean prostate Dice by slice position along the prostate volume.
    pos = np.array([r["position"] for r in rows if "position" in r])
    dice = np.array([r["dice"] for r in rows if "position" in r])
    bins = np.minimum((pos * n_bins).astype(int), n_bins - 1)
    centres = (np.arange(n_bins) + 0.5) / n_bins
    means = [dice[bins == b].mean() if np.any(bins == b) else np.nan for b in range(n_bins)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(centres, means, width=0.9 / n_bins)
    ax.set_xlabel("position along the prostate (0 and 1 = end slices, 0.5 = mid-gland)")
    ax.set_ylabel("mean slice DSC (prostate)")
    ax.set_title("Prostate Dice by slice position")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_failures(model, dataset, rows, device, num_classes, path):
    #Multi-panel figure for the given slices (rows from per_slice)
    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 3.2 * len(rows)), squeeze=False)
    model.eval()
    cache = {}
    for ax_row, r in zip(axes, rows):
        if r["index"] not in cache:
            image, mask, _, _, _ = dataset[r["index"]]
            with torch.no_grad():
                pred = model(image[None].to(device)).argmax(dim=1)[0].cpu().numpy()
            cache[r["index"]] = (image[0].numpy(), mask.numpy(), pred)
        image, mask, pred = cache[r["index"]]
        if image.ndim == 3:  # 3D volume (X, Y, Z): take slice z
            z = r["slice"]
            image, mask, pred = image[..., z], mask[..., z], pred[..., z]

        # Prostate error overlay: green = correct, red = false positive
        # (leakage into neighbouring tissue), blue = false negative (missed).
        lo, hi = np.percentile(image, [1, 99])
        grey = np.clip((image - lo) / (hi - lo + 1e-8), 0, 1)
        overlay = np.stack([grey] * 3, axis=-1)
        p, t = pred == PROSTATE, mask == PROSTATE
        overlay[p & t] = [0.0, 0.8, 0.0]
        overlay[p & ~t] = [1.0, 0.0, 0.0]
        overlay[~p & t] = [0.0, 0.3, 1.0]

        ax_row[0].imshow(grey, cmap="gray")
        ax_row[0].set_title(f"case {r['case']} week {r['week']} slice {r['slice']}", fontsize=9)
        ax_row[1].imshow(mask, cmap="viridis", vmin=0, vmax=num_classes - 1)
        ax_row[1].set_title("ground truth", fontsize=9)
        ax_row[2].imshow(pred, cmap="viridis", vmin=0, vmax=num_classes - 1)
        ax_row[2].set_title("prediction", fontsize=9)
        ax_row[3].imshow(overlay)
        ax_row[3].set_title(f"prostate errors (DSC {r['dice']:.2f}): red FP, blue FN", fontsize=9)
        for ax in ax_row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def main():
    # Parse command-line arguments, set up the model, data loaders, loss, optimizer, and training loop.
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes, class_names = ckpt["num_classes"], ckpt["class_names"]
    model = build_model(ckpt["model"], num_classes, base=ckpt["base"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {ckpt['model']} from epoch {ckpt['epoch']} "
        f"(val prostate DSC {ckpt['val_prostate_dice']:.4f})")

    dims = ckpt.get("dims", 2)
    if args.data_root is None:
        args.data_root = DATA_ROOT_3D if dims == 3 else DEFAULT_DATA_ROOT
    split_fn = get_volume_splits if dims == 3 else get_splits
    pairs = split_fn(args.data_root)["test"]

    if args.limit:
        pairs = pairs[:args.limit]
    if dims == 3:
        dataset = HipMRI3D(pairs, tuple(ckpt["img_size"]), ckpt.get("downsample", DOWNSAMPLE))
    else:
        dataset = HipMRI2D(pairs, ckpt["img_size"])
    batch_size = args.batch_size or (1 if dims == 3 else 32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=args.workers)

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    meter, _, forward_time = evaluate(model, loader, device, num_classes)
    vram = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else 0.0
    print("\nTest results (per volume)")
    print(format_results(meter, class_names))
    print(f"Inference time per case: {forward_time / len(meter.volumes):.3f} s")
    print(f"Peak VRAM, inference: {vram:.0f} MiB (batch size {batch_size})")

    # Per-volume table.
    keys, dice, iou = meter.per_volume()
    with open(os.path.join(args.out_dir, "per_volume_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "week"]
                        + [f"dice_{n}" for n in class_names]
                        + [f"iou_{n}" for n in class_names])
        for (case, week), d_row, i_row in zip(keys, dice, iou):
            writer.writerow([case, week]
                            + [f"{d:.4f}" for d in d_row]
                            + [f"{i:.4f}" for i in i_row])


    worst = np.argsort(dice[:, PROSTATE])[:3]
    print("Lowest prostate DSC volumes:",
        ", ".join(f"case {keys[i][0]} week {keys[i][1]} ({dice[i, PROSTATE]:.3f})" for i in worst))

    # Per-slice prostate table and the position analysis.
    rows = meter.per_slice(PROSTATE)
    add_positions(rows)
    with open(os.path.join(args.out_dir, "per_slice_prostate.csv"), "w", newline="") as f:
        fields = ["index", "case", "week", "slice", "dice", "target_px", "pred_px", "position"]
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    plot_dice_by_position(rows, os.path.join(args.out_dir, "dice_by_position.png"))

    # Failure cases: the prostate-containing slices with the lowest Dice.
    with_prostate = [r for r in rows if r["target_px"] > 0]
    failures = sorted(with_prostate, key=lambda r: r["dice"])[:args.n_failures]
    if failures:
        plot_failures(model, dataset, failures, device, num_classes,
                    os.path.join(args.out_dir, "failure_cases.png"))
    print(f"\nFigures and tables saved to {args.out_dir}")

if __name__ == "__main__":
    main()
