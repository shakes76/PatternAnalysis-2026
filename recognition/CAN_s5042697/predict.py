"""
Run trained models on the HipMRI test split and analyse where they fail.

Example:
    python predict.py --checkpoints runs/unet/best.pt runs/can/best.pt

For each checkpoint this prints per-class Dice / IoU, prostate Dice by
anatomical position, false-positive slices, inference latency and peak GPU
memory, and saves to --out-dir:
    <model>_per_slice.csv   one row per test slice
    <model>_worst.png       the worst prostate slices (input, truth,
                            prediction, error overlay)
    <model>_worst_fp.png    the worst over-predictions: slices with no true
                            prostate, ranked by false prostate pixels
    compare_worst.png       the worst slices for every model side by side
    compare_worst_fp.png    the same for over-predictions

The slices are coronal (front to back), so position is analysed on
two axes:
  * Superior-inferior runs within each slice. Per scan, the prostate's
    vertical extent is split into thirds: the superior third is the base
    (next to the bladder) and the inferior third is the apex. Dice is pooled
    over all pixels in each band, so false positives above or below the
    gland count against the nearest band.
  * Anterior-posterior runs across slices. Per scan, the slices containing
    prostate are split into an anterior edge, central and posterior edge
    third. The anterior end is the one with more bladder just beyond it.
"""

import argparse
import csv
import os
import re
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from dataset import BLADDER, CLASS_NAMES, NUM_CLASSES, PROSTATE, get_datasets
from modules import CAN, ResidualCAN, UNet
from utils import SegmentationMeter, get_device

MODELS = {"unet": UNet, "can": CAN, "can_res": ResidualCAN}
HERE = os.path.dirname(os.path.abspath(__file__))
_NAME_RE = re.compile(r"case_(\d+)_week_(\d+)_slice_(\d+)")

# Label colours for masks: background, body, bone, bladder, rectum, prostate
LABEL_CMAP = ListedColormap(["black", "#4d4d4d", "#e0e0a0", "#3b8bd9", "#a05a2c", "#e0336f"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--data-root", default=os.path.join(HERE, "data", "keras_slices_data"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "runs", "predict"))
    p.add_argument("--n-worst", type=int, default=5)
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    # Older checkpoints have no kwargs or run name; they used the defaults.
    model = MODELS[ckpt["model"]](**ckpt.get("kwargs", {})).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return ckpt.get("run_name", ckpt["model"]), model


def slice_info(path):
    """('040', '0', 12) -> scan id 'case_040_week_0' and slice index 12."""
    patient, week, index = _NAME_RE.search(os.path.basename(path)).groups()
    return f"case_{patient}_week_{week}", int(index)


@torch.no_grad()
def predict_all(model, dataset, device, batch_size=32):
    """Return images, true masks and predicted masks for a dataset as numpy arrays."""
    images, truths, preds = [], [], []
    meter = SegmentationMeter()
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        logits = model(x.to(device))
        meter.update(logits, y.to(device))
        images.append(x[:, 0].numpy())
        truths.append(y.numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(images), np.concatenate(truths), np.concatenate(preds), meter


def group_scans(paths):
    """Map scan id -> dataset indices sorted by slice number."""
    scans = defaultdict(list)
    for i, path in enumerate(paths):
        scan, index = slice_info(path)
        scans[scan].append((index, i))
    return {scan: [i for _, i in sorted(members)] for scan, members in scans.items()}


def assign_slice_regions(scans, truths):
    """
    Label each slice 'anterior', 'central', 'posterior' (prostate present in
    ground truth) or 'none'. The anterior end of the gland is the one with
    more bladder in the 5 slices beyond it.
    """
    regions = ["none"] * len(truths)
    for order in scans.values():
        has_prostate = [k for k, i in enumerate(order) if (truths[i] == PROSTATE).any()]
        if not has_prostate:
            continue
        first, last = has_prostate[0], has_prostate[-1]

        def bladder_beyond(ks):
            return sum(int((truths[order[k]] == BLADDER).sum()) for k in ks if 0 <= k < len(order))

        low_is_anterior = bladder_beyond(range(first - 5, first)) >= bladder_beyond(range(last + 1, last + 6))
        third = len(has_prostate) / 3
        for rank, k in enumerate(has_prostate):
            if rank < third:
                region = "anterior" if low_is_anterior else "posterior"
            elif rank >= 2 * third:
                region = "posterior" if low_is_anterior else "anterior"
            else:
                region = "central"
            regions[order[k]] = region
    return regions


def vertical_band_dice(scans, truths, preds):
    """
    Pooled prostate Dice in the base / mid / apex bands of each scan.

    Image rows run along array axis 1. Per scan, the prostate's row extent
    (over all its slices) is split into thirds, and the base band is the
    third on the bladder side. The outer bands extend to the image edge so
    every predicted pixel belongs to some band. Returns totals over all scans
    and a per-scan breakdown.
    """
    totals = {band: np.zeros(3) for band in ("base", "mid", "apex")}  # inter, pred, truth
    per_scan = []
    rows = np.arange(truths.shape[2])
    for scan, order in scans.items():
        t, p = truths[order] == PROSTATE, preds[order] == PROSTATE
        if not t.any():
            continue
        prostate_rows = rows[t.any(axis=(0, 1))]
        lo, hi = prostate_rows.min(), prostate_rows.max() + 1
        bladder_rows = rows[(truths[order] == BLADDER).any(axis=(0, 1))]
        bladder_above = bladder_rows.size == 0 or bladder_rows.mean() >= prostate_rows.mean()

        cut1, cut2 = lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3
        low_band, high_band = rows < cut1, rows >= cut2
        bands = {
            "base": high_band if bladder_above else low_band,
            "mid": ~low_band & ~high_band,
            "apex": low_band if bladder_above else high_band,
        }
        record = {"scan": scan}
        for band, mask in bands.items():
            tb, pb = t[:, :, mask], p[:, :, mask]
            counts = np.array([(tb & pb).sum(), pb.sum(), tb.sum()], dtype=float)
            totals[band] += counts
            record[band] = 2 * counts[0] / max(counts[1] + counts[2], 1)
        per_scan.append(record)

    pooled = {band: 2 * c[0] / max(c[1] + c[2], 1) for band, c in totals.items()}
    return pooled, per_scan


def per_slice_rows(paths, truths, preds, regions):
    rows = []
    for path, t, p, region in zip(paths, truths, preds, regions):
        scan, index = slice_info(path)
        gt, pr = t == PROSTATE, p == PROSTATE
        inter = int((gt & pr).sum())
        denom = int(gt.sum()) + int(pr.sum())
        rows.append({
            "scan": scan,
            "slice": index,
            "region": region,
            "gt_pixels": int(gt.sum()),
            "pred_pixels": int(pr.sum()),
            # Dice is undefined when both are empty; NaN keeps those out of averages.
            "prostate_dice": 2 * inter / denom if denom else float("nan"),
        })
    return rows


def region_summary(rows):
    """Mean and worst per-slice prostate Dice for each slice region."""
    summary = {}
    for region in ("anterior", "central", "posterior"):
        sel = [r for r in rows if r["region"] == region]
        dices = [r["prostate_dice"] for r in sel]
        summary[region] = {
            "slices": len(sel),
            "mean_slice_dice": float(np.nanmean(dices)) if sel else float("nan"),
            "worst_slice_dice": float(np.nanmin(dices)) if sel else float("nan"),
        }
    fp = [r for r in rows if r["region"] == "none" and r["pred_pixels"] > 0]
    summary["false_positive_slices"] = len(fp)
    summary["no_prostate_slices"] = sum(r["region"] == "none" for r in rows)
    return summary


@torch.no_grad()
def measure_latency(model, dataset, device, n=50, warmup=5):
    """
    Mean seconds per slice at batch size 1, after warm-up, and peak GPU
    memory in MiB during those passes (weights plus activations; None when
    not on CUDA).
    """
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    x = dataset[0][0].unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        model(x)
    sync()
    start = time.perf_counter()
    for _ in range(n):
        model(x)
    sync()
    seconds = (time.perf_counter() - start) / n
    peak_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    return seconds, peak_mib


def error_overlay(image, truth, pred):
    """
    RGB image: prostate true positives green, false positives red, false
    negatives orange. 
    """
    base = (image - image.min()) / (np.ptp(image) + 1e-8)
    rgb = np.stack([base] * 3, axis=-1) * 0.6
    gt, pr = truth == PROSTATE, pred == PROSTATE
    rgb[gt & pr] = [0.1, 0.9, 0.1]
    rgb[~gt & pr] = [1.0, 0.1, 0.1]
    rgb[gt & ~pr] = [1.0, 0.6, 0.0]
    return rgb


def show(ax, img, title, **kwargs):
    # Arrays are (256, 128) with axis 0 left-right; transpose so superior is up.
    ax.imshow(np.swapaxes(img, 0, 1), origin="lower", **kwargs)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def error_title(row):
    """Dice where the slice has prostate; otherwise the false pixel count."""
    if row["gt_pixels"] > 0:
        return f"prostate Dice {row['prostate_dice']:.2f}"
    return f"no prostate, {row['pred_pixels']} px false"


def plot_worst(name, idx, rows, images, truths, preds, path):
    fig, axes = plt.subplots(len(idx), 4, figsize=(10, 1.9 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        row = rows[i]
        label = f"{row['scan']} s{row['slice']} ({row['region']})"
        show(axes[r, 0], images[i], label, cmap="gray")
        show(axes[r, 1], truths[i], "ground truth", cmap=LABEL_CMAP, vmin=0, vmax=NUM_CLASSES - 1)
        show(axes[r, 2], preds[i], f"{name} prediction", cmap=LABEL_CMAP, vmin=0, vmax=NUM_CLASSES - 1)
        show(axes[r, 3], error_overlay(images[i], truths[i], preds[i]),
             f"{error_title(row)}\nTP green, FP red, FN orange")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_compare(idx, results, images, truths, path):
    names = list(results)
    fig, axes = plt.subplots(len(idx), 2 + len(names), figsize=(2.6 * (2 + len(names)), 1.9 * len(idx)),
                             squeeze=False)
    first = results[names[0]]["rows"]
    for r, i in enumerate(idx):
        show(axes[r, 0], images[i], f"{first[i]['scan']} s{first[i]['slice']} ({first[i]['region']})", cmap="gray")
        show(axes[r, 1], truths[i], "ground truth", cmap=LABEL_CMAP, vmin=0, vmax=NUM_CLASSES - 1)
        for c, name in enumerate(names):
            res = results[name]
            show(axes[r, 2 + c], error_overlay(images[i], truths[i], res["preds"][i]),
                 f"{name}: {error_title(res['rows'][i])}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    test = get_datasets(args.data_root, augment=False)["test"]
    paths = [image_path for image_path, _ in test.pairs]

    scans = group_scans(paths)

    results = {}
    for ckpt_path in args.checkpoints:
        name, model = load_model(ckpt_path, device)
        images, truths, preds, meter = predict_all(model, test, device)
        regions = assign_slice_regions(scans, truths)
        rows = per_slice_rows(paths, truths, preds, regions)
        summary = region_summary(rows)
        bands, bands_per_scan = vertical_band_dice(scans, truths, preds)
        latency, peak_mib = measure_latency(model, test, device)
        results[name] = {"rows": rows, "preds": preds}

        print(f"\n=== {name} ({ckpt_path}) on {device} ===")
        for c, (d, iou) in enumerate(zip(meter.dice(), meter.iou())):
            print(f"  {CLASS_NAMES[c]:10s} Dice {d:.4f}  IoU {iou:.4f}")
        print("  prostate Dice by superior-inferior band (pooled over test scans):")
        print("    " + "  ".join(f"{band} {bands[band]:.3f}" for band in ("base", "mid", "apex")))
        print("  prostate by slice position (mean / worst per-slice Dice):")
        for region in ("anterior", "central", "posterior"):
            s = summary[region]
            print(f"    {region:9s} {s['slices']:3d} slices  "
                  f"{s['mean_slice_dice']:.3f} / {s['worst_slice_dice']:.3f}")
        print(f"  false-positive prostate on {summary['false_positive_slices']} of "
              f"{summary['no_prostate_slices']} slices with no prostate")
        print(f"  latency {latency * 1000:.1f} ms per slice (batch 1)")
        if peak_mib is not None:
            print(f"  peak GPU memory {peak_mib:.0f} MiB at inference (batch 1)")

        write_csv(rows, os.path.join(args.out_dir, f"{name}_per_slice.csv"))
        write_csv(bands_per_scan, os.path.join(args.out_dir, f"{name}_bands_per_scan.csv"))
        scored = [i for i, r in enumerate(rows) if r["gt_pixels"] > 0]
        worst = sorted(scored, key=lambda i: rows[i]["prostate_dice"])[:args.n_worst]
        plot_worst(name, worst, rows, images, truths, preds, os.path.join(args.out_dir, f"{name}_worst.png"))

        # Dice cannot rank slices with no true prostate (it is 0 for any false
        # prostate), so rank over-predictions by the number of false pixels.
        false_slices = [i for i, r in enumerate(rows) if r["gt_pixels"] == 0 and r["pred_pixels"] > 0]
        worst_fp = sorted(false_slices, key=lambda i: -rows[i]["pred_pixels"])[:args.n_worst]
        if worst_fp:
            plot_worst(name, worst_fp, rows, images, truths, preds,
                       os.path.join(args.out_dir, f"{name}_worst_fp.png"))

    if len(results) > 1:
        # Slices that are worst on average across models, shown for every model.
        rows0 = next(iter(results.values()))["rows"]
        scored = [i for i, r in enumerate(rows0) if r["gt_pixels"] > 0]
        mean_dice = {i: np.mean([res["rows"][i]["prostate_dice"] for res in results.values()]) for i in scored}
        worst = sorted(scored, key=mean_dice.get)[:args.n_worst]
        plot_compare(worst, results, images, truths, os.path.join(args.out_dir, "compare_worst.png"))

        # Over-predictions: empty slices with the most false prostate on average.
        empty = [i for i, r in enumerate(rows0) if r["gt_pixels"] == 0]
        mean_fp = {i: np.mean([res["rows"][i]["pred_pixels"] for res in results.values()]) for i in empty}
        worst_fp = [i for i in sorted(empty, key=lambda i: -mean_fp[i]) if mean_fp[i] > 0][:args.n_worst]
        if worst_fp:
            plot_compare(worst_fp, results, images, truths, os.path.join(args.out_dir, "compare_worst_fp.png"))

    print(f"\nSaved CSVs and figures to {args.out_dir}")


if __name__ == "__main__":
    main()
