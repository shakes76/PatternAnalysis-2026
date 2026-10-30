"""
predict.py - Evaluate a trained model on the TEST patients and make the report figures.

WHAT IT DOES
    1. Loads a saved checkpoint (outputs/best_unet2d.pt or outputs/best_improved3d.pt).
    2. Predicts every test scan (cases 040-042, 9 scans in total) at FULL native resolution:
         - 2D model: each 256x128 slice, exactly as in the 2D dataset
         - 3D model: predicts at 128x128x64, then the class scores are upsampled back to the
           original 256x256x128 and compared with the ORIGINAL masks (so the 3D model is not
           marked on its own easier, downsampled masks)
    3. Reports:
         - per-class Dice and IoU over the whole test set
         - prostate Dice for EACH scan (the same 9 scans for both models -> paired comparison)
         - prostate Dice on the boundary slices of the prostate vs the middle slices
           (the dilemma: do the edges of the prostate get cut off?)
         - inference time per scan (model only, no file loading) and peak GPU memory (CUDA only)
    4. Saves figures of the worst slices (image | ground truth | prediction | error map)
       for the failure autopsy, and everything as JSON for the README tables.

WHY 2D AND 3D ARE SCORED SEPARATELY
    The course's 2D slices are not exact cuts of the 3D volumes (they were preprocessed
    separately), so we can't compare voxel-for-voxel. Instead both models are scored on the
    same 9 scans, each at its own native full resolution.

HOW TO RUN (from the project folder)
    python predict.py --model unet2d     --data-root ~/Datasets/HipMRI/keras_slices_data
    python predict.py --model improved3d --data-root ~/Datasets/HipMRI
    (On a Mac, put PYTORCH_ENABLE_MPS_FALLBACK=1 in front of the 3D command.)
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from dataset import (HipMRI2DDataset, get_volume_pairs, split_of_case, volume_key, load_nifti_3d,
                     resize_volume, DATA_ROOT, DATA_ROOT_3D, TARGET_SHAPE_3D, NUM_CLASSES, PROSTATE_LABEL)
from modules import count_parameters
from train import build_model, get_device
from utils import SegmentationMetrics, CLASS_NAMES

EDGE_FRACTION = 0.2  # first/last 20% of the prostate-containing slices count as "boundary" slices


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model on the test set")
    p.add_argument("--model", required=True, choices=["unet2d", "improved3d", "plain3d"])
    p.add_argument("--ckpt", default=None, help="checkpoint path (default: outputs/best_<model>.pt)")
    p.add_argument("--data-root", default=None, help="data folder (default: the Rangpur path)")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--n-worst", type=int, default=4, help="how many worst slices to plot for the autopsy")
    return p.parse_args()


def sync(device):
    """Wait for the GPU to finish, so timings measure real work (GPU calls run asynchronously)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def dice(pred, truth):
    """Dice for one class from two boolean arrays. None if the class is in neither (0/0)."""
    denom = pred.sum() + truth.sum()
    return None if denom == 0 else float(2 * (pred & truth).sum() / denom)


def edge_vs_middle(pred_stack, true_stack):
    """Split one scan's slices into prostate 'boundary' vs 'middle' slices and pool Dice for each.

    pred_stack / true_stack: boolean arrays (n_slices, H, W) saying "is prostate".
    Only slices whose ground truth contains prostate are used. The first and last
    EDGE_FRACTION of them (along the slice order) are the boundary slices; the rest are the middle.
    Returns pixel counts so results can be pooled across scans: {"edge": [inter, pred, true], ...}
    """
    idx = np.where(true_stack.reshape(len(true_stack), -1).any(axis=1))[0]
    counts = {"edge": np.zeros(3), "middle": np.zeros(3)}
    if len(idx) == 0:
        return counts
    n_edge = max(1, int(round(EDGE_FRACTION * len(idx))))
    edge = set(idx[:n_edge]) | set(idx[-n_edge:])
    for i in idx:
        p, t = pred_stack[i], true_stack[i]
        counts["edge" if i in edge else "middle"] += [(p & t).sum(), p.sum(), t.sum()]
    return counts


def pooled_dice(c):
    """Dice from pooled [intersection, predicted, true] counts."""
    return float(2 * c[0] / max(c[1] + c[2], 1))


# ---------------------------------------------------------------------------
# 2D model: predict slice by slice, then group the slices back into their scans
# ---------------------------------------------------------------------------
def predict_2d(model, device, root):
    ds = HipMRI2DDataset("test", root)
    name_re = re.compile(r"case_(\d+)_week_(\d+)_slice_(\d+)")
    scans = defaultdict(list)  # "040_w0" -> list of (slice number, dataset index)
    for i, (img_path, _) in enumerate(ds.pairs):
        case, week, sl = name_re.search(os.path.basename(img_path)).groups()
        scans[f"{case}_w{week}"].append((int(sl), i))

    metrics = SegmentationMetrics(NUM_CLASSES)
    per_scan, slices, times = {}, [], []
    edge_counts = {"edge": np.zeros(3), "middle": np.zeros(3)}

    for scan, items in sorted(scans.items()):
        items.sort()  # by slice number (slice_10 after slice_9, not after slice_1)
        imgs = torch.stack([ds[i][0] for _, i in items])   # (60, 1, 256, 128)
        masks = torch.stack([ds[i][1] for _, i in items])  # (60, 256, 128)

        sync(device); t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(imgs.to(device))
        sync(device); times.append(time.perf_counter() - t0)

        logits = logits.float().cpu()
        metrics.update(logits, masks)
        preds = logits.argmax(1).numpy()
        truth = masks.numpy()
        p_pro, t_pro = preds == PROSTATE_LABEL, truth == PROSTATE_LABEL
        per_scan[scan] = dice(p_pro, t_pro)
        c = edge_vs_middle(p_pro, t_pro)
        for k in edge_counts:
            edge_counts[k] += c[k]
        for j, (sl, _) in enumerate(items):  # keep slices with prostate for the autopsy
            d = dice(p_pro[j], t_pro[j])
            if t_pro[j].any():
                slices.append({"scan": scan, "slice": sl, "dice": d, "img": imgs[j, 0].numpy(),
                               "true": truth[j], "pred": preds[j]})
    return metrics, per_scan, edge_counts, slices, times


# ---------------------------------------------------------------------------
# 3D model: predict each volume at 128x128x64, upsample back to native size
# ---------------------------------------------------------------------------
def predict_3d(model, device, root):
    pairs = [(img, seg, volume_key(img)) for img, seg, case in get_volume_pairs(root)
             if split_of_case(case) == "test"]
    metrics = SegmentationMetrics(NUM_CLASSES)        # full native resolution (the fair score)
    metrics_small = SegmentationMetrics(NUM_CLASSES)  # at 128x128x64 (what train.py reported)
    per_scan, slices, times = {}, [], []
    edge_counts = {"edge": np.zeros(3), "middle": np.zeros(3)}

    for img_path, seg_path, (case, week) in sorted(pairs, key=lambda x: x[2]):
        scan = f"{case}_w{week}"
        img = load_nifti_3d(img_path)
        truth = np.rint(load_nifti_3d(seg_path)).astype(np.int64)  # native-resolution mask
        x = (img - img.mean()) / (img.std() + 1e-8)  # same preprocessing as training
        x = torch.from_numpy(resize_volume(x, TARGET_SHAPE_3D, is_mask=False))[None, None]

        sync(device); t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(x.to(device))
        sync(device); times.append(time.perf_counter() - t0)

        logits = logits.float().cpu()
        small_truth = resize_volume(truth.astype(np.float32), TARGET_SHAPE_3D, is_mask=True)
        metrics_small.update(logits, torch.from_numpy(small_truth).long()[None])
        # Upsample the class SCORES (smooth) rather than the labels (blocky), then pick the class
        big = F.interpolate(logits, size=truth.shape, mode="trilinear", align_corners=False)
        metrics.update(big, torch.from_numpy(truth)[None])
        preds = big.argmax(1)[0].numpy()
        del big

        p_pro, t_pro = preds == PROSTATE_LABEL, truth == PROSTATE_LABEL
        per_scan[scan] = dice(p_pro, t_pro)
        # slices along the last axis (128 of them), as (n_slices, H, W)
        p_st, t_st = np.moveaxis(p_pro, -1, 0), np.moveaxis(t_pro, -1, 0)
        c = edge_vs_middle(p_st, t_st)
        for k in edge_counts:
            edge_counts[k] += c[k]
        img_st, pred_st, true_st = (np.moveaxis(a, -1, 0) for a in (img, preds, truth))
        for j in np.where(t_st.reshape(len(t_st), -1).any(axis=1))[0]:
            slices.append({"scan": scan, "slice": int(j), "dice": dice(p_st[j], t_st[j]),
                           "img": img_st[j].astype(np.float32),  # small dtypes: keeps RAM use low
                           "true": true_st[j].astype(np.uint8), "pred": pred_st[j].astype(np.uint8)})
    return metrics, metrics_small, per_scan, edge_counts, slices, times


def pick_worst(slices, n):
    """The n lowest-Dice slices, taking at most one per scan so the autopsy shows
    different scans instead of n neighbouring slices of the same bad scan."""
    worst, used = [], set()
    for s in sorted(slices, key=lambda s: s["dice"]):
        if s["scan"] not in used:
            worst.append(s)
            used.add(s["scan"])
        if len(worst) == n:
            break
    return worst

def pick_examples(slices, n):
    """Typical examples for the README: in each scan, the slice with the most prostate
    (usually the middle of the gland), for up to n different scans."""
    biggest = {}
    for s in slices:
        area = int((s["true"] == PROSTATE_LABEL).sum())
        if s["scan"] not in biggest or area > biggest[s["scan"]][0]:
            biggest[s["scan"]] = (area, s)
    return [s for _, s in list(biggest.values())[:n]]


# ---------------------------------------------------------------------------
# Figure for the failure autopsy
# ---------------------------------------------------------------------------
def plot_worst(slices, n, path, title, chooser=None):
    """Rows = the n worst prostate slices (one per scan). Columns = image | ground truth | prediction | error map.
    Error map: green = correct prostate, red = false positive (extra), blue = false negative (missed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worst = (chooser or pick_worst)(slices, n)  # which slices to show (default: the worst ones)
    fig, axes = plt.subplots(len(worst), 4, figsize=(12, 3.2 * len(worst)), squeeze=False)
    for row, s in zip(axes, worst):
        t, p = s["true"] == PROSTATE_LABEL, s["pred"] == PROSTATE_LABEL
        err = np.zeros(t.shape + (3,))
        err[t & p] = (0, 0.8, 0)   # green: hit
        err[p & ~t] = (1, 0, 0)    # red: false positive
        err[t & ~p] = (0, 0.4, 1)  # blue: missed
        row[0].imshow(s["img"], cmap="gray")
        row[0].set_title(f"scan {s['scan']} slice {s['slice']}")
        row[1].imshow(s["true"], cmap="tab10", vmin=0, vmax=9)
        row[1].set_title("ground truth")
        row[2].imshow(s["pred"], cmap="tab10", vmin=0, vmax=9)
        row[2].set_title("prediction")
        row[3].imshow(s["img"], cmap="gray")
        row[3].imshow(err, alpha=0.6)
        row[3].set_title(f"prostate errors (Dice {s['dice']:.2f})")
        for a in row:
            a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    args = parse_args()
    is3d = args.model != "unet2d"
    root = args.data_root or (DATA_ROOT_3D if is3d else DATA_ROOT)
    root = os.path.expanduser(root)
    ckpt_path = args.ckpt or os.path.join(args.out_dir, f"best_{args.model}.pt")
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(args.model, ckpt["base"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()  # eval: dropout off, normalisation uses learned statistics
    print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}) on {device}")

    # Warm-up pass so one-off GPU setup cost isn't counted in the timing
    dummy = torch.zeros((1, 1) + (TARGET_SHAPE_3D if is3d else (256, 128)), device=device)
    with torch.no_grad():
        model(dummy)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if is3d:
        metrics, metrics_small, per_scan, edges, slices, times = predict_3d(model, device, root)
    else:
        metrics, per_scan, edges, slices, times = predict_2d(model, device, root)
        metrics_small = None
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else None

    # ---- Print results ----
    d, iou = metrics.dice(), metrics.iou()
    print(f"\nTest set, full native resolution ({len(per_scan)} scans)")
    print(f"{'class':<12}{'Dice':>8}{'IoU':>8}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"{name:<12}{d[c]:>8.4f}{iou[c]:>8.4f}")
    if metrics_small is not None:
        print(f"(prostate Dice at 128x128x64, as train.py reported: {metrics_small.dice()[PROSTATE_LABEL]:.4f})")

    vals = [v for v in per_scan.values() if v is not None]
    print("\nProstate Dice per scan:")
    for scan, v in per_scan.items():
        print(f"  {scan}: {v:.4f}" if v is not None else f"  {scan}: no prostate")
    print(f"  mean {np.mean(vals):.4f} +/- {np.std(vals):.4f} (std)")

    edge_d, mid_d = pooled_dice(edges["edge"]), pooled_dice(edges["middle"])
    print(f"\nProstate Dice on boundary slices: {edge_d:.4f} | middle slices: {mid_d:.4f}")
    print(f"Inference time per scan (model only): {1000 * np.mean(times):.1f} ms")
    if peak_gb is not None:
        print(f"Peak GPU memory (inference): {peak_gb:.2f} GB")

    # ---- Figures and JSON ----
    plot_worst(slices, args.n_worst, os.path.join(args.out_dir, f"{args.model}_worst_cases.png"),
               f"{args.model}: worst prostate slices on the test set")
    plot_worst(slices, args.n_worst, os.path.join(args.out_dir, f"{args.model}_example_cases.png"),
               f"{args.model}: typical test slices (largest prostate per scan)", chooser=pick_examples)
    results = {
        "model": args.model, "checkpoint_epoch": ckpt["epoch"], "params": count_parameters(model),
        "test_dice": dict(zip(CLASS_NAMES, d)), "test_iou": dict(zip(CLASS_NAMES, iou)),
        "prostate_dice_downsampled": metrics_small.dice()[PROSTATE_LABEL] if metrics_small else None,
        "prostate_dice_per_scan": per_scan,
        "prostate_dice_per_scan_mean": float(np.mean(vals)), "prostate_dice_per_scan_std": float(np.std(vals)),
        "prostate_dice_boundary_slices": edge_d, "prostate_dice_middle_slices": mid_d,
        "inference_ms_per_scan": 1000 * float(np.mean(times)), "device": device.type,
        "peak_inference_vram_gb": peak_gb,
        "worst_slices": [{k: s[k] for k in ("scan", "slice", "dice")}
                         for s in pick_worst(slices, args.n_worst)],
    }
    out = os.path.join(args.out_dir, f"{args.model}_eval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()