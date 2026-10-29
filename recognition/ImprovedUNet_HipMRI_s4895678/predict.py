"""Inference and failure analysis for the HipMRI prostate segmentation models.

Examples (run from this folder):
  python predict.py --model unet2d --analyse
  python predict.py --model improved3d --figures W029_Week5 G021_Week5
  python predict.py --model improved3d --tag matched_s1 --analyse   # a repeat run
Loads checkpoints/<model>_best.pth, predicts whole test volumes, prints Dice,
Dice by prostate third (base/middle/apex), leakage, and saves 4-panel figures.
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import PROSTATE, load_case, load_split, patient_of
from modules import ImprovedUNet, UNet2D

# label values of the two organs next to the prostate (the prostate label, 5, comes from dataset.py)
BLADDER, RECTUM = 3, 4
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]


def build(name, base3d=16):
    """Create the model chosen on the command line (same settings as in train.py)."""
    if name == "unet2d":
        return UNet2D()
    if name == "improved2d":
        return ImprovedUNet(dim=2, base=32, depth=5)
    if name == "improved3d":
        return ImprovedUNet(dim=3, base=base3d, depth=5)
    raise ValueError(f"unknown model: {name}")


def load_model(name, device, base3d=16, tag=""):
    """Load checkpoints/<name>[_<tag>]_best.pth (tag marks a repeat run, e.g. matched_s1)."""
    model = build(name, base3d).to(device)
    suffix = f"_{tag}" if tag else ""
    model.load_state_dict(torch.load(f"checkpoints/{name}{suffix}_best.pth", map_location=device))
    return model.eval()


@torch.no_grad()
def predict_volume(model, name, img, device, bs=16):
    """img: float32 [D,H,W] (z-scored) -> predicted labels uint8 [D,H,W]."""
    x = torch.from_numpy(img)
    if name.endswith("3d"):      # 3D model: whole volume in one pass (add batch and channel axes)
        return model(x[None, None].to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)
    # 2D models: predict the slices in batches, then stack them back into a volume
    out = []
    for i in range(0, x.shape[0], bs):
        out.append(model(x[i:i + bs, None].to(device)).argmax(1).cpu())
    return torch.cat(out).numpy().astype(np.uint8)


def dice(a, b):
    """Dice overlap of two boolean arrays (1.0 if both are empty)."""
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * (a & b).sum() / s)


def iou(a, b):
    """Intersection-over-union of two boolean arrays (1.0 if both are empty)."""
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def prostate_thirds(lab):
    """Split the true prostate's slices into base / middle / apex thirds.

    The base is the end nearer the bladder, found from the labels themselves
    (mean slice index of the bladder vs the prostate), so we do not rely on
    the image orientation metadata.
    """
    zs = np.flatnonzero((lab == PROSTATE).any(axis=(1, 2)))
    bladder_z = np.nonzero(lab == BLADDER)[0].mean()
    prostate_z = np.nonzero(lab == PROSTATE)[0].mean()
    low, mid, high = np.array_split(zs, 3)            # zs is sorted ascending
    if bladder_z > prostate_z:                        # bladder at high z -> base is high
        return {"apex": low, "middle": mid, "base": high}, zs
    return {"base": low, "middle": mid, "apex": high}, zs


def analyse(model, name, device, split_name="test", tag=""):
    """Score every volume of a split and save results/<name>[_<tag>]_analysis.csv.

    Per volume: prostate Dice, per-class Dice and IoU, Dice in the base/middle/apex thirds,
    leakage of the predicted prostate onto the rectum and bladder, and inference time.
    """
    split = load_split()
    cases = sorted(c for c, s in split.items() if s == split_name)
    rows = []
    for case in cases:
        img, lab = load_case(case)
        # time the prediction only (synchronise so queued GPU work is finished before stopping)
        t0 = time.time()
        pred = predict_volume(model, name, img, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - t0
        # boolean prostate masks: ground truth (gt_p) and prediction (pr_p); n_pred is the
        # predicted size, at least 1 so that the ratios below never divide by zero
        gt_p, pr_p = lab == PROSTATE, pred == PROSTATE
        thirds, zs = prostate_thirds(lab)
        n_pred = max(int(pr_p.sum()), 1)
        # leak_* = share of the predicted prostate voxels that lie on the TRUE rectum / bladder;
        # pred_outside_range = share predicted on slices where the true prostate is absent
        row = {"case": case, "patient": patient_of(case),
               "prostate_dice": dice(pr_p, gt_p),
               "leak_rectum": float((pr_p & (lab == RECTUM)).sum() / n_pred),
               "leak_bladder": float((pr_p & (lab == BLADDER)).sum() / n_pred),
               "pred_outside_range": float((int(pr_p.sum()) - int(pr_p[zs].sum())) / n_pred),
               "seconds": sec}
        for c, cname in enumerate(CLASS_NAMES):          # per-class Dice and IoU
            row[f"dice_{cname}"] = dice(pred == c, lab == c)
            row[f"iou_{cname}"] = iou(pred == c, lab == c)
        for part, idx in thirds.items():      # Dice inside the base / middle / apex slices
            row[f"dice_{part}"] = dice(pr_p[idx], gt_p[idx])
        rows.append(row)
        print(f"{case}: prostate Dice {row['prostate_dice']:.3f} | "
              f"base {row['dice_base']:.3f} mid {row['dice_middle']:.3f} apex {row['dice_apex']:.3f} | "
              f"rectum leak {row['leak_rectum']:.3f}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}{'_' + tag if tag else ''}_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n===== {name} on {split_name} ({len(rows)} volumes) =====")
    for key in ("prostate_dice", "iou_prostate", "dice_rectum", "iou_rectum", "dice_base", "dice_middle", "dice_apex",
                "leak_rectum", "leak_bladder", "pred_outside_range", "seconds"):
        print(f"{key:20s} mean {np.mean([r[key] for r in rows]):.4f}")
    return rows


def four_panel(model, name, device, case, z=None, rotate=True, outdir="images", tag=""):
    """Save input / ground truth / prediction / error overlay for one slice."""
    img, lab = load_case(case)
    pred = predict_volume(model, name, img, device)
    gt_p, pr_p = lab == PROSTATE, pred == PROSTATE
    _, zs = prostate_thirds(lab)
    if z is None:     # default: the worst slice among those that truly contain prostate
        z = int(zs[np.argmin([dice(pr_p[i], gt_p[i]) for i in zs])])
    # rotate for display only (the saved arrays keep their orientation)
    show = (lambda a: np.rot90(a, -1)) if rotate else (lambda a: a)

    # RGBA overlay that colours each pixel by its error type
    err = np.zeros(gt_p[z].shape + (4,))
    err[pr_p[z] & gt_p[z]] = (0.0, 0.8, 0.0, 0.5)     # correct: green
    err[pr_p[z] & ~gt_p[z]] = (1.0, 0.0, 0.0, 0.7)    # false positive: red
    err[~pr_p[z] & gt_p[z]] = (0.0, 0.3, 1.0, 0.7)    # false negative: blue

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.6))
    for a in ax:
        a.imshow(show(img[z]), cmap="gray")
        a.axis("off")
    ax[0].set_title(f"{case}, slice {z}")
    ax[1].contour(show(gt_p[z].astype(float)), levels=[0.5], colors="red")
    ax[1].set_title("ground truth (red)")
    ax[2].contour(show(pr_p[z].astype(float)), levels=[0.5], colors="yellow")
    ax[2].set_title(f"{name} prediction (yellow)")
    ax[3].imshow(show(err))
    ax[3].set_title(f"slice Dice {dice(pr_p[z], gt_p[z]):.2f}: red FP, blue FN")
    os.makedirs(outdir, exist_ok=True)
    path = f"{outdir}/{name}{'_' + tag if tag else ''}_{case}_z{z}.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path} | whole-volume prostate Dice {dice(pr_p, gt_p):.3f}")


def main():
    """Command line entry point: load a checkpoint, then score the split and/or draw figures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["unet2d", "improved2d", "improved3d"])
    ap.add_argument("--analyse", action="store_true", help="score every volume of --split")
    ap.add_argument("--figures", nargs="*", default=[], help="case names to draw")
    ap.add_argument("--z", type=int, default=None, help="slice to draw (default: worst)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--base3d", type=int, default=16)
    ap.add_argument("--tag", default="", help="suffix used in training, e.g. matched_s1")
    ap.add_argument("--no-rotate", action="store_true", help="draw in array orientation")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = load_model(args.model, device, args.base3d, args.tag)
    if args.analyse:
        analyse(model, args.model, device, args.split, args.tag)
    for case in args.figures:
        four_panel(model, args.model, device, case, args.z, not args.no_rotate, tag=args.tag)


if __name__ == "__main__":
    main()
