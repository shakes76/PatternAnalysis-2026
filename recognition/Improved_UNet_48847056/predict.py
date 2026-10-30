"""
predict.py - load the trained 2D and 3D U-Nets, compare them on the same test volumes,
and produce the figures used in the README:

  outputs/comparison.md        per-class Dice/IoU + resource table for both models
  outputs/slice_position.png   prostate Dice vs position from base (bladder side) to apex
  outputs/failure_*.png        multi-panel overlays of the worst prostate slices

Usage:
    python predict.py [--root PATH/TO/HipMRI_Study_open] [--out outputs] [--n-fail 4]
Works with just one checkpoint present (e.g. only unet2d_best.pt) as well.
"""

import argparse
import os

import matplotlib

# plots saved as files
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import NUM_CLASSES, get_data_3d
from modules import predict_volume
# reuse model builder and eval loop from training
from train import CLASS_NAMES, PROSTATE, RANGPUR_ROOT, build, evaluate

BLADDER = 3 # used to tell which end of the prostate is the base
N_BINS = 10  # bins along the prostate's superior-inferior extent


def load_model(ckpt_path, device):
    """
    rebuild model from checkpoint saved by train.py
    """
    ck = torch.load(ckpt_path, map_location=device)
    a = ck["args"]
    model = build(a["dims"], a["base"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, a


def base_to_apex_positions(gt):
    """
    For one ground truth volume (H, W, D), return {slice_index: position in [0, 1]} over the
    slices that contain prostate, with 0 = base (the end nearest the bladder) and
    1 = apex. Orientation is decided per volume from the bladder centroid.
    """

    # indices of axial slices that contain any prostate voxel
    z = np.where((gt == PROSTATE).any(axis=(0, 1)))[0]
    if len(z) == 0:
        return {}
    lo, hi = z.min(), z.max() # first and last prostate slices
    # z coords of all bladder voxels, the mean is the bladder's central slice
    bz = np.where(gt == BLADDER)[2]
    # the base of the prostate sits against the bladder, thus, whichever end is closer to the bladder is the base.
    # defaults to "low end is base"
    base_is_low = len(bz) == 0 or abs(bz.mean() - lo) <= abs(bz.mean() - hi)
    span = max(hi - lo, 1) # avoide divide by zero for single-slice prostate
    # returns positions in order from base to apex
    return {int(k): ((k - lo) / span if base_is_low else (hi - k) / span) for k in z}


def slice_position_curve(slice_dice, labels):
    """
    Average per-slice prostate Dice into N_BINS bins from base (0) to apex (1).

    slice_dice: per vol, an array of prostate Dice for each slice
    labels: matching ground truth volumes

    """
    bins = [[] for _ in range(N_BINS)]
    for sd, gt in zip(slice_dice, labels):
        for k, pos in base_to_apex_positions(gt).items():
            d = sd[k]
            if not np.isnan(d):
                bins[min(int(pos * N_BINS), N_BINS - 1)].append(d)
    return np.array([np.mean(b) if b else np.nan for b in bins]), [len(b) for b in bins]


def comparison_table(res):
    """Markdown table: per-class Dice/IoU, per-volume prostate Dice and resources."""
    names = list(res)
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    # start from 1 to skip background
    for c in range(1, NUM_CLASSES):
        lines.append(f"| {CLASS_NAMES[c]} Dice / IoU | "
                     + " | ".join(f"{res[n]['test']['dice'][c]:.3f} / {res[n]['test']['iou'][c]:.3f}" for n in names)
                     + " |")
    # mean shows typical performance, min shows worst patient which is important clinically
    lines.append("| Prostate Dice per volume, mean (min) | " + " | ".join(
        f"{np.nanmean(res[n]['test']['vol_prostate_dice']):.3f} ({np.nanmin(res[n]['test']['vol_prostate_dice']):.3f})"
        for n in names) + " |")
    lines.append("| Parameters | " + " | ".join(f"{res[n]['params']:,}" for n in names) + " |")
    lines.append("| Peak VRAM inference (MB) | " + " | ".join(f"{res[n]['peak_mb']:.0f}" for n in names) + " |")
    lines.append("| Inference time per volume (ms) | "
                 + " | ".join(f"{res[n]['test']['sec_per_volume'] * 1000:.1f}" for n in names) + " |")
    return "\n".join(lines)


def overlay(ax, img, gt, pred, title):
    """Image with prostate error overlay: green = correct, red = false positive,
    blue = false negative.
    
    img, gt, pred: 2D (H, W) arrays for one slice.
    """
    # .T + origin="lower" rotates the array into the usual radiological view
    ax.imshow(img.T, cmap="gray", origin="lower")
    rgba = np.zeros(img.shape + (4,)) # transparent colour layer 
    g, p = gt == PROSTATE, pred == PROSTATE
    rgba[g & p] = (0, 1, 0, 0.5) # true positive
    rgba[~g & p] = (1, 0, 0, 0.7) # false positive
    rgba[g & ~p] = (0, 0.4, 1, 0.7) # false negative 
    # line up colour layer same as image
    ax.imshow(rgba.transpose(1, 0, 2), origin="lower")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=RANGPUR_ROOT, help="HipMRI_Study_open folder (default: Rangpur path)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--n-fail", type=int, default=4, help="number of failure slices to plot")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    # load whichever checkpoints exist
    models = {}
    for name in ("unet2d", "unet3d"):
        path = os.path.join(args.out, f"{name}_best.pt")
        if os.path.exists(path):
            models[name] = load_model(path, device)
    if not models:
        raise SystemExit(f"no checkpoints found in {args.out}/")

    # load test split using first model's data settings. both are then scored on these volumes.
    a = next(iter(models.values()))[1]
    _, _, (tx, ty), test_files = get_data_3d(args.root, dims=a["dims"], size=tuple(a["size"]),
                                             early_stop=a["early_stop"])

    # metrics on identical test volumes 
    res, preds = {}, {}
    for name, (model, ma) in models.items():
        # reset peak memory counter to measure each model's peak separately 
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        test = evaluate(model, tx, ty, ma["dims"], device, amp)
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
        res[name] = {"test": test, "peak_mb": peak, "params": sum(p.numel() for p in model.parameters())}
        with torch.no_grad(), torch.autocast(device.type, enabled=amp):
            preds[name] = [predict_volume(model, torch.from_numpy(v).to(device), ma["dims"]).argmax(0).cpu().numpy()
                           for v in tx]

    table = comparison_table(res)
    print(table)
    with open(os.path.join(args.out, "comparison.md"), "w") as f:
        f.write(table + "\n")

    # prostate Dice from base to apex
    # tests whether 3d context helps at the base/apex ends where global Dice hides failures.
    fig, ax = plt.subplots(figsize=(7, 4))
    centres = (np.arange(N_BINS) + 0.5) / N_BINS
    for name in res:
        curve, counts = slice_position_curve(res[name]["test"]["slice_prostate_dice"], ty)
        ax.plot(centres, curve, marker="o", label=name)
        print(f"{name} Dice by position (base->apex):", np.round(curve, 3).tolist(), "slices/bin:", counts)
    ax.set_xlabel("position within prostate (0 = base, bladder side; 1 = apex)")
    ax.set_ylabel("mean per-slice prostate Dice")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "slice_position.png"), dpi=130)
    plt.close(fig)

    # failure autopsy: worst prostate slices of reference model
    ref = "unet3d" if "unet3d" in res else "unet2d"
    cands = []  # (dice, volume index, slice index)
    for v, sd in enumerate(res[ref]["test"]["slice_prostate_dice"]):
        pos = base_to_apex_positions(ty[v])
        for k, d in enumerate(sd):
            # candidate slices: those containing ground truth prostate that was missed, or a pure false negative
            if not np.isnan(d) and (k in pos or (preds[ref][v][..., k] == PROSTATE).any()):
                cands.append((d, v, k))
    cands.sort() # tuples sorted by Dice so worst comes first
    for i, (d, v, k) in enumerate(cands[: args.n_fail]):
        pos = base_to_apex_positions(ty[v]).get(k) # None if slice has no GT prostate
        where = f"position {pos:.2f}" if pos is not None else "outside GT prostate (false positive)"
        cols = 2 + len(preds)
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.4))
        axes[0].imshow(tx[v][..., k].T, cmap="gray", origin="lower")
        axes[0].set_title("input"); axes[0].axis("off")
        axes[1].imshow(ty[v][..., k].T, cmap="tab10", vmin=0, vmax=9, origin="lower", interpolation="nearest")
        axes[1].set_title("ground truth (all labels)"); axes[1].axis("off")
        for j, name in enumerate(preds):
            p = preds[name][v][..., k]
            dn = res[name]["test"]["slice_prostate_dice"][v][k]
            overlay(axes[2 + j], tx[v][..., k], ty[v][..., k], p, f"{name} prostate Dice {dn:.2f}")
        fig.suptitle(f"{os.path.basename(test_files[v])}, slice {k} ({where})\n"
                     "green = correct, red = false positive, blue = false negative", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"failure_{i + 1}.png"), dpi=130)
        plt.close(fig)
        print(f"failure_{i + 1}.png: {os.path.basename(test_files[v])} slice {k}, {where}, {ref} Dice {d:.3f}")


if __name__ == "__main__":
    main()