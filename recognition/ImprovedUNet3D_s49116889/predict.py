"""
predict.py - Load the trained models, run them on the held-out test patients
and produce the figures and tables used in README.md.

Usage (after both train.py runs have finished):
    python predict.py                    # both models (runs/3d, runs/2d)
    python predict.py --models 3d        # only the 3D Improved U-Net

Outputs in figures/:
    comparison_<case>.png    axial + coronal: MRI, ground truth, 2D, 3D
    failure_<n>_<case>.png   worst prostate slices of the 3D model with
                             false-positive / false-negative overlays
    slice_profile.png        prostate Dice vs position along the prostate
                             (apex -> base) for both models
    results_table.md         per-class Dice/IoU and resource table (markdown)

NumPy is used here only for plotting, which the brief allows in predict.py.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from dataset import (CLASS_NAMES, DEFAULT_CACHE, DEFAULT_ROOT, PROSTATE,
                     load_data, patient_split)
from modules import ImprovedUNet3D, UNet2D
from utils import evaluate

# background transparent-black, body grey, bone white, bladder yellow,
# rectum brown, prostate red
LABEL_CMAP = ListedColormap(["#000000", "#5a5a5a", "#e8e8e8", "#f2c94c",
                             "#a0522d", "#e63946"])
NAMES = {"3d": "3D Improved U-Net", "2d": "2D U-Net (baseline)"}


def load_model(kind: str, run_dir: str, device):
    model = ImprovedUNet3D() if kind == "3d" else UNet2D()
    path = os.path.join(run_dir, kind, "best.pt")
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def show_labels(ax, img, lab, title):
    ax.imshow(img, cmap="gray")
    ax.imshow(np.ma.masked_where(lab == 0, lab), cmap=LABEL_CMAP,
              vmin=0, vmax=5, alpha=0.55, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def show_errors(ax, img, gt, pred, title, label=PROSTATE):
    """Prostate overlay: green = correct, red = false positive, blue = false negative."""
    g, p = gt == label, pred == label
    rgb = np.zeros(img.shape + (4,))
    rgb[g & p] = (0.2, 0.8, 0.2, 0.7)
    rgb[~g & p] = (1.0, 0.1, 0.1, 0.8)
    rgb[g & ~p] = (0.1, 0.4, 1.0, 0.8)
    ax.imshow(img, cmap="gray")
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def slice_dice(gt2d, pred2d, label=PROSTATE):
    g, p = gt2d == label, pred2d == label
    denom = g.sum() + p.sum()
    return 2 * (g & p).sum() / denom if denom else np.nan


def comparison_figure(key, img, gt, preds, out):
    """Axial slice through the prostate centre + coronal view through it (shows continuity across slices; superior = up)."""
    zs = np.nonzero((gt == PROSTATE).any(axis=(1, 2)))[0]
    z = int(zs[len(zs) // 2]) if len(zs) else gt.shape[0] // 2
    ys, xs = np.nonzero(gt[z] == PROSTATE)
    x = int(xs.mean()) if len(xs) else gt.shape[2] // 2
    cols = 2 + len(preds)
    fig, ax = plt.subplots(2, cols, figsize=(3 * cols, 6.2))
    for r, (cut, label) in enumerate([(lambda v: v[z], f"axial z={z}"),
                                      (lambda v: v[:, :, x][::-1], f"coronal, col {x}")]):
        ax[r, 0].imshow(cut(img), cmap="gray")
        ax[r, 0].set_title(f"MRI ({label})", fontsize=9)
        ax[r, 0].axis("off")
        show_labels(ax[r, 1], cut(img), cut(gt), "ground truth")
        for c, (kind, pred) in enumerate(preds.items(), start=2):
            show_labels(ax[r, c], cut(img), cut(pred), NAMES[kind])
    fig.suptitle(f"Test case {key}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def failure_figures(results, data, idx_of, kind, n, out_dir):
    """
    Select n representative failure slices of `kind` and plot
    MRI / ground truth / `kind` prediction / other model / error overlay.

    Candidates are axial slices with prostate in the ground truth or the
    prediction (>= 10 voxels, to ignore single-voxel noise), ranked by
    per-slice prostate Dice. To be representative rather than n copies of one
    problem, at most one slice is taken per PATIENT, and the set mixes two
    failure types:
      * under-/mis-segmentation INSIDE the true prostate extent (FN-heavy)
      * over-extension OUTSIDE the true prostate extent (FP-only), i.e. the
        model continuing the gland past the true base or apex.
    """
    from dataset import patient_id
    other = next((k for k in results if k != kind), None)
    inside, outside = [], []
    for key, pred in results[kind]["predictions"].items():
        gt = data["labels"][idx_of[key]].numpy()
        pred = pred.numpy()
        gt_z = np.nonzero((gt == PROSTATE).any(axis=(1, 2)))[0]
        zs = np.nonzero(((gt == PROSTATE) | (pred == PROSTATE)).any(axis=(1, 2)))[0]
        for z in zs:
            area = (gt[z] == PROSTATE).sum() + (pred[z] == PROSTATE).sum()
            if area < 10:
                continue
            d = slice_dice(gt[z], pred[z])
            is_in = len(gt_z) > 0 and gt_z.min() <= z <= gt_z.max()
            (inside if is_in else outside).append((d, key, int(z)))
    inside.sort(key=lambda t: t[0])
    outside.sort(key=lambda t: t[0])

    def pick(pool, k, used):
        out = []
        for d, key, z in pool:
            if patient_id(key) not in used:
                out.append((d, key, z))
                used.add(patient_id(key))
            if len(out) == k:
                break
        return out

    used = set()
    n_in = (n + 1) // 2                       # e.g. 3 inside + 2 outside
    chosen = pick(inside, n_in, used) + pick(outside, n - n_in, used)
    if len(chosen) < n:                       # top up if a pool ran short
        chosen += pick(inside + outside, n - len(chosen), used)
    chosen.sort(key=lambda t: t[0])

    lines = []
    for i, (d, key, z) in enumerate(chosen, start=1):
        img = data["images"][idx_of[key]].float().numpy()
        gt = data["labels"][idx_of[key]].numpy()
        pred = results[kind]["predictions"][key].numpy()
        zs = np.nonzero((gt == PROSTATE).any(axis=(1, 2)))[0]
        if len(zs) == 0 or z < zs.min() or z > zs.max():
            where = (f"{zs.min() - z} slice(s) below GT prostate" if len(zs) and z < zs.min()
                     else f"{z - zs.max()} slice(s) above GT prostate" if len(zs)
                     else "no GT prostate in volume")
        else:
            where = f"{(z - zs.min()) / max(1, zs.max() - zs.min()):.0%} along GT extent"
        fp = int(((pred[z] == PROSTATE) & (gt[z] != PROSTATE)).sum())
        fn = int(((gt[z] == PROSTATE) & (pred[z] != PROSTATE)).sum())
        # what the prostate was confused with (for the README diagnosis)
        fn_as = np.bincount(pred[z][(gt[z] == PROSTATE) & (pred[z] != PROSTATE)],
                            minlength=len(CLASS_NAMES))
        fp_from = np.bincount(gt[z][(pred[z] == PROSTATE) & (gt[z] != PROSTATE)],
                              minlength=len(CLASS_NAMES))
        conf = []
        if fn:
            conf.append("missed prostate labelled as " + ", ".join(
                f"{CLASS_NAMES[c]} {fn_as[c]}" for c in np.argsort(-fn_as) if fn_as[c]))
        if fp:
            conf.append("false prostate taken from " + ", ".join(
                f"{CLASS_NAMES[c]} {fp_from[c]}" for c in np.argsort(-fp_from) if fp_from[c]))
        other_txt = ""
        cols = 4 + (other is not None)
        fig, ax = plt.subplots(1, cols, figsize=(3 * cols, 3.4))
        ax[0].imshow(img[z], cmap="gray"); ax[0].set_title("MRI", fontsize=9); ax[0].axis("off")
        show_labels(ax[1], img[z], gt[z], "ground truth")
        show_labels(ax[2], img[z], pred[z], f"{NAMES[kind]} (Dice {d:.2f})")
        if other is not None:
            op = results[other]["predictions"][key].numpy()
            od = slice_dice(gt[z], op[z])
            other_txt = f"; {NAMES[other]} slice Dice {od:.2f}"
            show_labels(ax[3], img[z], op[z], f"{NAMES[other]} (Dice {od:.2f})")
        show_errors(ax[-1], img[z], gt[z], pred[z],
                    f"{NAMES[kind]}: FP {fp} red / FN {fn} blue")
        fig.suptitle(f"Failure {i}: {key}, axial slice {z} ({where})", fontsize=10)
        fig.tight_layout()
        path = os.path.join(out_dir, f"failure_{i}_{key}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        lines.append(f"{i}. {key} slice {z}: Dice {d:.2f}, FP {fp}, FN {fn}, {where}"
                     f"{other_txt}; " + "; ".join(conf))
    return lines


def slice_profile(results, data, idx_of, out):
    """Prostate Dice as a function of relative position along the prostate (0 = first GT slice, 1 = last)."""
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for kind, res in results.items():
        per_bin = [[] for _ in range(len(bins) - 1)]
        for key, pred in res["predictions"].items():
            gt = data["labels"][idx_of[key]].numpy()
            pred = pred.numpy()
            zs = np.nonzero((gt == PROSTATE).any(axis=(1, 2)))[0]
            if len(zs) < 2:
                continue
            for z in range(zs.min(), zs.max() + 1):
                rel = (z - zs.min()) / (zs.max() - zs.min())
                b = min(int(rel * (len(bins) - 1)), len(bins) - 2)
                per_bin[b].append(slice_dice(gt[z], pred[z]))
        means = [np.nanmean(v) if v else np.nan for v in per_bin]
        ax.plot((bins[:-1] + bins[1:]) / 2, means, marker="o", label=NAMES[kind])
    ax.axvspan(0, 0.25, color="grey", alpha=0.12)
    ax.axvspan(0.75, 1, color="grey", alpha=0.12)
    ax.set(xlabel="relative slice position along prostate (shaded = extremity slices)",
           ylabel="mean per-slice prostate Dice", ylim=(0, 1),
           title="Where along the prostate do the models fail?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def results_table(results, run_dir):
    """Markdown tables for the README, combining test metrics and train-time resources."""
    kinds = list(results)
    meta = {}
    for k in kinds:
        path = os.path.join(run_dir, k, "test_metrics.json")
        meta[k] = json.load(open(path)) if os.path.exists(path) else {}
    head = "| Class | " + " | ".join(f"{NAMES[k]} Dice | {NAMES[k]} IoU" for k in kinds) + " |"
    sep = "|---" * (1 + 2 * len(kinds)) + "|"
    rows = [head, sep]
    for c, name in enumerate(CLASS_NAMES):
        cells = [f"{results[k]['dice_mean'][c]:.3f} | {results[k]['iou_mean'][c]:.3f}" for k in kinds]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    fg = [f"**{torch.nanmean(results[k]['dice_mean'][1:]):.3f}** | " for k in kinds]
    rows.append("| **foreground mean** | " + " | ".join(fg) + " |")
    rows.append("")
    rows.append("| Prostate region | " + " | ".join(NAMES[k] for k in kinds) + " |")
    rows.append("|---" * (1 + len(kinds)) + "|")
    rows.append("| extremity slices (base + apex) | " +
                " | ".join(f"{results[k]['prostate_extremity_dice']:.3f}" for k in kinds) + " |")
    rows.append("| central slices | " +
                " | ".join(f"{results[k]['prostate_central_dice']:.3f}" for k in kinds) + " |")
    rows.append("")
    rows.append("| Resource | " + " | ".join(NAMES[k] for k in kinds) + " |")
    rows.append("|---" * (1 + len(kinds)) + "|")
    for label, field, fmt in [("Parameters", "parameters", "{:,}"),
                              ("Peak training VRAM (GB)", "peak_train_vram_gb", "{:.2f}"),
                              ("Peak inference VRAM (GB)", "peak_inference_vram_gb", "{:.2f}"),
                              ("Inference time per volume (ms)", "seconds_per_volume", None),
                              ("Training time (min)", "train_minutes", "{:.0f}"),
                              ("Best epoch", "best_epoch", "{}")]:
        cells = []
        for k in kinds:
            v = meta[k].get(field)
            if v is None:
                cells.append("n/a")
            elif fmt is None:
                cells.append(f"{v * 1000:.0f}")
            else:
                cells.append(fmt.format(v))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description="Test-set inference and README figures")
    p.add_argument("--models", nargs="+", default=["3d", "2d"], choices=["3d", "2d"])
    p.add_argument("--runs", default="runs")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--out", default="figures")
    p.add_argument("--n-failures", type=int, default=5)
    p.add_argument("--n-comparisons", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args.root, args.cache)
    split = patient_split(data["keys"], seed=args.seed)       # same split as training
    idx_of = {data["keys"][i]: i for i in split["test"]}
    print(f"Test set: {len(split['test_patients'])} patients, {len(split['test'])} volumes")

    results = {}
    for kind in args.models:
        model = load_model(kind, args.runs, device)
        results[kind] = evaluate(model, data, split["test"], kind, device,
                                 keep_predictions=True)
        d = results[kind]["dice_mean"]
        print(f"\n{NAMES[kind]}")
        for c, name in enumerate(CLASS_NAMES):
            print(f"  {name:10s} Dice {d[c]:.4f}")
        print(f"  prostate extremity {results[kind]['prostate_extremity_dice']:.4f} | "
              f"central {results[kind]['prostate_central_dice']:.4f}")

    # Comparison figures: best, median and worst 3D prostate case
    main_kind = "3d" if "3d" in results else args.models[0]
    order = sorted(range(len(split["test"])),
                   key=lambda i: np.nan_to_num(results[main_kind]["dice_per_case"][i, PROSTATE].item(), nan=-1))
    picks = sorted({order[0], order[len(order) // 2], order[-1]})[:args.n_comparisons]
    for i in picks:
        key = results[main_kind]["keys"][i]
        img = data["images"][idx_of[key]].float().numpy()
        gt = data["labels"][idx_of[key]].numpy()
        preds = {k: results[k]["predictions"][key].numpy() for k in results}
        comparison_figure(key, img, gt, preds, os.path.join(args.out, f"comparison_{key}.png"))

    for f in os.listdir(args.out):            # clear figures from earlier runs
        if f.startswith("failure_"):
            os.remove(os.path.join(args.out, f))
    print(f"\nRepresentative failure slices ({NAMES[main_kind]}):")
    for line in failure_figures(results, data, idx_of, main_kind, args.n_failures, args.out):
        print("  " + line)

    slice_profile(results, data, idx_of, os.path.join(args.out, "slice_profile.png"))
    table = results_table(results, args.runs)
    with open(os.path.join(args.out, "results_table.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nFigures written to {args.out}/")


if __name__ == "__main__":
    main()
