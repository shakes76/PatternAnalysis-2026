"""
Make the README figures from finished training runs and predict.py output.

Example (after training all runs and running predict.py on all 12 into runs/predict_all12):
    python plot_figures.py --runs-dir runs --predict-dir runs/predict_all12

Saves to --out-dir (default images/):
    architecture.png   U-Net vs CAN layout and what dilation samples
    curves.png         validation loss and prostate Dice, mean +/- std over seeds
    slice_thirds.png   one test scan's prostate slices grouped into thirds,
                       with the U-Net and CAN predictions
    fp_distance.png    false-positive prostate slices by how far they lie
                       beyond the true end of the gland
    example.png        input, ground truth and full predictions on one central
                       slice (only with --checkpoints)
    failure_crops.png  zoomed failure cases (only with --checkpoints; needs the
                       data and trained weights), e.g.
                       python plot_figures.py --checkpoints runs/unet/best.pt runs/can/best.pt
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch

from dataset import PROSTATE

HERE = os.path.dirname(os.path.abspath(__file__))

COL = {"U-Net": "#1f77b4", "CAN": "#d62728", "CAN-128": "#ff7f0e", "ResidualCAN": "#2ca02c"}
GROUPS = {
    "U-Net": ["unet", "unet_s1", "unet_s2"],
    "CAN": ["can", "can_s1", "can_s2"],
    "CAN-128": ["can_c128", "can_c128_s1", "can_c128_s2"],
    "ResidualCAN": ["can_res", "can_res_s1", "can_res_s2"],
}

# ---------- Architecture ----------
def box(ax, x, y, w, h, text, color, fs=8):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.02",
                                fc=color, ec="black", lw=0.8))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs)


def architecture(out_dir):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2))
    for ax in (a1, a2):
        ax.set_xlim(0, 11.3)
        ax.axis("off")

    # U-Net: box height shows spatial resolution, label shows channels
    a1.set_ylim(-2.4, 3.4)
    a1.set_title("Baseline U-Net (7.76M parameters): shrink, then grow back", fontsize=11, loc="left")
    res = ["256×128", "128×64", "64×32", "32×16", "16×8"]
    ch = [32, 64, 128, 256, 512]
    heights = [3.2, 2.2, 1.5, 1.0, 0.7]
    xs_enc = [0.6, 1.8, 3.0, 4.2, 5.4]
    for i in range(5):
        box(a1, xs_enc[i], 0, 0.8, heights[i], f"{ch[i]}\n{res[i]}", "#cfe2f3", 7)
    xs_dec = [6.6, 7.8, 9.0, 10.2]
    for j, i in enumerate([3, 2, 1, 0]):
        box(a1, xs_dec[j], 0, 0.8, heights[i], f"{ch[i]}\n{res[i]}", "#d9ead3", 7)
        # skip connection
        a1.annotate("", xy=(xs_dec[j] + 0.4, heights[i] / 2 + 0.05), xytext=(xs_enc[i] + 0.4, heights[i] / 2 + 0.05),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=0.8,
                                    connectionstyle=f"arc3,rad=-{0.12 + 0.03 * i}"))
    a1.text(3.0, -2.15, "encoder: conv + max-pool (halve size)", ha="center", fontsize=8)
    a1.text(8.4, -2.15, "decoder: up-conv + skip connection (grey)", ha="center", fontsize=8)

    # CAN: all boxes the same height (full resolution)
    a2.set_ylim(-2.4, 2.4)
    a2.set_title("CAN (0.33M parameters): stay at full 256×128 resolution, widen the view with dilation",
                 fontsize=11, loc="left")
    labels = ["3×3", "3×3"] + [f"d={d}" for d in (1, 1, 2, 4, 8, 16, 32, 1)] + ["1×1"]
    colors = ["#cfe2f3"] * 2 + ["#f4cccc"] * 8 + ["#d9ead3"]
    x = 0.45
    for lab, c in zip(labels, colors):
        box(a2, x, 0, 0.75, 3.2, lab, c, 8)
        x += 0.95
    a2.text(1.25, -2.15, "front-end", ha="center", fontsize=8)
    a2.text(5.55, -2.15, "context module: dilated 3×3 convs, 64 channels", ha="center", fontsize=8)
    a2.text(10.22, -2.15, "head", ha="center", fontsize=8)

    # inset: what dilation means
    fig.text(0.02, 0.06, "Where a 3×3 filter samples\n(red) at each dilation:", fontsize=8, va="center")
    for k, (d, x0) in enumerate([(1, 0.3), (2, 0.5), (4, 0.7)]):
        ins = fig.add_axes([x0 - 0.05, 0.005, 0.1, 0.1])
        g = np.zeros((9, 9))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = 4 + dy * d, 4 + dx * d
                if 0 <= yy < 9 and 0 <= xx < 9:
                    g[yy, xx] = 1
        ins.imshow(g, cmap="Reds", vmin=0, vmax=1.3)
        ins.set_xticks(np.arange(-0.5, 9, 1), minor=True)
        ins.set_yticks(np.arange(-0.5, 9, 1), minor=True)
        ins.grid(which="minor", color="lightgrey", lw=0.4)
        ins.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ins.set_title(f"dilation {d}", fontsize=7, pad=2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(out_dir, "architecture.png"), dpi=110)
    plt.close(fig)


# ---------- Training curves ----------
def curves(runs_dir, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, runs in GROUPS.items():
        hs = [json.load(open(os.path.join(runs_dir, r, "history.json"))) for r in runs]
        val = np.array([[e[PROSTATE] for e in h["val_dice"]] for h in hs])
        vl = np.array([h["val_loss"] for h in hs])
        ep = np.arange(1, val.shape[1] + 1)
        for ax, arr in ((axes[0], vl), (axes[1], val)):
            m, s = arr.mean(0), arr.std(0)
            ax.plot(ep, m, color=COL[name], label=name, lw=1.6)
            ax.fill_between(ep, m - s, m + s, color=COL[name], alpha=0.15)
    axes[0].set_title("Validation loss (mean ± std, 3 seeds)")
    axes[1].set_title("Validation prostate Dice (mean ± std, 3 seeds)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_ylabel("Dice + CE loss")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0.75, 0.9)
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=110)
    plt.close(fig)


# ---------- Slice thirds for one scan ----------
def read_scan(predict_dir, model, scan):
    """Per-slice rows of one scan from predict.py's CSV, in slice order."""
    rows = [r for r in csv.DictReader(open(os.path.join(predict_dir, f"{model}_per_slice.csv"))) if r["scan"] == scan]
    return sorted(rows, key=lambda r: int(r["slice"]))


def slice_thirds(predict_dir, scan, out_dir):
    unet = read_scan(predict_dir, "unet", scan)
    can = read_scan(predict_dir, "can", scan)
    gt = unet
    s = np.array([int(r["slice"]) for r in gt])
    g = np.array([int(r["gt_pixels"]) for r in gt])
    has = np.where(g > 0)[0]
    lo, hi = has[0] - 4, has[-1] + 5
    reg_col = {"anterior": "#f6b26b", "central": "#93c47d", "posterior": "#8e7cc3", "none": "#dddddd"}
    fig, ax = plt.subplots(figsize=(11, 4))
    for i in range(lo, hi):
        ax.bar(s[i], g[i], color=reg_col[gt[i]["region"]], width=0.8)
    ax.plot(s[lo:hi], [int(r["pred_pixels"]) for r in unet[lo:hi]], "o-", color=COL["U-Net"], ms=4, lw=1, label="U-Net prediction")
    ax.plot(s[lo:hi], [int(r["pred_pixels"]) for r in can[lo:hi]], "s--", color=COL["CAN"], ms=4, lw=1, label="CAN prediction")
    handles = [Patch(color=reg_col[k], label=f"true prostate: {k} third") for k in ("anterior", "central", "posterior")]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], fontsize=8, loc="upper left")
    ax.set_xlabel("slice index in the scan (front → back of the body)")
    ax.set_ylabel("prostate pixels in the slice")
    ax.set_title(f"How slices are grouped into thirds ({scan}, seed 42 models)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "slice_thirds.png"), dpi=110)
    plt.close(fig)


# ---------- False positives beyond the gland ----------
def fp_distances(predict_dir, model):
    """
    For each slice with no true prostate but some predicted prostate, the
    number of slices between it and the nearest end of that scan's gland
    (0 would mean a gap inside the gland).
    """
    rows = list(csv.DictReader(open(os.path.join(predict_dir, f"{model}_per_slice.csv"))))
    gland = {}
    for r in rows:
        if int(r["gt_pixels"]) > 0:
            lo, hi = gland.get(r["scan"], (10**9, -1))
            gland[r["scan"]] = (min(lo, int(r["slice"])), max(hi, int(r["slice"])))
    out = []
    for r in rows:
        if int(r["gt_pixels"]) == 0 and int(r["pred_pixels"]) > 0:
            k = int(r["slice"])
            lo, hi = gland[r["scan"]]
            out.append(lo - k if k < lo else k - hi if k > hi else 0)
    return out


def fp_distance(predict_dir, out_dir, max_bin=7):
    """Bar chart of false-positive slices by distance, summed over seeds."""
    bins = np.arange(0, max_bin + 1)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    width = 0.2
    for k, (name, runs) in enumerate(GROUPS.items()):
        d = np.concatenate([fp_distances(predict_dir, r) for r in runs])
        counts = [(np.minimum(d, max_bin) == b).sum() for b in bins]
        ax.bar(bins + (k - 1.5) * width, counts, width, color=COL[name], label=f"{name} ({len(d)} slices)")
    ax.set_xticks(bins, [str(b) for b in bins[:-1]] + [f"{max_bin}+"])
    ax.set_xlabel("slices beyond the true end of the prostate (0 = gap inside the prostate)")
    ax.set_ylabel("false-positive slices\n(total over 3 seeds)")
    ax.set_title("Where false prostate appears: slices with no true prostate but some predicted")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fp_distance.png"), dpi=110)
    plt.close(fig)


# ---------- Zoomed failure cases ----------
# (scan, slice, short description) for each row of failure_crops.png
FAILURE_CASES = [
    ("case_040_week_0", 38, "posterior tip missed (65 px)"),
    ("case_040_week_3", 14, "anterior tip missed (47 px)"),
    ("case_041_week_0", 19, "3-px label at the anterior end"),
    ("case_040_week_3", 34, "no prostate: false tip on the rectum"),
]


def _load_for_figures(checkpoints, data_root):
    """Test dataset, (scan, slice) -> index map, loaded models and device."""
    from dataset import get_datasets
    from predict import load_model, slice_info
    from utils import get_device

    device = get_device(None)
    test = get_datasets(data_root, augment=False)["test"]
    index = {slice_info(path): i for i, (path, _) in enumerate(test.pairs)}
    return test, index, [load_model(c, device) for c in checkpoints], device


def example_output(checkpoints, data_root, out_dir, scan="case_040_week_0", sl=27):
    """Input, ground truth and each model's predicted labels on one slice."""
    import torch
    from dataset import CLASS_NAMES, NUM_CLASSES
    from predict import LABEL_CMAP

    test, index, models, device = _load_for_figures(checkpoints, data_root)
    image, truth = test[index[(scan, sl)]]
    with torch.no_grad():
        preds = [m(image.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy() for _, m in models]

    panels = [(image[0].numpy(), f"input ({scan} s{sl})", dict(cmap="gray")),
              (truth.numpy(), "ground truth", dict(cmap=LABEL_CMAP, vmin=0, vmax=NUM_CLASSES - 1))]
    panels += [(pred, f"{name} prediction", dict(cmap=LABEL_CMAP, vmin=0, vmax=NUM_CLASSES - 1))
               for (name, _), pred in zip(models, preds)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 2.4))
    for ax, (img, title, kw) in zip(axes, panels):
        # Arrays are (256, 128) with axis 0 left-right; transpose so superior is up.
        ax.imshow(np.swapaxes(img, 0, 1), origin="lower", **kw)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    handles = [Patch(color=LABEL_CMAP(c), label=CLASS_NAMES[c]) for c in range(1, NUM_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=NUM_CLASSES - 1, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out_dir, "example.png"), dpi=120)
    plt.close(fig)


def _label_counts(labels):
    """{'rectum': 47, ...} for an array of class indices."""
    from dataset import CLASS_NAMES
    values, counts = np.unique(labels, return_counts=True)
    return {CLASS_NAMES[v]: int(c) for v, c in zip(values, counts)}


def failure_crops(checkpoints, data_root, out_dir, half=24):
    """
    For each case in FAILURE_CASES, show a (2*half)-pixel crop of the input,
    the ground truth and each model's predicted labels, centred on the true
    prostate (or on any predicted prostate when there is none). A white
    outline marks the true prostate in every label panel, so a missed
    prostate shows what it was labelled as instead.
    """
    import torch
    from dataset import NUM_CLASSES
    from dataset import CLASS_NAMES
    from predict import LABEL_CMAP

    test, index, models, device = _load_for_figures(checkpoints, data_root)

    cols = 2 + len(models)
    fig, axes = plt.subplots(len(FAILURE_CASES), cols, figsize=(2.1 * cols, 2.3 * len(FAILURE_CASES)), squeeze=False)
    for r, (scan, sl, desc) in enumerate(FAILURE_CASES):
        image, truth = test[index[(scan, sl)]]
        with torch.no_grad():
            preds = [m(image.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy() for _, m in models]
        image, truth = image[0].numpy(), truth.numpy()

        # Printed breakdown backing the README case write-ups: what the
        # models called the true prostate, and what lies under false prostate.
        for (name, _), pred in zip(models, preds):
            missed = (truth == PROSTATE) & (pred != PROSTATE)
            false = (truth != PROSTATE) & (pred == PROSTATE)
            print(f"{scan} s{sl} {name}: missed {missed.sum()} px predicted as "
                  f"{_label_counts(pred[missed])}; false {false.sum()} px over truth {_label_counts(truth[false])}")

        target = truth == PROSTATE
        if not target.any():
            target = np.any([p == PROSTATE for p in preds], axis=0)
        cy, cx = np.argwhere(target).mean(0).round().astype(int) if target.any() else np.array(image.shape) // 2
        y0 = int(np.clip(cy - half, 0, image.shape[0] - 2 * half))
        x0 = int(np.clip(cx - half, 0, image.shape[1] - 2 * half))
        crop = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))

        # Arrays are (256, 128) with axis 0 left-right; transpose so superior is up.
        def draw(ax, img, title, outline=False, **kw):
            ax.imshow(np.swapaxes(img[crop], 0, 1), origin="lower", **kw)
            if outline and (truth == PROSTATE).any():
                ax.contour(np.swapaxes((truth == PROSTATE)[crop], 0, 1).astype(float), levels=[0.5],
                           colors="white", linewidths=1.2, origin="lower")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        labels = dict(cmap=LABEL_CMAP, vmin=0, vmax=NUM_CLASSES - 1)
        draw(axes[r, 0], image, f"{scan} s{sl}", cmap="gray")
        axes[r, 0].set_ylabel(desc, fontsize=8)
        draw(axes[r, 1], truth, "ground truth", outline=True, **labels)
        for k, ((name, _), pred) in enumerate(zip(models, preds)):
            draw(axes[r, 2 + k], pred, f"{name} prediction", outline=True, **labels)
    handles = [Patch(color=LABEL_CMAP(c), label=CLASS_NAMES[c]) for c in range(1, NUM_CLASSES)]
    handles.append(Patch(facecolor="none", edgecolor="black", label="true prostate (white outline)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle("Zoomed failure cases: each model's labels, with the true prostate outlined in white", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(out_dir, "failure_crops.png"), dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    p.add_argument("--predict-dir", default=os.path.join(HERE, "runs", "predict_all12"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "images"))
    p.add_argument("--scan", default="case_040_week_0", help="test scan for slice_thirds.png")
    p.add_argument("--checkpoints", nargs="+", default=None, help="models to show in failure_crops.png")
    p.add_argument("--data-root", default=os.path.join(HERE, "data", "keras_slices_data"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    architecture(args.out_dir)
    curves(args.runs_dir, args.out_dir)
    slice_thirds(args.predict_dir, args.scan, args.out_dir)
    fp_distance(args.predict_dir, args.out_dir)
    if args.checkpoints:
        example_output(args.checkpoints, args.data_root, args.out_dir)
        failure_crops(args.checkpoints, args.data_root, args.out_dir)
    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
