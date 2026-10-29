"""Per-slice prostate Dice along the slice axis for all three models.

Usage: python slice_profile.py [CASE ...]   (default: W029_Week5 G021_Week5 G021_Week7)
Prints a table and saves images/slice_profile.png.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import PROSTATE, load_case
from predict import dice, load_model, predict_volume, prostate_thirds

# Cases come from the command line; the defaults are the patients used in the failure analysis.
cases = sys.argv[1:] or ["W029_Week5", "G021_Week5", "G021_Week7"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names = ("unet2d", "improved2d", "improved3d")
models = {m: load_model(m, dev) for m in names}

fig, axes = plt.subplots(1, len(cases), figsize=(5.5 * len(cases), 4), squeeze=False)
for ax, case in zip(axes[0], cases):
    img, lab = load_case(case)
    gt = lab == PROSTATE
    thirds, zs = prostate_thirds(lab)
    # which end of the slice axis is the base (used to label the plot)
    base_high = thirds["base"].mean() > zs.mean()
    print(f"\n{case}: prostate on slices {zs.min()}-{zs.max()} | base end is the "
          f"{'HIGH' if base_high else 'LOW'}-index end")
    print("slice true_px | " + " | ".join(f"{m:>16s}" for m in names) + "   (Dice, predicted px)")
    # predict the whole volume once per model, then score every prostate slice
    preds = {m: predict_volume(models[m], m, img, dev) == PROSTATE for m in names}
    for z in zs:
        cells = [f"{dice(preds[m][z], gt[z]):.2f} ({int(preds[m][z].sum()):4d})" for m in names]
        print(f"{z:5d} {int(gt[z].sum()):7d} | " + " | ".join(f"{c:>16s}" for c in cells))
    # one Dice-per-slice line per model
    for m in names:
        ax.plot(zs, [dice(preds[m][z], gt[z]) for z in zs], marker="o", ms=3, label=m)
    ax.set_title(case)
    ax.set_xlabel("slice index" + (" (base = high end)" if base_high else " (base = low end)"))
    ax.set_ylabel("prostate Dice per slice")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
plt.tight_layout()
plt.savefig("images/slice_profile.png", dpi=100)
print("\nsaved images/slice_profile.png")
