"""
plot_curves.py - Training curves from the CSV logs -> images/training_curves.png

Usage (from the project folder):
    python eda/plot_curves.py
Missing logs are skipped.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else None


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for name, path in [("VQ-VAE2", "runs/vqvae2/log.csv"), ("baseline", "runs/baseline/log.csv")]:
        log = read(path)
        if log:
            line, = ax1.plot(log["epoch"], log["train_recon"], marker=".", label=f"{name} train")
            ax1.plot(log["epoch"], log["val_recon"], ls="--", marker=".", color=line.get_color(),
                     label=f"{name} val")
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("reconstruction MSE (log scale)")
    ax1.set_title("Stage 1: reconstruction")
    ax1.legend()

    baselines = {"top": 3.97, "bottom": 3.89}  # code-frequency entropies of the merged codes
    for level in ["top", "bottom"]:
        log = read(f"runs/prior_merged/{level}_log.csv")
        if log:
            line, = ax2.plot(log["epoch"], log["train_ce"], marker="o", label=f"{level} train")
            ax2.plot(log["epoch"], log["val_ce"], ls="--", marker="o", color=line.get_color(),
                     label=f"{level} val")
            ax2.axhline(baselines[level], color=line.get_color(), ls=":", lw=1,
                        label=f"{level} frequency baseline")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("cross-entropy (nats)")
    ax2.set_title("Stage 2: PixelCNN priors (merged codes, after fix)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/training_curves.png", dpi=120)
    print("Saved images/training_curves.png")


if __name__ == "__main__":
    main()