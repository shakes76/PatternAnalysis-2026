import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AE = re.compile(r"epoch:\s*(\d+):\s*train MSE\s*([\d.]+),\s*validate SSIM\s*([\d.]+)")
PRIOR = re.compile(r"epoch\s+(\d+):\s*train CE\s*([\d.]+),\s*val CE\s*([\d.]+)")


def parse(path, pattern):
    """Return a list of runs; each run is a list of (epoch, a, b). A new run starts when epoch resets."""
    runs, last = [], 0
    for line in open(path, errors="ignore"):
        m = pattern.search(line)
        if not m:
            continue
        ep, a, b = int(m.group(1)), float(m.group(2)), float(m.group(3))
        if not runs or ep <= last:
            runs.append([])
        runs[-1].append((ep, a, b))
        last = ep
    return runs


def plot_ae(items):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for label, runs in items:
        for k, run in enumerate(runs):
            name = label if len(runs) == 1 else f"{label} (run {k + 1})"
            ep = [r[0] for r in run]
            ax1.plot(ep, [r[1] for r in run], label=name)
            ax2.plot(ep, [r[2] for r in run], label=name)
    ax1.set_yscale("log")
    ax1.set(title="Training loss (MSE + extra loss)", xlabel="epoch", ylabel="train loss (log scale)")
    ax2.set(title="Validation SSIM (paired reconstruction)", xlabel="epoch", ylabel="SSIM")
    ax2.axhline(0.6, color="grey", ls="--", lw=1, label="0.6 target")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig("ae_curves.png", dpi=150)
    plt.close()
    print("saved ae_curves.png")


def plot_prior(items):
    fig, axes = plt.subplots(1, len(items), figsize=(5.5 * len(items), 4), squeeze=False)
    for ax, (label, runs) in zip(axes[0], items):
        for k, run in enumerate(runs):
            ep = [r[0] for r in run]
            val = [r[2] for r in run]
            tag = "" if len(runs) == 1 else f" run {k + 1}"
            ax.plot(ep, [r[1] for r in run], label=f"train{tag}")
            ax.plot(ep, val, label=f"validation{tag}", ls="--")
            best = min(range(len(val)), key=val.__getitem__)
            ax.plot(ep[best], val[best], "k*", ms=10)  # star marks the saved (lowest-validation) epoch
        ax.set(title=f"{label}: cross-entropy per code", xlabel="epoch", ylabel="CE (nats)")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig("prior_curves.png", dpi=150)
    plt.close()
    print("saved prior_curves.png")


if __name__ == "__main__":
    ae_items, prior_items = [], []
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        ae_runs, prior_runs = parse(path, AE), parse(path, PRIOR)
        if ae_runs:
            ae_items.append((label, ae_runs))
        if prior_runs:
            prior_items.append((label, prior_runs))
        if not ae_runs and not prior_runs:
            print(f"warning: no epoch lines found in {path}")
    if ae_items:
        plot_ae(ae_items)
    if prior_items:
        plot_prior(prior_items)