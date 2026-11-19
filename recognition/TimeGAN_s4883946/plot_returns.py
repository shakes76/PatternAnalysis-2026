"""Plot the real mid-price return distribution on the 1-second grid (feasibility evidence).

Shows why returns are hard to model: a sharp spike at zero and very fat tails,
compared with a Gaussian of the same standard deviation.

Usage: python plot_returns.py <data_dir> [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import build_features, load_lob, resample_to_grid


def main(data_dir, out="figures/return_distribution.png"):
    times, book = load_lob(data_dir)
    _, snaps = resample_to_grid(times, book, step=1.0)
    ret = build_features(snaps)[0][1:, 0]   # drop first step (no previous snapshot)

    zero = (ret == 0).mean()
    kurt = ((ret - ret.mean()) ** 4).mean() / ret.var() ** 2
    print(f"1 s returns: n={len(ret)} zero fraction={zero:.3f} std={ret.std():.2e} "
          f"kurtosis={kurt:.1f} max |return|={np.abs(ret).max():.2e} "
          f"(= {np.abs(ret).max() / ret.std():.0f} std)")

    gauss = np.random.default_rng(0).normal(ret.mean(), ret.std(), len(ret))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    lim = 5 * ret.std()
    bins = np.linspace(-lim, lim, 81)
    axes[0].hist(ret, bins, density=True, histtype="step", color="black", label="real")
    axes[0].hist(gauss, bins, density=True, histtype="step", color="tab:red",
                 label="Gaussian, same std")
    axes[0].set_yscale("log"); axes[0].set_xlabel("1 s log mid-return")
    axes[0].set_title(f"Spike at zero ({100 * zero:.0f}% exact zeros)")
    axes[0].legend(fontsize=8)

    # Tail comparison: P(|r| > x) on log-log axes
    # Probabilities use *all* returns (including the zeros); only x > 0 is drawn.
    x = np.sort(np.abs(ret))
    p_real = 1 - np.arange(1, len(x) + 1) / len(x) + 1e-9
    axes[1].loglog(x[x > 0], p_real[x > 0], color="black", label="real")
    xg = np.sort(np.abs(gauss))
    axes[1].loglog(xg, 1 - np.arange(1, len(xg) + 1) / len(xg) + 1e-9, color="tab:red",
                   label="Gaussian, same std")
    axes[1].set_xlim(1e-6, 3e-3); axes[1].set_ylim(1e-5, 2)  # ignore float-noise returns
    axes[1].set_xlabel("|return|"); axes[1].set_ylabel("P(|r| > x)")
    axes[1].set_title(f"Fat tails (kurtosis {kurt:.0f}; Gaussian = 3)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:])
