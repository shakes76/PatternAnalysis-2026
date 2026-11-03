"""Reconstruction quality of the phase 1 autoencoder checkpoint (embedder -> recovery).

Prints reconstruction metrics over EVAL_WINDOWS random windows, and plots one real
window against its reconstruction for each feature group: mid, spread, gaps and
sizes. Gaps and sizes have one column per book level, so they are shown as the mean
over their levels. All plotted values are in the scaled model space.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.constants import DEVICE
from common.misc import load_model
from dataset import FEATURE_COLS
from dataset import decode
from dataset import encode
from dataset import load_books


CHECKPOINT_PATH = Path("models/checkpoint_autoencoder.pt")
TICKER: str = "AMZN"            # ticker to evaluate (must be in the checkpoint's cfg.tickers)
SPLIT: str = "val"              # split to take the real windows from
SEED: int = 1
EVAL_WINDOWS: int = 1000        # random windows for the printed metrics
OUT_PATH = Path("figures/autoencoder_reconstruction.png")

# Feature groups in the scaled model space (the change flags follow the features)
N_FEAT = len(FEATURE_COLS)
GROUPS = {
    "mid": [FEATURE_COLS.index("mid_rel")],
    "spread": [FEATURE_COLS.index("spread")],
    "gaps": [i for i, c in enumerate(FEATURE_COLS) if "_gap_" in c],
    "sizes": [i for i, c in enumerate(FEATURE_COLS) if "_sz_" in c],
}
# Plot rows: group -> y label
SHOW = {"mid": "Mid (relative)", "spread": "Spread", "gaps": "Gaps (mean)", "sizes": "Sizes (mean)"}


def reconstruct(model, x: np.ndarray, labels: np.ndarray, batch: int = 256) -> np.ndarray:
    """Real windows through the autoencoder only: embedder -> latent -> recovery."""
    with torch.no_grad():
        xt = torch.from_numpy(x).float().to(DEVICE)
        c = model.one_hot(torch.from_numpy(labels).long().to(DEVICE))
        out = [model.recovery(model.embedder(xb, cb), cb) for xb, cb in zip(xt.split(batch), c.split(batch))]
    return torch.cat(out).cpu().numpy()


def metrics(x: np.ndarray, recon: np.ndarray, refs: np.ndarray, labels: np.ndarray, scalers,
            rng: np.random.Generator) -> dict[str, float]:
    """Reconstruction quality of a set of windows."""
    out = {"MSE, all inputs (scaled)": ((recon - x) ** 2).mean()}

    # R^2 per feature, averaged within each group: 1 = perfect, 0 = no better than the mean
    for name, cols in GROUPS.items():
        mse = ((recon[..., cols] - x[..., cols]) ** 2).mean(axis=(0, 1))
        var = x[..., cols].var(axis=(0, 1)) + 1e-8
        out[f"R² {name}"] = (1 - mse / var).mean()

    # Change flags: are change / no-change steps recovered?
    real_f, recon_f = x[..., N_FEAT:] > 0.5, recon[..., N_FEAT:] > 0.5
    out["Flag accuracy"] = (real_f == recon_f).mean()
    out["Flag recall (changed steps)"] = recon_f[real_f].mean()

    # Decoded books in real units
    real_b = decode(x, refs, labels, scalers, rng)
    recon_b = decode(recon, refs, labels, scalers, rng)
    out["|Error| best ask price (ticks)"] = np.abs(recon_b["ask_px"][..., 0] - real_b["ask_px"][..., 0]).mean()
    out["|Error| best ask size (shares)"] = np.abs(recon_b["ask_sz"][..., 0] - real_b["ask_sz"][..., 0]).mean()
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    model, scalers, cfg = load_model(CHECKPOINT_PATH, device=DEVICE)
    model.eval()
    k = cfg.tickers.index(TICKER)

    # Every window of the split, then this ticker's
    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes,
                       cfg.train_frac, cfg.val_frac, cfg.test_frac)[SPLIT]
    x, refs, labels = encode(books=books, scalers=scalers, window_length=cfg.window_length)
    pool = np.flatnonzero(labels == k)

    # Metrics over random windows
    idx = rng.choice(pool, size=min(EVAL_WINDOWS, len(pool)), replace=False)
    recon = reconstruct(model, x[idx], labels[idx])
    scores = metrics(x[idx], recon, refs[idx], labels[idx], scalers, rng)

    print(f"{CHECKPOINT_PATH.name}: {TICKER}, {SPLIT} split, {len(idx)} windows\n")
    for name, value in scores.items():
        print(f"  {name:<32}{value:>10.4f}")

    # Plot one random window: real vs reconstruction, one row per feature group
    i = rng.choice(pool)
    real, rec = x[i], reconstruct(model, x[i:i + 1], labels[i:i + 1])[0]

    fig, axes = plt.subplots(len(SHOW), 1, figsize=(12, 1.2 * len(SHOW)), sharex=True, layout="constrained")
    for ax, (group, label) in zip(axes, SHOW.items()):
        cols = GROUPS[group]
        ax.plot(real[:, cols].mean(axis=1), color="black", lw=1.3, alpha=0.8, label="Real", zorder=1)
        ax.plot(rec[:, cols].mean(axis=1), color="C1", lw=1.3, alpha=0.9, label="Reconstruction", zorder=2)
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is not axes[-1]:   # only the bottom row keeps its time axis ticks
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()