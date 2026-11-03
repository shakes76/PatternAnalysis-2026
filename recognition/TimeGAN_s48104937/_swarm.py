"""Simulate many short TimeGAN windows and compare their mid-price paths with real ones.

Draws N_PATHS real windows of WINDOW_LENGTH steps from the validation split and
generates the same number of synthetic windows, each decoded at its real partner's
reference. Each window's mid-price path is shown relative to its own first step, so
paths are comparable regardless of price level.

Mid price is the average of the best ask and best bid at each step.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.constants import DEVICE
from common.misc import load_model
from dataset import decode
from dataset import encode
from dataset import load_books


CHECKPOINT_PATH = Path("models/saved/primary.pt")
WINDOW_LENGTH: int = 128        # steps per path (1 s each)
N_PATHS: int = 1000            # real and synthetic paths each
BURN_IN: int = 10              # steps to let the GRU warm up, then dropped
TICKER: str = "AMZN"           # ticker to compare (must be in the checkpoint's tickers)
SPLIT: str = "train"
SEED: int = 66
OUT_PATH = Path(f"figures/mid_paths_{TICKER}_{WINDOW_LENGTH}s.png")


def mid_paths(book: dict) -> np.ndarray:
    """Mid-price change from each window's first step, in cents: shape (n, T)."""
    mid = (book["ask_px"][..., 0] + book["bid_px"][..., 0]).astype(np.float64) / 2
    return mid - mid[:, :1]


def kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis: 0 for a normal distribution, positive for fat tails."""
    z = (x - x.mean()) / x.std()
    return float((z ** 4).mean() - 3)


def describe(paths: np.ndarray) -> dict[str, float]:
    """Summary of the mid-price change at the end of each window."""
    end = paths[:, -1]
    lo, hi = np.percentile(end, [5, 95])
    return {
        "Mean change (c)": end.mean(),
        "Std change (c)": end.std(),
        "5th percentile (c)": lo,
        "95th percentile (c)": hi,
        "Excess kurtosis": kurtosis(end),
        "Max |change| (c)": np.abs(paths).max(),
    }


def plot(real: np.ndarray, synth: np.ndarray, path: Path) -> None:
    """Real paths, synthetic paths, and their median / 5-95% bands over time."""
    t = np.arange(real.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, layout="constrained")

    for ax, paths, name, colour in ((axes[0], real, "Real", "black"), (axes[1], synth, "TimeGAN", "C1")):
        ax.plot(t, paths.T, color=colour, lw=0.4, alpha=0.04)   # faint lines, so density shows through
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.set_title(f"{name} ({len(paths)} paths)", fontsize=12, weight="bold")

    ax = axes[2]
    for paths, name, colour in ((real, "Real", "black"), (synth, "TimeGAN", "C1")):
        lo, med, hi = np.percentile(paths, [5, 50, 95], axis=0)
        ax.fill_between(t, lo, hi, color=colour, alpha=0.15, lw=0)
        ax.plot(t, med, color=colour, lw=1.3, label=f"{name} median, 5-95%")
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.set_title("Comparison", fontsize=12, weight="bold")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    axes[0].set_ylabel("Mid-price change (¢)")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.spines[["top", "right"]].set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    model, scalers, cfg = load_model(CHECKPOINT_PATH, device=DEVICE)
    model.eval()
    k = cfg.tickers.index(TICKER)

    # Real windows of this ticker
    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes, cfg.train_frac, cfg.val_frac, cfg.test_frac)
    real_x, refs, labels = encode(books=books[SPLIT], scalers=scalers, window_length=WINDOW_LENGTH)
    pool = np.flatnonzero(labels == k)
    idx = rng.choice(pool, size=min(N_PATHS, len(pool)), replace=False)

    # Synthetic windows, each decoded at its real partner's reference
    with torch.no_grad():
        synth_x = model.sample(n=len(idx), window_length=WINDOW_LENGTH + BURN_IN, ticker=k).cpu().numpy()
    real = decode(real_x[idx], refs[idx], labels[idx], scalers, rng)
    synth = decode(synth_x[:, BURN_IN:], refs[idx], labels[idx], scalers, rng)

    real_paths, synth_paths = mid_paths(real), mid_paths(synth)

    print(f"{TICKER}: {len(idx)} paths of {WINDOW_LENGTH} s, {SPLIT} split, seed {SEED}")
    print(f"Mid-price change over the window\n\n{'':<24}{'Real':>10}{'TimeGAN':>10}")
    r, s = describe(real_paths), describe(synth_paths)
    for name in r:
        print(f"  {name:<22}{r[name]:>10.2f}{s[name]:>10.2f}")

    plot(real_paths, synth_paths, OUT_PATH)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()