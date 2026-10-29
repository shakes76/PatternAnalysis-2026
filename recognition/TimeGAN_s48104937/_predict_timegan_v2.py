from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.constants import DEPTH
from common.constants import DEVICE
from common.constants import LEVELS
from common.metrics import depth_heatmap
from common.metrics import max_log_size
from common.misc import load_model
from dataset import decode
from dataset import encode
from dataset import load_books


CHECKPOINT = Path("models/saved/primary.pt")
WINDOW_LENGTH: int = 512            # steps to generate and compare
BURN_IN: int = 10                   # number of steps to let the GRU warm up
TICKER: str = "AMZN"                # ticker to show (must be in cfg.tickers)
SEEDS: list[int] = list(range(59, 200))  # one figure per seed; close the window to see the next

SOURCES = ("Real", "Synthetic")
LEVEL_LABELS = [f"bid {l}" for l in reversed(LEVELS)] + [f"ask {l}" for l in LEVELS]
SHADES = np.linspace(1.0, 0.35, DEPTH)   # level 1 darkest, deeper levels lighter


# --- Plotting ---
def draw_prices(ax, book: dict, i: int, title: str) -> None:
    """Every ask (red) and bid (blue) price level of window i, best level darkest."""
    for side, cmap in (("ask", plt.cm.Reds), ("bid", plt.cm.Blues)):
        px = book[f"{side}_px"][i] / 100   # cents -> dollars, shape (step, DEPTH)
        for level in range(DEPTH):
            ax.plot(px[:, level], color=cmap(SHADES[level]), lw=1,
                    label=f"{side} 1" if level == 0 else None)
    ax.set(title=title, xlabel="step (1s)", ylabel="price ($)")
    ax.legend(loc="upper left", fontsize=8)

def draw_depth(ax, book: dict, i: int, vmax: float, title: str):
    """Depth heatmap of window i: bids below the white line, asks above."""
    im = ax.imshow(depth_heatmap(book, i), aspect="auto", origin="lower", cmap="viridis",
                   vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(2 * DEPTH), LEVEL_LABELS, fontsize=7)
    ax.axhline(DEPTH - 0.5, color="white", lw=1)   # the spread
    ax.set(title=title, xlabel="step (1s)")
    return im

def show_sample_pair(real: dict, synth: dict, i: int, title: str) -> None:
    """Real vs synthetic window i, prices on top and depth below, shown in a window (blocks until closed)."""
    vmax = max_log_size(real, synth)   # same colour scale for both heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    im = None
    for col, (book, name) in enumerate(zip((real, synth), SOURCES)):
        draw_prices(axes[0, col], book, i, f"{name}: price")
        im = draw_depth(axes[1, col], book, i, vmax, f"{name}: volume")
    fig.colorbar(im, ax=axes[1, :], label="log(1 + size)")
    fig.suptitle(title)
    plt.show()
    plt.close(fig)


# --- Main ---
def main() -> None:
    # Model and real windows are loaded once and reused for every seed
    model, scalers, cfg = load_model(CHECKPOINT, device=DEVICE)
    model.eval()
    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes, cfg.train_frac, cfg.val_frac, cfg.test_frac)
    real_x, refs, labels = encode(books=books["val"], scalers=scalers, window_length=WINDOW_LENGTH)
    k = cfg.tickers.index(TICKER)
    pool = np.flatnonzero(labels == k)
    print(f"{len(pool)} {TICKER} val windows of {WINDOW_LENGTH} steps (trained on {cfg.window_length})")

    for seed in SEEDS:
        print(f"seed {seed}")
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # One random real window and one synthetic window of the same ticker
        idx = rng.choice(pool, size=1)
        with torch.no_grad():
            synth_x = model.sample(n=1, window_length=WINDOW_LENGTH + BURN_IN, ticker=k).cpu().numpy()[:, BURN_IN:]

        real = decode(real_x[idx], refs[idx], labels[idx], scalers, rng)
        synth = decode(synth_x, refs[idx], labels[idx], scalers, rng)
        show_sample_pair(real, synth, 0, f"{TICKER}  seed {seed}  ({WINDOW_LENGTH} steps, burn-in {BURN_IN})")


if __name__ == "__main__":
    main()


# 9
# 16

# 29

# 31 (calm)

# 40 clean

# 60 eh
# 81 eh
# 83 mechanics
# 84 similar
# 95 still