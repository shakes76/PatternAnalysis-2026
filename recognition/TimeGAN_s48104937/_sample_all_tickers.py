"""One real and one TimeGAN window for every ticker, stacked in a single figure.

Each ticker is one row drawn exactly in the plot_book_pair_v2 style: prices above depth,
real on the left and synthetic on the right, a colour bar hugging the left heatmap and
Bid/Ask labels in the gap between the heatmaps. Both windows of a row are decoded at the
same reference, so they start from the same book. There is no title; the column titles
say which ticker and which side is real or synthetic.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.constants import DEVICE
from common.metrics import depth_heatmap
from common.misc import load_model
from dataset import decode
from dataset import encode
from dataset import load_books


CHECKPOINT = Path("models/saved/primary.pt")
WINDOW_LENGTH: int = 128       # steps to generate and compare
BURN_IN: int = 10              # steps to let the GRU warm up, then dropped
SPLIT: str = "val"
SEED: int = 21                 # default seed for any ticker not in SEEDS
SEEDS: dict[str, int] = {      # per-ticker seeds: each drives that row's real window, noise and decode
    "AMZN": 98,
    "AAPL": 987,
    "GOOG": 465,  # 387
    "INTC": 125,
    "MSFT": 199,
}
ROW_HEIGHT: float = 3.6        # inches per ticker, the height of one plot_book_pair_v2 figure
OUT_PATH = Path(f"figures/timegan_all_tickers_{WINDOW_LENGTH}s.png")


def plot_book_pairs(pairs: dict[str, tuple[dict, dict]], path: Path, idx: int = 0) -> None:
    """plot_book_pair_v2 for several (real, synthetic) pairs, one row per ticker, in one figure."""
    n = len(pairs)
    fig, axes = plt.subplots(2 * n, 2, figsize=(12, ROW_HEIGHT * n), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1] * n}, layout="constrained")
    fig.get_layout_engine().set(w_pad=1 / 72, wspace=0)

    for r, (ticker, (book_a, book_b)) in enumerate(pairs.items()):
        row = axes[2 * r:2 * r + 2]
        cols = [(book_a, idx), (book_b, idx)]
        col_titles = (f"{ticker} {WINDOW_LENGTH}s Real Sample", f"{ticker} {WINDOW_LENGTH}s TimeGAN Sample")
        depth = book_a["ask_px"].shape[2]
        shades = np.linspace(1.0, 0.35, depth)  # level 1 darkest, deeper levels lighter
        centres = [depth / 2 - 0.5, 1.5 * depth - 0.5]  # middle of the bid half, ask half

        prices = [{s: b[f"{s}_px"][i] / 100 for s in ("ask", "bid")} for b, i in cols]
        heatmaps = [depth_heatmap(b, i) for b, i in cols]
        vmax = max(h.max() for h in heatmaps)
        span = 1.1 * max(np.ptp(np.concatenate(list(p.values()))) for p in prices)

        for col, (px, heat) in enumerate(zip(prices, heatmaps)):
            ax_px, ax_vol = row[:, col]

            for side, cmap in (("ask", plt.cm.Reds), ("bid", plt.cm.Blues)):
                for level in reversed(range(depth)):  # best level last, so it sits on top
                    ax_px.plot(px[side][:, level], color=cmap(shades[level]),
                               lw=1.5 if level == 0 else 0.8,
                               label=f"{side.capitalize()}s" if level == 0 else None)
            all_px = np.concatenate(list(px.values()))
            mid = (all_px.min() + all_px.max()) / 2
            ax_px.set_ylim(mid - span / 2, mid + span / 2)
            ax_px.ticklabel_format(axis="y", useOffset=False, style="plain")
            ax_px.spines[["top", "right"]].set_visible(False)
            ax_px.set_title(col_titles[col], fontsize=12, weight="bold")

            # Depth: bids below the white line, asks above
            im = ax_vol.imshow(heat, aspect="auto", origin="lower", cmap="viridis",
                               vmin=0, vmax=vmax, interpolation="nearest")
            ax_vol.axhline(depth - 0.5, color="white", lw=1)
            # Dashes pointing into the middle gap at the Bid/Ask labels
            ax_vol.set_yticks(centres)
            ax_vol.tick_params(axis="y", left=col == 1, right=col == 0, labelleft=False,
                               length=6, width=1)

        row[0, 0].set_ylabel("Price ($)")
        # One legend inside the first price panel of the row, so it adds no height
        row[0, 0].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)

        # Colour bar hugging the left heatmap, in the space under the price labels
        cax = row[1, 0].inset_axes([-0.03, 0, 0.012, 1])
        fig.colorbar(im, cax=cax, label="log(1 + size)")
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")

    # Bid/Ask text placed after layout so it can be centred in each row's gap without widening it
    fig.canvas.draw()
    for r in range(n):
        left, right = axes[2 * r + 1, 0].get_position(), axes[2 * r + 1, 1].get_position()
        x = (left.x1 + right.x0) / 2
        for name, frac in (("Bid", 0.25), ("Ask", 0.75)):
            fig.text(x, left.y0 + frac * left.height, name, ha="center", va="center")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_pair(model, scalers, cfg, real_x, refs, labels, k: int, seed: int) -> tuple[dict, dict]:
    """The (real, synthetic) pair for ticker k drawn from this seed alone, decoded at the same reference."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # One random real window and one synthetic window of this ticker
    idx = rng.choice(np.flatnonzero(labels == k), size=1)
    with torch.no_grad():
        synth_x = model.sample(n=1, window_length=WINDOW_LENGTH + BURN_IN, ticker=k).cpu().numpy()[:, BURN_IN:]

    # Both decoded at the same reference, so they start from the same book
    return (decode(real_x[idx], refs[idx], labels[idx], scalers, rng),
            decode(synth_x, refs[idx], labels[idx], scalers, rng))


def load_windows():
    """Model, scalers, config and every encoded window of SPLIT."""
    model, scalers, cfg = load_model(CHECKPOINT, device=DEVICE)
    model.eval()
    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes, cfg.train_frac, cfg.val_frac, cfg.test_frac)
    real_x, refs, labels = encode(books=books[SPLIT], scalers=scalers, window_length=WINDOW_LENGTH)
    return model, scalers, cfg, real_x, refs, labels


def main() -> None:
    model, scalers, cfg, real_x, refs, labels = load_windows()

    pairs = {}
    for k, ticker in enumerate(cfg.tickers):
        # Each row depends only on its own seed
        seed = SEEDS.get(ticker, SEED)
        print(f"{ticker}  seed {seed}")
        pairs[ticker] = make_pair(model, scalers, cfg, real_x, refs, labels, k, seed)

    plot_book_pairs(pairs, OUT_PATH)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()