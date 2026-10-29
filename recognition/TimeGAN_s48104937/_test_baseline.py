"""Load the trained baseline and compare real and synthetic windows on a held-out split."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.metrics import depth_heatmap
from tests.smoke_test_model import CHECKPOINT_PATH
from tests.smoke_test_model import COLS
from tests.smoke_test_model import DEVICE
from tests.smoke_test_model import FIGURE_DIR
from tests.smoke_test_model import Generator
from tests.smoke_test_model import MinMax
from tests.smoke_test_model import evaluate
from tests.smoke_test_model import load_book
from tests.smoke_test_model import sample_books
from tests.smoke_test_model import to_books
from tests.smoke_test_model import to_features


SPLIT: str = "test"              # "train", "val" or "test"
SPLIT_FRACS = {"train": (0.00, 0.70), "val": (0.70, 0.85), "test": (0.85, 1.00)}
NUM_SAMPLES: int = 1000          # real windows compared against as many synthetic ones
SHOWN_WINDOW: int = 0            # which pair goes in the figure
SEED: int = 0


def plot_book_pair(
        book_a: dict,
        book_b: dict,
        path: Path,
        idx_a: int = 0,
        idx_b: int = 0,
        title: str | None = None,
        col_titles: tuple[str, str] | None = None,
) -> None:
    cols = [(book_a, idx_a), (book_b, idx_b)]
    depth = book_a["ask_px"].shape[2]
    shades = np.linspace(1.0, 0.35, depth)  # level 1 darkest, deeper levels lighter
    centres = [depth / 2 - 0.5, 1.5 * depth - 0.5]  # middle of the bid half, ask half

    prices = [{s: b[f"{s}_px"][i] / 100 for s in ("ask", "bid")} for b, i in cols]
    heatmaps = [depth_heatmap(b, i) for b, i in cols]
    vmax = max(h.max() for h in heatmaps)
    span = 1.1 * max(np.ptp(np.concatenate(list(p.values()))) for p in prices)

    fig, axes = plt.subplots(2, 2, figsize=(12, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    fig.get_layout_engine().set(w_pad=1 / 72, wspace=0)

    for col, (px, heat) in enumerate(zip(prices, heatmaps)):
        ax_px, ax_vol = axes[:, col]

        # Prices: every level, best level darkest and drawn last so it sits on top
        for side, cmap in (("ask", plt.cm.Reds), ("bid", plt.cm.Blues)):
            for level in reversed(range(depth)):
                ax_px.plot(px[side][:, level], color=cmap(shades[level]),
                           lw=1.2 if level == 0 else 0.8, alpha=0.9,
                           label=f"{side.title()}s" if level == 0 else None)
        all_px = np.concatenate(list(px.values()))
        mid = (all_px.min() + all_px.max()) / 2
        ax_px.set_ylim(mid - span / 2, mid + span / 2)
        ax_px.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax_px.spines[["top", "right"]].set_visible(False)

        # Bold column title, with a minimal legend left-aligned just below it
        if col_titles:
            ax_px.text(0.5, 1.14, col_titles[col], transform=ax_px.transAxes,
                       ha="center", va="bottom", fontsize=9, weight="bold")
        ax_px.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8,
                     frameon=False, borderaxespad=0.2, handlelength=1.5)

        # Depth: bids below the white line, asks above
        im = ax_vol.imshow(heat, aspect="auto", origin="lower", cmap="viridis",
                           vmin=0, vmax=vmax, interpolation="nearest")
        ax_vol.axhline(depth - 0.5, color="white", lw=1)
        ax_vol.set_xlabel("Time (s)", fontsize=9)
        # Dashes pointing into the middle gap at the Bid/Ask labels
        ax_vol.set_yticks(centres)
        ax_vol.tick_params(axis="y", left=col == 1, right=col == 0, labelleft=False,
                           length=6, width=1)

    axes[0, 0].set_ylabel("Price ($)", fontsize=9)
    suptitle = fig.suptitle(title, fontsize=10) if title else None

    # Colour bar hugging the left heatmap, in the space under the price labels
    cax = axes[1, 0].inset_axes([-0.03, 0, 0.012, 1])
    fig.colorbar(im, cax=cax)
    cax.set_ylabel("log(1 + size)", fontsize=9)
    cax.tick_params(labelsize=8)
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")

    # Text placed after layout so it can be centred without widening the gaps
    fig.canvas.draw()
    left, right = axes[1, 0].get_position(), axes[1, 1].get_position()
    x = (left.x1 + right.x0) / 2
    for name, frac in (("Bid", 0.25), ("Ask", 0.75)):
        fig.text(x, left.y0 + frac * left.height, name, ha="center", va="center", fontsize=9)
    if suptitle:  # centre over the panels, not the whole figure with the colour bar
        suptitle.set_x((axes[0, 0].get_position().x0 + axes[0, 1].get_position().x1) / 2)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def real_windows(tickers: list[str], window_length: int, split: str) -> np.ndarray:
    """Every stride-1 window of the chosen chronological split, shape (N, step, 40)."""
    lo, hi = SPLIT_FRACS[split]
    windows = []
    for ticker in tickers:
        book = load_book(ticker)
        values = book[COLS].to_numpy(np.float32)[int(len(book) * lo):int(len(book) * hi)]
        w = np.lib.stride_tricks.sliding_window_view(values, window_length, axis=0)
        windows.append(w.transpose(0, 2, 1))
    return np.ascontiguousarray(np.concatenate(windows))


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    # Model and scaler exactly as saved by the training script
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    cfg = ckpt["config"]
    gen = Generator(cfg["n_features"], cfg["hidden_dim"], cfg["num_layers"]).to(DEVICE)
    gen.load_state_dict(ckpt["generator"])
    gen.eval()
    scaler = MinMax(ckpt["scaler_lo"], ckpt["scaler_hi"])
    window_length = cfg["window_length"]

    # Real windows from the held-out split
    raw = real_windows(cfg["tickers"], window_length, SPLIT)
    idx = rng.choice(len(raw), size=min(NUM_SAMPLES, len(raw)), replace=False)
    feats, refs = to_features(raw[idx])
    real = to_books(feats, refs)

    # Synthetic windows, each anchored at the starting mid of its real counterpart
    synth = sample_books(gen, scaler, refs, window_length)

    print(f"Baseline vs real: {len(idx)} {SPLIT} windows of {window_length} s "
          f"({', '.join(cfg['tickers'])})")
    evaluate(real, synth)

    # Report figure: real window on the left, baseline window on the right
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"baseline_{SPLIT}_pair.png"
    plot_book_pair(real, synth, path, SHOWN_WINDOW, SHOWN_WINDOW,
                   col_titles=("Real Sample", "Baseline Model"))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()