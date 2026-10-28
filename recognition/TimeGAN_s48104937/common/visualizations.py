from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.constants import DEPTH
from common.constants import LEVELS
from common.metrics import abs_return_autocorr
from common.metrics import central_bins
from common.metrics import depth_heatmap
from common.metrics import max_log_size
from common.metrics import mid_returns
from common.metrics import spread
from common.metrics import tick_bins
from common.metrics import vwap_returns


SOURCES = ("Real", "Synthetic")
LEVEL_LABELS = [f"bid {l}" for l in reversed(LEVELS)] + [f"ask {l}" for l in LEVELS]
SHADES = np.linspace(1.0, 0.35, DEPTH)   # level 1 darkest, deeper levels lighter


# Training Plots
def plot_losses(embedding_losses: dict, supervisor_losses: dict, joint_losses: dict, path: Path) -> None:
    """One panel per training phase, saved for the README."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, hist, title in [(axes[0], embedding_losses, "Phase 1: embedding (recon MSE)"),
                            (axes[1], supervisor_losses, "Phase 2: supervised (next-step MSE)"),
                            (axes[2], joint_losses, "Phase 3: joint training")]:
        for name, values in hist.items():
            ax.plot(values, label=name)
        ax.set(title=title, xlabel="Epoch")
        ax.legend()
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# Evaluation Plots
def draw_depth(ax, book: dict, i: int, vmax: float, title: str):
    """Depth heatmap of window i: bids below the white line, asks above."""
    im = ax.imshow(depth_heatmap(book, i), aspect="auto", origin="lower", cmap="viridis",
                   vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(2 * DEPTH), LEVEL_LABELS, fontsize=7)
    ax.axhline(DEPTH - 0.5, color="white", lw=1)   # the spread
    ax.set(title=title, xlabel="step (1s)")
    return im

def draw_prices(ax, book: dict, i: int, title: str) -> None:
    """Every ask (red) and bid (blue) price level of window i, best level darkest."""
    for side, cmap in (("ask", plt.cm.Reds), ("bid", plt.cm.Blues)):
        px = book[f"{side}_px"][i] / 100   # cents -> dollars, shape (step, DEPTH)
        for level in range(DEPTH):
            ax.plot(px[:, level], color=cmap(SHADES[level]), lw=1,
                    label=f"{side} 1" if level == 0 else None)
    ax.set(title=title, xlabel="step (1s)", ylabel="price ($)")
    ax.legend(loc="upper left", fontsize=8)

def plot_spread_by_step(real: dict, synth: dict, path: Path, train_len: int | None = None) -> None:
    """Median and 5-95% band of the spread at each step, to expose drift.

    If the windows are longer than train_len (the length the model was trained
    on), a dotted line marks where generation becomes extrapolation.
    """
    steps = np.arange(spread(real).shape[1])

    fig, ax = plt.subplots(figsize=(8, 4))
    for book, name in zip((real, synth), SOURCES):
        lo, med, hi = np.percentile(spread(book), [5, 50, 95], axis=0)
        (line,) = ax.plot(steps, med, label=f"{name} median")
        ax.fill_between(steps, lo, hi, color=line.get_color(), alpha=0.2, label=f"{name} 5-95%")
    ax.axhline(0, color="red", lw=0.8, ls="--", label="Crossed below")
    if train_len is not None and len(steps) > train_len:
        ax.axvline(train_len - 1, color="grey", ls=":", label="Training length")
    ax.set(xlabel="Time step within window", ylabel="Spread (ticks)")
    # ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_distributions(real: dict, synth: dict, path: Path) -> None:
    """Histograms of spread, mid and VWAP returns, plus volatility clustering."""
    histograms = {
        "Spread (ticks)": (spread, tick_bins),          # one bin per tick
        "Mid-price log-returns": (mid_returns, central_bins),
        "Book VWAP log-returns": (vwap_returns, central_bins),
    }
    fig, axes = plt.subplots(1, len(histograms) + 1, figsize=(20, 4), constrained_layout=True)

    for ax, (title, (stat, make_bins)) in zip(axes, histograms.items()):
        values = [stat(real), stat(synth)]
        bins = make_bins(*values)                       # shared, so the two histograms align
        for v, name in zip(values, SOURCES):
            ax.hist(v.ravel(), bins=bins, alpha=0.5, density=True, label=name)
        ax.set_title(title)

    ax = axes[-1]
    for book, name in zip((real, synth), SOURCES):
        ax.plot(range(1, 11), abs_return_autocorr(mid_returns(book)), marker="o", label=name)
    ax.set(title="Autocorrelation of |mid returns|", xlabel="Lag")

    for ax in axes:
        ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_sample_pair(real: dict, synth: dict, i: int, path: Path) -> None:
    """Real vs synthetic window i: prices on top, depth below.

    The real windows are a random sample of the split and the synthetic ones
    are independent draws, so pairing them by index is already random.
    """
    vmax = max_log_size(real, synth)   # same colour scale for both heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    im = None
    for col, (book, name) in enumerate(zip((real, synth), SOURCES)):
        draw_prices(axes[0, col], book, i, f"{name} #{i}: price")
        im = draw_depth(axes[1, col], book, i, vmax, f"{name} #{i}: volume")
    fig.colorbar(im, ax=axes[1, :], label="log(1 + size)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

        for side, cmap in (("ask", plt.cm.Reds), ("bid", plt.cm.Blues)):
            for level in reversed(range(depth)):  # best level last, so it sits on top
                ax_px.plot(px[side][:, level], color=cmap(shades[level]),
                           lw=1.5 if level == 0 else 0.8,
                           label=f"best {side}" if level == 0 else None)
        all_px = np.concatenate(list(px.values()))
        mid = (all_px.min() + all_px.max()) / 2
        ax_px.set_ylim(mid - span / 2, mid + span / 2)
        ax_px.ticklabel_format(axis="y", useOffset=False, style="plain")
        if col_titles:
            ax_px.set_title(col_titles[col])

        # Depth: bids below the white line, asks above
        im = ax_vol.imshow(heat, aspect="auto", origin="lower", cmap="viridis",
                           vmin=0, vmax=vmax, interpolation="nearest")
        ax_vol.axhline(depth - 0.5, color="white", lw=1)
        ax_vol.set_xlabel("Time (s)")
        # Dashes pointing into the middle gap at the Bid/Ask labels
        ax_vol.set_yticks(centres)
        ax_vol.tick_params(axis="y", left=col == 1, right=col == 0, labelleft=False,
                           length=6, width=1)

    axes[0, 0].set_ylabel("Price ($)")
    axes[0, 0].legend(loc="upper left")
    suptitle = fig.suptitle(title) if title else None

    # Colour bar hugging the left heatmap, in the space under the price labels
    cax = axes[1, 0].inset_axes([-0.03, 0, 0.012, 1])
    fig.colorbar(im, cax=cax, label="log(1 + size)")
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")

    # Text placed after layout so it can be centred without widening the gaps
    fig.canvas.draw()
    left, right = axes[1, 0].get_position(), axes[1, 1].get_position()
    x = (left.x1 + right.x0) / 2
    for name, frac in (("Bid", 0.25), ("Ask", 0.75)):
        fig.text(x, left.y0 + frac * left.height, name, ha="center", va="center")
    if suptitle:  # centre over the panels, not the whole figure with the colour bar
        suptitle.set_x((axes[0, 0].get_position().x0 + axes[0, 1].get_position().x1) / 2)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)