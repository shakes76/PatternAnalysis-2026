"""Plot a stretch of real test-period order book data next to generated TimeGAN data.

Both columns cover the same length of time at the same 1 s grid and show, top to bottom:
mid-price with the best bid/ask band, the spread in ticks, the depth heatmap (bid levels
10..1 then ask levels 1..10, colour = log(1 + volume)) and the size of each step's price
move. The generated column is built by joining independent 24-step windows, so state is
not carried across a window boundary (faint dashed lines mark the joins).

Usage: python plot_comparison.py --data-dir <dir> --model-dir outputs/final \
           --model timegan --minutes 10 --out figures/real_vs_timegan_window.png
"""
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import metrics
from dataset import TICK, denormalise, get_dataloaders
from predict import load_model


def series(raw, start_mid, ladder):
    """Turn (T, 41) raw features into mid, best bid/ask, spread (ticks), depth, returns."""
    ask_d, bid_d = metrics.distances(raw[None], ladder)
    ask_d, bid_d = ask_d[0], bid_d[0]
    mid = start_mid * np.exp(np.cumsum(raw[:, 0]))   # rebuild prices from the log-returns
    return {
        "mid": mid,
        "ask": mid + ask_d[:, 0], "bid": mid - bid_d[:, 0],
        "spread": (ask_d[:, 0] + bid_d[:, 0]) / TICK,
        "depth": metrics.depth_heatmaps(raw[None])[0],
        "ret_bp": raw[:, 0] * 1e4,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-dir", default="outputs/final")
    p.add_argument("--model", default="timegan")
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--offset-minutes", type=float, default=0.0,
                   help="start this many minutes into the test period")
    p.add_argument("--continuous", action="store_true",
                   help="generate the whole stretch as one sequence instead of joining windows")
    p.add_argument("--enforce", action="store_true",
                   help="snap generated prices and volumes onto the market rules first")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="figures/real_vs_timegan_window.png")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    model, cfg = load_model(f"{args.model_dir}/{args.model}.pt", torch.device("cpu"))
    ladder = cfg.get("ladder", "levels")
    _, info = get_dataloaders(args.data_dir, cfg["seq_len"], cfg["step"],
                              return_transform=cfg.get("return_transform", "zscore"),
                              clip_q=cfg.get("clip_q", 0.5), ladder=ladder)
    step, seq_len = cfg["step"], cfg["seq_len"]
    n_steps = int(args.minutes * 60 / step)
    s0 = info["splits"]["test"][0] + int(args.offset_minutes * 60 / step)

    real_raw = info["feats_raw"][s0:s0 + n_steps]
    start_mid = info["mid"][s0 - 1]
    n_win = 1 if args.continuous else math.ceil(n_steps / seq_len)
    win_len = n_steps if args.continuous else seq_len
    synth = denormalise(model.sample(n_win, win_len).numpy(), info)
    synth_raw = synth.reshape(-1, synth.shape[-1])[:n_steps]
    if args.enforce:
        synth_raw = metrics.snap_to_rules(synth_raw, ladder)

    clock0 = info["grid"][s0] if "grid" in info else 0.0
    start_clock = clock0  # seconds after midnight
    fmt = lambda t: f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}"
    title_real = f"Real test data ({fmt(start_clock)} to {fmt(start_clock + n_steps * step)})"
    title_synth = (f"{args.model} generated (one continuous sequence)" if args.continuous else
                   f"{args.model} generated ({n_win} windows of {seq_len * step:.0f} s joined)")

    data = [series(real_raw, start_mid, ladder), series(synth_raw, start_mid, ladder)]
    t = np.arange(n_steps) * step / 60   # minutes
    vmin, vmax = np.percentile(data[0]["depth"], [1, 99])
    price_lo = min(d["bid"].min() for d in data) - 0.05
    price_hi = max(d["ask"].max() for d in data) + 0.05
    spread_hi = max(d["spread"].max() for d in data) * 1.05
    ret_hi = max(np.abs(d["ret_bp"]).max() for d in data) * 1.05

    # third column holds only the colour bar, so all time axes line up
    fig, axes = plt.subplots(4, 3, figsize=(13, 10),
                             gridspec_kw={"height_ratios": [1.3, 0.8, 1.6, 0.8],
                                          "width_ratios": [1, 1, 0.025]})
    t_end = t[-1] + step / 60
    for r in range(4):
        for c in range(2):
            axes[r, c].set_xlim(0, t_end)
    for col, (d, title) in enumerate(zip(data, [title_real, title_synth])):
        ax = axes[0, col]
        ax.fill_between(t, d["bid"], d["ask"], color="tab:blue", alpha=0.25, label="bid-ask band")
        ax.plot(t, d["mid"], color="tab:blue", lw=1, label="mid-price")
        ax.set_ylim(price_lo, price_hi); ax.set_title(title, fontsize=10)
        if col == 0:
            ax.set_ylabel("price ($)"); ax.legend(fontsize=7, loc="upper left")
        ax = axes[1, col]
        ax.step(t, d["spread"], where="post", color="tab:orange", lw=0.9)
        ax.set_ylim(0, spread_hi)
        if col == 0:
            ax.set_ylabel("spread (ticks)")
        ax = axes[2, col]
        im = ax.imshow(d["depth"], aspect="auto", origin="lower", vmin=vmin, vmax=vmax,
                       extent=[0, t[-1] + step / 60, -0.5, 19.5], cmap="viridis")
        ax.axhline(9.5, color="white", lw=0.8)
        if col == 0:
            ax.set_ylabel("level (bid 10..1 | ask 1..10)", fontsize=8)
        ax = axes[3, col]
        ax.vlines(t, 0, np.abs(d["ret_bp"]), color="tab:red", lw=0.6)
        ax.set_ylim(0, ret_hi); ax.set_xlabel("time (minutes)")
        if col == 0:
            ax.set_ylabel("|1 s move| (bp)")
    for k in range(1, 1 if args.continuous else n_win):   # window joins in the generated column
        for r in range(4):
            axes[r, 1].axvline(k * seq_len * step / 60, color="grey", lw=0.5, ls="--", alpha=0.35)
    fig.colorbar(im, cax=axes[2, 2], label="log(1 + volume)")
    for r in (0, 1, 3):
        axes[r, 2].axis("off")
    fig.suptitle("Real versus generated order book over the same length of time "
                 "(dashed lines = joins between independent generated windows)", fontsize=11)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
