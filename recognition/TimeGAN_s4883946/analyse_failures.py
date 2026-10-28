"""Failure analysis of generated order book windows (supports the README autopsy).

For each model in --model-dir it measures:
  * tick grid: real prices sit on a $0.01 grid, so real spreads are whole ticks. How
    many generated spreads are off the grid, below one tick, and how much of the spread
    KL disappears if generated spreads are rounded to the nearest tick;
  * ladder: how large the ladder violations are (in ticks) and where in the book they sit;
  * negative depth: how many snapshots contain a negative volume;
and saves figures/spread_ticks.png.

Usage: python analyse_failures.py --data-dir <dir> --model-dir outputs/final
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import metrics
from dataset import TICK, denormalise, get_dataloaders, real_windows
from predict import COLOURS, load_model


def analyse(name, raw, real_spread_ticks, ladder):
    """Print the tick-grid, ladder and negative-depth failure rates of one model's windows."""
    ask_d, bid_d = metrics.distances(raw, ladder)
    spread_ticks = ((ask_d[..., 0] + bid_d[..., 0]) / TICK).ravel()
    rounded = np.maximum(np.round(spread_ticks), 1.0)   # what the spread would be on the grid
    off_grid = np.abs(spread_ticks - np.round(spread_ticks)) > 0.1
    print(f"\n== {name}")
    print(f"spread: {100 * off_grid.mean():.1f}% of generated spreads are more than 0.1 tick "
          f"off the tick grid; {100 * (spread_ticks < 1).mean():.1f}% are below one tick")
    print(f"spread KL: raw {metrics.kl_divergence(real_spread_ticks, spread_ticks):.3f}  "
          f"rounded to whole ticks {metrics.kl_divergence(real_spread_ticks, rounded):.3f}")

    # Ladder: size and place of violations (gap between successive levels, in ticks)
    gaps = np.concatenate([np.diff(ask_d, axis=-1), np.diff(bid_d, axis=-1)], axis=-1) / TICK
    bad = (gaps < 0).any(-1)
    worst_dip = -np.minimum(gaps, 0).min(-1)[bad]
    first_level = (np.concatenate([(np.diff(ask_d, axis=-1) < 0)[..., :2],
                                   (np.diff(bid_d, axis=-1) < 0)[..., :2]], -1).any(-1))[bad]
    print(f"ladder: {100 * bad.mean():.1f}% of snapshots invalid; of those, "
          f"{100 * (worst_dip <= 1).mean():.0f}% dip by at most 1 tick and "
          f"{100 * (worst_dip <= 3).mean():.0f}% by at most 3 ticks; "
          f"{100 * first_level.mean():.0f}% involve levels 1 to 3 (the touch)")
    print(f"ladder: {100 * (gaps < 1).any(-1).mean():.1f}% of snapshots have a gap under one tick "
          "(real data: 0%)")
    # log1p(volume) below 0 decodes to a negative share count
    neg = (raw[..., metrics.ASK_VOL] < 0) | (raw[..., metrics.BID_VOL] < 0)
    print(f"depth: {100 * neg.any(-1).mean():.2f}% of snapshots have a negative volume "
          f"({100 * neg.mean():.3f}% of all level volumes)")
    return spread_ticks


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-dir", default="outputs/final")
    p.add_argument("--models", nargs="+", default=["timegan", "baseline"])
    p.add_argument("--n-samples", type=int, default=3000)
    p.add_argument("--out", default="figures/spread_ticks.png")
    args = p.parse_args()
    torch.manual_seed(0)

    models, cfgs = {}, {}
    for n in args.models:
        models[n], cfgs[n] = load_model(f"{args.model_dir}/{n}.pt", torch.device("cpu"))
    cfg = next(iter(cfgs.values()))
    ladder = cfg.get("ladder", "levels")
    loaders, info = get_dataloaders(args.data_dir, cfg["seq_len"], cfg["step"],
                                    return_transform=cfg.get("return_transform", "zscore"),
                                    clip_q=cfg.get("clip_q", 0.5), ladder=ladder)
    real = real_windows(info, "test")
    # real spreads in ticks: the reference the generated spreads are compared with
    real_ticks = (metrics.spreads(real, ladder) / TICK).ravel()
    print(f"real test spreads: {100 * (np.abs(real_ticks - np.round(real_ticks)) > 0.1).mean():.1f}% "
          f"off the tick grid; mean {real_ticks.mean():.2f} ticks")

    spreads = {"real": real_ticks}
    for n, m in models.items():
        synth = denormalise(m.sample(args.n_samples, cfg["seq_len"]).numpy(), info)
        spreads[n] = analyse(n, synth, real_ticks, ladder)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.arange(0.5, 15.5, 0.25)
    for n, v in spreads.items():
        ax.hist(v, bins, density=True, histtype="step", color=COLOURS[n], label=n)
    ax.set_xlabel("spread (ticks of $0.01)"); ax.set_ylabel("density")
    ax.set_title("Real spreads sit on whole ticks; generated spreads fall between ticks")
    ax.set_xticks(range(1, 15)); ax.legend()
    fig.tight_layout(); fig.savefig(args.out, dpi=130)


if __name__ == "__main__":
    main()
