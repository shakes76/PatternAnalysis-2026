"""Show strong, middle and weak generated windows, each next to its closest real window.

Every generated depth heatmap is scored with SSIM against its nearest real test heatmap.
Windows are then ranked by that score. "Strong" windows sit at the 99th percentile,
"middle" ones at the median and "weak" ones at the 1st percentile, so the picture shows
the spread of quality and not only the best cases.

Rows: generated depth heatmap, closest real depth heatmap, spread in ticks (generated
against real). Usage:
    python plot_examples.py --data-dir <dir> --model-dir outputs --model timegan
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity

import metrics
from dataset import TICK, denormalise, get_dataloaders, real_windows
from predict import load_model


def nearest_real(synth_maps, real_maps):
    """Index of the closest real heatmap (smallest L2 distance) for every generated one."""
    d = torch.cdist(torch.from_numpy(synth_maps.reshape(len(synth_maps), -1)).float(),
                    torch.from_numpy(real_maps.reshape(len(real_maps), -1)).float())
    return d.argmin(dim=1).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-dir", default="outputs")
    p.add_argument("--model", default="timegan")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--per-group", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enforce", action="store_true",
                   help="snap generated windows onto the market rules before plotting")
    p.add_argument("--out", default="figures/examples_strong_middle_weak.png")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    model, cfg = load_model(f"{args.model_dir}/{args.model}.pt", torch.device("cpu"))
    ladder = cfg.get("ladder", "levels")
    _, info = get_dataloaders(args.data_dir, cfg["seq_len"], cfg["step"],
                              return_transform=cfg.get("return_transform", "zscore"),
                              clip_q=cfg.get("clip_q", 0.5), ladder=ladder)
    real = real_windows(info, "test")
    synth = denormalise(model.sample(args.n_samples, cfg["seq_len"]).numpy(), info)
    if args.enforce:
        synth = metrics.snap_to_rules(synth, ladder)

    real_maps, synth_maps = metrics.depth_heatmaps(real), metrics.depth_heatmaps(synth)
    match = nearest_real(synth_maps, real_maps)
    data_range = float(real_maps.max() - real_maps.min())
    scores = np.array([structural_similarity(s, real_maps[j], data_range=data_range, win_size=7)
                       for s, j in zip(synth_maps, match)])

    order = np.argsort(scores)   # worst first
    n = len(order)
    groups = {}
    for label, pct in (("strong (top 1%)", 99), ("middle (median)", 50), ("weak (bottom 1%)", 1)):
        centre = int(round(pct / 100 * (n - 1)))
        lo = min(max(centre - args.per_group // 2, 0), n - args.per_group)
        groups[label] = order[lo:lo + args.per_group]
    picks = [(label, i) for label, idx in groups.items() for i in idx]

    vmin, vmax = np.percentile(real_maps, [1, 99])
    fig, axes = plt.subplots(3, len(picks), figsize=(2.6 * len(picks), 7.2), squeeze=False)
    for c, (label, i) in enumerate(picks):
        j = match[i]
        for r, (maps, name) in enumerate(((synth_maps[i], "generated"), (real_maps[j], "closest real"))):
            axes[r, c].imshow(maps, aspect="auto", origin="lower", vmin=vmin, vmax=vmax,
                              cmap="viridis")
            axes[r, c].axhline(9.5, color="white", lw=0.8)
            axes[r, c].set_xticks([])
            if c == 0:
                axes[r, c].set_ylabel(f"{name}\nlevel (bid 10..1 | ask 1..10)", fontsize=8)
        axes[0, c].set_title(f"{label}\nSSIM {scores[i]:.2f}", fontsize=9)
        gen_sp = metrics.spreads(synth[i][None], ladder)[0] / TICK
        real_sp = metrics.spreads(real[j][None], ladder)[0] / TICK
        axes[2, c].plot(real_sp, color="black", label="closest real")
        axes[2, c].plot(gen_sp, color="tab:blue", label="generated")
        axes[2, c].set_xlabel("time step (s)")
        if c == 0:
            axes[2, c].set_ylabel("spread (ticks)"); axes[2, c].legend(fontsize=7)
    fig.suptitle(f"{args.model}: generated windows ranked by similarity to the closest real window "
                 f"({n} windows)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}; score range {scores.min():.2f} to {scores.max():.2f}, "
          f"median {np.median(scores):.2f}")


if __name__ == "__main__":
    main()
