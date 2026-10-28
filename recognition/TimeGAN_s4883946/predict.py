"""Load trained models, generate synthetic order book windows, print the comparison
against real held-out data and save figures for inspection.

    python predict.py --data-dir <dir> --model-dir outputs

Figures written to <model-dir>/figures:
    depth_heatmaps.png     real vs synthetic depth surfaces (level x time)
    distributions.png      spread and mid-return histograms
    volatility.png         autocorrelation of |returns| and example mid-price paths
    violations.png         rates of crossed books / invalid ladders / negative depth
    autopsy.png            worst generated snapshots by rule violation, with real reference
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import metrics
from dataset import LEVELS, N_FEATURES, TICK, denormalise, get_dataloaders, real_windows
from modules import build_model

LADDER = "levels"   # set from the checkpoint in main()
COLOURS = {"real": "black", "timegan": "tab:blue", "baseline": "tab:orange"}


def load_model(path, device):
    """Rebuild a model from the settings saved in its checkpoint, then load its weights."""
    ckpt = torch.load(path, map_location=device)
    a = ckpt["args"]
    model = build_model(a["model"], N_FEATURES, a["hidden"], a["layers"],
                        feature_adv=a.get("feature_adv", False),
                        hold_gate=a.get("hold_gate", False)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), a


def fig_heatmaps(real_raw, synth, out, n_cols=4, seed=0):
    """Rows: real, then each model; columns: independent random windows."""
    rng = np.random.default_rng(seed)
    rows = {"real": real_raw[rng.choice(len(real_raw), n_cols, replace=False)]}
    for name, raw in synth.items():
        rows[name] = raw[rng.choice(len(raw), n_cols, replace=False)]
    hm = {k: metrics.depth_heatmaps(v) for k, v in rows.items()}
    ref = metrics.depth_heatmaps(real_raw)
    vmin, vmax = np.percentile(ref, [1, 99])
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.2 * n_cols, 2.6 * len(rows)),
                             squeeze=False)
    for r, (name, h) in enumerate(hm.items()):
        for c in range(n_cols):
            im = axes[r, c].imshow(h[c], aspect="auto", origin="lower", vmin=vmin, vmax=vmax,
                                   cmap="viridis")
            axes[r, c].axhline(LEVELS - 0.5, color="white", lw=0.8)
            axes[r, c].set_title(f"{name} #{c + 1}", fontsize=9)
            if c == 0:
                axes[r, c].set_ylabel("level (bid 10..1 | ask 1..10)", fontsize=7)
            if r == len(rows) - 1:
                axes[r, c].set_xlabel("time step (s)")
    fig.colorbar(im, ax=axes, label="log(1 + volume)", shrink=0.8)
    fig.suptitle("Depth surfaces: real vs synthetic (white line = best bid/ask boundary)")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_distributions(real_raw, synth, out):
    """Histograms of spread and of 1-step returns, real against each model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    sp_real = metrics.spreads(real_raw, LADDER).ravel()
    # histogram range: the real 0.5th to 99.5th percentile, padded so model tails stay visible
    lo, hi = np.percentile(sp_real, [0.5, 99.5])
    pad = (hi - lo) * 0.5
    bins = np.linspace(lo - pad, hi + pad, 60)
    axes[0].hist(sp_real, bins, density=True, histtype="step", color=COLOURS["real"], label="real")
    for n, raw in synth.items():
        axes[0].hist(metrics.spreads(raw, LADDER).ravel(), bins, density=True, histtype="step",
                     color=COLOURS[n], label=n)
    axes[0].set_xlabel("spread ($)"); axes[0].set_title("Spread distribution"); axes[0].legend()
    r_real = real_raw[..., 0].ravel()
    lo, hi = np.percentile(r_real, [0.5, 99.5]); pad = (hi - lo) * 0.5
    bins = np.linspace(lo - pad, hi + pad, 80)
    axes[1].hist(r_real, bins, density=True, histtype="step", color=COLOURS["real"], label="real")
    for n, raw in synth.items():
        axes[1].hist(raw[..., 0].ravel(), bins, density=True, histtype="step",
                     color=COLOURS[n], label=n)
    axes[1].set_yscale("log"); axes[1].set_xlabel("1-step log mid-return")
    axes[1].set_title("Mid-price return distribution (log y)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_volatility(real_raw, synth, out, seed=0):
    """Left: autocorrelation of |return| (volatility clustering). Right: sample price paths."""
    lags = (1, 2, 3, 5, 8, 10)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    width = 0.8 / (1 + len(synth))
    all_sets = {"real": real_raw, **synth}
    for i, (n, raw) in enumerate(all_sets.items()):
        acf = metrics.abs_return_acf(raw, lags)
        axes[0].bar(np.arange(len(lags)) + i * width, [acf[k] for k in lags], width,
                    color=COLOURS[n], label=n)
    axes[0].set_xticks(np.arange(len(lags)) + 0.4 - width / 2); axes[0].set_xticklabels(lags)
    axes[0].set_xlabel("lag (steps)"); axes[0].set_ylabel("ACF of |return|")
    axes[0].set_title("Volatility clustering"); axes[0].legend()
    rng = np.random.default_rng(seed)
    for n, raw in all_sets.items():
        for j, k in enumerate(rng.choice(len(raw), 3, replace=False)):
            axes[1].plot(np.cumsum(raw[k, :, 0]) * 1e4, color=COLOURS[n], alpha=0.7,
                         label=n if j == 0 else None)
    axes[1].set_xlabel("time step (s)"); axes[1].set_ylabel("cumulative return (bp)")
    axes[1].set_title("Example mid-price paths (3 windows each)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_violations(audits, out):
    """Bar chart of how often each market rule is broken, per model."""
    keys = ["crossed", "ladder", "neg_depth", "any"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.8 / len(audits)
    for i, (n, a) in enumerate(audits.items()):
        ax.bar(np.arange(len(keys)) + i * width, [100 * a[k] for k in keys], width,
               color=COLOURS[n], label=n)
    ax.set_xticks(np.arange(len(keys)) + 0.4 - width / 2); ax.set_xticklabels(keys)
    ax.set_ylabel("% of snapshots"); ax.set_title("Microstructure rule violations"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def violation_score(raw):
    """Per-snapshot badness: how far the book is from valid (0 means valid)."""
    ask_d, bid_d = metrics.distances(raw, LADDER)
    crossed = np.clip(-(ask_d[..., 0] + bid_d[..., 0]), 0, None)
    ladder = (np.clip(-np.diff(ask_d, axis=-1), 0, None).sum(-1)
              + np.clip(-np.diff(bid_d, axis=-1), 0, None).sum(-1))
    return crossed + ladder


def fig_autopsy(real_raw, synth, out, k=3):
    """Worst generated snapshots per model: price distance by level vs a real snapshot."""
    ask_r, bid_r = metrics.distances(real_raw[:1], LADDER)   # one real snapshot as the reference
    levels = np.arange(1, LEVELS + 1)
    fig, axes = plt.subplots(len(synth), k, figsize=(3.6 * k, 3 * len(synth)), squeeze=False)
    for r, (name, raw) in enumerate(synth.items()):
        score = violation_score(raw).ravel()
        worst = np.argsort(score)[::-1][:k]   # indices of the k most invalid snapshots
        ask_d, bid_d = metrics.distances(raw, LADDER)
        for c, idx in enumerate(worst):
            w, t = divmod(idx, raw.shape[1])   # flat index -> (window, time step)
            ax = axes[r, c]
            ax.plot(levels, ask_r[0, 0] / TICK, "k--", label="real ask")
            ax.plot(levels, bid_r[0, 0] / TICK, "k:", label="real bid")
            ax.plot(levels, ask_d[w, t] / TICK, "o-", color="tab:red", label="synthetic ask")
            ax.plot(levels, bid_d[w, t] / TICK, "o-", color="tab:green", label="synthetic bid")
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_title(f"{name}: violation score {score[idx]:.2f}", fontsize=9)
            ax.set_xlabel("level"); ax.set_ylabel("distance from mid (ticks)")
            if r == 0 and c == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Failure autopsy: worst synthetic snapshots (ladders should rise with level)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-dir", default="outputs")
    p.add_argument("--models", nargs="+", default=["timegan", "baseline"])
    p.add_argument("--n-samples", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")   # sampling is cheap, no GPU needed
    fig_dir = f"{args.model_dir}/figures"
    os.makedirs(fig_dir, exist_ok=True)

    models, cfgs = {}, {}
    for name in args.models:
        models[name], cfgs[name] = load_model(f"{args.model_dir}/{name}.pt", device)
    # rebuild the data pipeline with the same settings the models were trained with, so the
    # normalisation and return transform match
    cfg = next(iter(cfgs.values()))
    loaders, info = get_dataloaders(args.data_dir, cfg["seq_len"], cfg["step"],
                                    return_transform=cfg.get("return_transform", "zscore"),
                                    clip_q=cfg.get("clip_q", 0.5),
                                    ladder=cfg.get("ladder", "levels"))
    global LADDER
    LADDER = cfg.get("ladder", "levels")
    train_raw, test_raw = real_windows(info, "train"), real_windows(info, "test")

    # generate windows and undo the normalisation to get raw features
    synth = {n: denormalise(m.sample(args.n_samples, cfg["seq_len"], device).numpy(), info)
             for n, m in models.items()}

    # Real-data reference: how far a *real* later period is from the training period,
    # and the SSIM a perfect generator could reach (held-out real vs nearest real).
    # (train windows act as the "synthetic" candidates; test windows are the references,
    # so no window is compared with itself)
    ref = metrics.evaluate(test_raw, train_raw[:: max(1, len(train_raw) // args.n_samples)],
                           reference_raw=test_raw, ladder=LADDER)
    ref_row = {"real train period -> test (reference)": ref}
    results = {n: metrics.evaluate(test_raw, s, reference_raw=train_raw, ladder=LADDER) for n, s in synth.items()}
    results_enforced = {n: metrics.evaluate(test_raw, s, reference_raw=train_raw, ladder=LADDER,
                                            enforce_rules=True) for n, s in synth.items()}

    # table of test-split metrics: raw output, rules-enforced output, then the real-data reference
    cols = ["audit_crossed", "audit_ladder", "audit_neg_depth", "kl_spread", "kl_return",
            "acf_lag1", "acf_lag5", "ssim", "ssim_mean_surface"]
    print("\nTest-split comparison (synthetic vs real test windows)")
    print(f"{'':42s}" + "".join(f"{c.replace('audit_', '')[:13]:>14s}" for c in cols))
    real_acf = metrics.abs_return_acf(test_raw)
    enforced_rows = {f"{n} (rules enforced)": r for n, r in results_enforced.items()}
    for label, row in {**results, **enforced_rows, **ref_row}.items():
        print(f"{label:42s}" + "".join(f"{row[c]:14.4f}" for c in cols))
    print(f"{'real test windows (ACF only)':42s}" + " " * 14 * 5
          + f"{real_acf[1]:14.4f}{real_acf[5]:14.4f}")

    for n in models:
        sj = f"{args.model_dir}/{n}_summary.json"
        if os.path.exists(sj):
            prof = json.load(open(sj))["profile"]
            print(f"{n}: {prof['parameters']:,} params, trained in {prof['train_seconds']:.0f}s on "
                  f"{prof['device']}, 1000 windows sampled in {prof['sample_1000_windows_seconds']:.3f}s")

    # all figures compare against the real *test* windows
    fig_heatmaps(test_raw, synth, f"{fig_dir}/depth_heatmaps.png")
    fig_distributions(test_raw, synth, f"{fig_dir}/distributions.png")
    fig_volatility(test_raw, synth, f"{fig_dir}/volatility.png")
    fig_violations({n: {k[6:]: v for k, v in r.items() if k.startswith("audit_")}
                    for n, r in results.items()}, f"{fig_dir}/violations.png")
    fig_autopsy(test_raw, synth, f"{fig_dir}/autopsy.png")
    with open(f"{args.model_dir}/comparison.json", "w") as f:
        json.dump({"models": results, "models_enforced": results_enforced, "real_reference": ref,
                   "real_acf": real_acf}, f, indent=2)
    print(f"\nFigures written to {fig_dir}")


if __name__ == "__main__":
    main()
