from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.constants import DEVICE
from common.metrics import unchanged_rates
from common.metrics import violation_rates
from common.misc import load_model
from common.visualizations import plot_sample_pair
from dataset import FEATURE_COLS
from dataset import decode
from dataset import encode
from dataset import load_books


WINDOW_LENGTH: int | None = None   # None = training window length; set e.g. 512 to test long windows
NUM_SAMPLES: int = 500             # real windows to reconstruct
TICKER: str = "AMZN"               # ticker to evaluate (must be in cfg.tickers)
SPLIT: str = "val"                 # split to take real windows from
SEED: int = 48                     # seed for np and torch rng
OUT_DIR = Path("dummy/figures")

N_FEAT = len(FEATURE_COLS)
GROUPS = {
    "mid_rel": [0],
    "spread": [1],
    "gaps": list(range(2, 20)),
    "sizes": list(range(20, 40)),
    "flags": list(range(N_FEAT, 2 * N_FEAT)),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng: np.random.Generator = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    model, scalers, cfg = load_model(Path("models/timegan.pt"), device=DEVICE)
    model.eval()
    window_length = WINDOW_LENGTH or cfg.window_length
    k = cfg.tickers.index(TICKER)

    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes, cfg.train_frac, cfg.val_frac, cfg.test_frac)
    real_x, refs, labels = encode(books=books[SPLIT], scalers=scalers, window_length=window_length)
    idx = rng.choice(np.flatnonzero(labels == k), size=NUM_SAMPLES, replace=False)
    real_x, refs, labels = real_x[idx], refs[idx], labels[idx]

    # Real windows through the autoencoder only: embedder -> latent -> recovery
    with torch.no_grad():
        x = torch.from_numpy(real_x).float().to(DEVICE)
        c = model.one_hot(torch.from_numpy(labels).long().to(DEVICE))
        h = torch.cat([model.embedder(xb, cb) for xb, cb in zip(x.split(256), c.split(256))])
        recon = torch.cat([model.recovery(hb, cb) for hb, cb in zip(h.split(256), c.split(256))])
    h, recon = h.cpu().numpy(), recon.cpu().numpy()

    print(f"{TICKER}: {NUM_SAMPLES} {SPLIT} windows of {window_length} steps "
          f"(trained on {cfg.window_length})\n")

    # Reconstruction error and how much variation survives, per feature group (scaled space)
    print(f"{'Group':<10}{'MSE':>10}{'std real':>11}{'std recon':>11}{'ratio':>8}   (std across time)")
    for name, cols in GROUPS.items():
        mse = ((recon[..., cols] - real_x[..., cols]) ** 2).mean()
        s_real = real_x[..., cols].std(axis=1).mean()
        s_recon = recon[..., cols].std(axis=1).mean()
        print(f"{name:<10}{mse:>10.5f}{s_real:>11.4f}{s_recon:>11.4f}{s_recon / (s_real + 1e-8):>8.2f}")

    # Worst channels, by MSE relative to that channel's own variance
    mse_c = ((recon[..., :N_FEAT] - real_x[..., :N_FEAT]) ** 2).mean(axis=(0, 1))
    var_c = real_x[..., :N_FEAT].var(axis=(0, 1)) + 1e-8
    print("\nWorst channels (MSE / variance; 1.0 = no better than predicting the mean):")
    for i in np.argsort(-mse_c / var_c)[:8]:
        print(f"  {FEATURE_COLS[i]:<14}{mse_c[i] / var_c[i]:>8.3f}")

    # Flags: does the autoencoder keep how often things change?
    real_f, recon_f = real_x[..., N_FEAT:], recon[..., N_FEAT:]
    acc = ((recon_f > 0.5) == (real_f > 0.5)).mean()
    print(f"\nFlags: accuracy at 0.5 {acc:.3f}   real change rate {real_f.mean():.3f}   "
          f"mean recon prob {recon_f.mean():.3f}")
    print(f"  on changed steps   mean prob {recon_f[real_f > 0.5].mean():.3f}")
    print(f"  on unchanged steps mean prob {recon_f[real_f < 0.5].mean():.3f}")

    # Latent space health: saturated or dead units, and how many dimensions are actually used
    flat = h.reshape(-1, h.shape[-1])
    saturated = ((flat < 0.02) | (flat > 0.98)).mean(axis=0)
    dead = flat.std(axis=0) < 1e-3
    eig = np.linalg.eigvalsh(np.cov(flat.T))[::-1]
    explained = np.cumsum(eig) / eig.sum()
    print(f"\nLatent: {h.shape[-1]} dims   dead {dead.sum()}   "
          f"mostly saturated (>50% of steps) {(saturated > 0.5).sum()}")
    print(f"  dims for 90% / 99% of variance: {np.searchsorted(explained, 0.9) + 1} / "
          f"{np.searchsorted(explained, 0.99) + 1}")

    # Decoded books: real vs reconstruction, anchored at the same reference
    real_b = decode(real_x, refs, labels, scalers, rng)
    recon_b = decode(recon, refs, labels, scalers, rng)
    print(f"\n{'Decoded books':<24}{'Real':>10}{'Recon':>10}")
    for name, real_v, recon_v in [("violations", violation_rates(real_b), violation_rates(recon_b)),
                                  ("unchanged", unchanged_rates(real_b), unchanged_rates(recon_b))]:
        for key in real_v:
            print(f"  {name + ' ' + key:<22}{real_v[key]:>10.2%}{recon_v[key]:>10.2%}")
    for key in ("ask_px", "bid_px", "ask_sz", "bid_sz"):
        err = np.abs(recon_b[key] - real_b[key])
        unit = "ticks" if key.endswith("px") else "shares"
        print(f"  mean |error| {key:<8} L1 {err[..., 0].mean():>8.2f}   all levels {err.mean():>8.2f} {unit}")

    # Error over the window: GRU warm-up at the start, drift past the training length
    err_t = ((recon - real_x) ** 2)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for name, cols in GROUPS.items():
        ax.plot(err_t[..., cols].mean(axis=(0, 2)), label=name)
    ax.axvline(cfg.window_length, color="grey", ls="--", lw=1, label="training length")
    ax.set(xlabel="step", ylabel="MSE (scaled)", yscale="log", title=f"{TICKER} reconstruction error by step")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"embedder_{TICKER}_error_by_step.png", dpi=150)

    # A few channels of one window: real vs reconstruction
    show = ["mid_rel", "spread", "ask_gap_5", "ask_sz_1", "bid_sz_5"]
    fig, axes = plt.subplots(len(show), 1, figsize=(9, 1.8 * len(show)), sharex=True)
    for ax, col in zip(axes, show):
        i = FEATURE_COLS.index(col)
        ax.plot(real_x[0, :, i], label="real", lw=1.2)
        ax.plot(recon[0, :, i], label="recon", lw=1.2)
        ax.set_ylabel(col, fontsize=8)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"embedder_{TICKER}_channels.png", dpi=150)

    # Decoded book of the same window (right panel is the reconstruction, not a synthetic sample)
    plot_sample_pair(real_b, recon_b, 0, OUT_DIR / f"embedder_{TICKER}_recon_pair.png")
    print(f"\nSaved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()