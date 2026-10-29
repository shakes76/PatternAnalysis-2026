"""What is giving the generator away? Probes the TimeGAN discriminator.

Run from the project root (same place as train.py / predict.py). Set CHECKPOINT below;
compare a checkpoint from before a collapse (epoch 50) with one after (epoch 60).

Tests
  1. Per-step D score       : is the gap concentrated at t=0 (start constraint) or spread out (dynamics)?
  2. Latent smoothness      : do real latents hold still (|dh| near 0) while synthetic ones jitter?
  3. Channel swap           : put one synthetic channel into otherwise-real windows; the channels whose
                              swap drops D's "real" score the most are what D keys on.
  4. Gradient saliency      : which input channels / steps D's real-score is most sensitive to.
  5. Dynamics interventions : add synthetic-sized jitter to held steps of real windows (should look fake
                              if jitter is the tell); freeze synthetic windows by their own flags (should
                              look more real if jitter is the tell).

Tests 3-5 score data-space windows through the embedder, i.e. D(embedder(x)). That is exactly the path
real data takes during training; for synthetic windows it uses recovery(h_hat) as a stand-in.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.constants import DEVICE
from common.misc import load_model
from dataset import FEATURE_COLS
from dataset import encode
from dataset import load_books


# --- Config ---
CHECKPOINT = Path("models/checkpoints/timegan_joint_epoch040.pt")
TICKER = 0          # ticker index (k-th ticker in the config)
NUM_WINDOWS = 512   # number of real and synthetic windows to compare
SEED = 0

N_FEAT = len(FEATURE_COLS)
CHANNELS = FEATURE_COLS + [f"flag:{c}" for c in FEATURE_COLS]
GROUPS = {
    "mid_rel": [0],
    "spread": [1],
    "all price gaps": [i for i, c in enumerate(FEATURE_COLS) if "_gap_" in c],
    "all sizes": [i for i, c in enumerate(FEATURE_COLS) if "_sz_" in c],
    "all features": list(range(N_FEAT)),
    "all flags": list(range(N_FEAT, 2 * N_FEAT)),
}


# --- Scoring helpers ---
@torch.no_grad()
def d_prob(model, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """D's per-step probability of 'real', shape (n, step)."""
    return torch.sigmoid(model.discriminator(h, c)).squeeze(-1)

@torch.no_grad()
def data_score(model, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Per-step 'real' probability of data-space windows, via the embedder (the real-data path)."""
    return d_prob(model, model.embedder(x, c), c)

def hold_by_flags(x: torch.Tensor) -> torch.Tensor:
    """Sample binary flags from the flag channels and hold features where the flag is 0 (like decode)."""
    flags = torch.bernoulli(x[..., N_FEAT:].clamp(0, 1))
    flags[:, 0] = 1
    steps = torch.arange(x.shape[1], device=x.device).view(1, -1, 1).expand_as(flags)
    last = torch.cummax(torch.where(flags.bool(), steps, torch.zeros_like(steps)), dim=1).values
    feats = torch.gather(x[..., :N_FEAT], 1, last)
    return torch.cat([feats, flags], dim=-1)

def saliency(model, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """|d (sum of D logits) / d x| for real windows, shape (n, step, channel)."""
    model.train()   # cuDNN GRUs only run backward in train mode; no dropout, so outputs are unchanged
    x_req = x.clone().requires_grad_(True)
    logits = model.discriminator(model.embedder(x_req, c), c).sum()
    (grad,) = torch.autograd.grad(logits, x_req)
    model.eval()
    return grad.abs()


# --- Main ---
def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    Path("figures").mkdir(exist_ok=True)

    # Model and real training windows
    model, scalers, cfg = load_model(CHECKPOINT, device=DEVICE)
    model.eval()
    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes,
                       cfg.train_frac, cfg.val_frac, cfg.test_frac)
    windows, _, labels = encode(books["train"], scalers, cfg.window_length)
    pool = np.flatnonzero(labels == TICKER)
    idx = np.sort(rng.choice(pool, size=min(NUM_WINDOWS, len(pool)), replace=False))

    x = torch.from_numpy(windows[idx]).float().to(DEVICE)
    n, seq_len = x.shape[:2]
    c = model.one_hot(torch.full((n,), TICKER, device=DEVICE))

    # Real and synthetic latents / windows, exactly as in train_joint
    with torch.no_grad():
        h = model.embedder(x, c)
        e_hat = model.generator(model.noise(n, seq_len), c)
        h_hat = model.supervisor(e_hat, c)
        x_hat = model.recovery(h_hat, c)

    p_real, p_hhat, p_ehat = d_prob(model, h, c), d_prob(model, h_hat, c), d_prob(model, e_hat, c)
    p_xhat = data_score(model, x_hat, c)
    base = p_real.mean().item()

    print(f"\nCheckpoint {CHECKPOINT}  ticker {cfg.tickers[TICKER]}  windows {n}")

    # 1. Per-step D score
    print("\n[1] D 'real' probability by step")
    print(f"  {'':<22}{'t=0':>8}{'t=1-4':>8}{'t>=5':>8}{'all':>8}")
    for name, p in (("real h", p_real), ("synthetic h_hat", p_hhat), ("synthetic e_hat", p_ehat),
                    ("embed(x_hat)", p_xhat)):
        print(f"  {name:<22}{p[:, 0].mean():8.3f}{p[:, 1:5].mean():8.3f}"
              f"{p[:, 5:].mean():8.3f}{p.mean():8.3f}")
    gap0 = (p_real[:, 0] - p_hhat[:, 0]).mean().item()
    gap_rest = (p_real[:, 5:] - p_hhat[:, 5:]).mean().item()
    print(f"  real - h_hat gap: t=0 {gap0:.3f}   t>=5 {gap_rest:.3f}"
          f"   -> {'start constraint' if gap0 > 2 * gap_rest else 'spread over the window (dynamics)'}")

    # 2. Latent and data-space smoothness
    dh_real = (h[:, 1:] - h[:, :-1]).norm(dim=-1).flatten().cpu().numpy()
    dh_fake = (h_hat[:, 1:] - h_hat[:, :-1]).norm(dim=-1).flatten().cpu().numpy()
    small = np.quantile(dh_real, 0.25)
    print("\n[2] Latent step size |h_t - h_{t-1}|")
    print(f"  {'':<22}{'p10':>9}{'p50':>9}{'p90':>9}{'< real p25':>12}")
    for name, d in (("real h", dh_real), ("synthetic h_hat", dh_fake)):
        print(f"  {name:<22}{np.quantile(d, 0.1):9.4f}{np.quantile(d, 0.5):9.4f}"
              f"{np.quantile(d, 0.9):9.4f}{(d < small).mean():12.2%}")
    real_held = (x[:, 1:, :N_FEAT] == x[:, :-1, :N_FEAT]).float().mean().item()
    fake_held = ((x_hat[:, 1:, :N_FEAT] - x_hat[:, :-1, :N_FEAT]).abs() < 1e-4).float().mean().item()
    print(f"  data-space held feature-steps: real {real_held:.2%}  synthetic {fake_held:.2%}")
    print(f"  flag channels: real mean {x[..., N_FEAT:].mean():.3f} (binary)   "
          f"synthetic mean {x_hat[..., N_FEAT:].mean():.3f}, "
          f"share in (0.05, 0.95): {((x_hat[..., N_FEAT:] > 0.05) & (x_hat[..., N_FEAT:] < 0.95)).float().mean():.2%}")

    # 3. Channel swap
    drops = []
    for j in range(2 * N_FEAT):
        xs = x.clone()
        xs[..., j] = x_hat[..., j]
        drops.append(base - data_score(model, xs, c).mean().item())
    drops = np.array(drops)
    print(f"\n[3] Swap one synthetic channel into real windows (base real score {base:.3f})")
    print("  Largest drops = what D keys on:")
    for j in np.argsort(-drops)[:15]:
        print(f"    {CHANNELS[j]:<20}{drops[j]:+.4f}")
    print("  Groups:")
    for name, cols in GROUPS.items():
        xs = x.clone()
        xs[..., cols] = x_hat[..., cols]
        print(f"    {name:<20}{base - data_score(model, xs, c).mean().item():+.4f}")
    xs = x.clone()
    xs[:, 0] = x_hat[:, 0]
    print(f"    {'step 0 only':<20}{base - data_score(model, xs, c).mean().item():+.4f}")

    # 4. Gradient saliency on real windows
    sal = saliency(model, x, c)
    sal_ch = sal.mean(dim=(0, 1)).cpu().numpy()
    sal_t = sal.mean(dim=(0, 2)).cpu().numpy()
    print("\n[4] Gradient saliency on real windows (top channels)")
    for j in np.argsort(-sal_ch)[:10]:
        print(f"    {CHANNELS[j]:<20}{sal_ch[j]:.4e}")
    print(f"  by step: t=0 {sal_t[0]:.3e}   mean t>=1 {sal_t[1:].mean():.3e}")

    # 5. Dynamics interventions
    with torch.no_grad():
        jitter = (x_hat[:, 1:, :N_FEAT] - x_hat[:, :-1, :N_FEAT]).abs().mean(dim=(0, 1))   # per feature
        held = (x[:, 1:, N_FEAT:] == 0).float()
        xj = x.clone()
        xj[:, 1:, :N_FEAT] += torch.randn_like(xj[:, 1:, :N_FEAT]) * jitter * held
        jittered = data_score(model, xj, c).mean().item()
        frozen = data_score(model, hold_by_flags(x_hat), c).mean().item()
    print("\n[5] Dynamics interventions")
    print(f"  real                          {base:.3f}")
    print(f"  real + synthetic-size jitter  {jittered:.3f}   (big drop -> jitter is the tell)")
    print(f"  synthetic (embed(x_hat))      {p_xhat.mean().item():.3f}")
    print(f"  synthetic, frozen by flags    {frozen:.3f}   (big rise -> jitter is the tell)")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    for name, p in (("real h", p_real), ("h_hat", p_hhat), ("e_hat", p_ehat)):
        ax.plot(p.mean(dim=0).cpu().numpy(), label=name)
    ax.set(title="D 'real' probability by step", xlabel="Step", ylim=(0, 1))
    ax.legend()

    ax = axes[0, 1]
    bins = np.linspace(0, np.quantile(np.concatenate([dh_real, dh_fake]), 0.99), 60)
    ax.hist(dh_real, bins=bins, alpha=0.6, label="real h", density=True)
    ax.hist(dh_fake, bins=bins, alpha=0.6, label="h_hat", density=True)
    ax.set(title="Latent step size |h_t - h_{t-1}|", yscale="log")
    ax.legend()

    ax = axes[1, 0]
    top = np.argsort(-drops)[:15][::-1]
    ax.barh([CHANNELS[j] for j in top], drops[top])
    ax.set(title="Drop in D real-score when channel is swapped to synthetic")

    ax = axes[1, 1]
    top = np.argsort(-sal_ch)[:15][::-1]
    ax.barh([CHANNELS[j] for j in top], sal_ch[top])
    ax.set(title="Gradient saliency on real windows")

    fig.tight_layout()
    out = Path(f"figures/disc_diagnostics_{CHECKPOINT.stem}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()