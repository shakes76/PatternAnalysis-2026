"""What gives the generator away? Probes the TimeGAN discriminator to show where to work.

Each probe asks the trained discriminator D a different question:
  1. When:      D's per-step "real" probability for real and synthetic latents. A gap only at
                t=0 points at the starting conditions; a gap across the window points at dynamics.
  2. Smoothness: step sizes of real vs synthetic latents, and how often each holds still.
  3. What:      swap one synthetic channel (or group) into otherwise-real windows and measure how
                much D's real-score drops. The biggest drops are what D keys on.
  4. Saliency:  which channels D's real-score is most sensitive to on real windows.
  5. Jitter:    add synthetic-sized jitter to real windows' held steps, and freeze synthetic
                windows by their own flags. If jitter is the tell, the first looks fake and the
                second looks more real.

Probes 3-5 score data-space windows through the embedder, D(embedder(x)), which is exactly the
path real data takes in training; synthetic windows use recovery(h_hat) as a stand-in.
The run ends with a ranked summary of where to work, and saves a figure.
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


CHECKPOINT = Path("models/saved/primary.pt")
TICKER: str = "AMZN"        # ticker to probe (must be in the checkpoint's tickers)
SPLIT: str = "train"        # D was trained on this split
NUM_WINDOWS: int = 512      # real and synthetic windows each
TOP_CHANNELS: int = 10      # channels shown in the per-channel ranking
SEED: int = 0
OUT_PATH = Path(f"figures/discriminator_diagnostics_{CHECKPOINT.stem}.png")

# Model channels: the features, then one change flag per feature
N_FEAT = len(FEATURE_COLS)
CHANNELS = FEATURE_COLS + [f"flag:{c}" for c in FEATURE_COLS]
FLAGS = list(range(N_FEAT, 2 * N_FEAT))
GROUPS = {
    "Mid": [FEATURE_COLS.index("mid_rel")],
    "Spread": [FEATURE_COLS.index("spread")],
    "Price gaps": [i for i, c in enumerate(FEATURE_COLS) if "_gap_" in c],
    "Sizes": [i for i, c in enumerate(FEATURE_COLS) if "_sz_" in c],
    "Change flags": FLAGS,
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


def swap_drop(model, x: torch.Tensor, x_hat: torch.Tensor, c: torch.Tensor, base: float,
              cols: list[int] | slice, steps: slice = slice(None)) -> float:
    """Drop in D's mean real-score when these channels (and steps) of real windows are made synthetic."""
    xs = x.clone()
    xs[:, steps, cols] = x_hat[:, steps, cols]
    return base - data_score(model, xs, c).mean().item()


def hold_by_flags(x: torch.Tensor) -> torch.Tensor:
    """Sample binary flags from the flag channels and hold features where the flag is 0 (like decode)."""
    flags = torch.bernoulli(x[..., N_FEAT:].clamp(0, 1))
    flags[:, 0] = 1
    steps = torch.arange(x.shape[1], device=x.device).view(1, -1, 1).expand_as(flags)
    last = torch.cummax(torch.where(flags.bool(), steps, torch.zeros_like(steps)), dim=1).values
    feats = torch.gather(x[..., :N_FEAT], 1, last)
    return torch.cat([feats, flags], dim=-1)


def saliency(model, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """|d (sum of D logits) / d x| on real windows, shape (n, step, channel)."""
    model.train()   # cuDNN GRUs only run backward in train mode; no dropout, so outputs are unchanged
    x_req = x.clone().requires_grad_(True)
    logits = model.discriminator(model.embedder(x_req, c), c).sum()
    (grad,) = torch.autograd.grad(logits, x_req)
    model.eval()
    return grad.abs()


# --- Plot ---
def style(ax, title: str, xlabel: str, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def barh(ax, labels: list[str], values: np.ndarray) -> None:
    """Horizontal bars, largest at the top, with a zero line."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=["C3" if v > 0 else "C0" for v in values], height=0.7)
    ax.set_yticks(y, labels, fontsize=8)
    ax.axvline(0, color="grey", lw=0.6)


def plot(p_real, p_hhat, p_ehat, top: list[tuple[str, float]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), layout="constrained")

    # Per-step real probability
    ax = axes[0]
    for p, name, colour in ((p_real, "Real", "black"), (p_hhat, "Synthetic (supervised)", "C1"),
                            (p_ehat, "Synthetic (generator)", "C2")):
        ax.plot(p.mean(dim=0).cpu().numpy(), color=colour, lw=1.5, label=name)
    ax.set_ylim(0, 1)
    style(ax, "Discriminator Score by Step", "Time (s)", "P(real)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # The single channels D relies on most
    barh(axes[1], [name for name, _ in top], np.array([v for _, v in top]))
    style(axes[1], "Channel Impact", "Drop in P(real) when swapped")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Main ---
def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    model, scalers, cfg = load_model(CHECKPOINT, device=DEVICE)
    model.eval()
    k = cfg.tickers.index(TICKER)

    # Real windows at the training length, as D saw them
    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes, cfg.train_frac, cfg.val_frac, cfg.test_frac)
    windows, _, labels = encode(books[SPLIT], scalers, cfg.window_length)
    assert windows.shape[-1] == len(CHANNELS), "expected one change flag per feature after the features"
    pool = np.flatnonzero(labels == k)
    idx = np.sort(rng.choice(pool, size=min(NUM_WINDOWS, len(pool)), replace=False))

    x = torch.from_numpy(windows[idx]).float().to(DEVICE)
    n, seq_len = x.shape[:2]
    c = model.one_hot(torch.full((n,), k, device=DEVICE))

    # Real and synthetic latents / windows, exactly as in train_joint
    with torch.no_grad():
        h = model.embedder(x, c)
        e_hat = model.generator(model.noise(n, seq_len), c)
        h_hat = model.supervisor(e_hat, c)
        x_hat = model.recovery(h_hat, c)

    p_real, p_hhat, p_ehat = d_prob(model, h, c), d_prob(model, h_hat, c), d_prob(model, e_hat, c)
    p_xhat = data_score(model, x_hat, c)
    base = p_real.mean().item()
    print(f"{CHECKPOINT}  ticker {TICKER}  {n} windows of {seq_len} steps ({SPLIT} split)")

    # 1. When
    print("\n[1] When: D 'real' probability by step")
    print(f"  {'':<26}{'t=0':>8}{'t=1-4':>8}{'t>=5':>8}{'all':>8}")
    for name, p in (("Real", p_real), ("Synthetic (supervised)", p_hhat), ("Synthetic (generator)", p_ehat),
                    ("Synthetic, re-embedded", p_xhat)):
        print(f"  {name:<26}{p[:, 0].mean():8.3f}{p[:, 1:5].mean():8.3f}{p[:, 5:].mean():8.3f}{p.mean():8.3f}")
    gap0 = (p_real[:, 0] - p_hhat[:, 0]).mean().item()
    gap_rest = (p_real[:, 5:] - p_hhat[:, 5:]).mean().item()

    # 2. Smoothness
    dh_real = (h[:, 1:] - h[:, :-1]).norm(dim=-1).flatten().cpu().numpy()
    dh_fake = (h_hat[:, 1:] - h_hat[:, :-1]).norm(dim=-1).flatten().cpu().numpy()
    small = np.quantile(dh_real, 0.25)
    real_held = (x[:, 1:, :N_FEAT] == x[:, :-1, :N_FEAT]).float().mean().item()
    fake_held = ((x_hat[:, 1:, :N_FEAT] - x_hat[:, :-1, :N_FEAT]).abs() < 1e-4).float().mean().item()
    soft_flags = ((x_hat[..., N_FEAT:] > 0.05) & (x_hat[..., N_FEAT:] < 0.95)).float().mean().item()
    print("\n[2] Smoothness: latent step size |h_t - h_{t-1}|")
    print(f"  {'':<26}{'p10':>9}{'p50':>9}{'p90':>9}{'< real p25':>12}")
    for name, d in (("Real", dh_real), ("Synthetic (supervised)", dh_fake)):
        print(f"  {name:<26}{np.quantile(d, 0.1):9.4f}{np.quantile(d, 0.5):9.4f}"
              f"{np.quantile(d, 0.9):9.4f}{(d < small).mean():12.2%}")
    print(f"  Feature-steps held still:  real {real_held:.2%}   synthetic {fake_held:.2%}")
    print(f"  Change flags:  real mean {x[..., N_FEAT:].mean():.3f} (binary)   synthetic mean "
          f"{x_hat[..., N_FEAT:].mean():.3f}, {soft_flags:.2%} not near 0 or 1")

    # 3. What: channel and group swaps
    drops = np.array([swap_drop(model, x, x_hat, c, base, [j]) for j in range(len(CHANNELS))])
    group_drops = {name: swap_drop(model, x, x_hat, c, base, cols) for name, cols in GROUPS.items()}
    group_drops["Step 0 only"] = swap_drop(model, x, x_hat, c, base, slice(None), slice(0, 1))
    group_drops = dict(sorted(group_drops.items(), key=lambda kv: -kv[1]))
    top = [(CHANNELS[j], drops[j]) for j in np.argsort(-drops)[:TOP_CHANNELS]]
    print(f"\n[3] What: swap synthetic channels into real windows (real score {base:.3f})")
    print("  Groups:")
    for name, d in group_drops.items():
        print(f"    {name:<22}{d:+.4f}")
    print("  Top channels:")
    for name, d in top:
        print(f"    {name:<22}{d:+.4f}")

    # 4. Saliency
    sal = saliency(model, x, c)
    sal_ch = sal.mean(dim=(0, 1)).cpu().numpy()
    sal_t = sal.mean(dim=(0, 2)).cpu().numpy()
    print("\n[4] Saliency: D sensitivity on real windows")
    for name, cols in GROUPS.items():
        print(f"    {name:<22}{sal_ch[cols].mean():.3e}")
    print(f"  By step: t=0 {sal_t[0]:.3e}   mean t>=1 {sal_t[1:].mean():.3e}")

    # 5. Jitter
    with torch.no_grad():
        jitter = (x_hat[:, 1:, :N_FEAT] - x_hat[:, :-1, :N_FEAT]).abs().mean(dim=(0, 1))   # per feature
        held = (x[:, 1:, N_FEAT:] == 0).float()
        xj = x.clone()
        xj[:, 1:, :N_FEAT] += torch.randn_like(xj[:, 1:, :N_FEAT]) * jitter * held
        jittered = data_score(model, xj, c).mean().item()
        frozen = data_score(model, hold_by_flags(x_hat), c).mean().item()
    synth = p_xhat.mean().item()
    print("\n[5] Jitter interventions")
    print(f"  Real                           {base:.3f}")
    print(f"  Real + synthetic-size jitter   {jittered:.3f}   (big drop -> jitter is a tell)")
    print(f"  Synthetic, re-embedded         {synth:.3f}")
    print(f"  Synthetic, frozen by flags     {frozen:.3f}   (big rise -> jitter is a tell)")

    # Summary: where to work, most important first
    print("\nWhere to work")
    print(f"  Overall gap: real {base:.3f} vs synthetic {p_hhat.mean():.3f}")
    print(f"  Timing: {'starting conditions (gap concentrated at t=0)' if gap0 > 2 * gap_rest else 'dynamics across the whole window'}"
          f"  (gap t=0 {gap0:+.3f}, t>=5 {gap_rest:+.3f})")
    jitter_effect = (base - jittered) + (frozen - synth)
    print(f"  Jitter: {'a likely tell' if jitter_effect > 0.05 else 'not a major tell'}"
          f"  (combined effect {jitter_effect:+.3f})")
    print("  Feature groups, by how much they give the generator away:")
    for rank, (name, d) in enumerate(((n, d) for n, d in group_drops.items() if d > 0), start=1):
        print(f"    {rank}. {name:<20}{d:+.4f}")

    plot(p_real, p_hhat, p_ehat, top, OUT_PATH)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()