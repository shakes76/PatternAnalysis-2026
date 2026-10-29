"""What exactly is wrong with spread, mid_rel and flag:mid_rel in the synthetic windows?

Run from the project root. Uses the original dataset.py: windows are
[40 scaled features | 40 change flags], with the flag for feature j at channel 40 + j.

Compares real training windows with freshly generated ones, both decoded with dataset.decode,
so the numbers are what predict.py's metrics see. Raw (pre-hold, pre-rounding) synthetic
values are inspected separately where that tells you whether the problem is in the value
channels or in the flags.

Sections
  A. Spread      : distribution, persistence, transitions, drift over the window, raw grid fit
  B. Mid         : size of moves, bid-ask bounce, volatility clustering, random-walk growth, step 0
  C. Mid flag    : change rate, flag confidence, burstiness (gaps between changes), value/flag
                   agreement, and how mid and spread changes co-occur
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.constants import DEVICE
from common.misc import load_model
from dataset import FEATURE_COLS
from dataset import decode
from dataset import encode
from dataset import load_books


# --- Config ---
CHECKPOINT = Path("models_v1_norm/checkpoints/timegan_joint_epoch010.pt")
TICKER = 0           # ticker index (k-th ticker in the config)
NUM_WINDOWS = 1000   # real and synthetic windows to compare
BURN_IN = 0          # generator steps to discard; 0 matches what the discriminator sees in training
SEED = 0

# Window layout from dataset.encode: [scaled features | one change flag per feature]
N_FEAT = len(FEATURE_COLS)
MID = FEATURE_COLS.index("mid_rel")
SPREAD = FEATURE_COLS.index("spread")
MID_FLAG = N_FEAT + MID
SPREAD_FLAG = N_FEAT + SPREAD


# --- Helpers ---
def top_of_book(books: dict, refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decoded books -> (mid_rel, spread) in ticks, each (window, step)."""
    ask1 = books["ask_px"][..., 0].astype(float)
    bid1 = books["bid_px"][..., 0].astype(float)
    return (ask1 + bid1) / 2 - refs[:, None], ask1 - bid1

def gaps_between_changes(changed: np.ndarray) -> np.ndarray:
    """Steps between consecutive changes within each window; changed is (window, step) bool."""
    gaps = [np.diff(np.flatnonzero(row)) for row in changed]
    gaps = [g for g in gaps if len(g)]
    return np.concatenate(gaps) if gaps else np.array([], dtype=int)

def run_lengths(values: np.ndarray) -> np.ndarray:
    """Lengths of runs of identical consecutive values within each window."""
    out = []
    for row in values:
        breaks = np.flatnonzero(np.diff(row) != 0)
        edges = np.concatenate([[-1], breaks, [len(row) - 1]])
        out.append(np.diff(edges))
    return np.concatenate(out)

def autocorr(x: np.ndarray, lag: int) -> float:
    """Pooled lag autocorrelation of (window, step) series, pairs taken within windows."""
    a, b = x[:, :-lag].ravel(), x[:, lag:].ravel()
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")

def header(title: str) -> None:
    print(f"\n{title}")
    print(f"  {'':<40}{'Real':>10}{'Synth':>10}")

def row(name: str, real: float, synth: float, fmt: str = "{:.3f}") -> None:
    print(f"  {name:<40}{fmt.format(real):>10}{fmt.format(synth):>10}")


# --- Main ---
def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    Path("figures").mkdir(exist_ok=True)

    # Real windows and as many synthetic ones, anchored at the same references
    model, scalers, cfg = load_model(CHECKPOINT, device=DEVICE)
    model.eval()
    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes,
                       cfg.train_frac, cfg.val_frac, cfg.test_frac)
    windows, refs, labels = encode(books["train"], scalers, cfg.window_length)
    pool = np.flatnonzero(labels == TICKER)
    idx = np.sort(rng.choice(pool, size=min(NUM_WINDOWS, len(pool)), replace=False))
    refs, labels = refs[idx], labels[idx]

    real_x = windows[idx]
    with torch.no_grad():
        synth_x = model.sample(len(idx), cfg.window_length + BURN_IN, TICKER).cpu().numpy()[:, BURN_IN:]
    n, seq_len = real_x.shape[:2]

    r_mid, r_spr = top_of_book(decode(real_x, refs, labels, scalers, rng), refs)
    s_mid, s_spr = top_of_book(decode(synth_x, refs, labels, scalers, rng), refs)

    # Raw synthetic values before flags hold them and before rounding
    scaler = list(scalers.values())[TICKER]
    s_raw = scaler.inverse_transform(synth_x[..., :N_FEAT])
    s_raw_mid, s_raw_spr = s_raw[..., MID], s_raw[..., SPREAD]

    print(f"\nCheckpoint {CHECKPOINT}  ticker {cfg.tickers[TICKER]}  windows {n}  steps {seq_len}")

    # ---------------- A. Spread ----------------
    header("[A] Spread (ticks, decoded)")
    for v in range(1, 7):
        row(f"share spread = {v}", (r_spr == v).mean(), (s_spr == v).mean(), "{:.2%}")
    row("share spread >= 7", (r_spr >= 7).mean(), (s_spr >= 7).mean(), "{:.2%}")
    row("share spread <= 0 (crossed/locked)", (r_spr <= 0).mean(), (s_spr <= 0).mean(), "{:.2%}")
    row("mean", r_spr.mean(), s_spr.mean())
    row("std", r_spr.std(), s_spr.std())
    row("P(unchanged step to step)", (np.diff(r_spr) == 0).mean(), (np.diff(s_spr) == 0).mean(), "{:.2%}")
    row("mean run length (steps)", run_lengths(r_spr).mean(), run_lengths(s_spr).mean(), "{:.2f}")
    for v, label in ((1, "= 1"), (2, "= 2")):
        r_from, s_from = r_spr[:, :-1] == v, s_spr[:, :-1] == v
        r_stay = (r_spr[:, 1:][r_from] == v).mean() if r_from.any() else np.nan
        s_stay = (s_spr[:, 1:][s_from] == v).mean() if s_from.any() else np.nan
        row(f"P(stay | spread {label})", r_stay, s_stay, "{:.2%}")
    k = max(1, seq_len // 10)
    row("mean spread, first 10% of steps", r_spr[:, :k].mean(), s_spr[:, :k].mean())
    row("mean spread, last 10% of steps", r_spr[:, -k:].mean(), s_spr[:, -k:].mean())
    off_grid = np.abs(s_raw_spr - np.round(s_raw_spr)) > 0.25
    print(f"  raw synthetic spread off the tick grid (>0.25 tick): {off_grid.mean():.2%}   "
          f"below 1 tick: {(s_raw_spr < 1).mean():.2%}")

    # ---------------- B. Mid ----------------
    r_dmid, s_dmid = np.diff(r_mid, axis=1), np.diff(s_mid, axis=1)
    header("[B] Mid moves (ticks, decoded)")
    for lo, hi, label in ((0, 0, "|dmid| = 0"), (0.5, 0.5, "|dmid| = 0.5"), (1, 1, "|dmid| = 1"),
                          (1.5, 2, "|dmid| 1.5-2"), (2.5, np.inf, "|dmid| > 2")):
        row(label, ((np.abs(r_dmid) >= lo) & (np.abs(r_dmid) <= hi)).mean(),
            ((np.abs(s_dmid) >= lo) & (np.abs(s_dmid) <= hi)).mean(), "{:.2%}")
    r_moves, s_moves = r_dmid[r_dmid != 0], s_dmid[s_dmid != 0]
    row("mean |dmid| given a move", np.abs(r_moves).mean() if r_moves.size else np.nan,
        np.abs(s_moves).mean() if s_moves.size else np.nan)
    row("share up-moves given a move", (r_moves > 0).mean() if r_moves.size else np.nan,
        (s_moves > 0).mean() if s_moves.size else np.nan, "{:.2%}")
    row("dmid autocorr lag 1 (bounce < 0)", autocorr(r_dmid, 1), autocorr(s_dmid, 1))
    for lag in (1, 2, 5):
        row(f"|dmid| autocorr lag {lag} (clustering)", autocorr(np.abs(r_dmid), lag),
            autocorr(np.abs(s_dmid), lag))
    for t in sorted({1, 10, seq_len // 2, seq_len - 1}):
        row(f"std of mid_rel at step {t}", r_mid[:, t].std(), s_mid[:, t].std())
    row("mean |mid_rel| over window", np.abs(r_mid).mean(), np.abs(s_mid).mean())
    print(f"  raw synthetic mid_rel at step 0: mean {s_raw_mid[:, 0].mean():+.3f}  "
          f"std {s_raw_mid[:, 0].std():.3f}   (real is exactly 0)")
    half_off = np.abs(s_raw_mid * 2 - np.round(s_raw_mid * 2)) / 2 > 0.125
    print(f"  raw synthetic mid_rel off the half-tick grid (>0.125 tick): {half_off.mean():.2%}")

    # ---------------- C. Mid flag ----------------
    r_flag = real_x[:, 1:, MID_FLAG]
    s_prob = synth_x[:, 1:, MID_FLAG]
    r_changed, s_changed = r_dmid != 0, s_dmid != 0
    header("[C] Mid change flag")
    row("change rate: real flag / synth flag prob", r_flag.mean(), s_prob.mean(), "{:.2%}")
    row("change rate: decoded books", r_changed.mean(), s_changed.mean(), "{:.2%}")
    for lo, hi in ((0, 0.05), (0.05, 0.5), (0.5, 0.95), (0.95, 1.0001)):
        print(f"  synth flag prob in [{lo:.2f}, {min(hi, 1):.2f}): {((s_prob >= lo) & (s_prob < hi)).mean():.2%}")
    r_gaps, s_gaps = gaps_between_changes(r_changed), gaps_between_changes(s_changed)
    row("mean steps between mid changes", r_gaps.mean() if r_gaps.size else np.nan,
        s_gaps.mean() if s_gaps.size else np.nan, "{:.2f}")
    row("median steps between mid changes", np.median(r_gaps) if r_gaps.size else np.nan,
        np.median(s_gaps) if s_gaps.size else np.nan, "{:.1f}")
    row("p90 steps between mid changes", np.quantile(r_gaps, 0.9) if r_gaps.size else np.nan,
        np.quantile(s_gaps, 0.9) if s_gaps.size else np.nan, "{:.1f}")
    row("share of back-to-back changes (gap 1)", (r_gaps == 1).mean() if r_gaps.size else np.nan,
        (s_gaps == 1).mean() if s_gaps.size else np.nan, "{:.2%}")
    row("change indicator autocorr lag 1", autocorr(r_changed.astype(float), 1),
        autocorr(s_changed.astype(float), 1))
    row("change rate, first 5 steps", r_changed[:, :5].mean(), s_changed[:, :5].mean(), "{:.2%}")
    row("change rate, remaining steps", r_changed[:, 5:].mean(), s_changed[:, 5:].mean(), "{:.2%}")

    # Do the raw synthetic mid values agree with the synthetic flag?
    s_raw_dmid = np.abs(np.diff(s_raw_mid, axis=1))
    low, high = s_prob < 0.2, s_prob > 0.8
    print(f"  raw synth |dmid| where flag prob < 0.2: {s_raw_dmid[low].mean() if low.any() else np.nan:.3f}   "
          f"> 0.8: {s_raw_dmid[high].mean() if high.any() else np.nan:.3f} ticks "
          f"(values should barely move when the flag says no change)")

    # Joint mid / spread changes
    r_sc, s_sc = np.diff(r_spr, axis=1) != 0, np.diff(s_spr, axis=1) != 0
    header("[C] Mid and spread changes together (decoded)")
    for name, r_v, s_v in (("P(both change)", r_changed & r_sc, s_changed & s_sc),
                           ("P(mid only)", r_changed & ~r_sc, s_changed & ~s_sc),
                           ("P(spread only)", ~r_changed & r_sc, ~s_changed & s_sc),
                           ("P(neither)", ~r_changed & ~r_sc, ~s_changed & ~s_sc)):
        row(name, r_v.mean(), s_v.mean(), "{:.2%}")
    row("P(spread change | mid change)", r_sc[r_changed].mean() if r_changed.any() else np.nan,
        s_sc[s_changed].mean() if s_changed.any() else np.nan, "{:.2%}")
    row("P(mid change | spread change)", r_changed[r_sc].mean() if r_sc.any() else np.nan,
        s_changed[s_sc].mean() if s_sc.any() else np.nan, "{:.2%}")

    # ---------------- Figure ----------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    vals = np.arange(0, 9)
    width = 0.4
    ax.bar(vals - width / 2, [(np.clip(r_spr, 0, 8) == v).mean() for v in vals], width, label="real")
    ax.bar(vals + width / 2, [(np.clip(s_spr, 0, 8) == v).mean() for v in vals], width, label="synth")
    ax.set(title="Spread distribution (0 = crossed/locked, 8 = 8+)", xlabel="Spread (ticks)")
    ax.legend()

    ax = axes[0, 1]
    vals = np.arange(-3, 3.5, 0.5)
    ax.bar(vals - 0.1, [(np.clip(r_dmid, -3, 3) == v).mean() for v in vals], 0.2, label="real")
    ax.bar(vals + 0.1, [(np.clip(s_dmid, -3, 3) == v).mean() for v in vals], 0.2, label="synth")
    ax.set(title="Mid move per step (clipped at ±3)", xlabel="dmid (ticks)", yscale="log")
    ax.legend()

    ax = axes[0, 2]
    ax.plot(r_mid.std(axis=0), label="real")
    ax.plot(s_mid.std(axis=0), label="synth")
    ax.set(title="Std of mid_rel by step (random-walk growth)", xlabel="Step")
    ax.legend()

    ax = axes[1, 0]
    bins = np.arange(1, 32)
    ax.hist(r_gaps, bins=bins, alpha=0.6, density=True, label="real")
    ax.hist(s_gaps, bins=bins, alpha=0.6, density=True, label="synth")
    ax.set(title="Steps between mid changes", xlabel="Gap (steps)", yscale="log")
    ax.legend()

    ax = axes[1, 1]
    ax.hist(s_prob.ravel(), bins=50, alpha=0.8, label="synth flag prob")
    ax.axvline(r_flag.mean(), color="k", linestyle="--", label=f"real rate {r_flag.mean():.2f}")
    ax.set(title="Synthetic mid flag probabilities", xlabel="P(mid changes)", yscale="log")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(np.arange(1, seq_len), r_changed.mean(axis=0), label="real")
    ax.plot(np.arange(1, seq_len), s_changed.mean(axis=0), label="synth")
    ax.set(title="Mid change rate by step", xlabel="Step")
    ax.legend()

    fig.tight_layout()
    out = Path(f"figures/mid_spread_diagnostics_{CHECKPOINT.stem}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()