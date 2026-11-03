import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]   # TimeGAN_s48104937/
sys.path.insert(0, str(ROOT))                # so `common` imports from the project root

from common.metrics import abs_return_autocorr
from common.metrics import garch_fit
from common.metrics import kl_divergence
from common.metrics import mean_ssim
from common.metrics import mid_returns
from common.metrics import spread
from common.metrics import unchanged_rates
from common.metrics import violation_rates
from common.metrics import vwap_returns
from common.visualizations import plot_book_pair_v2
from common.visualizations import plot_distributions
from common.visualizations import plot_spread_by_step


# --- Data Config ---
BOOK_PATH = str(ROOT / "data/{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv")
MSG_PATH = str(ROOT / "data/{ticker}_2012-06-21_34200000_57600000_message_10.csv")
TICKERS = ["AMZN"]
RESAMPLE_FREQ = "1s"
WINDOW_LENGTH = 128
TRAIN_FRAC = 0.7      # chronological, per ticker; the rest is left for val/test
DEPTH = 10
PX_TICK = 100         # raw LOBSTER prices are $ x 10000; // 100 gives cents (ticks)

# --- Model Config ---
HIDDEN_DIM = 64
NUM_LAYERS = 2

# --- Training Config ---
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 60
SEED = 0

# --- Evaluation Config ---
NUM_EVAL = 1000
SHOWN_WINDOW = 0         # which evaluation window goes in the report figure
CHECKPOINT_PATH = ROOT / "models/checkpoints/baseline.pt"
LOSS_PLOT_PATH = ROOT / "figures/baseline_losses.png"
FIGURE_DIR = ROOT / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Column order: [ask_px x10, ask_sz x10, bid_px x10, bid_sz x10], level 1 first
LEVELS = range(1, DEPTH + 1)
LOBSTER_COLS = [f"{s}_{k}_{l}" for l in LEVELS for s in ("ask", "bid") for k in ("px", "sz")]
BLOCKS = [("ask", "px"), ("ask", "sz"), ("bid", "px"), ("bid", "sz")]
COLS = [f"{s}_{k}_{l}" for s, k in BLOCKS for l in LEVELS]
PX_IDX = np.r_[0:DEPTH, 2 * DEPTH:3 * DEPTH]
SZ_IDX = np.r_[DEPTH:2 * DEPTH, 3 * DEPTH:4 * DEPTH]


# --- Data ---
def load_book(ticker: str) -> pd.DataFrame:
    """One ticker's book: valid events only, prices in ticks, resampled to RESAMPLE_FREQ."""
    books = pd.read_csv(BOOK_PATH.format(ticker=ticker), header=None, names=LOBSTER_COLS)
    msgs = pd.read_csv(MSG_PATH.format(ticker=ticker), header=None,
                       names=["time", "event_type", "order_id", "size", "price", "direction"])
    is_valid = msgs["event_type"] <= 4
    books = books[is_valid]
    px_cols = [c for c in LOBSTER_COLS if "_px_" in c]
    sz_cols = [c for c in LOBSTER_COLS if "_sz_" in c]
    books[px_cols] //= PX_TICK
    books.index = pd.to_timedelta(msgs.loc[is_valid, "time"], unit="s")
    books = books[(books[sz_cols] > 0).all(axis=1)]
    return books.resample(RESAMPLE_FREQ).last().ffill().astype("int64")

def train_windows(tickers: list[str], window_length: int) -> np.ndarray:
    """Every stride-1 window of the chronological training split, shape (N, step, 40)."""
    windows = []
    for ticker in tickers:
        book = load_book(ticker)
        values = book[COLS].to_numpy(np.float32)[:int(len(book) * TRAIN_FRAC)]
        w = np.lib.stride_tricks.sliding_window_view(values, window_length, axis=0)
        windows.append(w.transpose(0, 2, 1))
    return np.ascontiguousarray(np.concatenate(windows))

def to_features(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw windows -> (features, refs): prices relative to the first mid, log1p sizes."""
    refs = (raw[:, 0, 0] + raw[:, 0, 2 * DEPTH]) / 2      # (ask_px_1 + bid_px_1) / 2 at step 0
    feats = raw.copy()
    feats[..., PX_IDX] -= refs[:, None, None]
    feats[..., SZ_IDX] = np.log1p(feats[..., SZ_IDX])
    return feats, refs

def to_books(feats: np.ndarray, refs: np.ndarray) -> dict[str, np.ndarray]:
    """Undo to_features and round to whole ticks/shares.

    Returns {"ask_px", "ask_sz", "bid_px", "bid_sz"}, each int64 of shape
    (window, step, level), the same format dataset.decode produces.
    """
    raw = feats.astype(np.float64)
    raw[..., PX_IDX] += refs[:, None, None]
    raw[..., SZ_IDX] = np.expm1(raw[..., SZ_IDX])
    raw = np.rint(raw).astype("int64")
    return {f"{s}_{k}": raw[..., i * DEPTH:(i + 1) * DEPTH] for i, (s, k) in enumerate(BLOCKS)}

class MinMax:
    """Per-column min-max scaling to [-1, 1]."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray):
        self.lo, self.hi = np.asarray(lo, np.float32), np.asarray(hi, np.float32)

    @classmethod
    def fit(cls, data: np.ndarray) -> "MinMax":
        flat = data.reshape(-1, data.shape[-1])
        return cls(flat.min(0), flat.max(0))

    def transform(self, data: np.ndarray) -> np.ndarray:
        return 2 * (data - self.lo) / (self.hi - self.lo + 1e-8) - 1

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        return (scaled + 1) / 2 * (self.hi - self.lo + 1e-8) + self.lo


# --- Model ---
class Generator(nn.Module):
    """Autoregressive GRU: a Gaussian over the next step, given the steps so far."""

    def __init__(self, n_features: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, 2 * n_features)   # mean and log-variance per column

    def step(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """Inputs (batch, step, F) -> mean, log-variance of each next step, and the GRU state."""
        y, h = self.gru(x, h)
        mean, logvar = self.out(y).chunk(2, dim=-1)
        return mean, logvar.clamp(-10, 5), h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher forcing: predict every step of x from a zero start step and the real steps before it."""
        inputs = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        mean, logvar, _ = self.step(inputs)
        return mean, logvar

    @torch.no_grad()
    def sample(self, n: int, window_length: int) -> torch.Tensor:
        """Generate n windows one step at a time, feeding each sample back in."""
        device = next(self.parameters()).device
        x = torch.zeros(n, 1, self.out.out_features // 2, device=device)   # start step
        h, steps = None, []
        for _ in range(window_length):
            mean, logvar, h = self.step(x, h)
            x = (mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)).clamp(-1, 1)
            steps.append(x)
        return torch.cat(steps, dim=1)

def nll_loss(gen: Generator, x: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood of the real windows under teacher forcing."""
    mean, logvar = gen(x)
    return F.gaussian_nll_loss(mean, x, logvar.exp())


# --- Training ---
def train(gen: Generator, loader: DataLoader, epochs: int) -> dict:
    opt = torch.optim.Adam(gen.parameters(), lr=LR)
    losses = {"nll": []}

    for epoch in range(1, epochs + 1):
        total = 0.0
        for (x,) in loader:
            loss = nll_loss(gen, x.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        losses["nll"].append(total / len(loader))
        print(f"[Baseline {epoch:3d}/{epochs}] NLL {losses['nll'][-1]:.4f}")

    return losses


# --- Plotting ---
def plot_losses(losses: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(losses), figsize=(5 * len(losses), 4), squeeze=False)
    for ax, (name, values) in zip(axes[0], losses.items()):
        ax.plot(values)
        ax.set(title=name, xlabel="Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Generation and Evaluation ---
def sample_books(gen: Generator, scaler: MinMax, refs: np.ndarray, window_length: int) -> dict:
    """Generate len(refs) windows and decode them as books anchored at refs."""
    gen.eval()
    scaled = gen.sample(len(refs), window_length).cpu().numpy()
    gen.train()
    return to_books(scaler.inverse_transform(scaled), refs)

def evaluate(real: dict, synth: dict) -> None:
    """Same metrics as predict.py, real vs baseline."""
    print(f"\nComparing {len(real['ask_px'])} windows against the train split\n")
    print(f"{'Invariant violations':<24}{'Real':>10}{'Baseline':>12}")
    real_v, synth_v = violation_rates(real), violation_rates(synth)
    for key in real_v:
        print(f"  {key:<22}{real_v[key]:>10.2%}{synth_v[key]:>12.2%}")

    print(f"\n{'Unchanged step-to-step':<24}{'Real':>10}{'Baseline':>12}")
    real_u, synth_u = unchanged_rates(real), unchanged_rates(synth)
    for key in real_u:
        print(f"  {key:<22}{real_u[key]:>10.2%}{synth_u[key]:>12.2%}")

    print(f"\nKL spread       {kl_divergence(spread(real), spread(synth), bins=50):.4f}   (target <= 0.1)")
    print(f"KL mid returns  {kl_divergence(mid_returns(real), mid_returns(synth)):.4f}   (target <= 0.1)")
    print(f"KL VWAP returns {kl_divergence(vwap_returns(real), vwap_returns(synth)):.4f}")
    print(f"Depth SSIM      {mean_ssim(real, synth):.4f}   (target > 0.6)")

    print("\n|return| autocorrelation, lags 1-5:")
    print(f"  Real       {np.round(abs_return_autocorr(mid_returns(real))[:5], 3)}")
    print(f"  Baseline   {np.round(abs_return_autocorr(mid_returns(synth))[:5], 3)}")

    print(f"\n{'GARCH(1,1), VWAP ret (bp)':<26}{'alpha':>8}{'beta':>8}{'a+b':>8}{'half-life':>11}")
    for name, book in (("Real", real), ("Baseline", synth)):
        f = garch_fit(vwap_returns(book))
        flag = "" if f["converged"] else "  (not converged)"
        print(f"  {name:<24}{f['alpha']:>8.3f}{f['beta']:>8.3f}{f['persistence']:>8.3f}"
              f"{f['half_life']:>11.1f}{flag}")


# --- Main ---
def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    # Data: the scaler is fitted on the training split only
    raw = train_windows(TICKERS, WINDOW_LENGTH)
    feats, refs = to_features(raw)
    scaler = MinMax.fit(feats)
    loader = DataLoader(TensorDataset(torch.from_numpy(scaler.transform(feats).astype(np.float32))),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Model and training
    n_features = len(COLS)
    gen = Generator(n_features, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses = train(gen, loader, EPOCHS)
    train_time = time.perf_counter() - start

    # Resource profile
    print(f"\nGenerator params {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Training time {train_time / 60:.1f} min")
    if DEVICE.type == "cuda":
        print(f"Peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")

    # Save
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "generator": gen.state_dict(),
        "scaler_lo": scaler.lo.tolist(),
        "scaler_hi": scaler.hi.tolist(),
        "config": {"n_features": n_features, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
                   "window_length": WINDOW_LENGTH, "tickers": TICKERS, "resample_freq": RESAMPLE_FREQ,
                   "train_frac": TRAIN_FRAC, "columns": COLS},
    }, CHECKPOINT_PATH)
    plot_losses(losses, LOSS_PLOT_PATH)

    # Evaluate on random train windows, synthetic ones anchored at the same refs
    idx = rng.choice(len(raw), size=min(NUM_EVAL, len(raw)), replace=False)
    start = time.perf_counter()
    synth = sample_books(gen, scaler, refs[idx], WINDOW_LENGTH)
    print(f"Generation: {len(idx)} windows in {time.perf_counter() - start:.2f} s")
    real = to_books(feats[idx], refs[idx])
    evaluate(real, synth)

    # Aggregate plots, shared with predict.py so they compare directly with TimeGAN's
    plot_spread_by_step(real, synth, FIGURE_DIR / "baseline_spread_by_step.png")
    plot_distributions(real, synth, FIGURE_DIR / "baseline_distributions.png")

    # Report figure: a real window (left) beside a baseline window (right), on a shared
    i = SHOWN_WINDOW
    plot_book_pair_v2(real, synth, FIGURE_DIR / "baseline_real_vs_synth.png", i, i,
                   title=None,
                   col_titles=("Real Sample", "AR-GRU Sample"))
    print(f"\nSaved the window figure and the spread/distribution plots to {FIGURE_DIR}")

if __name__ == "__main__":
    main()
