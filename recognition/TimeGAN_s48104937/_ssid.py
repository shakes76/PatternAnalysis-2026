"""Depth SSIM of TimeGAN against real, next to the real-vs-real reference.

Real-vs-real pairs real windows at random and re-anchors each pair to start at
the same mid, so the score compares depth shape, not price level. Real-vs-TimeGAN
does the same with synthetic windows decoded at their real partner's reference,
as predict.py does. Both use the same mean_ssim, so they sit on one scale: the
closer TimeGAN is to the real-vs-real score, the better.
"""
from pathlib import Path

import numpy as np
import torch

from common.constants import DEVICE
from common.metrics import mean_ssim
from common.misc import load_model
from dataset import BOOK_COLS
from dataset import decode
from dataset import encode
from dataset import load_books


MODEL_PATH = Path("models/saved/primary.pt")
WINDOW_LENGTH: int | None = None   # None = the model's sample length
NUM_SAMPLES: int = 500             # pairs per draw, per ticker (same as predict.py)
NUM_DRAWS: int = 5                 # independent random draws, to show the spread
SPLIT: str = "train"               # same split predict.py compares against
SEED: int = 48

KEYS = {
    "ask_px": [c for c in BOOK_COLS if "ask_px" in c],
    "ask_sz": [c for c in BOOK_COLS if "ask_sz" in c],
    "bid_px": [c for c in BOOK_COLS if "bid_px" in c],
    "bid_sz": [c for c in BOOK_COLS if "bid_sz" in c],
}


def real_vs_real(ticker_books, length: int, rng: np.random.Generator) -> float:
    """SSIM between two disjoint random sets of real windows, re-anchored to the same mid."""
    windows = {key: np.lib.stride_tricks.sliding_window_view(
                   ticker_books[cols].to_numpy(np.int64), length, axis=0).transpose(0, 2, 1)
               for key, cols in KEYS.items()}
    idx = rng.permutation(len(windows["ask_px"]))[:2 * NUM_SAMPLES]
    a, b = idx[:NUM_SAMPLES], idx[NUM_SAMPLES:]
    book_a = {key: w[a].copy() for key, w in windows.items()}
    book_b = {key: w[b].copy() for key, w in windows.items()}

    # Re-anchor b to start at a's mid (whole ticks), like synthetic windows in predict.py
    mid_a = (book_a["ask_px"][:, 0, 0] + book_a["bid_px"][:, 0, 0]) / 2
    mid_b = (book_b["ask_px"][:, 0, 0] + book_b["bid_px"][:, 0, 0]) / 2
    shift = np.rint(mid_a - mid_b).astype(np.int64)[:, None, None]
    book_b["ask_px"] += shift
    book_b["bid_px"] += shift
    return mean_ssim(book_a, book_b)


def real_vs_timegan(model, scalers, cfg, real_x, refs, labels, k: int, length: int,
                    rng: np.random.Generator) -> float:
    """SSIM between random real windows and TimeGAN windows decoded at the same references."""
    pool = np.flatnonzero(labels == k)
    idx = rng.choice(pool, size=min(NUM_SAMPLES, len(pool)), replace=False)
    with torch.no_grad():
        synth_x = model.sample(n=len(idx), window_length=length + cfg.burn_in, ticker=k).cpu().numpy()
    real = decode(real_x[idx], refs[idx], labels[idx], scalers, rng)
    synth = decode(synth_x[:, cfg.burn_in:], refs[idx], labels[idx], scalers, rng)
    return mean_ssim(real, synth)


def summary(scores: list[float]) -> str:
    s = np.array(scores)
    return f"{s.mean():.4f} ± {s.std():.4f}  (range {s.min():.4f}-{s.max():.4f})"


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    model, scalers, cfg = load_model(MODEL_PATH, device=DEVICE)
    model.eval()
    length = WINDOW_LENGTH or cfg.sample_length

    # Books built with the model's own data settings, so both comparisons see the same split
    books = load_books(cfg.tickers, resample_freq=cfg.resample_freq, trim_regimes=cfg.trim_regimes,
                       train_frac=cfg.train_frac, val_frac=cfg.val_frac, test_frac=cfg.test_frac)[SPLIT]
    real_x, refs, labels = encode(books=books, scalers=scalers, window_length=length)

    print(f"Depth SSIM, {SPLIT} split, {NUM_DRAWS} draws of {NUM_SAMPLES} pairs, {length}-step windows\n")
    for ticker, ticker_books in books.groupby("ticker", sort=False):
        k = cfg.tickers.index(ticker)
        rr = [real_vs_real(ticker_books, length, rng) for _ in range(NUM_DRAWS)]
        rt = [real_vs_timegan(model, scalers, cfg, real_x, refs, labels, k, length, rng)
              for _ in range(NUM_DRAWS)]
        print(f"{ticker}")
        print(f"  Real vs real     {summary(rr)}")
        print(f"  Real vs TimeGAN  {summary(rt)}")
        print(f"  Ratio            {np.mean(rt) / np.mean(rr):.2f}")


if __name__ == "__main__":
    main()
