"""Real-vs-real depth SSIM: the reference the synthetic SSIM should be judged against.

Pairs real windows at random, exactly as predict.py pairs real with synthetic
windows, and scores them with the same mean_ssim. Each window in a pair is
re-anchored to start at its partner's mid, as synthetic windows are anchored to
their real partner's reference, so the score compares depth shape, not price level.
"""
import tomllib
from pathlib import Path

import numpy as np

from common.metrics import mean_ssim
from common.misc import Config
from dataset import BOOK_COLS
from dataset import load_books


WINDOW_LENGTH: int = 128       # same as predict.py
NUM_SAMPLES: int = 500         # pairs per draw, per ticker (same as predict.py)
NUM_DRAWS: int = 5             # independent random pairings, to show the spread
SPLIT: str = "train"           # same split predict.py compares against
SEED: int = 48

KEYS = {
    "ask_px": [c for c in BOOK_COLS if "ask_px" in c],
    "ask_sz": [c for c in BOOK_COLS if "ask_sz" in c],
    "bid_px": [c for c in BOOK_COLS if "bid_px" in c],
    "bid_sz": [c for c in BOOK_COLS if "bid_sz" in c],
}


def main() -> None:
    rng = np.random.default_rng(SEED)
    with open(Path("config.toml"), "rb") as f:
        cfg = Config(**tomllib.load(f))

    splits = load_books(cfg.tickers, resample_freq=cfg.resample_freq, trim_regimes=cfg.trim_regimes,
                        train_frac=cfg.train_frac, val_frac=cfg.val_frac, test_frac=cfg.test_frac)

    for ticker, ticker_books in splits[SPLIT].groupby("ticker", sort=False):
        # Every stride-1 window of this ticker's book, as (window, step, level) arrays
        windows = {key: np.lib.stride_tricks.sliding_window_view(
                       ticker_books[cols].to_numpy(np.int64), WINDOW_LENGTH, axis=0).transpose(0, 2, 1)
                   for key, cols in KEYS.items()}
        n = len(windows["ask_px"])

        scores = []
        for _ in range(NUM_DRAWS):
            # Two disjoint random sets of windows, paired up
            idx = rng.permutation(n)[:2 * NUM_SAMPLES]
            a, b = idx[:NUM_SAMPLES], idx[NUM_SAMPLES:]
            book_a = {key: w[a].copy() for key, w in windows.items()}
            book_b = {key: w[b].copy() for key, w in windows.items()}

            # Re-anchor b to start at a's mid (whole ticks), like synthetic windows in predict.py
            mid_a = (book_a["ask_px"][:, 0, 0] + book_a["bid_px"][:, 0, 0]) / 2
            mid_b = (book_b["ask_px"][:, 0, 0] + book_b["bid_px"][:, 0, 0]) / 2
            shift = np.rint(mid_a - mid_b).astype(np.int64)[:, None, None]
            book_b["ask_px"] += shift
            book_b["bid_px"] += shift

            scores.append(mean_ssim(book_a, book_b))

        scores = np.array(scores)
        print(f"{ticker}: real-vs-real depth SSIM {scores.mean():.4f} ± {scores.std():.4f}   "
              f"(range {scores.min():.4f}-{scores.max():.4f}, {NUM_DRAWS} draws of {NUM_SAMPLES} pairs)")


if __name__ == "__main__":
    main()