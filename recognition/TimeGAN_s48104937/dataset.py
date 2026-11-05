from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from components.scaler import LogMinMaxScaler
from common.constants import TICKERS
from common.constants import DEPTH
from common.constants import LEVELS
from common.constants import PX_TICK
from common.constants import BOOK_PATH
from common.constants import MSG_PATH
from common.misc import download_dataset


# Dataset columns
MSG_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]
BOOK_COLS = [f"{s}_{k}_{l}" for l in LEVELS for s in ("ask", "bid") for k in ("px", "sz")]

# Feature columns
SZ_COLS = [f"{s}_sz_{l}" for l in LEVELS for s in ("ask", "bid")]
FEATURE_COLS = (
    ["mid_rel", "spread"]
    + [f"{s}_gap_{l}" for l in range(2, DEPTH + 1) for s in ("ask", "bid")]
    + SZ_COLS
)
FLAG_GROUPS = [[i] for i in range(len(FEATURE_COLS))]
SZ_IDX = [FEATURE_COLS.index(c) for c in SZ_COLS]


# Dataset Functions
def load_books(
        tickers: list[str],
        resample_freq: str | None = None,
        trim_regimes: str | None = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Load, clean and chronologically split each ticker's book.

    Returns {"train", "val", "test"}, each holding every ticker's share of its own day.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1")

    parts = {"train": [], "val": [], "test": []}
    for ticker in tickers:
        if ticker not in TICKERS:
            raise ValueError(f"Invalid ticker {ticker}")

        book_path = Path(BOOK_PATH.format(ticker=ticker))
        msgs_path = Path(MSG_PATH.format(ticker=ticker))
        if not (book_path.is_file() and msgs_path.is_file()):
            download_dataset(ticker, book_path.parent)

        books = pd.read_csv(BOOK_PATH.format(ticker=ticker), header=None, names=BOOK_COLS)
        msgs = pd.read_csv(MSG_PATH.format(ticker=ticker), header=None, names=MSG_COLS)

        is_valid = msgs["event_type"] <= 4
        books = books[is_valid]
        books[[c for c in BOOK_COLS if "px" in c]] //= PX_TICK
        books.index = pd.to_timedelta(msgs.loc[is_valid, "time"], unit="s")

        if trim_regimes:
            trim = pd.Timedelta(trim_regimes)
            books = books.loc[pd.Timedelta("09:30:00") + trim: pd.Timedelta("16:00:00") - trim]

        if resample_freq:
            books = books.resample(resample_freq).last().ffill()

        books = books.astype("int64")
        books["ticker"] = ticker

        # Chronological split of this ticker's day
        n = len(books)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        parts["train"].append(books.iloc[:train_end])
        parts["val"].append(books.iloc[train_end:val_end])
        parts["test"].append(books.iloc[val_end:])

    return {split: pd.concat(frames) for split, frames in parts.items()}

def encode(
        books: pd.DataFrame,
        scalers: dict[str, LogMinMaxScaler],
        window_length: int,
        stride: int = 1,
        fit: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn books into model-space windows, ticker by ticker.

    `scalers` maps each ticker to its own scaler; its key order defines the
    ticker labels (label k = k-th key). With fit=True each scaler is fitted on
    that ticker's windows. Returns (windows, refs, labels): windows of scaled
    features followed by change flags, the reference mid of each window, and
    each window's ticker label.
    """
    tickers = list(scalers)
    all_windows, all_refs, all_labels = [], [], []
    for ticker, ticker_books in books.groupby("ticker", sort=False):
        # Windows of raw book values
        values = ticker_books[BOOK_COLS].to_numpy(float)
        w = np.lib.stride_tricks.sliding_window_view(values, window_length, axis=0)
        w = dict(zip(BOOK_COLS, np.moveaxis(w[::stride].transpose(0, 2, 1), -1, 0)))

        # Features
        refs = (w["ask_px_1"][:, 0] + w["bid_px_1"][:, 0]) / 2
        feats = {
            "mid_rel": (w["ask_px_1"] + w["bid_px_1"]) / 2 - refs[:, None],
            "spread": w["ask_px_1"] - w["bid_px_1"],
        }
        for l in range(2, DEPTH + 1):
            feats[f"ask_gap_{l}"] = w[f"ask_px_{l}"] - w[f"ask_px_{l - 1}"]
            feats[f"bid_gap_{l}"] = w[f"bid_px_{l - 1}"] - w[f"bid_px_{l}"]
        for c in SZ_COLS:
            feats[c] = w[c]
        features = np.stack([feats[c] for c in FEATURE_COLS], axis=-1)

        scaler = scalers[ticker]
        if fit is True:
            scaler.fit(features)
        scaled = scaler.transform(features)

        # Change flags
        changed = np.ones(features.shape, dtype=bool)
        changed[:, 1:] = features[:, 1:] != features[:, :-1]
        flags = np.stack([changed[..., g].any(axis=-1) for g in FLAG_GROUPS], axis=-1)

        all_windows.append(np.concatenate([scaled, flags], axis=-1))
        all_refs.append(refs)
        all_labels.append(np.full(len(refs), tickers.index(ticker)))

    return np.concatenate(all_windows), np.concatenate(all_refs), np.concatenate(all_labels)

def decode(
        windows: np.ndarray,
        refs: np.ndarray,
        labels: np.ndarray,
        scalers: dict[str, LogMinMaxScaler],
        rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Turn model-space windows back into order books.

    Each window is unscaled with the scaler of its ticker label (the k-th key
    of `scalers`). Returns {"ask_px", "ask_sz", "bid_px", "bid_sz"}, each an
    int64 array of shape (window, step, level) with level 0 the best level.
    """
    seq_len = windows.shape[1]
    n_feat = len(FEATURE_COLS)

    # Unscale, each window with its own ticker's scaler
    features = np.empty(windows[..., :n_feat].shape)
    for k, scaler in enumerate(scalers.values()):
        is_ticker = labels == k
        features[is_ticker] = scaler.inverse_transform(windows[is_ticker, :, :n_feat])

    # Sample flags and hold unchanged groups
    probs = windows[..., n_feat:]
    flags = rng.random(probs.shape) < probs
    keep = np.ones(features.shape, dtype=bool)   # features not in a group are never held
    for k, group in enumerate(FLAG_GROUPS):
        keep[..., group] = flags[..., k:k + 1]
    keep[:, 0] = True
    steps = np.arange(seq_len)[None, :, None]
    last_change = np.maximum.accumulate(np.where(keep, steps, 0), axis=1)
    features = np.take_along_axis(features, last_change, axis=1)

    # Rebuild prices
    f = dict(zip(FEATURE_COLS, np.moveaxis(features, -1, 0)))
    mid = refs[:, None] + f["mid_rel"]
    books = {
        "ask_px_1": mid + f["spread"] / 2,
        "bid_px_1": mid - f["spread"] / 2,
    }
    for l in range(2, DEPTH + 1):
        books[f"ask_px_{l}"] = books[f"ask_px_{l - 1}"] + f[f"ask_gap_{l}"]
        books[f"bid_px_{l}"] = books[f"bid_px_{l - 1}"] - f[f"bid_gap_{l}"]
    for c in SZ_COLS:
        books[c] = f[c]

    # Group levels by side and kind, rounded to whole ticks and shares
    return {
        f"{side}_{kind}": np.rint(np.stack([books[f"{side}_{kind}_{l}"] for l in LEVELS], axis=-1)).astype("int64")
        for side in ("ask", "bid") for kind in ("px", "sz")
    }
