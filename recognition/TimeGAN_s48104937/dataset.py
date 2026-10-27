from __future__ import annotations

import numpy as np
import pandas as pd

from common.utils import LogMinMaxScaler

TICKERS: list[str] = ["AMZN", "AAPL", "GOOG", "INTC", "MSFT"]
BOOK_PATH = "data/{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
MSG_PATH = "data/{ticker}_2012-06-21_34200000_57600000_message_10.csv"

# Message file
MSG_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]

# Orderbook file
DEPTH = 10
LEVELS = range(1, DEPTH + 1)
BOOK_COLS = [f"{s}_{k}_{l}" for l in LEVELS for s in ("ask", "bid") for k in ("px", "sz")]
ASK_COLS = [f"ask_{k}_{l}" for l in LEVELS for k in ("px", "sz")]
BID_COLS = [f"bid_{k}_{l}" for l in LEVELS for k in ("px", "sz")]
PX_COLS = [f"{s}_px_{l}" for l in LEVELS for s in ("ask", "bid")]
SZ_COLS = [f"{s}_sz_{l}" for l in LEVELS for s in ("ask", "bid")]
ASK_PX_COLS = [f"ask_px_{l}" for l in LEVELS]
ASK_SZ_COLS = [f"ask_sz_{l}" for l in LEVELS]
BID_PX_COLS = [f"bid_px_{l}" for l in LEVELS]
BID_SZ_COLS = [f"bid_sz_{l}" for l in LEVELS]
PX_TICK = 100

# --- Model features ---
FEATURE_COLS = (
    ["mid_rel", "spread"]
    + [f"{s}_gap_{l}" for l in range(2, DEPTH + 1) for s in ("ask", "bid")]
    + SZ_COLS
)
FLAG_GROUPS = [[i] for i in range(len(FEATURE_COLS))]


def load_books(tickers: list[str], resample_freq: str | None = None) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        books = pd.read_csv(BOOK_PATH.format(ticker=ticker), header=None, names=BOOK_COLS)
        msgs = pd.read_csv(MSG_PATH.format(ticker=ticker), header=None, names=MSG_COLS)

        is_valid = msgs["event_type"] <= 4
        books = books[is_valid]
        books[PX_COLS] //= PX_TICK
        books.index = pd.to_timedelta(msgs.loc[is_valid, "time"], unit="s")
        books = books[(books[SZ_COLS] > 0).all(axis=1)]

        if resample_freq:
            books = books.resample(resample_freq).last().ffill()

        books = books.astype("int64")
        books["ticker"] = ticker
        frames.append(books)

    return pd.concat(frames)


def split_books(
        books: pd.DataFrame,
        train_frac: float,
        val_frac: float,
        test_frac: float,
) -> dict[str, pd.DataFrame]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1")

    parts = {"train": [], "val": [], "test": []}
    for _, ticker_books in books.groupby("ticker", sort=False):
        n = len(ticker_books)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        parts["train"].append(ticker_books.iloc[:train_end])
        parts["val"].append(ticker_books.iloc[train_end:val_end])
        parts["test"].append(ticker_books.iloc[val_end:])
    return {split: pd.concat(frames) for split, frames in parts.items()}


def encode(
        books: pd.DataFrame,
        scaler: LogMinMaxScaler,
        window_length: int,
        stride: int = 1,
        fit: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    # Windows of raw book values, per ticker
    windows = []
    for _, ticker_books in books.groupby("ticker", sort=False):
        values = ticker_books[BOOK_COLS].to_numpy(float)
        w = np.lib.stride_tricks.sliding_window_view(values, window_length, axis=0)
        windows.append(w[::stride].transpose(0, 2, 1))
    w = dict(zip(BOOK_COLS, np.moveaxis(np.concatenate(windows), -1, 0)))

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

    if fit is True:
        scaler.fit(features)
    scaled = scaler.transform(features)

    # Change flags
    changed = np.ones(features.shape, dtype=bool)
    changed[:, 1:] = features[:, 1:] != features[:, :-1]
    flags = np.stack([changed[..., g].any(axis=-1) for g in FLAG_GROUPS], axis=-1)

    return np.concatenate([scaled, flags], axis=-1), refs


def decode(
        windows: np.ndarray,
        refs: np.ndarray,
        scaler: LogMinMaxScaler,
        rng: np.random.Generator,
) -> pd.DataFrame:
    n, seq_len = windows.shape[:2]
    n_feat = len(FEATURE_COLS)

    # Unscale
    features = scaler.inverse_transform(windows[..., :n_feat])

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

    index = pd.MultiIndex.from_product([range(n), range(seq_len)], names=["window", "step"])
    values = np.stack([books[c] for c in BOOK_COLS], axis=-1).reshape(n * seq_len, -1)
    return pd.DataFrame(values, columns=BOOK_COLS, index=index).round().astype("int64")
