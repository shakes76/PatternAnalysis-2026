"""Data loading and preprocessing for LOBSTER AMZN Level-10 order book data.

Pipeline:
    raw LOBSTER csv -> fixed time grid snapshots -> per-snapshot features
    -> chronological train/val/test split -> z-score (train statistics only)
    -> overlapping windows -> torch Dataset / DataLoader

Feature vector per time step (41 values, see `FEATURE_NAMES`):
    0       log mid-price return since the previous grid step
    1..10   log1p(ask distance from mid, in ticks) for levels 1..10
    11..20  log1p(bid distance from mid, in ticks) for levels 1..10
    21..30  log1p(ask volume) for levels 1..10
    31..40  log1p(bid volume) for levels 1..10

Prices are stored as distances from the mid-price, so the microstructure rules
become sign/ordering conditions on the features instead of being baked into the
representation: a generated book is crossed when ask_dist_1 + bid_dist_1 <= 0,
and a price ladder is invalid when the distances are not increasing with level.
This lets us measure whether a model learns the rules rather than enforcing them
by construction.

Usage: python dataset.py <data_dir>   (prints a summary and a round-trip check)
"""
import sys

import numpy as np
import torch
from scipy.special import ndtr, ndtri
from torch.utils.data import DataLoader, Dataset

PREFIX = "AMZN_2012-06-21_34200000_57600000_"
LEVELS = 10
PRICE_SCALE = 1e4   # LOBSTER stores prices in units of 1/10,000 dollar
TICK = 0.01         # minimum price increment in dollars
N_FEATURES = 1 + 4 * LEVELS
FEATURE_NAMES = (
    ["mid_return"]
    + [f"ask_dist_{i}" for i in range(1, LEVELS + 1)]
    + [f"bid_dist_{i}" for i in range(1, LEVELS + 1)]
    + [f"ask_vol_{i}" for i in range(1, LEVELS + 1)]
    + [f"bid_vol_{i}" for i in range(1, LEVELS + 1)]
)


def load_lob(data_dir):
    """Load event times (seconds after midnight) and the raw order book.

    The orderbook file has 4 columns per level: ask price, ask size,
    bid price, bid size. Row i is the book state after event i.
    """
    book = np.loadtxt(f"{data_dir}/{PREFIX}orderbook_10.csv", delimiter=",")
    times = np.loadtxt(f"{data_dir}/{PREFIX}message_10.csv", delimiter=",", usecols=0)
    return times, book


def resample_to_grid(times, book, step=1.0):
    """Sample the book on a fixed time grid using the last event at or before each time.

    The raw event stream is irregular and ~90% of events do not move the mid-price,
    so a fixed grid gives the model evenly spaced steps.
    """
    grid = np.arange(times[0], times[-1], step)
    idx = np.searchsorted(times, grid, side="right") - 1  # last event <= grid time
    return grid, book[idx]


def book_to_levels(book):
    """Split a (K, 40) book array into ask/bid prices (dollars) and volumes."""
    ask_p = book[:, 0::4] / PRICE_SCALE
    ask_v = book[:, 1::4]
    bid_p = book[:, 2::4] / PRICE_SCALE
    bid_v = book[:, 3::4]
    return ask_p, ask_v, bid_p, bid_v


def decode_distances(block, ladder="levels"):
    """Price distances in ticks from a (..., 10) block of raw distance features.

    levels:     feature = log1p(distance from mid at that level)
    increments: first feature = log1p(distance of level 1); the others are
                log1p(gap to the previous level). Real gaps are always >= 1 tick, so a
                model has to output sensible gaps for the ladder to stay ordered; a
                negative decoded gap is a ladder violation (nothing is enforced).
    """
    d = np.expm1(block)
    return np.cumsum(d, axis=-1) if ladder == "increments" else d


def build_features(book, ladder="levels"):
    """Turn (K, 40) book snapshots into a (K, 41) raw feature array.

    Returns the features and the mid-price series (needed to rebuild prices).
    """
    if ladder not in ("levels", "increments"):
        raise ValueError(f"unknown ladder representation '{ladder}'")
    ask_p, ask_v, bid_p, bid_v = book_to_levels(book)
    mid = (ask_p[:, 0] + bid_p[:, 0]) / 2
    ret = np.zeros(len(mid))
    ret[1:] = np.diff(np.log(mid))   # first step has no previous snapshot
    # distances from the mid in ticks (a tick is the smallest price step, $0.01)
    ask_ticks = (ask_p - mid[:, None]) / TICK
    bid_ticks = (mid[:, None] - bid_p) / TICK
    if ladder == "increments":   # keep level 1, replace the rest by gaps between levels
        ask_ticks = np.concatenate([ask_ticks[:, :1], np.diff(ask_ticks, axis=1)], axis=1)
        bid_ticks = np.concatenate([bid_ticks[:, :1], np.diff(bid_ticks, axis=1)], axis=1)
    # log1p shrinks the long right tail of distances and volumes so z-scoring behaves
    ask_dist, bid_dist = np.log1p(ask_ticks), np.log1p(bid_ticks)
    feats = np.concatenate(
        [ret[:, None], ask_dist, bid_dist, np.log1p(ask_v), np.log1p(bid_v)], axis=1
    )
    return feats, mid


def fit_return_transform(train_returns, kind="zscore", clip_q=0.5):
    """Fit an optional transform for the mid-return feature on training data only.

    zscore:   no transform (the usual z-scoring happens later for every feature)
    clip:     winsorise at the clip_q and 100-clip_q percentiles, so a few extreme
              returns cannot dominate the mean/std or the losses
    quantile: map returns to normal scores through their empirical CDF. Ties get their
              mid-rank, so the ~70% exact zeros all map to one value; inverting through
              the empirical quantile function sends a whole band of normal scores back
              to exactly 0, so a generator does not have to output exact zeros.
    """
    if kind not in ("zscore", "clip", "quantile"):
        raise ValueError(f"unknown return transform '{kind}'")
    tf = {"kind": kind}
    if kind == "clip":
        tf["lo"], tf["hi"] = np.percentile(train_returns, [clip_q, 100 - clip_q])
    elif kind == "quantile":
        tf["sorted"] = np.sort(train_returns)
    return tf


def apply_return_transform(r, tf):
    if tf["kind"] == "clip":
        return np.clip(r, tf["lo"], tf["hi"])
    if tf["kind"] == "quantile":
        s = tf["sorted"]
        mid_rank = (np.searchsorted(s, r, "left") + np.searchsorted(s, r, "right")) / 2
        return ndtri(np.clip(mid_rank / len(s), 0.5 / len(s), 1 - 0.5 / len(s)))
    return r


def invert_return_transform(z, tf):
    """Undo apply_return_transform (clipping cannot be undone and is left as is)."""
    if tf["kind"] == "quantile":
        s = tf["sorted"]
        return np.interp(ndtr(z), (np.arange(len(s)) + 0.5) / len(s), s)
    return z


def chronological_split(n_steps, fractions=(0.6, 0.2, 0.2)):
    """Contiguous (start, end) index ranges for train/val/test in time order.

    No shuffling across time: the model only ever trains on earlier data than it
    is validated and tested on, which avoids look-ahead bias.
    """
    assert abs(sum(fractions) - 1.0) < 1e-9
    a = int(n_steps * fractions[0])
    b = int(n_steps * (fractions[0] + fractions[1]))
    return {"train": (0, a), "val": (a, b), "test": (b, n_steps)}


def make_windows(feats, seq_len, stride=1):
    """Overlapping windows of shape (n_windows, seq_len, n_features).

    Windows are built inside one split only, so none straddles a split boundary.
    """
    # one window per start index; the last start leaves room for a full window
    starts = np.arange(0, len(feats) - seq_len + 1, stride)
    return np.stack([feats[s:s + seq_len] for s in starts]).astype(np.float32)


class LOBWindowDataset(Dataset):
    """Windows of normalised order book features, shape (seq_len, n_features)."""

    def __init__(self, windows):
        self.windows = torch.from_numpy(windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return self.windows[i]


def get_dataloaders(data_dir, seq_len=24, step=1.0, stride=1, batch_size=128,
                    fractions=(0.6, 0.2, 0.2), return_transform="zscore", clip_q=0.5,
                    ladder="levels"):
    """Build train/val/test DataLoaders and return them with preprocessing metadata.

    Normalisation statistics (mean/std per feature) come from the training split
    only, so no information from validation or test leaks into training.
    Returns (loaders, info) where info holds the stats and the split ranges.
    """
    times, book = load_lob(data_dir)
    grid, snaps = resample_to_grid(times, book, step)
    feats, mid = build_features(snaps, ladder)
    splits = chronological_split(len(feats), fractions)

    tr0, tr1 = splits["train"]
    # Optional transform of the return feature (fitted on the training split only)
    tf = fit_return_transform(feats[tr0:tr1, 0], return_transform, clip_q)
    model_feats = feats.copy()
    model_feats[:, 0] = apply_return_transform(feats[:, 0], tf)
    mean = model_feats[tr0:tr1].mean(axis=0)
    std = model_feats[tr0:tr1].std(axis=0) + 1e-8   # epsilon guards against constant features
    norm = (model_feats - mean) / std

    loaders = {}
    for name, (s0, s1) in splits.items():
        windows = make_windows(norm[s0:s1], seq_len, stride)
        loaders[name] = DataLoader(
            LOBWindowDataset(windows),
            batch_size=batch_size,
            shuffle=(name == "train"),   # shuffling windows within train is fine
            drop_last=(name == "train"),
        )
    info = {"mean": mean, "std": std, "splits": splits, "grid": grid, "mid": mid,
            "feats_raw": feats, "seq_len": seq_len, "step": step, "return_tf": tf,
            "stride": stride, "ladder": ladder}
    return loaders, info


def denormalise(windows, info):
    """Undo normalisation: (..., n_features) model space -> raw features."""
    raw = np.asarray(windows) * info["std"] + info["mean"]
    raw[..., 0] = invert_return_transform(raw[..., 0], info["return_tf"])
    return raw


def real_windows(info, split):
    """Real raw (untransformed) windows of a split, for use as the evaluation reference."""
    s0, s1 = info["splits"][split]
    return make_windows(info["feats_raw"][s0:s1], info["seq_len"], info["stride"])


def features_to_book(raw, start_mid, ladder="levels"):
    """Rebuild prices (dollars) and volumes from raw features of one window.

    raw: (seq_len, 41) raw features; start_mid: mid-price before the first step.
    Returns ask_p, ask_v, bid_p, bid_v, each (seq_len, 10). Invalid outputs
    (crossed books, non-monotonic ladders) are kept as-is so they can be audited.
    """
    mid = start_mid * np.exp(np.cumsum(raw[:, 0]))
    ask_dist = decode_distances(raw[:, 1:1 + LEVELS], ladder) * TICK
    bid_dist = decode_distances(raw[:, 1 + LEVELS:1 + 2 * LEVELS], ladder) * TICK
    ask_v = np.expm1(raw[:, 1 + 2 * LEVELS:1 + 3 * LEVELS])
    bid_v = np.expm1(raw[:, 1 + 3 * LEVELS:])
    return mid[:, None] + ask_dist, ask_v, mid[:, None] - bid_dist, bid_v


if __name__ == "__main__":
    loaders, info = get_dataloaders(sys.argv[1])
    for name, loader in loaders.items():
        s0, s1 = info["splits"][name]
        print(f"{name}: {len(loader.dataset)} windows, grid steps {s0}-{s1}, "
              f"batch shape {tuple(next(iter(loader)).shape)}")

    # Round-trip check: rebuilding a real window must recover the original book.
    times, book = load_lob(sys.argv[1])
    _, snaps = resample_to_grid(times, book, info["step"])
    ask_p, ask_v, bid_p, bid_v = book_to_levels(snaps)
    s = 1000
    raw = info["feats_raw"][s:s + 24]
    a_p, a_v, b_p, b_v = features_to_book(raw, info["mid"][s - 1])
    print("round-trip max abs error: ask price", np.abs(a_p - ask_p[s:s + 24]).max(),
          "bid price", np.abs(b_p - bid_p[s:s + 24]).max(),
          "ask vol", np.abs(a_v - ask_v[s:s + 24]).max())
    print("real windows crossed books:", int((a_p[:, 0] <= b_p[:, 0]).sum()))
    print("fraction of zero mid-returns on the grid:",
          float((info["feats_raw"][1:, 0] == 0).mean()))
