import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from dataset_v4 import prepare_dataset_v4

TICK = 0.01
OUT = Path("outputs_v5_41_price_gaps")
OUT.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

train = norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
)
real = norm.inverse_transform(
    np.asarray(data["val"][:512], dtype=np.float64)
)

source = np.load(
    "outputs_v5_38_integrated/generated_lob.npz"
)
original = source["relative_orderbook"].astype(np.float64)

def prices(book):
    return book[..., 0:40:4], book[..., 2:40:4]

def gaps(book):
    ask, bid = prices(book)
    return (
        np.diff(ask, axis=-1),
        -np.diff(bid, axis=-1)
    )

train_ask_gap, train_bid_gap = gaps(train)
fake_ask_gap, fake_bid_gap = gaps(original)

def mean_per_gap(x):
    return x.mean(axis=(0, 1))

# Calibrate using FULL TRAIN only.
# No Validation-based fit.
ask_target = mean_per_gap(train_ask_gap)
bid_target = mean_per_gap(train_bid_gap)

ask_source = mean_per_gap(fake_ask_gap)
bid_source = mean_per_gap(fake_bid_gap)

ask_ratio = ask_target / np.maximum(ask_source, 1e-8)
bid_ratio = bid_target / np.maximum(bid_source, 1e-8)

def calibrated_gaps(g, ratio):
    # Quantize adjacent prices to integer tick
    # increments, with at least one tick.
    n_ticks = np.maximum(
        np.rint(g * ratio / TICK).astype(np.int64),
        1
    )
    return n_ticks * TICK

new_ask_gap = calibrated_gaps(
    fake_ask_gap, ask_ratio
)
new_bid_gap = calibrated_gaps(
    fake_bid_gap, bid_ratio
)

updated = original.copy()
ask, bid = prices(original)

# Preserve best bid, best ask, midpoint and spread.
new_ask = np.concatenate(
    [
        ask[..., :1],
        ask[..., :1] + np.cumsum(new_ask_gap, axis=-1)
    ],
    axis=-1
)

new_bid = np.concatenate(
    [
        bid[..., :1],
        bid[..., :1] - np.cumsum(new_bid_gap, axis=-1)
    ],
    axis=-1
)

updated[..., 0:40:4] = new_ask
updated[..., 2:40:4] = new_bid

def relative_vwap(book):
    ask, bid = prices(book)
    av = np.maximum(np.expm1(book[..., 1:40:4]), 0)
    bv = np.maximum(np.expm1(book[..., 3:40:4]), 0)
    mid = (ask[..., 0] + bid[..., 0]) / 2

    a = (
        (ask * av).sum(-1)
        / np.maximum(av.sum(-1), 1e-12)
    ) - mid

    b = mid - (
        (bid * bv).sum(-1)
        / np.maximum(bv.sum(-1), 1e-12)
    )
    return a, b

real_a, real_b = relative_vwap(real)

def metrics(book):
    a, b = relative_vwap(book)
    ap, bp = prices(book)

    return {
        "ask_vwap_mean": float(a.mean()),
        "bid_vwap_distance_mean": float(b.mean()),
        "ask_vwap_ks": float(
            ks_2samp(real_a.ravel(), a.ravel()).statistic
        ),
        "bid_vwap_ks": float(
            ks_2samp(real_b.ravel(), b.ravel()).statistic
        ),
        "ask_vwap_wasserstein": float(
            wasserstein_distance(real_a.ravel(), a.ravel())
        ),
        "bid_vwap_wasserstein": float(
            wasserstein_distance(real_b.ravel(), b.ravel())
        ),
        "ask10_distance_mean": float(
            (ap[..., -1] - (ap[..., 0] + bp[..., 0])/2).mean()
        ),
        "bid10_distance_mean": float(
            ((ap[..., 0] + bp[..., 0])/2 - bp[..., -1]).mean()
        )
    }

old_ask, old_bid = prices(original)
new_ask, new_bid = prices(updated)

checks = {
    "best_ask_preserved": bool(
        np.allclose(new_ask[..., 0], old_ask[..., 0])
    ),
    "best_bid_preserved": bool(
        np.allclose(new_bid[..., 0], old_bid[..., 0])
    ),
    "asks_ascending": bool(
        np.all(np.diff(new_ask, axis=-1) > 0)
    ),
    "bids_descending": bool(
        np.all(np.diff(new_bid, axis=-1) < 0)
    ),
    "volumes_preserved": bool(
        np.array_equal(
            updated[..., 1:40:2],
            original[..., 1:40:2]
        )
    ),
}

results = {
    "train_mean_ask_gaps": ask_target.tolist(),
    "train_mean_bid_gaps": bid_target.tolist(),
    "generated_mean_ask_gaps": ask_source.tolist(),
    "generated_mean_bid_gaps": bid_source.tolist(),
    "ask_scale_ratios": ask_ratio.tolist(),
    "bid_scale_ratios": bid_ratio.tolist(),
    "real_validation": metrics(real),
    "original_v5_38": metrics(original),
    "calibrated_v5_41": metrics(updated),
    "constraints": checks
}

print("=== V5.41 PRICE GAP CALIBRATION ===")
print(json.dumps(results, indent=2))

features = np.concatenate(
    [
        updated,
        source["log_returns"][..., None].astype(np.float64)
    ],
    axis=-1
)

np.savez_compressed(
    OUT / "generated_lob.npz",
    features=features.astype(np.float32),
    relative_orderbook=updated.astype(np.float32),
    log_returns=source["log_returns"],
    volumes=source["volumes"],
    midprice=source["midprice"]
)

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.41 price gap experiment complete!")
