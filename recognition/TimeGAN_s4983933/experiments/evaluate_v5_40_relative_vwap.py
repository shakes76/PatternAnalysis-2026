import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from dataset_v4 import prepare_dataset_v4

OUT = Path("outputs_v5_40_vwap")
OUT.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

real = norm.inverse_transform(
    np.asarray(data["val"][:512], dtype=np.float64)
)

generated = np.load(
    "outputs_v5_38_integrated/generated_lob.npz"
)["relative_orderbook"].astype(np.float64)

def calculate(book):
    ask_price = book[..., 0:40:4]
    bid_price = book[..., 2:40:4]

    ask_vol = np.maximum(
        np.expm1(book[..., 1:40:4]), 0
    )
    bid_vol = np.maximum(
        np.expm1(book[..., 3:40:4]), 0
    )

    mid = (ask_price[..., 0] + bid_price[..., 0]) / 2

    ask_total = ask_vol.sum(axis=-1)
    bid_total = bid_vol.sum(axis=-1)

    ask_valid = ask_total > 0
    bid_valid = bid_total > 0

    ask_vwap = np.divide(
        (ask_price * ask_vol).sum(axis=-1),
        ask_total,
        out=np.full_like(mid, np.nan),
        where=ask_valid
    )

    bid_vwap = np.divide(
        (bid_price * bid_vol).sum(axis=-1),
        bid_total,
        out=np.full_like(mid, np.nan),
        where=bid_valid
    )

    return {
        "ask": ask_vwap - mid,
        "bid": bid_vwap - mid,
        "total_volume": ask_total + bid_total,
        "invalid_ask": int((~ask_valid).sum()),
        "invalid_bid": int((~bid_valid).sum()),
    }

r = calculate(real)
g = calculate(generated)

def describe(x):
    x = x[np.isfinite(x)]
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p05": float(np.quantile(x, .05)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, .95)),
    }

def compare(a, b):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    return {
        "wasserstein": float(
            wasserstein_distance(a, b)
        ),
        "ks_statistic": float(
            ks_2samp(a, b).statistic
        ),
    }

results = {
    "real": {
        "ask_relative_vwap": describe(r["ask"]),
        "bid_relative_vwap": describe(r["bid"]),
        "total_volume": describe(r["total_volume"]),
        "invalid_ask": r["invalid_ask"],
        "invalid_bid": r["invalid_bid"],
    },
    "generated": {
        "ask_relative_vwap": describe(g["ask"]),
        "bid_relative_vwap": describe(g["bid"]),
        "total_volume": describe(g["total_volume"]),
        "invalid_ask": g["invalid_ask"],
        "invalid_bid": g["invalid_bid"],
    },
    "comparison": {
        "ask": compare(r["ask"], g["ask"]),
        "bid": compare(r["bid"], g["bid"]),
        "total_volume": compare(
            r["total_volume"], g["total_volume"]
        ),
    },
}

print("=== V5.40 RELATIVE VWAP RESULTS ===")
print(json.dumps(results, indent=2))

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.40 relative VWAP complete!")
