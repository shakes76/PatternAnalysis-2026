import json
from pathlib import Path

import numpy as np
from dataset_v4 import prepare_dataset_v4

OUT = Path("outputs_v5_42_temporal")
OUT.mkdir(exist_ok=True)

N = 512
MAX_LAG = 5

data = prepare_dataset_v4("data")
norm = data["normalizer"]

real = norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
)

source = np.load(
    "outputs_v5_38_integrated/generated_lob.npz"
)

generated = source[
    "relative_orderbook"
].astype(np.float64)

generated_returns = source[
    "log_returns"
].astype(np.float64)

assert real.shape == (N, 24, 41)
assert generated.shape == (N, 24, 40)

def extract(book, returns=None):
    ask = book[..., 0:40:4]
    bid = book[..., 2:40:4]

    ask_vol = np.maximum(
        np.expm1(book[..., 1:40:4]), 0
    )
    bid_vol = np.maximum(
        np.expm1(book[..., 3:40:4]), 0
    )

    mid = (ask[..., 0] + bid[..., 0]) / 2

    ask_vwap = (
        (ask * ask_vol).sum(axis=-1)
        / np.maximum(ask_vol.sum(axis=-1), 1e-12)
    ) - mid

    bid_vwap = mid - (
        (bid * bid_vol).sum(axis=-1)
        / np.maximum(bid_vol.sum(axis=-1), 1e-12)
    )

    if returns is None:
        returns = book[..., 40]

    return {
        "spread": ask[..., 0] - bid[..., 0],
        "return": returns,
        "absolute_return": np.abs(returns),
        "ask_relative_vwap": ask_vwap,
        "bid_relative_vwap_distance": bid_vwap,
        "total_depth": (
            ask_vol.sum(-1) + bid_vol.sum(-1)
        ),
    }

r = extract(real)
g = extract(generated, generated_returns)

# Pool within-sequence lagged pairs.
# This avoids artificial transitions between
# different 24-step sequences.
def acf(x, max_lag=MAX_LAG):
    results = []

    for lag in range(1, max_lag + 1):
        a = x[:, :-lag].reshape(-1)
        b = x[:, lag:].reshape(-1)

        a = a - a.mean()
        b = b - b.mean()

        denominator = np.sqrt(
            np.sum(a*a) * np.sum(b*b)
        )

        value = (
            float(np.sum(a*b) / denominator)
            if denominator > 1e-12 else 0.0
        )

        results.append(value)

    return results

def describe(x):
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "acf_lag_1_to_5": acf(x),
    }

results = {}

print("=== V5.42 TEMPORAL ACF RESULTS ===")

for name in r:
    real_result = describe(r[name])
    gen_result = describe(g[name])

    error = np.abs(
        np.array(real_result["acf_lag_1_to_5"])
        - np.array(gen_result["acf_lag_1_to_5"])
    )

    results[name] = {
        "real": real_result,
        "generated": gen_result,
        "acf_mean_absolute_error": float(
            error.mean()
        ),
    }

    print(f"\n{name.upper()}")
    print(
        "Real ACF:",
        np.round(
            real_result["acf_lag_1_to_5"], 4
        )
    )
    print(
        "Generated ACF:",
        np.round(
            gen_result["acf_lag_1_to_5"], 4
        )
    )
    print(
        "ACF MAE:",
        round(float(error.mean()), 5)
    )

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.42 temporal evaluation complete!")
