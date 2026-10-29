import json
from pathlib import Path
import numpy as np

OUT = Path("outputs_v5_39_audit")
OUT.mkdir(exist_ok=True)

data = np.load(
    "outputs_v5_38_integrated/generated_lob.npz"
)

features = data["features"].astype(np.float64)
book = data["relative_orderbook"].astype(np.float64)
returns = data["log_returns"].astype(np.float64)
volume = data["volumes"].astype(np.float64)

assert features.shape == (512, 24, 41)
assert book.shape == (512, 24, 40)

ask = book[..., 0:40:4]
bid = book[..., 2:40:4]

# V5.21 used an absolute reference price of 100.
mid_from_book = (
    (ask[..., 0] + bid[..., 0]) / 2 + 100.0
)

reconstructed_returns = np.diff(
    np.log(mid_from_book), axis=1
)

return_error = np.abs(
    reconstructed_returns - returns[:, 1:]
)

spread = ask[..., 0] - bid[..., 0]
tick_float = spread / 0.01

results = {
    "shape": list(features.shape),
    "return_max_abs_error": float(
        return_error.max()
    ),
    "return_mean_abs_error": float(
        return_error.mean()
    ),
    "return_consistent_at_1e_6": bool(
        np.all(return_error < 1e-6)
    ),
    "positive_absolute_midprice": bool(
        np.all(mid_from_book > 0)
    ),
    "positive_spread": bool(
        np.all(spread > 0)
    ),
    "ascending_asks": bool(
        np.all(np.diff(ask, axis=-1) > 0)
    ),
    "descending_bids": bool(
        np.all(np.diff(bid, axis=-1) < 0)
    ),
    "nonnegative_volumes": bool(
        np.all(volume >= 0)
    ),
    "tick_spread_max_rounding_error": float(
        np.abs(
            tick_float - np.rint(tick_float)
        ).max()
    ),
    "volume_feature_max_error": float(
        np.abs(
            np.log1p(volume)
            - book[..., 1:40:2]
        ).max()
    ),
    "features_match_book": bool(
        np.allclose(
            features[..., :40], book,
            atol=1e-6
        )
    ),
    "features_match_returns": bool(
        np.allclose(
            features[..., 40], returns,
            atol=1e-6
        )
    ),
}

print("=== V5.39 INTEGRATION AUDIT ===")
print(json.dumps(results, indent=2))

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.39 audit complete!")
