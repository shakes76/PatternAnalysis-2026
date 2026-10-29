import json
from pathlib import Path
import numpy as np
from dataset_v4 import prepare_dataset_v4

OUT = Path("outputs_v5_40_vwap")
OUT.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

real = norm.inverse_transform(
    np.asarray(data["val"][:512], dtype=np.float64)
)

fake = np.load(
    "outputs_v5_38_integrated/generated_lob.npz"
)["relative_orderbook"].astype(np.float64)

def analyse(book):
    ask_p = book[..., 0:40:4]
    bid_p = book[..., 2:40:4]

    ask_v = np.maximum(np.expm1(book[..., 1:40:4]), 0)
    bid_v = np.maximum(np.expm1(book[..., 3:40:4]), 0)

    mid = (ask_p[..., 0] + bid_p[..., 0]) / 2

    ask_distance = ask_p - mid[..., None]
    bid_distance = mid[..., None] - bid_p

    ask_weights = ask_v / np.maximum(
        ask_v.sum(axis=-1, keepdims=True), 1e-12
    )
    bid_weights = bid_v / np.maximum(
        bid_v.sum(axis=-1, keepdims=True), 1e-12
    )

    return {
        "ask_distance_by_level": ask_distance.mean(
            axis=(0, 1)
        ).round(5).tolist(),
        "bid_distance_by_level": bid_distance.mean(
            axis=(0, 1)
        ).round(5).tolist(),
        "ask_volume_share": ask_weights.mean(
            axis=(0, 1)
        ).round(5).tolist(),
        "bid_volume_share": bid_weights.mean(
            axis=(0, 1)
        ).round(5).tolist(),
        "ask_unweighted_mean_distance": float(
            ask_distance.mean()
        ),
        "bid_unweighted_mean_distance": float(
            bid_distance.mean()
        ),
        "ask_weighted_mean_distance": float(
            (ask_distance * ask_weights).sum(-1).mean()
        ),
        "bid_weighted_mean_distance": float(
            (bid_distance * bid_weights).sum(-1).mean()
        )
    }

results = {
    "real_validation": analyse(real),
    "generated_v5_38": analyse(fake)
}

print("=== V5.40 VWAP COMPONENT DIAGNOSTIC ===")

for name, result in results.items():
    print("\n", name.upper())
    for key, value in result.items():
        print(key, ":", value)

with open(OUT / "components.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.40 component diagnostic complete!")
