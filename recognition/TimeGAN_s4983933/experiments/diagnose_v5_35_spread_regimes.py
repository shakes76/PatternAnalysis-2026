import json
from pathlib import Path
import numpy as np
from dataset_v4 import prepare_dataset_v4

OUT = Path("outputs_v5_35_spread_regimes")
OUT.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def spread_ticks(x):
    raw = norm.inverse_transform(
        np.asarray(x, dtype=np.float64)
    )
    spread = raw[..., 0] - raw[..., 2]
    return np.maximum(
        np.rint(spread / 0.01).astype(int), 1
    )

train = spread_ticks(data["train"])
val = spread_ticks(data["val"][:512])

def describe(x):
    delta = np.abs(np.diff(x, axis=1))
    return {
        "sequences": int(len(x)),
        "mean_ticks": float(x.mean()),
        "first_step_mean": float(x[:, 0].mean()),
        "last_step_mean": float(x[:, -1].mean()),
        "unchanged_fraction": float(
            (delta == 0).mean()
        ),
        "mean_tick_change": float(delta.mean()),
        "p10": float(np.quantile(x, 0.1)),
        "p50": float(np.quantile(x, 0.5)),
        "p90": float(np.quantile(x, 0.9)),
    }

results = {
    "train_all": describe(train),
    "validation_reference": describe(val),
    "train_quarters": {}
}

for i, indices in enumerate(
    np.array_split(np.arange(len(train)), 4), 1
):
    results["train_quarters"][f"Q{i}"] = (
        describe(train[indices])
    )

print("=== V5.35 SPREAD REGIME AUDIT ===")

for name, result in results.items():
    if name != "train_quarters":
        print(name.upper(), result)

for name, result in results["train_quarters"].items():
    print(name, result)

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.35 spread regime audit complete!")
