import json
from pathlib import Path

import numpy as np
from dataset_v4 import prepare_dataset_v4

OUT = Path("outputs_v5_30_mdn_extremes")
OUT.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def extract(book):
    return np.maximum(
        np.expm1(book[..., 1:40:2]), 0
    )

train = extract(norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
))
val = extract(norm.inverse_transform(
    np.asarray(data["val"][:512], dtype=np.float64)
))

generated = np.load(
    "outputs_v5_29_mdn/rollout_candidates.npz"
)["probabilistic_v29"].astype(np.float64)

assert generated.shape == (512, 24, 20)

def summary(x):
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, .95)),
        "p99": float(np.quantile(x, .99)),
        "p99_9": float(np.quantile(x, .999)),
        "p99_99": float(np.quantile(x, .9999)),
        "max": float(x.max()),
        "above_2000": int((x > 2000).sum()),
        "above_10000": int((x > 10000).sum()),
        "above_100000": int((x > 100000).sum()),
        "nonfinite": int((~np.isfinite(x)).sum()),
    }

results = {
    "train": summary(train),
    "validation": summary(val),
    "v5_29": summary(generated),
}

print("=== V5.30 EXTREME-VALUE AUDIT ===")
for name, result in results.items():
    print(name.upper(), result)

print("\n=== TOP 15 GENERATED VOLUMES ===")

flat = generated.ravel()
top_ids = np.argsort(flat)[-15:][::-1]

top = []
for index in top_ids:
    sequence, step, level = np.unravel_index(
        index, generated.shape
    )

    side = "Ask" if level % 2 == 0 else "Bid"
    level_number = level // 2 + 1

    item = {
        "sequence": int(sequence),
        "step": int(step),
        "side": side,
        "level": int(level_number),
        "volume": float(flat[index]),
    }
    top.append(item)
    print(item)

results["top_15"] = top

results["time_step_means"] = [
    float(v) for v in generated.mean(axis=(0, 2))
]

results["time_step_maximums"] = [
    float(v) for v in generated.max(axis=(0, 2))
]

print("\n=== MEAN VOLUME BY TIME STEP ===")
print(np.round(results["time_step_means"], 2))

print("\n=== MAXIMUM VOLUME BY TIME STEP ===")
print(np.round(results["time_step_maximums"], 2))

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.30 extreme-value audit complete!")
