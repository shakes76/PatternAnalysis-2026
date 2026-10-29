import json
from pathlib import Path

import numpy as np
from dataset_v4 import prepare_dataset_v4

OUT = Path("outputs_v5_24_persistence")
OUT.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def extract(x):
    return np.maximum(
        np.expm1(x[..., 1:40:2]), 0
    )

train_raw = norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
)

val_raw = norm.inverse_transform(
    np.asarray(data["val"][:512], dtype=np.float64)
)

fake_raw = np.load(
    "outputs_v5_21_integrated/generated_lob.npz"
)["relative_orderbook"].astype(np.float64)

sets = {
    "TRAIN": extract(train_raw),
    "VALIDATION": extract(val_raw),
    "GENERATED": extract(fake_raw),
}

def metrics(x):
    differences = np.abs(np.diff(x, axis=1))

    # Original LOBSTER volume values are
    # integer shares; tolerate only floating
    # point reconstruction error.
    unchanged = differences < 1e-5
    changed = ~unchanged

    return {
        "unchanged_fraction": float(
            unchanged.mean()
        ),
        "mean_absolute_change": float(
            differences.mean()
        ),
        "conditional_change": float(
            differences[changed].mean()
        ) if changed.any() else 0.0,
        "median_conditional_change": float(
            np.median(differences[changed])
        ) if changed.any() else 0.0,
        "per_level_unchanged": [
            float(v) for v in unchanged.mean(
                axis=(0, 1)
            )
        ],
        "per_level_mean_change": [
            float(v) for v in differences.mean(
                axis=(0, 1)
            )
        ],
    }

results = {
    name: metrics(x)
    for name, x in sets.items()
}

print("=== V5.24 PERSISTENCE AUDIT ===")

for name, result in results.items():
    print(f"\n{name}")
    print(
        "Unchanged fraction:",
        round(result["unchanged_fraction"], 5)
    )
    print(
        "Mean absolute change:",
        round(result["mean_absolute_change"], 5)
    )
    print(
        "Conditional change:",
        round(result["conditional_change"], 5)
    )
    print(
        "Median conditional change:",
        round(result["median_conditional_change"], 5)
    )

print("\n=== PER-LEVEL UNCHANGED FRACTION ===")
print("Level | Train | Validation | Generated")

for i in range(20):
    side = "Ask" if i % 2 == 0 else "Bid"
    level = i // 2 + 1

    print(
        f"{side}{level:<2} | "
        f"{results['TRAIN']['per_level_unchanged'][i]:.4f} | "
        f"{results['VALIDATION']['per_level_unchanged'][i]:.4f} | "
        f"{results['GENERATED']['per_level_unchanged'][i]:.4f}"
    )

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.24 persistence audit complete!")
