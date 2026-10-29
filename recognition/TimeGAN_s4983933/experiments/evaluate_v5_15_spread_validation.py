import json
import numpy as np
from pathlib import Path
from dataset_v4 import prepare_dataset_v4

OUT = Path("outputs_v5_15_spread_validation")
OUT.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

train = norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
)
val = norm.inverse_transform(
    np.asarray(data["val"][:512], dtype=np.float64)
)

generated = np.load(
    "outputs_v5_10_full_lob/generated_full_lob.npz"
)["relative_orderbook"].astype(np.float64)

def spread(book):
    return book[..., 0] - book[..., 2]

def ticks(x):
    return np.maximum(
        1, np.rint(x / .01).astype(int)
    )

train_s = spread(train).ravel()
val_s = spread(val).ravel()
fake_s = spread(generated).ravel()

# Quantile map using TRAIN only.
sorted_train = np.sort(ticks(train_s))
order = np.argsort(fake_s, kind="stable")

positions = np.minimum(
    (
        (np.arange(len(fake_s)) + .5)
        / len(fake_s)
        * len(sorted_train)
    ).astype(int),
    len(sorted_train) - 1
)

mapped_ticks = np.empty(len(fake_s), dtype=int)
mapped_ticks[order] = sorted_train[positions]
mapped_s = mapped_ticks * .01

# Fixed validation evaluation range, common to
# every method. Do not change it per candidate.
low = min(
    val_s.min(), train_s.min(),
    fake_s.min(), mapped_s.min()
)
high = max(
    val_s.max(), train_s.max(),
    fake_s.max(), mapped_s.max()
)

edges = np.linspace(low, high, 81)

def kl_hist(real, candidate):
    p, _ = np.histogram(real, bins=edges)
    q, _ = np.histogram(candidate, bins=edges)

    p = p.astype(float) + 1e-8
    q = q.astype(float) + 1e-8

    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))

# Discrete tick KL, same method as V5.15.
max_tick = max(
    ticks(train_s).max(),
    ticks(val_s).max(),
    ticks(fake_s).max(),
    ticks(mapped_s).max()
)

def tick_probs(x):
    c = np.bincount(
        ticks(x), minlength=max_tick + 1
    )[1:].astype(float)
    c += 1
    return c / c.sum()

p_tick = tick_probs(val_s)

def kl_tick(candidate):
    q = tick_probs(candidate)
    return float(
        np.sum(p_tick * np.log(p_tick / q))
    )

candidates = {
    "train_distribution": train_s,
    "v5_10_original": fake_s,
    "v5_10_tick_quantile": mapped_s,
}

results = {}

for name, s in candidates.items():
    results[name] = {
        "histogram_kl": kl_hist(val_s, s),
        "tick_kl": kl_tick(s),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "distinct_ticks": int(
            np.unique(ticks(s)).size
        ),
    }

print("=== V5.15 VALIDATION SPREAD COMPARISON ===")
print(json.dumps(results, indent=2))

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved:", OUT / "results.json")
print("V5.15 validation check complete!")
