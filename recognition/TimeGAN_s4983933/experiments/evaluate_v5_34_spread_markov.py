import json
from pathlib import Path

import numpy as np
from scipy.special import rel_entr
from dataset_v4 import prepare_dataset_v4

SEED = 371034
N, T = 512, 24
TICK = 0.01

rng = np.random.default_rng(SEED)

out = Path("outputs_v5_34_spread_markov")
out.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

train = norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
)
val = norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
)
generated_book = np.load(
    "outputs_v5_32_integrated/generated_lob.npz"
)["relative_orderbook"].astype(np.float64)

def ticks(book):
    spread = book[..., 0] - book[..., 2]
    return np.maximum(
        np.rint(spread / TICK).astype(np.int64), 1
    )

tr = ticks(train)
va = ticks(val)
original = ticks(generated_book)

max_tick = int(tr.max())

# Train-only empirical transition matrix.
# Row = current tick, column = next tick.
counts = np.zeros(
    (max_tick + 1, max_tick + 1),
    dtype=np.float64
)

for current, following in zip(
    tr[:, :-1].ravel(),
    tr[:, 1:].ravel()
):
    counts[current, following] += 1

# Fallback for unobserved states:
# self-transition, rather than arbitrary jumps.
transitions = np.zeros_like(counts)

for s in range(1, max_tick + 1):
    total = counts[s].sum()
    if total > 0:
        transitions[s] = counts[s] / total
    else:
        transitions[s, s] = 1.0

def sample_sequence(initial):
    x = np.empty((N, T), dtype=np.int64)
    x[:, 0] = initial

    for t in range(1, T):
        for state in np.unique(x[:, t-1]):
            ids = np.flatnonzero(x[:, t-1] == state)
            x[ids, t] = rng.choice(
                max_tick + 1,
                size=len(ids),
                p=transitions[state]
            )

    return x

# Both variants use only TRAIN-defined transitions.
# Variant A retains the original generator's first tick.
# Variant B samples first ticks from TRAIN.
initial_train = rng.choice(
    tr.ravel(), size=N, replace=True
)

candidates = {
    "original_v5_32": original,
    "markov_generated_initial": sample_sequence(
        np.clip(original[:, 0], 1, max_tick)
    ),
    "markov_train_initial": sample_sequence(
        initial_train
    )
}

def histogram(x):
    clipped = np.clip(x.ravel(), 1, max_tick + 1)
    return np.bincount(
        clipped - 1,
        minlength=max_tick + 1
    ).astype(np.float64)

def kl(p, q):
    a = histogram(p) + 1
    b = histogram(q) + 1
    a /= a.sum()
    b /= b.sum()
    return float(np.sum(rel_entr(a, b)))

def describe(x):
    differences = np.abs(np.diff(x, axis=1))
    return {
        "mean_ticks": round(float(x.mean()), 4),
        "unchanged_fraction": round(
            float((differences == 0).mean()), 5
        ),
        "mean_abs_tick_change": round(
            float(differences.mean()), 5
        ),
        "max_ticks": int(x.max()),
        "kl_validation_to_generated": round(
            kl(va, x), 6
        ),
        "kl_generated_to_validation": round(
            kl(x, va), 6
        ),
    }

results = {
    "train": describe(tr),
    "validation": describe(va),
    "candidates": {}
}

print("=== V5.34 SPREAD MARKOV RESULTS ===")
print("TRAIN", results["train"])
print("VALIDATION", results["validation"])

for name, x in candidates.items():
    result = describe(x)
    results["candidates"][name] = result
    print(name.upper(), result)

    np.save(
        out / f"{name}.npy",
        x.astype(np.int16)
    )

with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.34 experiment complete!")
