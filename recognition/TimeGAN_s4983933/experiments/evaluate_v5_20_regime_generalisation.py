import json
from pathlib import Path

import numpy as np
from dataset_v4 import prepare_dataset_v4

SEED = 371020
N = 512
T = 24
THRESHOLD = 1e-7

out = Path("outputs_v5_20_regimes")
out.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

train = norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
)[..., 40]

val = norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
)[..., 40]

# IMPORTANT:
# Interpretation as market "regimes" requires
# train windows to be chronologically ordered.
# This script compares array quarters; it does
# not independently prove chronological order.

limit = max(
    float(np.quantile(np.abs(train.ravel()), .999)) * 1.2,
    1e-7
)
edges = np.linspace(-limit, limit, 82)

def kl(real, fake):
    a = np.clip(real.ravel(), edges[0], edges[-1])
    b = np.clip(fake.ravel(), edges[0], edges[-1])

    p = np.histogram(a, bins=edges)[0].astype(float) + 1e-8
    q = np.histogram(b, bins=edges)[0].astype(float) + 1e-8

    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))

def abs_lag1(x):
    a = np.abs(x[:, :-1]).ravel()
    b = np.abs(x[:, 1:]).ravel()

    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])

rng = np.random.default_rng(SEED)

initial_u = rng.random(N)
step_u = rng.random((N, T))
jump_u = rng.random((N, T))

def generate(source):
    state = np.abs(source) >= THRESHOLD

    previous = state[:, :-1].ravel()
    following = state[:, 1:].ravel()

    jump_rate = float(state.mean())

    transition = np.empty(2, dtype=float)

    for s in (0, 1):
        mask = previous == bool(s)
        transition[s] = (
            float(following[mask].mean())
            if mask.any() else jump_rate
        )

    jumps = np.sort(source[state])

    if len(jumps) == 0:
        raise ValueError("No jump samples")

    indices = np.minimum(
        (jump_u * len(jumps)).astype(int),
        len(jumps) - 1
    )
    draws = jumps[indices]

    result = np.zeros((N, T), dtype=float)

    active = initial_u < jump_rate
    result[active, 0] = draws[active, 0]

    for t in range(1, T):
        probability = transition[active.astype(int)]
        active = step_u[:, t] < probability
        result[active, t] = draws[active, t]

    return result, jump_rate, transition

def metrics(x):
    active = np.abs(x) >= THRESHOLD
    jumps = np.abs(x[active])

    return {
        "kl": round(kl(val, x), 6),
        "zero_fraction": round(
            float((~active).mean()), 5
        ),
        "std": round(float(x.std()), 9),
        "abs_lag1": round(abs_lag1(x), 5),
        "nonzero_median_abs": (
            round(float(np.median(jumps)), 9)
            if len(jumps) else None
        )
    }

count = len(train)
boundaries = np.linspace(
    0, count, 5, dtype=int
)

sources = {
    "full_history": train,
    "early_25pct": train[
        boundaries[0]:boundaries[1]
    ],
    "middle_early_25pct": train[
        boundaries[1]:boundaries[2]
    ],
    "middle_late_25pct": train[
        boundaries[2]:boundaries[3]
    ],
    "recent_25pct": train[
        boundaries[3]:boundaries[4]
    ],
}

results = {
    "real_validation": metrics(val),
    "candidates": {}
}

print("=== V5.20 REGIME GENERALISATION ===")
print("REAL:", results["real_validation"])

for name, source in sources.items():
    generated, rate, transition = generate(source)

    item = {
        "training_windows": len(source),
        "training_jump_rate": round(rate, 6),
        "transition_to_jump": [
            round(float(v), 6)
            for v in transition
        ],
        **metrics(generated)
    }

    results["candidates"][name] = item

    print(name.upper() + ":", item)

with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.20 evaluation complete!")
