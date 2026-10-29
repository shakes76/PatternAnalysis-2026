import json
from pathlib import Path

import numpy as np
from dataset_v4 import prepare_dataset_v4

SEED = 371019
N = 512
T = 24
THRESHOLD = 1e-7

out = Path("outputs_v5_19_recent_markov")
out.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

train = norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
)[..., 40]

val = norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
)[..., 40]

# Fixed evaluation protocol from V5.16.
limit = max(
    float(np.quantile(np.abs(train.ravel()), .999)) * 1.2,
    1e-7,
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

# Reuse identical random numbers for all candidates.
rng = np.random.default_rng(SEED)
initial_uniform = rng.random(N)
step_uniform = rng.random((N, T))
jump_uniform = rng.random((N, T))

def generate(source):
    active = np.abs(source) >= THRESHOLD
    current = active[:, :-1].ravel()
    following = active[:, 1:].ravel()

    jump_rate = float(active.mean())
    transition = np.empty(2, dtype=float)

    for previous in (False, True):
        selected = following[current == previous]
        transition[int(previous)] = (
            float(selected.mean())
            if len(selected) else jump_rate
        )

    jumps = source[active]
    if len(jumps) == 0:
        raise ValueError("No nonzero returns in source.")

    # Empirical jump magnitudes and signs,
    # using training data only.
    jumps_sorted = np.sort(jumps)

    indices = np.minimum(
        (jump_uniform * len(jumps_sorted)).astype(int),
        len(jumps_sorted) - 1
    )
    jump_draws = jumps_sorted[indices]

    generated = np.zeros((N, T), dtype=float)
    state = initial_uniform < jump_rate

    generated[state, 0] = jump_draws[state, 0]

    for t in range(1, T):
        probability = transition[state.astype(int)]
        state = step_uniform[:, t] < probability
        generated[state, t] = jump_draws[state, t]

    return generated, jump_rate, transition

def metrics(x):
    mask = np.abs(x) >= THRESHOLD
    jumps = np.abs(x[mask])

    return {
        "kl": round(kl(val, x), 6),
        "zero_fraction": round(float((~mask).mean()), 5),
        "std": round(float(x.std()), 9),
        "abs_lag1": round(abs_lag1(x), 5),
        "nonzero_median_abs": (
            round(float(np.median(jumps)), 9)
            if len(jumps) else None
        ),
    }

sources = {
    "full_history": train,
    "recent_50pct": train[len(train)//2:],
    "recent_25pct": train[3*len(train)//4:],
}

results = {
    "protocol": {
        "bins": 81,
        "range": [-limit, limit],
        "evaluation": "validation_first_512",
        "seed": SEED,
    },
    "real_validation": metrics(val),
    "candidates": {},
}

print("=== V5.19 RECENT MARKOV RESULTS ===")
print("REAL:", results["real_validation"])

for name, source in sources.items():
    generated, rate, transition = generate(source)

    item = {
        "training_windows": int(len(source)),
        "training_jump_rate": round(rate, 6),
        "transition_to_jump": [
            round(float(v), 6)
            for v in transition
        ],
        **metrics(generated),
    }

    results["candidates"][name] = item
    print(name.upper() + ":", item)

with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.19 evaluation complete!")
