import json
from pathlib import Path
import numpy as np

from dataset_v4 import prepare_dataset_v4

SEED = 371016
N = 512
T = 24
OUT = Path("outputs_v5_16_returns")
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

train = norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
)[..., 40]

val = norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
)[..., 40]

fake = np.asarray(
    np.load(
        "outputs_v5_10_full_lob/generated_full_lob.npz"
    )["log_returns"],
    dtype=np.float64
)

assert val.shape == fake.shape == (N, T)

THRESHOLD = 1e-7

def nonzero(x):
    return np.abs(x) >= THRESHOLD

train_flat = train.ravel()
train_jumps = train_flat[nonzero(train_flat)]

# Baseline 1: independent sampling from TRAIN.
iid = rng.choice(
    train_flat, size=(N, T), replace=True
)

# Baseline 2: complete TRAIN window replay.
replay_ids = rng.integers(0, len(train), size=N)
replay = train[replay_ids].copy()

# Baseline 3: two-state Markov model trained
# exclusively on TRAIN event transitions.
states = nonzero(train)
current = states[:, :-1].ravel()
following = states[:, 1:].ravel()

transition = np.zeros((2, 2), dtype=float)

for previous in (0, 1):
    selected = following[current == previous]
    transition[previous, 1] = (
        float(selected.mean())
        if len(selected) else float(states.mean())
    )
    transition[previous, 0] = (
        1.0 - transition[previous, 1]
    )

markov = np.zeros((N, T), dtype=np.float64)
state = rng.random(N) < states.mean()

for t in range(T):
    if t > 0:
        jump_probability = transition[
            state.astype(int), 1
        ]
        state = rng.random(N) < jump_probability

    count = int(state.sum())
    if count:
        markov[state, t] = rng.choice(
            train_jumps, size=count, replace=True
        )

# Fix the KL protocol before comparing models:
# symmetric range based only on TRAIN return tails,
# 81 histogram bins (central bin contains zero).
limit = float(
    np.quantile(np.abs(train_flat), 0.999)
)
limit = max(limit * 1.2, 1e-7)
edges = np.linspace(-limit, limit, 82)

def kl_divergence(real, candidate):
    r = np.clip(real.ravel(), edges[0], edges[-1])
    c = np.clip(
        candidate.ravel(), edges[0], edges[-1]
    )

    p, _ = np.histogram(r, bins=edges)
    q, _ = np.histogram(c, bins=edges)

    p = p.astype(float) + 1e-8
    q = q.astype(float) + 1e-8

    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))

def lag1_absolute(x):
    a = np.abs(x[:, :-1]).ravel()
    b = np.abs(x[:, 1:]).ravel()

    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def describe(x):
    jumps = np.abs(x[nonzero(x)])

    return {
        "kl_to_validation": kl_divergence(val, x),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "zero_fraction": float(
            np.mean(~nonzero(x))
        ),
        "nonzero_median_abs": (
            float(np.median(jumps))
            if len(jumps) else None
        ),
        "nonzero_p90_abs": (
            float(np.quantile(jumps, 0.9))
            if len(jumps) else None
        ),
        "abs_return_lag1": lag1_absolute(x),
    }

results = {
    "protocol": {
        "evaluation_split": "validation_first_512",
        "bins": 81,
        "range": [-limit, limit],
        "kl_direction": "KL(validation || generated)",
        "training_only_calibration": True,
        "zero_threshold": THRESHOLD,
        "random_seed": SEED,
    },
    "markov_transition": transition.tolist(),
    "methods": {
        "real_validation": describe(val),
        "v5_10_timegan": describe(fake),
        "train_iid_baseline": describe(iid),
        "train_window_replay": describe(replay),
        "train_markov_baseline": describe(markov),
    },
}

print("=== V5.16 RETURN KL VALIDATION ===")
print(json.dumps(results, indent=2))

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

np.savez_compressed(
    OUT / "generated_returns.npz",
    iid=iid,
    replay=replay,
    markov=markov,
)

print("\nV5.16 evaluation complete!")
