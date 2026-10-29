import json
from pathlib import Path

import numpy as np
from dataset_v4 import prepare_dataset_v4

SEED = 12345
THRESHOLD = 1e-7
rng = np.random.default_rng(SEED)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def inverse_returns(windows):
    z = np.asarray(windows, dtype=np.float64)[..., 40]
    return norm.return_scale * np.sinh(
        z * norm.return_std + norm.return_mean
    )

train = inverse_returns(data["train"])
real = inverse_returns(data["test"][:512])

generated_data = np.load(
    "outputs_v5_10_full_lob/generated_full_lob.npz"
)
original = generated_data["log_returns"]

# Markov states: 0 = no price change; 1 = jump.
states = (np.abs(train) >= THRESHOLD).astype(int)

transition_counts = np.ones((2, 2), dtype=np.float64)
for a, b in zip(states[:, :-1].ravel(),
                states[:, 1:].ravel()):
    transition_counts[a, b] += 1

transition = transition_counts / transition_counts.sum(
    axis=1, keepdims=True
)
initial_prob = states[:, 0].mean()

nonzero_train = train[states == 1]
if len(nonzero_train) == 0:
    raise RuntimeError("No training jumps available")

n, length = original.shape
baseline = np.zeros((n, length), dtype=np.float64)

for i in range(n):
    state = int(rng.random() < initial_prob)

    for t in range(length):
        if state == 1:
            baseline[i, t] = rng.choice(nonzero_train)

        if t + 1 < length:
            state = int(rng.random() < transition[state, 1])

def metrics(x):
    zero = np.abs(x) < THRESHOLD
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "near_zero_fraction": float(zero.mean()),
        "positive_fraction": float(
            np.mean(x > THRESHOLD)
        ),
        "negative_fraction": float(
            np.mean(x < -THRESHOLD)
        ),
        "temporal_change": float(
            np.abs(np.diff(x, axis=1)).mean()
        ),
        "across_sample_std": float(
            x.std(axis=0).mean()
        ),
        "nonzero_abs_median": (
            float(np.median(np.abs(x[~zero])))
            if (~zero).any() else 0.0
        ),
    }

results = {
    "method": "Train-calibrated two-state Markov return baseline",
    "seed": SEED,
    "transition_matrix": transition.tolist(),
    "initial_jump_probability": float(initial_prob),
    "real_test": metrics(real),
    "v5_10_original": metrics(original),
    "v5_11_markov_baseline": metrics(baseline),
}

out = Path("outputs_v5_11_return_baseline")
out.mkdir(parents=True, exist_ok=True)

np.save(out / "baseline_returns.npy", baseline)
with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("=== V5.11 RETURN BASELINE ===")
print(json.dumps(results, indent=2))
print("\nSaved:", out / "results.json")
print("V5.11 baseline complete!")
