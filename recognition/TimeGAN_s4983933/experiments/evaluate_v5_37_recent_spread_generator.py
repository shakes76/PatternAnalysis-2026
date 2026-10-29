import json
from pathlib import Path
import numpy as np
from scipy.special import rel_entr
from dataset_v4 import prepare_dataset_v4

SEED = 371037
N, T = 512, 24
TICK = 0.01

out = Path("outputs_v5_37_recent_spread")
out.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def ticks(book):
    spread = book[..., 0] - book[..., 2]
    return np.maximum(
        np.rint(spread / TICK).astype(int), 1
    )

def decode(split):
    return norm.inverse_transform(
        np.asarray(split, dtype=np.float64)
    )

train = ticks(decode(data["train"]))
val = ticks(decode(data["val"][:N]))

source = np.load(
    "outputs_v5_32_integrated/generated_lob.npz"
)
original = ticks(
    source["relative_orderbook"].astype(np.float64)
)

assert original.shape == val.shape == (N, T)

# V5.36 selected the recent 2/3 window using
# a backtest entirely inside TRAIN.
recent_count = round(len(train) * 2 / 3)
recent = train[-recent_count:]

max_tick = int(train.max())
states = np.arange(1, max_tick + 1)

# Empirical initial-state distribution from TRAIN.
initial_counts = np.bincount(
    recent[:, 0],
    minlength=max_tick + 1
)[1:].astype(np.float64)

initial_prob = initial_counts / initial_counts.sum()

# Train-only marginal distribution.
marginal_counts = np.bincount(
    recent.ravel(),
    minlength=max_tick + 1
)[1:].astype(np.float64)

marginal_prob = marginal_counts / marginal_counts.sum()

# Train-only transition counts.
counts = np.zeros(
    (max_tick + 1, max_tick + 1),
    dtype=np.float64
)

np.add.at(
    counts,
    (recent[:, :-1].ravel(),
     recent[:, 1:].ravel()),
    1
)

transitions = np.zeros_like(counts)

for s in states:
    total = counts[s].sum()
    if total > 0:
        transitions[s] = counts[s] / total
    else:
        transitions[s, s] = 1

def generate_iid(seed):
    rng = np.random.default_rng(seed)
    return rng.choice(
        states,
        size=(N, T),
        p=marginal_prob
    )

def generate_markov(seed):
    rng = np.random.default_rng(seed)

    result = np.empty((N, T), dtype=np.int64)

    result[:, 0] = rng.choice(
        states, size=N, p=initial_prob
    )

    for t in range(1, T):
        current = result[:, t-1]
        for s in np.unique(current):
            ids = np.flatnonzero(current == s)
            result[ids, t] = rng.choice(
                max_tick + 1,
                size=len(ids),
                p=transitions[s]
            )

    return result

# Same train-defined support and +1 smoothing
# as V5.33-V5.36.
def histogram(x):
    values = np.clip(
        x.ravel(), 1, max_tick + 1
    )
    return np.bincount(
        values - 1,
        minlength=max_tick + 1
    ).astype(np.float64)

def kl(a, b):
    p = histogram(a) + 1
    q = histogram(b) + 1
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(rel_entr(p, q)))

def metrics(x):
    d = np.abs(np.diff(x, axis=1))
    return {
        "mean_ticks": round(float(x.mean()), 5),
        "max_ticks": int(x.max()),
        "unchanged_fraction": round(
            float((d == 0).mean()), 5
        ),
        "mean_abs_tick_change": round(
            float(d.mean()), 5
        ),
        "kl_val_to_candidate": round(
            kl(val, x), 6
        ),
        "kl_candidate_to_val": round(
            kl(x, val), 6
        ),
        "first_step_mean": round(
            float(x[:, 0].mean()), 5
        ),
        "last_step_mean": round(
            float(x[:, -1].mean()), 5
        )
    }

iid = generate_iid(SEED)
markov = generate_markov(SEED + 1)

results = {
    "train_recent_reference": metrics(recent),
    "real_validation": metrics(val),
    "original_v5_32": metrics(original),
    "recent_iid": metrics(iid),
    "recent_markov": metrics(markov)
}

print("=== V5.37 RECENT SPREAD RESULTS ===")
for name, result in results.items():
    print(name.upper(), result)

np.savez_compressed(
    out / "generated_spread_ticks.npz",
    recent_iid=iid.astype(np.int16),
    recent_markov=markov.astype(np.int16)
)

with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.37 spread experiment complete!")
