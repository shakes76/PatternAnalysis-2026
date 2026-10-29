import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dataset_v4 import prepare_dataset_v4

SEED = 371028
N, T = 512, 24
BINS = 8
THRESHOLD = 1e-5

torch.manual_seed(SEED)
torch.set_num_threads(4)
out = Path("outputs_v5_28_stochastic_updates")
out.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def volumes(book):
    return np.maximum(
        np.expm1(book[..., 1:40:2]), 0
    )

train = volumes(norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
))
real = volumes(norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
))
fake = volumes(np.load(
    "outputs_v5_21_integrated/generated_lob.npz"
)["relative_orderbook"].astype(np.float64))

class StateDepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.mask_head = nn.Linear(128, 20)
        self.value_head = nn.Linear(128, 20)

    def forward(self, x):
        h = self.encoder(x)
        return self.mask_head(h), self.value_head(h)

ck = torch.load(
    "outputs_v5_27_state_depth/best_depth_state.pt",
    map_location="cpu",
    weights_only=False
)
model = StateDepthNet()
model.load_state_dict(ck["model_state_dict"])
model.eval()
mu = torch.tensor(ck["mu"], dtype=torch.float32)
sd = torch.tensor(ck["sd"], dtype=torch.float32)

# Build TRAIN-only per-level conditional pools.
before = train[:, :-1].reshape(-1, 20)
after = train[:, 1:].reshape(-1, 20)

pools = []
for j in range(20):
    change = np.abs(after[:, j] - before[:, j]) >= THRESHOLD
    source = before[change, j]
    target = after[change, j]

    if len(source) == 0:
        raise RuntimeError(f"No updates at level {j}")

    # Each level receives its own TRAIN-based
    # current-volume quantile boundaries.
    edges = np.quantile(
        source, np.linspace(0, 1, BINS + 1)
    )[1:-1]

    groups = []
    bucket = np.searchsorted(edges, source, side="right")
    for k in range(BINS):
        ids = np.flatnonzero(bucket == k)
        # Sparse bin fallback uses the same level's
        # full TRAIN update pool.
        if len(ids) < 20:
            ids = np.arange(len(source))
        groups.append((
            source[ids],
            target[ids]
        ))

    pools.append((edges, groups))

def sample_update(current, j):
    edges, groups = pools[j]
    categories = np.searchsorted(
        edges, current, side="right"
    )
    new = np.empty_like(current)

    for k in range(BINS):
        where = np.flatnonzero(categories == k)
        if not len(where):
            continue

        s, target = groups[k]
        ids = rng.integers(0, len(s), size=len(where))
        new[where] = target[ids]

    return new

def sample_delta(current, j):
    edges, groups = pools[j]
    categories = np.searchsorted(
        edges, current, side="right"
    )
    new = np.empty_like(current)

    for k in range(BINS):
        where = np.flatnonzero(categories == k)
        if not len(where):
            continue

        s, target = groups[k]
        ids = rng.integers(0, len(s), size=len(where))
        delta = target[ids] - s[ids]
        new[where] = np.maximum(current[where] + delta, 0)

    return new

# Shared random numbers for all rollout variants.
mask_rng = np.random.default_rng(SEED)
uniforms = mask_rng.random((N, T-1, 20))

def rollout(method):
    x = np.empty_like(fake)
    x[:, 0] = fake[:, 0]

    with torch.no_grad():
        for t in range(1, T):
            state = torch.tensor(
                np.log1p(x[:, t-1]),
                dtype=torch.float32
            )
            logits, values = model((state - mu) / sd)
            probabilities = torch.sigmoid(logits).numpy()
            update = uniforms[:, t-1] < probabilities

            next_state = x[:, t-1].copy()

            if method == "deterministic":
                proposed = np.maximum(
                    np.expm1(
                        np.clip(values.numpy(), 0, 15)
                    ), 0
                )
                next_state[update] = proposed[update]
            else:
                for j in range(20):
                    ids = np.flatnonzero(update[:, j])
                    if not len(ids):
                        continue
                    current = next_state[ids, j]
                    if method == "delta":
                        proposed = sample_delta(current, j)
                    else:
                        proposed = sample_update(current, j)
                    next_state[ids, j] = proposed

            x[:, t] = next_state

    return x

def metrics(x):
    diff = np.abs(np.diff(x, axis=1))
    changed = diff >= THRESHOLD

    logs = np.log1p(x).reshape(-1, 20)
    corr = np.corrcoef(logs, rowvar=False)
    offdiag = ~np.eye(20, dtype=bool)

    # Compare early and late steps to detect drift.
    return {
        "mean": round(float(x.mean()), 3),
        "std": round(float(x.std()), 3),
        "p99": round(float(np.quantile(x, .99)), 3),
        "unchanged_fraction": round(
            float((~changed).mean()), 5
        ),
        "temporal_change": round(
            float(diff.mean()), 3
        ),
        "conditional_change": round(
            float(diff[changed].mean()), 3
        ) if changed.any() else 0.0,
        "conditional_median": round(
            float(np.median(diff[changed])), 3
        ) if changed.any() else 0.0,
        "mean_abs_level_correlation": round(
            float(np.mean(np.abs(corr[offdiag]))), 5
        ),
        "first_step_mean": round(
            float(x[:, 0].mean()), 3
        ),
        "last_step_mean": round(
            float(x[:, -1].mean()), 3
        ),
    }

det = rollout("deterministic")
delta = rollout("delta")
replace = rollout("replace")

results = {
    "real_validation": metrics(real),
    "v5_21_original": metrics(fake),
    "v5_27_deterministic": metrics(det),
    "v5_28_conditional_delta": metrics(delta),
    "v5_28_conditional_replace": metrics(replace),
}

print("=== V5.28 STOCHASTIC UPDATE RESULTS ===")
for name, result in results.items():
    print(name.upper(), result)

with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

np.savez_compressed(
    out / "candidates.npz",
    deterministic=det.astype(np.float32),
    conditional_delta=delta.astype(np.float32),
    conditional_replace=replace.astype(np.float32)
)
print("\nV5.28 evaluation complete!")
