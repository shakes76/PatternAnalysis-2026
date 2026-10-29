import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42

SEED = 371018
N = 512
T = 24
THRESHOLD = 1e-7

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

out = Path("outputs_v5_18_conditional_returns")
out.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

train_x = np.asarray(data["train"], dtype=np.float32)
val_x = np.asarray(data["val"][:N], dtype=np.float32)

train_raw = norm.inverse_transform(
    train_x.astype(np.float64)
)
val_raw = norm.inverse_transform(
    val_x.astype(np.float64)
)

train_r = train_raw[..., 40]
val_r = val_raw[..., 40]

ck = torch.load(
    "outputs_v5_10_latent_matching/timegan_finetuned.pt",
    map_location="cpu",
    weights_only=True,
)

gan = TimeGANV42(
    feature_dim=ck["feature_dim"],
    hidden_dim=ck["hidden_dim"],
    noise_dim=ck["noise_dim"],
    num_layers=ck["num_layers"],
)
gan.load_state_dict(ck["model_state_dict"])
gan.eval()

# Extract real-data embeddings.
hidden_list = []

with torch.no_grad():
    for start in range(0, len(train_x), 128):
        xb = torch.from_numpy(
            train_x[start:start + 128]
        )
        hidden_list.append(
            gan.embedder(xb).numpy()
        )

real_h = np.concatenate(hidden_list, axis=0)

# Predict next-step jump from current hidden.
X = real_h[:, :-1, :].reshape(
    -1, real_h.shape[-1]
)
y = (
    np.abs(train_r[:, 1:]) >= THRESHOLD
).astype(np.float32).reshape(-1)

mu = X.mean(axis=0)
sd = X.std(axis=0) + 1e-6

X = (X - mu) / sd

head = nn.Linear(X.shape[1], 1)
optimizer = torch.optim.Adam(
    head.parameters(), lr=1e-3
)

x_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

for epoch in range(100):
    order = torch.randperm(len(x_tensor))

    for ids in order.split(2048):
        logits = head(x_tensor[ids]).flatten()
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y_tensor[ids]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

head.eval()

# Generate hidden states using the actual
# TimeGAN generator, not real-data embeddings.
with torch.no_grad():
    noise_rng = torch.Generator().manual_seed(SEED)
    z = torch.rand(
        N, T, ck["noise_dim"],
        generator=noise_rng,
    )
    generated_h = gan.supervisor(
        gan.generator(z)
    ).numpy()

    generated_features = (
        generated_h[:, :-1, :].reshape(
            -1, generated_h.shape[-1]
        )
    )

    generated_features = (
        generated_features - mu
    ) / sd

    scores = head(
        torch.from_numpy(
            generated_features.astype(np.float32)
        )
    ).numpy().reshape(N, T - 1)

# Standardise condition scores using only
# generated hidden-state statistics.
# This avoids a constant shift changing
# the overall jump frequency.
score_mean = float(scores.mean())
score_std = float(scores.std())

condition = np.clip(
    (scores - score_mean)
    / max(score_std, 1e-6),
    -2.0, 2.0
)

# Fit Markov transitions from TRAIN only.
states = np.abs(train_r) >= THRESHOLD
current = states[:, :-1].reshape(-1)
following = states[:, 1:].reshape(-1)

transition = np.zeros((2, 2))

for previous in (0, 1):
    subset = following[current == previous]
    p = (
        float(subset.mean())
        if len(subset)
        else float(states.mean())
    )
    transition[previous] = [1 - p, p]

train_flat = train_r.ravel()
jumps = train_flat[
    np.abs(train_flat) >= THRESHOLD
]

# Use identical random draws for fair comparison.
rng = np.random.default_rng(SEED)
initial_uniform = rng.random(N)
event_uniform = rng.random((N, T))
jump_indices = rng.integers(
    0, len(jumps), size=(N, T)
)
jump_values = jumps[jump_indices]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate(beta):
    result = np.zeros((N, T))
    active = (
        initial_uniform < float(states.mean())
    )

    result[active, 0] = jump_values[active, 0]

    for t in range(1, T):
        base_p = transition[
            active.astype(int), 1
        ]

        base_p = np.clip(base_p, 1e-6, 1 - 1e-6)

        base_logit = np.log(
            base_p / (1 - base_p)
        )

        adjusted_p = sigmoid(
            base_logit
            + beta * condition[:, t - 1]
        )

        active = (
            event_uniform[:, t] < adjusted_p
        )

        result[active, t] = (
            jump_values[active, t]
        )

    return result

limit = float(
    np.quantile(np.abs(train_flat), 0.999)
) * 1.2

limit = max(limit, 1e-7)

edges = np.linspace(-limit, limit, 82)

def kl(real, generated):
    a = np.clip(
        real.ravel(), edges[0], edges[-1]
    )
    b = np.clip(
        generated.ravel(), edges[0], edges[-1]
    )

    p, _ = np.histogram(a, bins=edges)
    q, _ = np.histogram(b, bins=edges)

    p = p.astype(float) + 1e-8
    q = q.astype(float) + 1e-8

    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))

def abs_lag1(x):
    a = np.abs(x[:, :-1]).ravel()
    b = np.abs(x[:, 1:]).ravel()

    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])

def summary(x):
    return {
        "kl": round(kl(val_r, x), 5),
        "zero_fraction": round(
            float(np.mean(
                np.abs(x) < THRESHOLD
            )), 5
        ),
        "abs_return_lag1": round(
            abs_lag1(x), 5
        ),
        "std": round(float(x.std()), 8),
    }

results = {
    "real_validation": summary(val_r),
    "markov_beta_0": summary(generate(0.0)),
    "hybrid_beta_0_5": summary(generate(0.5)),
    "hybrid_beta_1": summary(generate(1.0)),
    "generated_condition_std": score_std,
    "generated_condition_range": [
        float(condition.min()),
        float(condition.max())
    ],
}

print("=== V5.18 CONDITIONAL RETURN RESULTS ===")
print(json.dumps(results, indent=2))

with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.18 evaluation complete!")
