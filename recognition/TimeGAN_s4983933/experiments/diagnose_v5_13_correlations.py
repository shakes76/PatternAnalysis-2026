import json
from pathlib import Path

import numpy as np
import torch

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42
from lob_head_v5_8 import MLPConstraintAwareLOBHead
from train_v5_12_depth_aware import (
    StochasticVolumeHead, correlated_noise
)

SEED = 12346
N = 512
T = 24
ALPHA = 0.2
OUT = Path("outputs_v5_13_correlations")
OUT.mkdir(exist_ok=True)

torch.set_num_threads(4)

data = prepare_dataset_v4("data")
norm = data["normalizer"]
book_norm = norm.book_normalizer

real_book = norm.inverse_transform(
    np.asarray(data["test"][:N], dtype=np.float64)
)[..., :40]

ck = torch.load(
    "outputs_v5_10_latent_matching/timegan_finetuned.pt",
    map_location="cpu",
    weights_only=True
)

model = TimeGANV42(
    feature_dim=ck["feature_dim"],
    hidden_dim=ck["hidden_dim"],
    noise_dim=ck["noise_dim"],
    num_layers=ck["num_layers"]
)
model.load_state_dict(ck["model_state_dict"])
model.eval()

hc = torch.load(
    "outputs_v5_9_variance_aware/variance_aware_best.pt",
    map_location="cpu",
    weights_only=True
)

decoder = MLPConstraintAwareLOBHead(
    hidden_dim=hc["hidden_dim"],
    minimum=book_norm.minimum,
    scale=book_norm.scale,
    intermediate=hc["intermediate"],
    min_spread=0.01,
    min_gap=0.01
)
decoder.load_state_dict(hc["head_state_dict"])
decoder.eval()

vc = torch.load(
    "outputs_v5_12_depth_aware/best_volume_head.pt",
    map_location="cpu",
    weights_only=True
)

head = StochasticVolumeHead(
    hidden_dim=vc["hidden_dim"]
)
head.load_state_dict(vc["head_state_dict"])
head.eval()

rng = torch.Generator().manual_seed(SEED)

with torch.no_grad():
    z = torch.rand(
        N, T, ck["noise_dim"], generator=rng
    )
    hidden = model.supervisor(model.generator(z))
    base_book = decoder.forward_raw(hidden)

    noise = correlated_noise(
        N, T, generator=rng
    )
    generated_log = head(
        hidden,
        base_book[..., 1::2],
        noise
    )

generated = np.expm1(generated_log.numpy())

smoothed = np.empty_like(generated)
smoothed[:, 0] = generated[:, 0]
for t in range(1, T):
    smoothed[:, t] = (
        ALPHA * generated[:, t]
        + (1 - ALPHA) * smoothed[:, t - 1]
    )

real = np.expm1(real_book[..., 1::2])

# Feature order: Ask1, Bid1, ..., Ask10, Bid10.
names = [
    f"{side}{level}"
    for level in range(1, 11)
    for side in ("Ask", "Bid")
]

def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if a.std() < 1e-10 or b.std() < 1e-10:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])

def temporal_correlations(x, lag):
    return np.asarray([
        safe_corr(x[:, :-lag, j], x[:, lag:, j])
        for j in range(20)
    ])

def cross_level_correlations(x):
    # Within ask ladder and within bid ladder only.
    values = []
    for side_offset in (0, 1):
        for level in range(9):
            j = 2 * level + side_offset
            k = j + 2
            values.append(safe_corr(
                x[..., j], x[..., k]
            ))
    return np.asarray(values)

def summary(x):
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p99": float(np.quantile(x, 0.99)),
        "temporal_change": float(
            np.abs(np.diff(x, axis=1)).mean()
        ),
        "lag1_mean": float(
            temporal_correlations(x, 1).mean()
        ),
        "lag2_mean": float(
            temporal_correlations(x, 2).mean()
        ),
        "lag5_mean": float(
            temporal_correlations(x, 5).mean()
        ),
        "adjacent_level_corr_mean": float(
            cross_level_correlations(x).mean()
        )
    }

results = {
    "real": summary(real),
    "v5_12_raw": summary(generated),
    "v5_12_ema_02": summary(smoothed),
    "level_wise": [],
}

print("=== DEPTH CORRELATION SUMMARY ===")
for key in ("real", "v5_12_raw", "v5_12_ema_02"):
    print(
        f"\n{key}:"
        f"\n{json.dumps(results[key], indent=2)}"
    )

print("\n=== LEVEL-WISE DISTRIBUTION ===")
print(
    f"{'Level':<7}"
    f"{'Real mean':>11}"
    f"{'EMA mean':>11}"
    f"{'Real std':>11}"
    f"{'EMA std':>11}"
    f"{'Real lag1':>11}"
    f"{'EMA lag1':>11}"
)

real_lag1 = temporal_correlations(real, 1)
ema_lag1 = temporal_correlations(smoothed, 1)

for j, name in enumerate(names):
    r = real[..., j]
    s = smoothed[..., j]

    row = {
        "level": name,
        "real_mean": float(r.mean()),
        "ema_mean": float(s.mean()),
        "real_std": float(r.std()),
        "ema_std": float(s.std()),
        "real_p99": float(np.quantile(r, .99)),
        "ema_p99": float(np.quantile(s, .99)),
        "real_lag1": float(real_lag1[j]),
        "ema_lag1": float(ema_lag1[j]),
    }
    results["level_wise"].append(row)

    print(
        f"{name:<7}"
        f"{row['real_mean']:>11.1f}"
        f"{row['ema_mean']:>11.1f}"
        f"{row['real_std']:>11.1f}"
        f"{row['ema_std']:>11.1f}"
        f"{row['real_lag1']:>11.3f}"
        f"{row['ema_lag1']:>11.3f}"
    )

results["errors"] = {
    "lag1_mae": float(np.mean(np.abs(
        temporal_correlations(real, 1)
        - temporal_correlations(smoothed, 1)
    ))),
    "lag5_mae": float(np.mean(np.abs(
        temporal_correlations(real, 5)
        - temporal_correlations(smoothed, 5)
    ))),
    "adjacent_level_corr_mae": float(
        np.mean(np.abs(
            cross_level_correlations(real)
            - cross_level_correlations(smoothed)
        ))
    ),
}

print("\n=== CORRELATION ERRORS ===")
print(json.dumps(results["errors"], indent=2))

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved:", OUT / "results.json")
print("V5.13 correlation audit complete!")
