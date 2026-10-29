import json
import runpy
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

# Reuse the exact V5.13 model, data and EMA result.
s = runpy.run_path("diagnose_v5_13_correlations.py")

real = np.asarray(s["real"], dtype=np.float64)
generated = np.asarray(s["smoothed"], dtype=np.float64)

N, T, L = generated.shape
assert L == 20

rng = np.random.default_rng(202613)

# Preserve the entire 24-step trajectory of each
# individual level, but independently shuffle the
# sequence identities for all 20 levels.
shuffled = np.empty_like(generated)

for level in range(L):
    perm = rng.permutation(N)
    shuffled[:, :, level] = generated[perm, :, level]

def adjacent_corr(x):
    values = []
    for side in (0, 1):
        for level in range(9):
            j = 2 * level + side
            k = j + 2
            a = x[..., j].ravel()
            b = x[..., k].ravel()
            values.append(np.corrcoef(a, b)[0, 1])
    return np.array(values)

def lag_corr(x, lag=1):
    values = []
    for level in range(20):
        a = x[:, :-lag, level].ravel()
        b = x[:, lag:, level].ravel()
        values.append(np.corrcoef(a, b)[0, 1])
    return np.array(values)

def images(x):
    asks = x[..., 0::2]
    bids = x[..., 1::2]
    depth = np.concatenate(
        [bids[..., ::-1], asks], axis=-1
    )
    return np.log1p(
        np.maximum(depth, 0)
    ).transpose(0, 2, 1)

scale = max(
    float(np.quantile(images(real), 0.99)),
    1e-6
)

reference_img = np.clip(
    images(real), 0, scale
) / scale

pairing = np.random.default_rng(12346).permutation(N)

def measure_ssim(x):
    candidate = np.clip(
        images(x), 0, scale
    ) / scale

    return float(np.mean([
        structural_similarity(
            reference_img[i],
            candidate[pairing[i]],
            data_range=1.0,
            win_size=7
        )
        for i in range(N)
    ]))

def metrics(x):
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p99": float(np.quantile(x, .99)),
        "temporal_change": float(
            np.abs(np.diff(x, axis=1)).mean()
        ),
        "lag1_mean": float(lag_corr(x, 1).mean()),
        "lag5_mean": float(lag_corr(x, 5).mean()),
        "adjacent_corr_mean": float(
            adjacent_corr(x).mean()
        ),
        "adjacent_corr_abs_mean": float(
            np.abs(adjacent_corr(x)).mean()
        ),
        "random_pair_ssim": measure_ssim(x),
    }

results = {
    "real": {
        "adjacent_corr_mean": float(
            adjacent_corr(real).mean()
        ),
        "adjacent_corr_abs_mean": float(
            np.abs(adjacent_corr(real)).mean()
        ),
    },
    "original_ema": metrics(generated),
    "independent_level_shuffle": metrics(shuffled),
    "note": (
        "Diagnostic permutation only. Each level's "
        "24-step trajectory is preserved, but cross-level "
        "joint structure is artificially disrupted."
    ),
}

out = Path("outputs_v5_13_level_shuffle")
out.mkdir(exist_ok=True)

with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== LEVEL INDEPENDENCE EXPERIMENT ===")
print(json.dumps(results, indent=2))
print("\nSaved:", out / "results.json")
