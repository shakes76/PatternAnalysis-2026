import json
from pathlib import Path
import numpy as np

from dataset_v4 import prepare_dataset_v4

SEED = 371025
N, T = 512, 24
OUT = Path("outputs_v5_25_copy_update")
OUT.mkdir(exist_ok=True)

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
val = volumes(norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
))
fake = volumes(np.load(
    "outputs_v5_21_integrated/generated_lob.npz"
)["relative_orderbook"].astype(np.float64))

assert val.shape == fake.shape == (N, T, 20)

# Real train transitions, including their
# cross-level co-update patterns.
train_delta = np.diff(train, axis=1)
transition_pool = train_delta.reshape(-1, 20)

# Draw entire 20-level transition vectors.
indices = rng.integers(
    0, len(transition_pool), size=(N, T - 1)
)
sampled_delta = transition_pool[indices]
update_mask = np.abs(sampled_delta) >= 1e-5

copy_gan = np.empty_like(fake)
copy_delta = np.empty_like(fake)

copy_gan[:, 0] = fake[:, 0]
copy_delta[:, 0] = fake[:, 0]

for t in range(1, T):
    mask = update_mask[:, t - 1]

    # Method A: GAN candidate when update occurs.
    copy_gan[:, t] = np.where(
        mask,
        fake[:, t],
        copy_gan[:, t - 1]
    )

    # Method B: replay TRAIN change increments.
    copy_delta[:, t] = np.maximum(
        copy_delta[:, t - 1]
        + sampled_delta[:, t - 1],
        0
    )

def metrics(x):
    d = np.abs(np.diff(x, axis=1))
    changed = d >= 1e-5

    z = np.log1p(x).reshape(-1, 20)
    corr = np.corrcoef(z, rowvar=False)
    offdiag = ~np.eye(20, dtype=bool)

    return {
        "mean": round(float(x.mean()), 3),
        "std": round(float(x.std()), 3),
        "p99": round(
            float(np.quantile(x, .99)), 3
        ),
        "unchanged_fraction": round(
            float((~changed).mean()), 5
        ),
        "mean_temporal_change": round(
            float(d.mean()), 3
        ),
        "conditional_change": round(
            float(d[changed].mean()), 3
        ) if changed.any() else 0.0,
        "conditional_median": round(
            float(np.median(d[changed])), 3
        ) if changed.any() else 0.0,
        "mean_abs_level_correlation": round(
            float(np.mean(np.abs(
                corr[offdiag]
            ))), 5
        ),
    }

results = {
    "real_validation": metrics(val),
    "v5_21_original": metrics(fake),
    "copy_gan": metrics(copy_gan),
    "copy_train_delta": metrics(copy_delta),
    "train_mask_update_fraction": float(
        update_mask.mean()
    ),
}

print("=== V5.25 COPY-OR-UPDATE RESULTS ===")
for name, values in results.items():
    print(name.upper(), values)

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

np.savez_compressed(
    OUT / "depth_candidates.npz",
    copy_gan=copy_gan.astype(np.float32),
    copy_train_delta=copy_delta.astype(np.float32),
)

print("\nV5.25 experiment complete!")
