import json
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity
from dataset_v4 import prepare_dataset_v4

SEED = 371026
N, T = 512, 24
rng = np.random.default_rng(SEED)

out = Path("outputs_v5_26_copy_replace")
out.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def volume(book):
    return np.maximum(
        np.expm1(book[..., 1:40:2]), 0
    )

train = volume(norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
))
real = volume(norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
))
fake = volume(np.load(
    "outputs_v5_21_integrated/generated_lob.npz"
)["relative_orderbook"].astype(np.float64))

assert real.shape == fake.shape == (N, T, 20)

before = train[:, :-1].reshape(-1, 20)
after = train[:, 1:].reshape(-1, 20)

# Sample complete joint transition events
# from TRAIN only.
indices = rng.integers(0, len(before), (N, T-1))
sample_before = before[indices]
sample_after = after[indices]

update = np.abs(
    sample_after - sample_before
) >= 1e-5

# Model A: each updated level receives the
# empirical post-event level value.
replace = np.empty_like(fake)
replace[:, 0] = fake[:, 0]

for t in range(1, T):
    replace[:, t] = np.where(
        update[:, t-1],
        sample_after[:, t-1],
        replace[:, t-1]
    )

# Model B: retain GAN candidate values but
# use the same empirical event update masks.
copy_gan = np.empty_like(fake)
copy_gan[:, 0] = fake[:, 0]

for t in range(1, T):
    copy_gan[:, t] = np.where(
        update[:, t-1],
        fake[:, t],
        copy_gan[:, t-1]
    )

def heatmaps(x):
    ask = x[..., 0::2]
    bid = x[..., 1::2]
    surface = np.concatenate(
        [bid[..., ::-1], ask], axis=-1
    )
    return np.log1p(surface).transpose(0, 2, 1)

real_img = heatmaps(real)
scale = max(
    float(np.quantile(real_img, .99)), 1e-6
)
real_img = np.clip(real_img / scale, 0, 1)

pairing = np.random.default_rng(SEED).permutation(N)

def ssim_diagnostic(x):
    generated_img = np.clip(
        heatmaps(x) / scale, 0, 1
    )
    return float(np.mean([
        structural_similarity(
            real_img[i],
            generated_img[pairing[i]],
            data_range=1.0,
            win_size=7
        )
        for i in range(N)
    ]))

def metrics(x):
    diff = np.abs(np.diff(x, axis=1))
    changed = diff >= 1e-5

    logs = np.log1p(x).reshape(-1, 20)
    corr = np.corrcoef(logs, rowvar=False)
    mask = ~np.eye(20, dtype=bool)

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
        "mean_abs_correlation": round(
            float(np.mean(np.abs(corr[mask]))), 5
        ),
        "random_pair_ssim": round(
            ssim_diagnostic(x), 5
        ),
    }

results = {
    "real_validation": metrics(real),
    "original_v5_21": metrics(fake),
    "copy_gan": metrics(copy_gan),
    "copy_replace": metrics(replace),
}

print("=== V5.26 COPY-REPLACE RESULTS ===")
for name, result in results.items():
    print(name.upper(), result)

with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

np.savez_compressed(
    out / "candidates.npz",
    copy_gan=copy_gan.astype(np.float32),
    copy_replace=replace.astype(np.float32)
)

print("\nV5.26 experiment complete!")
