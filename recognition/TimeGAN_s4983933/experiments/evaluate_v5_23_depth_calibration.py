import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from skimage.metrics import structural_similarity

from dataset_v4 import prepare_dataset_v4

SEED = 371023
N, T = 512, 24

out = Path("outputs_v5_23_depth_calibration")
out.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def extract(x):
    return np.maximum(
        np.expm1(x[..., 1:40:2]), 0
    )

train_raw = norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
)
val_raw = norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
)

generated = np.load(
    "outputs_v5_21_integrated/generated_lob.npz"
)["relative_orderbook"].astype(np.float64)

train = extract(train_raw)
real = extract(val_raw)
fake = extract(generated)

assert fake.shape == real.shape == (N, T, 20)

# TRAIN provides the target marginal distribution.
# Generated samples provide only rank ordering.
# No validation statistics are used for fitting.
def calibrate(source):
    result = np.zeros_like(source)

    for level in range(20):
        target = np.sort(
            train[..., level].ravel()
        )

        values = source[..., level].ravel()

        quantiles = (
            rankdata(values, method="average") - 0.5
        ) / len(values)

        calibrated = np.quantile(
            target,
            quantiles,
            method="linear"
        )

        result[..., level] = calibrated.reshape(N, T)

    return result

def ema(x, alpha):
    result = np.empty_like(x)
    result[:, 0] = x[:, 0]

    for t in range(1, T):
        result[:, t] = (
            alpha * x[:, t]
            + (1 - alpha) * result[:, t - 1]
        )

    return result

def corr_abs(x):
    z = np.log1p(x).reshape(-1, 20)
    c = np.corrcoef(z, rowvar=False)
    mask = ~np.eye(20, dtype=bool)
    return float(np.mean(np.abs(c[mask])))

def heatmap(x):
    ask = x[..., 0::2]
    bid = x[..., 1::2]
    surface = np.concatenate(
        [bid[..., ::-1], ask], axis=-1
    )
    return np.log1p(surface).transpose(0, 2, 1)

real_img = heatmap(real)
scale = max(
    float(np.quantile(real_img, .99)),
    1e-6
)
real_img = np.clip(real_img / scale, 0, 1)

pairing = np.random.default_rng(SEED).permutation(N)

def random_ssim(x):
    imgs = np.clip(heatmap(x) / scale, 0, 1)
    return float(np.mean([
        structural_similarity(
            real_img[i],
            imgs[pairing[i]],
            data_range=1.0,
            win_size=7
        )
        for i in range(N)
    ]))

def metrics(x):
    real_means = real.mean(axis=(0, 1))
    fake_means = x.mean(axis=(0, 1))

    return {
        "mean": round(float(x.mean()), 3),
        "std": round(float(x.std()), 3),
        "p99": round(float(np.quantile(x, .99)), 3),
        "temporal_change": round(
            float(np.abs(np.diff(x, axis=1)).mean()),
            3
        ),
        "mean_abs_level_correlation": round(
            corr_abs(x), 5
        ),
        "level_mean_mape": round(
            float(np.mean(
                np.abs(fake_means - real_means)
                / np.maximum(real_means, 1e-8)
            )),
            5
        ),
        "random_pair_ssim_diagnostic": round(
            random_ssim(x), 5
        ),
    }

calibrated = calibrate(fake)

candidates = {
    "v5_21_original": fake,
    "level_quantile": calibrated,
    "level_quantile_ema_0_7": ema(calibrated, .7),
    "level_quantile_ema_0_4": ema(calibrated, .4),
}

results = {
    "real_validation": metrics(real),
    "candidates": {}
}

print("=== V5.23 DEPTH CALIBRATION RESULTS ===")
print("REAL:", results["real_validation"])

for name, values in candidates.items():
    result = metrics(values)
    results["candidates"][name] = result
    print(name.upper() + ":", result)

with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.23 evaluation complete!")
