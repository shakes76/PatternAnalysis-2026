import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42
from train_v5_14_multifactor import MultiFactorVolumeHead

N = 512
T = 24
SEED = 371114
ALPHAS = [1.0, 0.7, 0.4, 0.2, 0.1]
OUT = Path("outputs_v5_14_ssim_val")
OUT.mkdir(exist_ok=True)

torch.set_num_threads(4)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

raw = norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
)
real = np.maximum(
    np.expm1(raw[..., 1:40:2]), 0
)

gan_ck = torch.load(
    "outputs_v5_10_latent_matching/timegan_finetuned.pt",
    map_location="cpu",
    weights_only=True,
)

gan = TimeGANV42(
    feature_dim=gan_ck["feature_dim"],
    hidden_dim=gan_ck["hidden_dim"],
    noise_dim=gan_ck["noise_dim"],
    num_layers=gan_ck["num_layers"],
)
gan.load_state_dict(gan_ck["model_state_dict"])
gan.eval()

head_ck = torch.load(
    "outputs_v5_14_multifactor/best_multifactor_head.pt",
    map_location="cpu",
    weights_only=True,
)
head = MultiFactorVolumeHead(
    hidden_dim=head_ck["hidden_dim"],
    mean_log=torch.ones(20),
    std_log=torch.ones(20),
)
head.load_state_dict(head_ck["head_state_dict"])
head.eval()

rng = torch.Generator().manual_seed(SEED)

with torch.no_grad():
    z = torch.rand(
        N, T, gan_ck["noise_dim"],
        generator=rng,
    )
    h = gan.supervisor(gan.generator(z))
    eps = torch.randn(
        N, T, 20, generator=rng
    )
    generated = torch.expm1(
        head(h, eps)
    ).numpy()


def smooth(x, alpha):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0]
    for t in range(1, T):
        y[:, t] = (
            alpha * x[:, t]
            + (1 - alpha) * y[:, t-1]
        )
    return y


def heatmaps(x):
    asks = x[..., 0::2]
    bids = x[..., 1::2]
    depth = np.concatenate(
        [bids[..., ::-1], asks], axis=-1
    )
    return np.log1p(
        np.maximum(depth, 0)
    ).transpose(0, 2, 1)


real_img = heatmaps(real)
scale = max(
    float(np.quantile(real_img, .99)), 1e-6
)

def normalize(img):
    return np.clip(img, 0, scale) / scale

real_img = normalize(real_img)

pairing = np.random.default_rng(SEED).permutation(N)

def compare(a, b):
    return float(structural_similarity(
        a, b,
        data_range=1.0,
        win_size=7
    ))

def random_ssim(images):
    return float(np.mean([
        compare(real_img[i], images[pairing[i]])
        for i in range(N)
    ]))

def nearest_ssim(images, reference_count=128):
    # Diagnostic only: best SSIM among a fixed
    # subset of 128 real validation heatmaps.
    # This is NOT the rubric SSIM.
    reference = real_img[:reference_count]
    scores = []

    for i in range(min(N, 64)):
        candidate = images[i]
        best = max(
            compare(candidate, target)
            for target in reference
        )
        scores.append(best)

    return float(np.mean(scores))


results = {
    "real_self_ssim": float(np.mean([
        compare(x, x) for x in real_img
    ])),
    "real_random_ssim": random_ssim(real_img),
    "alphas": {},
}

print("=== V5.14 VALIDATION DEPTH SSIM ===")
print(
    "Real-to-self:",
    round(results["real_self_ssim"], 4)
)
print(
    "Real-to-random-real:",
    round(results["real_random_ssim"], 4)
)

for alpha in ALPHAS:
    x = smooth(generated, alpha)
    img = normalize(heatmaps(x))

    metrics = {
        "random_pair_ssim": random_ssim(img),
        "nearest_real_ssim_diagnostic": nearest_ssim(img),
        "mean_volume": float(x.mean()),
        "temporal_change": float(
            np.abs(np.diff(x, axis=1)).mean()
        ),
    }

    results["alphas"][str(alpha)] = metrics

    print(
        f"alpha={alpha:<4}"
        f" | random SSIM="
        f"{metrics['random_pair_ssim']:.4f}"
        f" | nearest SSIM="
        f"{metrics['nearest_real_ssim_diagnostic']:.4f}"
        f" | mean={metrics['mean_volume']:.1f}"
        f" | change={metrics['temporal_change']:.2f}"
    )

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved:", OUT / "results.json")
print("V5.14 validation SSIM check complete!")
