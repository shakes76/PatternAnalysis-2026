import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42
from lob_head_v5_8 import MLPConstraintAwareLOBHead
from train_v5_12_depth_aware import (
    StochasticVolumeHead,
    correlated_noise,
)

N = 512
T = 24
SEED = 12345
OUT = Path("outputs_v5_12_depth_evaluation")
OUT.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(4)
torch.manual_seed(SEED)

data = prepare_dataset_v4("data")
norm = data["normalizer"]
book_norm = norm.book_normalizer

real = norm.inverse_transform(
    np.asarray(data["test"][:N], dtype=np.float64)
)[..., :40]

# Frozen V5.10 TimeGAN.
ck = torch.load(
    "outputs_v5_10_latent_matching/timegan_finetuned.pt",
    map_location="cpu",
    weights_only=True,
)

model = TimeGANV42(
    feature_dim=ck["feature_dim"],
    hidden_dim=ck["hidden_dim"],
    noise_dim=ck["noise_dim"],
    num_layers=ck["num_layers"],
)
model.load_state_dict(ck["model_state_dict"])
model.eval()

# Frozen V5.9 constrained price/volume decoder.
hc = torch.load(
    "outputs_v5_9_variance_aware/variance_aware_best.pt",
    map_location="cpu",
    weights_only=True,
)

decoder = MLPConstraintAwareLOBHead(
    hidden_dim=hc["hidden_dim"],
    minimum=book_norm.minimum,
    scale=book_norm.scale,
    intermediate=hc["intermediate"],
    min_spread=0.01,
    min_gap=0.01,
)
decoder.load_state_dict(hc["head_state_dict"])
decoder.eval()

# V5.12 stochastic volume correction.
vc = torch.load(
    "outputs_v5_12_depth_aware/best_volume_head.pt",
    map_location="cpu",
    weights_only=True,
)
volume_head = StochasticVolumeHead(
    hidden_dim=vc["hidden_dim"]
)
volume_head.load_state_dict(vc["head_state_dict"])
volume_head.eval()

rng = torch.Generator().manual_seed(SEED)

with torch.no_grad():
    z = torch.rand(
        N, T, ck["noise_dim"],
        generator=rng
    )
    hidden = model.supervisor(model.generator(z))
    base_book = decoder.forward_raw(hidden)

    volume_noise = correlated_noise(
        N, T, generator=rng
    )

    base_volumes = base_book[..., 1::2]
    new_volumes = volume_head(
        hidden, base_volumes, volume_noise
    )

    improved_book = base_book.clone()
    improved_book[..., 1::2] = new_volumes

old = base_book.numpy()
new = improved_book.numpy()

def volumes(book):
    return np.expm1(book[..., 1::2])

def depth_image(book):
    asks = np.maximum(
        np.expm1(book[..., 1::4]), 0
    )
    bids = np.maximum(
        np.expm1(book[..., 3::4]), 0
    )

    depth = np.concatenate(
        [bids[..., ::-1], asks], axis=-1
    )
    return np.log1p(depth).transpose(0, 2, 1)

real_depth = depth_image(real)
scale = max(
    float(np.quantile(real_depth, 0.99)),
    1e-6
)

real_img = np.clip(
    real_depth, 0, scale
) / scale

pairing = np.random.default_rng(SEED).permutation(N)

def evaluate(name, generated):
    v = volumes(generated)
    img = np.clip(
        depth_image(generated), 0, scale
    ) / scale

    scores = [
        structural_similarity(
            real_img[i],
            img[pairing[i]],
            data_range=1.0,
            win_size=7
        )
        for i in range(N)
    ]

    return {
        "name": name,
        "volume_mean": float(v.mean()),
        "volume_std": float(v.std()),
        "volume_p99": float(np.quantile(v, .99)),
        "across_sample_std": float(
            v.std(axis=0).mean()
        ),
        "temporal_change": float(
            np.abs(np.diff(v, axis=1)).mean()
        ),
        "mean_ssim": float(np.mean(scores)),
        "median_ssim": float(np.median(scores)),
        "ssim_min": float(np.min(scores)),
        "ssim_max": float(np.max(scores)),
        "valid_volume_fraction": float(
            np.mean(np.isfinite(v) & (v >= 0))
        )
    }

results = {
    "real_test": {
        "volume_mean": float(volumes(real).mean()),
        "volume_std": float(volumes(real).std()),
        "volume_p99": float(
            np.quantile(volumes(real), .99)
        ),
        "across_sample_std": float(
            volumes(real).std(axis=0).mean()
        ),
        "temporal_change": float(
            np.abs(np.diff(
                volumes(real), axis=1
            )).mean()
        ),
    },
    "v5_10_original": evaluate("V5.10", old),
    "v5_12_volume_head": evaluate("V5.12", new),
    "best_training_epoch": vc["best_epoch"],
    "ssim_protocol": (
        "512 independently paired 20x24 log-volume "
        "images, shared real-test p99 scale, win_size=7"
    ),
}

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

np.save(
    OUT / "generated_relative_book.npy", new
)

print("=== V5.12 TEST DEPTH COMPARISON ===")
print(json.dumps(results, indent=2))

print("\nSaved:", OUT / "results.json")
print("V5.12 test evaluation complete!")
