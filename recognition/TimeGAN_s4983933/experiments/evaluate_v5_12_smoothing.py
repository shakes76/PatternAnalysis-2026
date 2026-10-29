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

SEED = 12345
N = 512
T = 24
ALPHAS = [1.0, 0.7, 0.4, 0.2, 0.1]
OUT = Path("outputs_v5_12_smoothing")
OUT.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(4)

data = prepare_dataset_v4("data")
norm = data["normalizer"]
book_norm = norm.book_normalizer

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

hc = torch.load(
    "outputs_v5_9_variance_aware/variance_aware_best.pt",
    map_location="cpu",
    weights_only=True,
)

decoder = MLPConstraintAwareLOBHead(
    hidden_dim=hc["hidden_dim"],
    minimum=book_norm.minimum,
    scale=book_norm.scale,
    min_spread=0.01,
    min_gap=0.01,
    intermediate=hc["intermediate"],
)
decoder.load_state_dict(hc["head_state_dict"])
decoder.eval()

vc = torch.load(
    "outputs_v5_12_depth_aware/best_volume_head.pt",
    map_location="cpu",
    weights_only=True,
)

head = StochasticVolumeHead(
    hidden_dim=vc["hidden_dim"]
)
head.load_state_dict(vc["head_state_dict"])
head.eval()


def generate(seed):
    rng = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        z = torch.rand(
            N, T, ck["noise_dim"], generator=rng
        )
        h = model.supervisor(model.generator(z))
        raw = decoder.forward_raw(h).numpy()

        noise = correlated_noise(
            N, T, generator=rng
        )
        base = torch.tensor(
            raw[..., 1::2], dtype=torch.float32
        )
        log_volume = head(h, base, noise).numpy()

    return raw, np.expm1(log_volume)


def smooth_volumes(volumes, alpha):
    # Causal exponential moving average.
    # alpha=1 means no smoothing.
    result = np.empty_like(volumes)
    result[:, 0] = volumes[:, 0]

    for t in range(1, volumes.shape[1]):
        result[:, t] = (
            alpha * volumes[:, t]
            + (1 - alpha) * result[:, t - 1]
        )
    return np.maximum(result, 0.0)


def real_book(split):
    x = np.asarray(data[split][:N], dtype=np.float64)
    return norm.inverse_transform(x)[..., :40]


def volumes_from_book(book):
    return np.expm1(book[..., 1::2])


def depth_images(volumes):
    # Original order: Ask1, Bid1, Ask2, Bid2...
    asks = volumes[..., 0::2]
    bids = volumes[..., 1::2]
    depth = np.concatenate(
        [bids[..., ::-1], asks], axis=-1
    )
    return np.log1p(depth).transpose(0, 2, 1)


def stats(x):
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p99": float(np.quantile(x, 0.99)),
        "temporal_change": float(
            np.abs(np.diff(x, axis=1)).mean()
        ),
        "across_sample_std": float(
            x.std(axis=0).mean()
        ),
    }


def ssim_scores(reference, candidate, pairing, scale):
    real_img = np.clip(
        depth_images(reference), 0, scale
    ) / scale
    candidate_img = np.clip(
        depth_images(candidate), 0, scale
    ) / scale

    return np.array([
        structural_similarity(
            real_img[i],
            candidate_img[pairing[i]],
            data_range=1.0,
            win_size=7,
        )
        for i in range(len(reference))
    ])


def evaluate_split(split, seed):
    real = volumes_from_book(real_book(split))
    _, generated = generate(seed)

    rng = np.random.default_rng(seed)
    pairing = rng.permutation(N)

    scale = max(
        float(np.quantile(depth_images(real), 0.99)),
        1e-6,
    )

    identity = np.arange(N)
    real_self = ssim_scores(
        real, real, identity, scale
    )
    real_random = ssim_scores(
        real, real, pairing, scale
    )

    records = {}
    for alpha in ALPHAS:
        candidate = smooth_volumes(generated, alpha)
        candidate_stats = stats(candidate)

        scores = ssim_scores(
            real, candidate, pairing, scale
        )

        records[str(alpha)] = {
            **candidate_stats,
            "mean_ssim": float(scores.mean()),
            "median_ssim": float(np.median(scores)),
            "ssim_std": float(scores.std()),
        }

    return {
        "real": stats(real),
        "ssim_sanity": {
            "real_self": float(real_self.mean()),
            "real_random": float(real_random.mean()),
        },
        "candidates": records,
    }


val = evaluate_split("val", SEED)

# Selection uses Validation only, based on matching
# raw-volume temporal change and std. We do not
# choose alpha using random-pair SSIM.
def selection_score(record, target):
    eps = 1e-12
    temporal_error = abs(
        np.log(
            (record["temporal_change"] + eps)
            / (target["temporal_change"] + eps)
        )
    )
    std_error = abs(
        np.log(
            (record["std"] + eps)
            / (target["std"] + eps)
        )
    )
    return temporal_error + 0.5 * std_error


best_alpha = min(
    ALPHAS,
    key=lambda a: selection_score(
        val["candidates"][str(a)], val["real"]
    ),
)

test = evaluate_split("test", SEED + 1)

results = {
    "experiment": "V5.12 causal EMA ablation",
    "model_selection": (
        "Validation temporal-change log-ratio "
        "+ 0.5 * volume-std log-ratio; "
        "no test-based selection"
    ),
    "selected_alpha": best_alpha,
    "validation": val,
    "test": test,
    "limitations": (
        "Random-pair SSIM compares unrelated sequences. "
        "EMA is post-processing, not a trained temporal model."
    ),
}

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("=== VALIDATION SMOOTHING ABLATION ===")
print("Real:", json.dumps(val["real"]))
print("SSIM sanity:", val["ssim_sanity"])

for alpha in ALPHAS:
    r = val["candidates"][str(alpha)]
    print(
        f"alpha={alpha:<4}"
        f" | temporal={r['temporal_change']:.3f}"
        f" | std={r['std']:.3f}"
        f" | p99={r['p99']:.1f}"
        f" | SSIM={r['mean_ssim']:.4f}"
    )

print("\nSelected on Validation:", best_alpha)

print("\n=== TEST SMOOTHING ABLATION ===")
print("Real:", json.dumps(test["real"]))
print("SSIM sanity:", test["ssim_sanity"])

for alpha in ALPHAS:
    r = test["candidates"][str(alpha)]
    marker = " <-- VAL SELECTED" if alpha == best_alpha else ""
    print(
        f"alpha={alpha:<4}"
        f" | temporal={r['temporal_change']:.3f}"
        f" | std={r['std']:.3f}"
        f" | p99={r['p99']:.1f}"
        f" | SSIM={r['mean_ssim']:.4f}"
        + marker
    )

print("\nSaved:", OUT / "results.json")
print("Smoothing ablation complete!")
