import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataset_v4 import prepare_dataset_v4
from depth_model_v5_29 import ProbabilisticDepthNet

SEED = 371029
N, T = 512, 24
THRESHOLD = 1e-5

torch.manual_seed(SEED)
torch.set_num_threads(4)

out = Path("outputs_v5_29_mdn")
out.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def extract(book):
    return np.maximum(
        np.expm1(book[..., 1:40:2]), 0
    )

real = extract(norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
))

fake = extract(np.load(
    "outputs_v5_21_integrated/generated_lob.npz"
)["relative_orderbook"].astype(np.float64))

assert real.shape == fake.shape == (N, T, 20)

class DeterministicDepthNet(nn.Module):
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

# Checkpoints come from our own training runs.
ck27 = torch.load(
    "outputs_v5_27_state_depth/best_depth_state.pt",
    map_location="cpu",
    weights_only=False
)
ck29 = torch.load(
    "outputs_v5_29_mdn/best_mdn.pt",
    map_location="cpu",
    weights_only=False
)

net27 = DeterministicDepthNet()
net27.load_state_dict(ck27["model_state_dict"])
net27.eval()

net29 = ProbabilisticDepthNet(
    hidden=ck29["hidden"],
    components=ck29["components"]
)
net29.load_state_dict(ck29["model_state_dict"])
net29.eval()

mu27 = torch.tensor(ck27["mu"], dtype=torch.float32)
sd27 = torch.tensor(ck27["sd"], dtype=torch.float32)

mu29 = torch.tensor(ck29["mu"], dtype=torch.float32)
sd29 = torch.tensor(ck29["sd"], dtype=torch.float32)

# Shared mask uniform random numbers.
mask_rng = np.random.default_rng(SEED)
uniforms = mask_rng.random((N, T-1, 20))

# Separate random stream for mixture sampling.
sample_rng = np.random.default_rng(SEED + 1)

def rollout(version):
    result = np.empty_like(fake)
    result[:, 0] = fake[:, 0]
    probabilities = []

    with torch.no_grad():
        for t in range(1, T):
            current = torch.from_numpy(
                np.log1p(result[:, t-1]).astype(np.float32)
            )

            if version == "v27":
                logits, values = net27(
                    (current - mu27) / sd27
                )
                proposed_log = np.clip(
                    values.numpy(), 0, 15
                )

            else:
                logits, weights, means, log_scales = net29(
                    (current - mu29) / sd29
                )

                weights = torch.softmax(
                    weights, dim=-1
                ).numpy()
                means = means.numpy()
                scales = torch.exp(log_scales).numpy()

                # Choose one Gaussian component per
                # sample and price level.
                cumulative = np.cumsum(
                    weights, axis=-1
                )

                u = sample_rng.random((N, 20, 1))
                component = (
                    u > cumulative[..., :-1]
                ).sum(axis=-1)

                chosen_mean = np.take_along_axis(
                    means,
                    component[..., None],
                    axis=-1
                )[..., 0]

                chosen_scale = np.take_along_axis(
                    scales,
                    component[..., None],
                    axis=-1
                )[..., 0]

                sampled_normalized = (
                    chosen_mean
                    + chosen_scale * sample_rng.standard_normal(
                        (N, 20)
                    )
                )

                # V5.29 target was standardized
                # during training, unlike V5.27.
                proposed_log = (
                    sampled_normalized
                    * sd29.numpy()
                    + mu29.numpy()
                )
                proposed_log = np.clip(
                    proposed_log, 0, 15
                )

            probabilities.append(
                torch.sigmoid(logits).numpy()
            )

            update = (
                uniforms[:, t-1]
                < probabilities[-1]
            )

            proposed_volume = np.expm1(
                proposed_log
            )

            result[:, t] = np.where(
                update,
                proposed_volume,
                result[:, t-1]
            )

    return result, np.stack(
        probabilities, axis=1
    )

def metrics(x):
    diff = np.abs(np.diff(x, axis=1))
    changed = diff >= THRESHOLD

    log = np.log1p(x).reshape(-1, 20)
    corr = np.corrcoef(log, rowvar=False)
    offdiag = ~np.eye(20, dtype=bool)

    level_means = x.mean(axis=(0, 1))
    real_means = real.mean(axis=(0, 1))

    return {
        "mean": round(float(x.mean()), 3),
        "std": round(float(x.std()), 3),
        "p99": round(
            float(np.quantile(x, .99)), 3
        ),
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
            float(np.abs(corr[offdiag]).mean()), 5
        ),
        "level_mean_mape": round(
            float(np.mean(
                np.abs(level_means - real_means)
                / np.maximum(real_means, 1e-8)
            )), 5
        ),
        "first_step_mean": round(
            float(x[:, 0].mean()), 3
        ),
        "last_step_mean": round(
            float(x[:, -1].mean()), 3
        )
    }

roll27, prob27 = rollout("v27")
roll29, prob29 = rollout("v29")

results = {
    "real_validation": metrics(real),
    "original_v5_21": metrics(fake),
    "v5_27_deterministic": metrics(roll27),
    "v5_29_mdn": metrics(roll29),
    "mean_update_probability": {
        "v5_27": round(float(prob27.mean()), 5),
        "v5_29": round(float(prob29.mean()), 5)
    },
    "best_mdn_epoch": int(ck29["epoch"])
}

print("=== V5.29 FREE ROLLOUT RESULTS ===")
for name, result in results.items():
    print(name.upper(), result)

with open(out / "rollout_results.json", "w") as f:
    json.dump(results, f, indent=2)

np.savez_compressed(
    out / "rollout_candidates.npz",
    deterministic_v27=roll27.astype(np.float32),
    probabilistic_v29=roll29.astype(np.float32)
)

print("\nV5.29 free rollout complete!")
