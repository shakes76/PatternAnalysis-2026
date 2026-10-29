import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dataset_v4 import prepare_dataset_v4

SEED = 371027
N, T = 512, 24
torch.manual_seed(SEED)
torch.set_num_threads(4)

out = Path("outputs_v5_27_state_depth")
data = prepare_dataset_v4("data")
norm = data["normalizer"]

def extract(x):
    return np.maximum(
        np.expm1(x[..., 1:40:2]), 0
    )

real = extract(norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
))

original = extract(np.load(
    "outputs_v5_21_integrated/generated_lob.npz"
)["relative_orderbook"].astype(np.float64))

class StateDepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mask_head = nn.Linear(128, 20)
        self.value_head = nn.Linear(128, 20)

    def forward(self, x):
        h = self.encoder(x)
        return self.mask_head(h), self.value_head(h)

# This checkpoint was produced by our own
# trusted V5.27 training script.
ck = torch.load(
    out / "best_depth_state.pt",
    map_location="cpu",
    weights_only=False
)

model = StateDepthNet()
model.load_state_dict(ck["model_state_dict"])
model.eval()

mu = torch.tensor(ck["mu"], dtype=torch.float32)
sd = torch.tensor(ck["sd"], dtype=torch.float32)

def rollout(stochastic):
    rng = np.random.default_rng(SEED)
    x = np.empty_like(original)
    x[:, 0] = original[:, 0]

    probabilities = []

    with torch.no_grad():
        for t in range(1, T):
            current = torch.from_numpy(
                np.log1p(x[:, t-1]).astype(np.float32)
            )
            inputs = (current - mu) / sd

            logits, predicted = model(inputs)
            prob = torch.sigmoid(logits).numpy()
            probabilities.append(prob)

            if stochastic:
                update = rng.random(prob.shape) < prob
            else:
                update = prob >= 0.5

            # Value head predicts log1p(volume).
            next_values = np.maximum(
                np.expm1(
                    np.clip(predicted.numpy(), 0, 15)
                ),
                0
            )

            x[:, t] = np.where(
                update,
                next_values,
                x[:, t-1]
            )

    return x, np.stack(probabilities, axis=1)

def metrics(x):
    difference = np.abs(np.diff(x, axis=1))
    updated = difference >= 1e-5

    z = np.log1p(x).reshape(-1, 20)
    corr = np.corrcoef(z, rowvar=False)
    offdiag = ~np.eye(20, dtype=bool)

    return {
        "mean": round(float(x.mean()), 3),
        "std": round(float(x.std()), 3),
        "p99": round(float(np.quantile(x, .99)), 3),
        "unchanged_fraction": round(
            float((~updated).mean()), 5
        ),
        "mean_temporal_change": round(
            float(difference.mean()), 3
        ),
        "conditional_change": (
            round(float(difference[updated].mean()), 3)
            if updated.any() else 0.0
        ),
        "conditional_median": (
            round(float(np.median(difference[updated])), 3)
            if updated.any() else 0.0
        ),
        "mean_abs_level_correlation": round(
            float(np.abs(corr[offdiag]).mean()), 5
        ),
        "first_step_mean": round(
            float(x[:, 0].mean()), 3
        ),
        "last_step_mean": round(
            float(x[:, -1].mean()), 3
        )
    }

sampled, sampled_p = rollout(True)
thresholded, threshold_p = rollout(False)

results = {
    "checkpoint_epoch": int(ck["epoch"]),
    "real_validation": metrics(real),
    "original_v5_21": metrics(original),
    "v5_27_probability_sampling": metrics(sampled),
    "v5_27_threshold_0_5": metrics(thresholded),
    "mean_predicted_update_probability": {
        "sampling": float(sampled_p.mean()),
        "threshold": float(threshold_p.mean())
    }
}

print("=== V5.27 FREE ROLLOUT RESULTS ===")
for name, value in results.items():
    print(name.upper(), value)

with open(out / "rollout_results.json", "w") as f:
    json.dump(results, f, indent=2)

np.savez_compressed(
    out / "rollout_candidates.npz",
    probability_sampling=sampled.astype(np.float32),
    threshold_0_5=thresholded.astype(np.float32)
)

print("\nV5.27 free rollout complete!")
