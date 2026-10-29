import json
from pathlib import Path

import numpy as np
import torch

from dataset_v4 import prepare_dataset_v4
from depth_model_v5_29 import ProbabilisticDepthNet

SEED = 371031
N, T = 512, 24
EPS = 1e-5

torch.set_num_threads(4)

out = Path("outputs_v5_31_bounded_mdn")
out.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def extract(book):
    return np.maximum(
        np.expm1(book[..., 1:40:2]), 0
    )

train = extract(norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
))
real = extract(norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
))
initial_source = extract(np.load(
    "outputs_v5_21_integrated/generated_lob.npz"
)["relative_orderbook"].astype(np.float64))

assert real.shape == initial_source.shape == (N, T, 20)

ck = torch.load(
    "outputs_v5_29_mdn/best_mdn.pt",
    map_location="cpu",
    weights_only=False
)

model = ProbabilisticDepthNet(
    hidden=ck["hidden"],
    components=ck["components"]
)
model.load_state_dict(ck["model_state_dict"])
model.eval()

mu = np.asarray(ck["mu"], dtype=np.float64)
sd = np.asarray(ck["sd"], dtype=np.float64)

# Per-level boundaries estimated ONLY from TRAIN.
train_flat = train.reshape(-1, 20)

caps = {
    "original_log_15": np.full(
        20, np.expm1(15.0)
    ),
    "train_p999": np.quantile(
        train_flat, 0.999, axis=0
    ),
    "train_p9999": np.quantile(
        train_flat, 0.9999, axis=0
    ),
    "train_maximum": train_flat.max(axis=0),
}

# Precompute identical random numbers across
# all methods for controlled comparison.
rng = np.random.default_rng(SEED)
mask_u = rng.random((T - 1, N, 20))
component_u = rng.random((T - 1, N, 20, 1))
gaussian_z = rng.standard_normal((T - 1, N, 20))

def rollout(cap):
    x = np.empty_like(initial_source)
    x[:, 0] = initial_source[:, 0]

    proposed_count = 0
    out_of_bounds = 0
    selected_count = 0
    selected_out_of_bounds = 0

    with torch.no_grad():
        for t in range(1, T):
            current = np.log1p(x[:, t-1])
            inputs = torch.tensor(
                ((current - mu) / sd).astype(
                    np.float32
                )
            )

            logits, weights, means, scales_log = model(
                inputs
            )

            probabilities = torch.sigmoid(
                logits
            ).numpy()

            weights = torch.softmax(
                weights, dim=-1
            ).numpy()

            means = means.numpy()
            scales = torch.exp(scales_log).numpy()

            cumulative = np.cumsum(
                weights, axis=-1
            )

            component = (
                component_u[t-1]
                > cumulative[..., :-1]
            ).sum(axis=-1)

            chosen_means = np.take_along_axis(
                means,
                component[..., None],
                axis=-1
            )[..., 0]

            chosen_scales = np.take_along_axis(
                scales,
                component[..., None],
                axis=-1
            )[..., 0]

            normalized = (
                chosen_means
                + chosen_scales * gaussian_z[t-1]
            )

            raw_log = normalized * sd + mu

            upper = np.log1p(cap)[None, :]

            exceeded = raw_log > upper
            proposed_count += exceeded.size
            out_of_bounds += int(exceeded.sum())

            update = mask_u[t-1] < probabilities
            selected_count += int(update.sum())
            selected_out_of_bounds += int(
                (update & exceeded).sum()
            )

            bounded_log = np.clip(
                raw_log, 0, upper
            )
            proposed = np.expm1(bounded_log)

            x[:, t] = np.where(
                update, proposed, x[:, t-1]
            )

    return x, {
        "proposal_upper_exceed_fraction": (
            out_of_bounds / max(proposed_count, 1)
        ),
        "selected_upper_exceed_fraction": (
            selected_out_of_bounds
            / max(selected_count, 1)
        ),
    }

def metrics(x):
    diff = np.abs(np.diff(x, axis=1))
    changed = diff >= EPS

    z = np.log1p(x).reshape(-1, 20)
    corr = np.corrcoef(z, rowvar=False)
    offdiag = ~np.eye(20, dtype=bool)

    return {
        "mean": round(float(x.mean()), 3),
        "std": round(float(x.std()), 3),
        "p99": round(
            float(np.quantile(x, .99)), 3
        ),
        "p999": round(
            float(np.quantile(x, .999)), 3
        ),
        "max": round(float(x.max()), 3),
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
            float(np.abs(corr[offdiag]).mean()), 5
        ),
        "first_step_mean": round(
            float(x[:, 0].mean()), 3
        ),
        "last_step_mean": round(
            float(x[:, -1].mean()), 3
        ),
    }

results = {
    "real_validation": metrics(real),
    "variants": {},
}

print("=== V5.31 BOUNDED MDN RESULTS ===")
print("REAL_VALIDATION", results["real_validation"])

for name, cap in caps.items():
    generated, boundary = rollout(cap)

    result = {
        **metrics(generated),
        **boundary,
    }

    results["variants"][name] = result
    print(name.upper(), result)

    np.save(
        out / f"{name}.npy",
        generated.astype(np.float32)
    )

with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.31 bounded MDN experiment complete!")
