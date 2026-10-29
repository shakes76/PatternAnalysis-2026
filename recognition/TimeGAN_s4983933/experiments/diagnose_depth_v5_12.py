import json
from pathlib import Path

import numpy as np
from dataset_v4 import prepare_dataset_v4

OUT = Path("outputs_depth_diagnostic")
OUT.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

real = norm.inverse_transform(
    np.asarray(data["test"][:512], dtype=np.float64)
)[..., :40]

saved = np.load(
    "outputs_v5_10_full_lob/generated_full_lob.npz"
)
fake = np.asarray(
    saved["relative_orderbook"], dtype=np.float64
)

assert real.shape == fake.shape == (512, 24, 40)

def extract_depth(book):
    # Relative book uses:
    # AskPrice, log1p(AskSize), BidPrice, log1p(BidSize)
    asks = np.expm1(book[..., 1::4])
    bids = np.expm1(book[..., 3::4])

    # Shape: N x T x 20
    return np.concatenate([bids, asks], axis=-1)

real_depth = extract_depth(real)
fake_depth = extract_depth(fake)

names = (
    [f"Bid{i}" for i in range(1, 11)]
    + [f"Ask{i}" for i in range(1, 11)]
)

def describe(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
        "pair_difference": float(
            np.abs(x[::2] - x[1::2]).mean()
        ),
        "temporal_change": float(
            np.abs(np.diff(x, axis=1)).mean()
        ),
        "across_sample_std": float(
            x.std(axis=0).mean()
        ),
    }

results = {
    "real": describe(real_depth),
    "fake": describe(fake_depth),
    "levels": [],
}

print("=== DEPTH DISTRIBUTION AUDIT ===")
print("\nREAL:")
print(json.dumps(results["real"], indent=2))
print("\nGENERATED:")
print(json.dumps(results["fake"], indent=2))

print("\n=== LEVEL-WISE COMPARISON ===")
print(
    f"{'Level':<8}"
    f"{'Real mean':>12}"
    f"{'Fake mean':>12}"
    f"{'Real std':>12}"
    f"{'Fake std':>12}"
)

for j, name in enumerate(names):
    r = real_depth[..., j]
    f = fake_depth[..., j]

    rec = {
        "level": name,
        "real_mean": float(r.mean()),
        "fake_mean": float(f.mean()),
        "real_std": float(r.std()),
        "fake_std": float(f.std()),
        "real_temporal_change": float(
            np.abs(np.diff(r, axis=1)).mean()
        ),
        "fake_temporal_change": float(
            np.abs(np.diff(f, axis=1)).mean()
        ),
    }
    results["levels"].append(rec)

    print(
        f"{name:<8}"
        f"{rec['real_mean']:>12.2f}"
        f"{rec['fake_mean']:>12.2f}"
        f"{rec['real_std']:>12.2f}"
        f"{rec['fake_std']:>12.2f}"
    )

# Assess the shape of the depth profile,
# independently of overall volume magnitude.
real_profile = real_depth.mean(axis=(0, 1))
fake_profile = fake_depth.mean(axis=(0, 1))

real_profile = real_profile / max(
    real_profile.sum(), 1e-12
)
fake_profile = fake_profile / max(
    fake_profile.sum(), 1e-12
)

results["depth_profile_l1"] = float(
    np.abs(real_profile - fake_profile).sum()
)

print("\n=== KEY DIAGNOSTICS ===")
print(
    "Depth profile L1 distance:",
    results["depth_profile_l1"]
)
print(
    "Generated / real across-sample std:",
    results["fake"]["across_sample_std"]
    / max(results["real"]["across_sample_std"], 1e-12)
)
print(
    "Generated / real temporal change:",
    results["fake"]["temporal_change"]
    / max(results["real"]["temporal_change"], 1e-12)
)

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved:", OUT / "results.json")
print("Depth diagnostic complete!")
