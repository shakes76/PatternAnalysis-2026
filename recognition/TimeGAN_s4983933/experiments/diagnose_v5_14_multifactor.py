import numpy as np
from dataset_v4 import prepare_dataset_v4

SEED = 371014
N = 512
rng = np.random.default_rng(SEED)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def load_volumes(split):
    raw = norm.inverse_transform(
        np.asarray(data[split], dtype=np.float64)
    )
    return np.maximum(np.expm1(raw[..., 1:40:2]), 0)

train = load_volumes("train")
val = load_volumes("val")[:N]

# Independent 24-event trajectories sampled from
# TRAIN for each of the 20 order-book levels.
# Never use validation or test samples to generate.
independent = np.empty_like(val)

for j in range(20):
    selected = rng.integers(0, len(train), size=N)
    independent[..., j] = train[selected, :, j]

# Entire book replay provides a joint-distribution
# upper baseline for this train-only resampling method.
joint_ids = rng.integers(0, len(train), size=N)
joint = train[joint_ids]

def pca_stats(x):
    a = np.log1p(x).reshape(-1, 20)
    a = (a - a.mean(0)) / np.maximum(a.std(0), 1e-8)
    eigen = np.linalg.eigvalsh(
        np.cov(a, rowvar=False)
    )[::-1]
    ratio = np.maximum(eigen, 1e-12)
    ratio /= ratio.sum()
    rank = np.exp(-(ratio * np.log(ratio)).sum())
    return float(ratio[0]), float(rank)

def adjacent_corr(x):
    a = np.log1p(x).reshape(-1, 20)
    c = np.corrcoef(a, rowvar=False)
    return float(np.mean([
        c[j, j + 2]
        for j in range(18)
    ]))

def lag1(x):
    a = x[:, :-1].reshape(-1, 20)
    b = x[:, 1:].reshape(-1, 20)
    return float(np.mean([
        np.corrcoef(a[:, j], b[:, j])[0, 1]
        for j in range(20)
    ]))

def report(name, x):
    pc1, rank = pca_stats(x)
    print(f"\n=== {name} ===")
    print(f"Mean: {x.mean():.3f}")
    print(f"Std: {x.std():.3f}")
    print(f"P99: {np.quantile(x, .99):.3f}")
    print(f"Temporal change: "
          f"{np.abs(np.diff(x, axis=1)).mean():.3f}")
    print(f"PC1: {pc1:.4f}")
    print(f"Effective rank: {rank:.3f}")
    print(f"Adjacent log-volume correlation: "
          f"{adjacent_corr(x):.4f}")
    print(f"Lag1 raw-volume correlation: "
          f"{lag1(x):.4f}")

print("=== V5.14 MULTIFACTOR FEASIBILITY ===")
report("REAL VALIDATION", val)
report("TRAIN JOINT REPLAY", joint)
report("TRAIN INDEPENDENT LEVEL TRAJECTORIES", independent)
print("\nMultifactor diagnostic complete!")
