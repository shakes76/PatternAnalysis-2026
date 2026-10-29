import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42

SEED = 12345
N = 512
BINS = 80
OUT = Path("outputs_v5_47_corrected_benchmark")
OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

real = norm.inverse_transform(
    np.asarray(data["test"][:N], dtype=np.float64)
)

train = norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
)

# Baseline A: replay complete training windows.
rng = np.random.default_rng(SEED)
indices = rng.integers(0, len(train), size=N)
empirical = train[indices]

# V4.3: original TimeGAN generator and recovery.
ck = torch.load(
    "checkpoints_v4_3_full/timegan_joint.pt",
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

noise = torch.rand(N, 24, ck["noise_dim"])

with torch.no_grad():
    hidden = model.supervisor(model.generator(noise))
    original_normalized = model.recovery(hidden).numpy()

v43 = norm.inverse_transform(original_normalized)

# V5.10: generated constrained relative book + returns.
saved = np.load(
    "outputs_v5_10_full_lob/generated_full_lob.npz"
)

v510 = np.concatenate(
    [
        saved["relative_orderbook"],
        saved["log_returns"][..., None],
    ],
    axis=-1,
).astype(np.float64)

# V5.11: same generated book, statistical returns.
markov_returns = np.load(
    "outputs_v5_11_return_baseline/baseline_returns.npy"
)

v511 = v510.copy()
v511[..., 40] = markov_returns


# V5.38: frozen final hybrid generator.
final = np.load(
    "outputs_v5_38_integrated/generated_lob.npz"
)

v538 = np.asarray(
    final["features"], dtype=np.float64
)

groups = {
    "real_test": real,
    "empirical_replay": empirical,
    "v4_3_timegan": v43,
    "v5_10_timegan": v510,
    "v5_11_hybrid": v511,
    "v5_38_final": v538,
}

for name, x in groups.items():
    assert x.shape == (N, 24, 41), (name, x.shape)

# Shared histogram ranges for ALL compared methods.
def make_edges(values):
    all_values = np.concatenate(
        [np.asarray(v).ravel() for v in values]
    )
    all_values = all_values[np.isfinite(all_values)]
    low, high = np.quantile(
        all_values, [0.001, 0.999]
    )
    if high <= low:
        high = low + 1e-8
    return np.linspace(low, high, BINS + 1)


def spread(x):
    return x[..., 0] - x[..., 2]


spread_edges = make_edges(
    [spread(x) for x in groups.values()]
)
return_edges = make_edges(
    [x[..., 40] for x in groups.values()]
)


def kl(real_values, fake_values, edges):
    p, _ = np.histogram(
        np.clip(real_values, edges[0], edges[-1]),
        bins=edges,
    )
    q, _ = np.histogram(
        np.clip(fake_values, edges[0], edges[-1]),
        bins=edges,
    )
    p = p.astype(np.float64) + 1e-8
    q = q.astype(np.float64) + 1e-8
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


# Shared real-test intensity scaling.
def depth_image(x):
    book = x[..., :40]

    ask_log = book[..., 1::4]
    bid_log = book[..., 3::4]

    # Protect evaluation from invalid/unbounded outputs.
    ask_size = np.maximum(np.expm1(
        np.clip(ask_log, -20, 20)
    ), 0.0)
    bid_size = np.maximum(np.expm1(
        np.clip(bid_log, -20, 20)
    ), 0.0)

    depth = np.concatenate(
        [bid_size[..., ::-1], ask_size],
        axis=-1
    )
    return np.log1p(depth).transpose(0, 2, 1)


real_depth = depth_image(real)
scale = max(
    float(np.quantile(real_depth, 0.99)),
    1e-6,
)

def depth_scaled(x):
    return np.clip(depth_image(x), 0, scale) / scale

real_img = depth_scaled(real)

# Identical random pairing across every model.
pairing = np.random.default_rng(SEED).permutation(N)


def structural_checks(x):
    book = x[..., :40]
    asks = book[..., 0::4]
    bids = book[..., 2::4]
    ask_sizes = book[..., 1::4]
    bid_sizes = book[..., 3::4]

    valid = (
        (asks[..., 0] > bids[..., 0])
        & (asks[..., 0] > 0)
        & (bids[..., 0] < 0)
        & np.all(np.diff(asks, axis=-1) > 0, axis=-1)
        & np.all(np.diff(bids, axis=-1) < 0, axis=-1)
        & np.all(ask_sizes >= 0, axis=-1)
        & np.all(bid_sizes >= 0, axis=-1)
        & np.isfinite(book).all(axis=-1)
    )
    return float(valid.mean())



def market_validity(x, tick=0.01, tol=1e-6):
    book = np.asarray(x[..., :40], dtype=np.float64)

    asks = book[..., 0::4]
    bids = book[..., 2::4]
    ask_log_sizes = book[..., 1::4]
    bid_log_sizes = book[..., 3::4]

    spread = asks[..., 0] - bids[..., 0]
    ask_gaps = np.diff(asks, axis=-1)
    bid_gaps = -np.diff(bids, axis=-1)

    valid = (
        (spread >= tick - tol)
        & np.all(ask_gaps >= tick - tol, axis=-1)
        & np.all(bid_gaps >= tick - tol, axis=-1)
        & np.all(ask_log_sizes >= 0, axis=-1)
        & np.all(bid_log_sizes >= 0, axis=-1)
        & np.isfinite(book).all(axis=-1)
    )

    return float(valid.mean())


def evaluate(name, x):
    x = np.asarray(x, dtype=np.float64)
    s = spread(x)
    r = x[..., 40]

    images = depth_scaled(x)
    scores = [
        structural_similarity(
            real_img[i],
            images[pairing[i]],
            data_range=1.0,
            win_size=7,
        )
        for i in range(N)
    ]

    near_zero = np.abs(r) < 1e-7

    return {
        "spread_mean": float(s.mean()),
        "spread_std": float(s.std()),
        "spread_kl": kl(
            spread(real), s, spread_edges
        ),
        "return_kl_thresholded": kl(
            np.where(
                np.abs(real[..., 40]) < 1e-7,
                0.0,
                real[..., 40],
            ),
            np.where(
                np.abs(r) < 1e-7,
                0.0,
                r,
            ),
            return_edges,
        ),
        "return_std": float(r.std()),
        "return_kl": kl(
            real[..., 40], r, return_edges
        ),
        "return_near_zero_fraction": float(
            near_zero.mean()
        ),
        "return_temporal_change": float(
            np.abs(np.diff(r, axis=1)).mean()
        ),
        "mean_heatmap_ssim": float(np.mean(scores)),
        "median_heatmap_ssim": float(np.median(scores)),
        "valid_snapshot_fraction": structural_checks(x),
        "market_valid_snapshot_fraction":
            market_validity(x),

    }


results = {
    "protocol": {
        "samples": N,
        "sequence_length": 24,
        "seed": SEED,
        "kl_bins": BINS,
        "kl_smoothing": 1e-8,
        "kl_direction": "KL(real || candidate)",
        "kl_range": (
            "Shared across all compared methods, "
            "pooled 0.1%-99.9% quantiles"
        ),
        "ssim": (
            "Independent randomly paired sequences; "
            "20x24 depth; log1p; real-test p99 scale; "
            "window=7"
        ),
        "v5_11_note": (
            "V5.10 book with separately sampled "
            "train-calibrated Markov returns"
        ),
    },
    "models": {},
}

print("\n=== UNIFIED FINAL BENCHMARK ===")

for name, x in groups.items():
    record = evaluate(name, x)
    results["models"][name] = record

    print(f"\n=== {name.upper()} ===")
    print(json.dumps(record, indent=2))

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== COMPARISON TABLE ===")
print(
    f"{'Model':<20}"
    f"{'Spread KL':>12}"
    f"{'Return KL':>12}"
    f"{'SSIM':>10}"
    f"{'Valid %':>10}"
)

for name, r in results["models"].items():
    print(
        f"{name:<20}"
        f"{r['spread_kl']:>12.4f}"
        f"{r['return_kl']:>12.4f}"
        f"{r['mean_heatmap_ssim']:>10.4f}"
        f"{100*r['valid_snapshot_fraction']:>10.2f}"
    )

print("\nSaved:", OUT / "results.json")
print("Unified benchmark complete!")
