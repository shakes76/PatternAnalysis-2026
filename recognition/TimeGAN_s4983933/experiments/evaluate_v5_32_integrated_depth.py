import json
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity
from dataset_v4 import prepare_dataset_v4

OUT = Path("outputs_v5_32_integrated")
OUT.mkdir(exist_ok=True)

SEED = 371032
N, T = 512, 24

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def extract_volumes(book):
    return np.maximum(
        np.expm1(book[..., 1:40:2]), 0
    )

val = norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
)

real_vol = extract_volumes(val)

source = np.load(
    "outputs_v5_21_integrated/generated_lob.npz"
)

original_book = source[
    "relative_orderbook"
].astype(np.float64)

original_vol = extract_volumes(original_book)

new_vol = np.load(
    "outputs_v5_31_bounded_mdn/train_p999.npy"
).astype(np.float64)

assert original_book.shape == (N, T, 40)
assert new_vol.shape == (N, T, 20)

# Preserve V5.21 price columns and replace
# only the 20 log-volume columns.
updated_book = original_book.copy()
updated_book[..., 1:40:2] = np.log1p(new_vol)

# 41-dimensional representation:
# 40 relative orderbook features plus returns.
returns = np.asarray(
    source["log_returns"], dtype=np.float64
)

assert returns.shape == (N, T)

features = np.concatenate(
    [updated_book, returns[..., None]],
    axis=-1
)

assert features.shape == (N, T, 41)

def make_heatmap(x):
    ask = x[..., 0::2]
    bid = x[..., 1::2]

    depth = np.concatenate(
        [bid[..., ::-1], ask], axis=-1
    )

    return np.log1p(depth).transpose(0, 2, 1)

# Same comparison protocol for all variants.
real_img = make_heatmap(real_vol)
scale = max(
    float(np.quantile(real_img, .99)),
    1e-6
)

real_img = np.clip(
    real_img / scale, 0, 1
)

rng = np.random.default_rng(SEED)
permutation = rng.permutation(N)

def random_ssim(x):
    imgs = np.clip(
        make_heatmap(x) / scale, 0, 1
    )

    return float(np.mean([
        structural_similarity(
            real_img[i],
            imgs[permutation[i]],
            data_range=1.0,
            win_size=7
        )
        for i in range(N)
    ]))

def metrics(x):
    d = np.abs(np.diff(x, axis=1))
    changed = d >= 1e-5

    z = np.log1p(x).reshape(-1, 20)
    corr = np.corrcoef(z, rowvar=False)
    offdiag = ~np.eye(20, dtype=bool)

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
            float(d.mean()), 3
        ),
        "mean_abs_level_correlation": round(
            float(np.abs(corr[offdiag]).mean()), 5
        ),
        "random_pair_ssim": round(
            random_ssim(x), 5
        ),
    }

# Check physical price ladder from V5.21.
ask = updated_book[..., 0::4]
bid = updated_book[..., 2::4]

checks = {
    "positive_spread": bool(
        np.all(ask[..., 0] > bid[..., 0])
    ),
    "ascending_asks": bool(
        np.all(np.diff(ask, axis=-1) > 0)
    ),
    "descending_bids": bool(
        np.all(np.diff(bid, axis=-1) < 0)
    ),
    "nonnegative_volumes": bool(
        np.all(new_vol >= 0)
    ),
    "finite_features": bool(
        np.all(np.isfinite(features))
    ),
    "prices_unchanged_from_v5_21": bool(
        np.array_equal(
            updated_book[..., 0:40:2],
            original_book[..., 0:40:2]
        )
    ),
}

results = {
    "real_validation": metrics(real_vol),
    "original_v5_21": metrics(original_vol),
    "integrated_v5_32": metrics(new_vol),
    "market_constraints": checks,
}

print("=== V5.32 INTEGRATED DEPTH RESULTS ===")
for name, value in results.items():
    print(name.upper(), value)

np.savez_compressed(
    OUT / "generated_lob.npz",
    features=features.astype(np.float32),
    relative_orderbook=updated_book.astype(
        np.float32
    ),
    log_returns=returns.astype(np.float32),
    midprice=np.asarray(source["midprice"]),
    volumes=new_vol.astype(np.float32)
)

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.32 integrated evaluation complete!")
