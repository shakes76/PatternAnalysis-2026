import json
from pathlib import Path
import numpy as np
from scipy.special import rel_entr
from dataset_v4 import prepare_dataset_v4

OUT = Path("outputs_v5_38_integrated")
OUT.mkdir(exist_ok=True)

TICK = 0.01
N, T = 512, 24

data = prepare_dataset_v4("data")
norm = data["normalizer"]

val = norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
)

v32 = np.load(
    "outputs_v5_32_integrated/generated_lob.npz"
)
book = np.asarray(
    v32["relative_orderbook"], dtype=np.float64
).copy()
returns = np.asarray(
    v32["log_returns"], dtype=np.float64
)

new_ticks = np.load(
    "outputs_v5_37_recent_spread/generated_spread_ticks.npz"
)["recent_markov"].astype(np.int64)

assert book.shape == (N, T, 40)
assert returns.shape == new_ticks.shape == (N, T)

# Preserve each timestep's midpoint, changing
# all ask and bid levels symmetrically.
old_spread = book[..., 0] - book[..., 2]
new_spread = new_ticks * TICK

delta_half = (
    (new_spread - old_spread) / 2
)[..., None]

# Prices occupy 0,2,4,...,38:
# Ask1, Bid1, Ask2, Bid2, ...
book[..., 0:40:4] += delta_half
book[..., 2:40:4] -= delta_half

ask = book[..., 0:40:4]
bid = book[..., 2:40:4]

spread = ask[..., 0] - bid[..., 0]
mid = (ask[..., 0] + bid[..., 0]) / 2

original = np.asarray(
    v32["relative_orderbook"], dtype=np.float64
)
original_mid = (
    original[..., 0] + original[..., 2]
) / 2

original_volumes = np.expm1(
    original[..., 1:40:2]
)
new_volumes = np.expm1(
    book[..., 1:40:2]
)

checks = {
    "positive_spread": bool(
        np.all(spread > 0)
    ),
    "ascending_asks": bool(
        np.all(np.diff(ask, axis=-1) > 0)
    ),
    "descending_bids": bool(
        np.all(np.diff(bid, axis=-1) < 0)
    ),
    "nonnegative_volumes": bool(
        np.all(new_volumes >= -1e-6)
    ),
    "midprice_preserved": bool(
        np.allclose(mid, original_mid, atol=1e-8)
    ),
    "volumes_preserved": bool(
        np.allclose(
            original_volumes,
            new_volumes,
            atol=1e-8
        )
    ),
    "finite": bool(
        np.all(np.isfinite(book))
    ),
}

def ticks(x):
    return np.maximum(
        np.rint(
            (x[..., 0] - x[..., 2]) / TICK
        ).astype(int),
        1
    )

tr = ticks(norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
))
va = ticks(val)
generated = ticks(book)

max_tick = int(tr.max())

def histogram(x):
    values = np.clip(
        x.ravel(), 1, max_tick + 1
    )
    return np.bincount(
        values - 1,
        minlength=max_tick + 1
    ).astype(np.float64)

def kl(a, b):
    p = histogram(a) + 1
    q = histogram(b) + 1
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(rel_entr(p, q)))

d = np.abs(np.diff(generated, axis=1))

features = np.concatenate(
    [book, returns[..., None]], axis=-1
)

results = {
    "shape": list(features.shape),
    "spread_kl_val_to_generated": kl(
        va, generated
    ),
    "spread_kl_generated_to_val": kl(
        generated, va
    ),
    "spread_unchanged_fraction": float(
        (d == 0).mean()
    ),
    "spread_mean_tick_change": float(
        d.mean()
    ),
    "spread_mean_ticks": float(
        generated.mean()
    ),
    "market_constraints": checks,
}

print("=== V5.38 INTEGRATED RESULTS ===")
print(json.dumps(results, indent=2))

np.savez_compressed(
    OUT / "generated_lob.npz",
    features=features.astype(np.float32),
    relative_orderbook=book.astype(np.float32),
    log_returns=returns.astype(np.float32),
    spread_ticks=generated.astype(np.int16),
    volumes=new_volumes.astype(np.float32),
    midprice=np.asarray(v32["midprice"])
)

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nV5.38 integration complete!")
