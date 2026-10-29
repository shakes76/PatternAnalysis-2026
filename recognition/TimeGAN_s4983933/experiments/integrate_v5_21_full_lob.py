import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42
from lob_head_v5_8 import MLPConstraintAwareLOBHead
from train_v5_14_multifactor import MultiFactorVolumeHead

SEED = 371021
N, T = 512, 24
TICK = 0.01
REFERENCE_PRICE = 100.0
THRESHOLD = 1e-7

torch.set_num_threads(4)
rng = np.random.default_rng(SEED)
out = Path("outputs_v5_21_integrated")
out.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

train = norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
)
val = norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
)

train_r = train[..., 40]
val_r = val[..., 40]

# -------------------------------------------------
# 1. Load TimeGAN
# -------------------------------------------------
ck = torch.load(
    "outputs_v5_10_latent_matching/timegan_finetuned.pt",
    map_location="cpu",
    weights_only=True,
)

gan = TimeGANV42(
    feature_dim=ck["feature_dim"],
    hidden_dim=ck["hidden_dim"],
    noise_dim=ck["noise_dim"],
    num_layers=ck["num_layers"],
)
gan.load_state_dict(ck["model_state_dict"])
gan.eval()

# -------------------------------------------------
# 2. Load price head
# -------------------------------------------------
pc = torch.load(
    "outputs_v5_9_variance_aware/variance_aware_best.pt",
    map_location="cpu",
    weights_only=True,
)

bn = norm.book_normalizer

price_head = MLPConstraintAwareLOBHead(
    hidden_dim=pc["hidden_dim"],
    minimum=bn.minimum,
    scale=bn.scale,
    min_spread=0.01,
    min_gap=0.01,
    intermediate=pc["intermediate"],
)
price_head.load_state_dict(pc["head_state_dict"])
price_head.eval()

# -------------------------------------------------
# 3. Load volume head
# -------------------------------------------------
vc = torch.load(
    "outputs_v5_14_multifactor/best_multifactor_head.pt",
    map_location="cpu",
    weights_only=True,
)

volume_head = MultiFactorVolumeHead(
    hidden_dim=vc["hidden_dim"],
    mean_log=torch.ones(20),
    std_log=torch.ones(20),
)
volume_head.load_state_dict(vc["head_state_dict"])
volume_head.eval()

with torch.no_grad():
    generator = torch.Generator().manual_seed(SEED)
    z = torch.rand(
        N, T, ck["noise_dim"],
        generator=generator,
    )
    hidden = gan.supervisor(gan.generator(z))

    base_book = (
        price_head.forward_raw(hidden)
        .detach().cpu().numpy()
        .astype(np.float64)
    )

    innovations = torch.randn(
        N, T, 20,
        generator=generator,
    )

    log_vol = volume_head(
        hidden, innovations
    ).detach().cpu().numpy()

volume = np.maximum(np.expm1(log_vol), 0)

# -------------------------------------------------
# 4. Recent-25% Markov returns
# -------------------------------------------------
recent = train_r[3 * len(train_r) // 4:]
state = np.abs(recent) >= THRESHOLD

previous = state[:, :-1].ravel()
following = state[:, 1:].ravel()

transition = np.zeros(2)

for s in (0, 1):
    selected = following[previous == bool(s)]
    transition[s] = (
        float(selected.mean())
        if len(selected)
        else float(state.mean())
    )

jumps = recent[state]
returns = np.zeros((N, T), dtype=np.float64)

# First return is zero, measured relative to
# an initial reference mid-price.
active = rng.random(N) < float(state.mean())

for t in range(1, T):
    active = (
        rng.random(N)
        < transition[active.astype(int)]
    )
    count = int(active.sum())

    if count:
        returns[active, t] = rng.choice(
            jumps, count, replace=True
        )

# -------------------------------------------------
# 5. Train-only spread quantile calibration
# -------------------------------------------------
train_spread = np.maximum(
    TICK,
    train[..., 0] - train[..., 2]
).ravel()

train_ticks = np.maximum(
    1, np.rint(train_spread / TICK).astype(int)
)

original_spread = (
    base_book[..., 0] - base_book[..., 2]
).ravel()

order = np.argsort(original_spread, kind="stable")
sorted_ticks = np.sort(train_ticks)

positions = np.minimum(
    (
        (np.arange(len(order)) + 0.5)
        / len(order) * len(sorted_ticks)
    ).astype(int),
    len(sorted_ticks) - 1
)

mapped = np.empty(len(order), dtype=int)
mapped[order] = sorted_ticks[positions]

spread = mapped.reshape(N, T) * TICK

# -------------------------------------------------
# 6. Build a self-consistent mid-price path
# -------------------------------------------------
# Relative prices are anchored to 100.0 purely as
# a reference. This is NOT an inferred AMZN price.
mid_abs = REFERENCE_PRICE * np.exp(
    np.cumsum(returns, axis=1)
)
mid_relative = mid_abs - REFERENCE_PRICE

ask1 = mid_relative + spread / 2
bid1 = mid_relative - spread / 2

# Retain the learned ladder gaps.
book = base_book.copy()

book[..., 0] = ask1
book[..., 2] = bid1

for level in range(1, 10):
    prev_ask = 4 * (level - 1)
    prev_bid = prev_ask + 2
    ask_idx = 4 * level
    bid_idx = ask_idx + 2

    ask_gap = np.maximum(
        base_book[..., ask_idx]
        - base_book[..., prev_ask],
        TICK
    )
    bid_gap = np.maximum(
        base_book[..., prev_bid]
        - base_book[..., bid_idx],
        TICK
    )

    book[..., ask_idx] = (
        book[..., prev_ask] + ask_gap
    )
    book[..., bid_idx] = (
        book[..., prev_bid] - bid_gap
    )

# Replace volume slots with the learned
# multi-factor volume generator.
book[..., 1:40:2] = np.log1p(volume)

# Recalculate actual returns from the new book.
reconstructed_mid = (
    book[..., 0] + book[..., 2]
) / 2 + REFERENCE_PRICE

reconstructed_returns = np.zeros((N, T))
reconstructed_returns[:, 1:] = np.diff(
    np.log(reconstructed_mid), axis=1
)

# -------------------------------------------------
# 7. Validate market structure
# -------------------------------------------------
asks = book[..., 0:40:4]
bids = book[..., 2:40:4]

checks = {
    "positive_spread": bool(
        np.all(asks[..., 0] > bids[..., 0])
    ),
    "ascending_asks": bool(
        np.all(np.diff(asks, axis=-1) > 0)
    ),
    "descending_bids": bool(
        np.all(np.diff(bids, axis=-1) < 0)
    ),
    "nonnegative_volumes": bool(
        np.all(volume >= 0)
    ),
    "returns_consistent": bool(
        np.allclose(
            reconstructed_returns,
            returns,
            atol=1e-10,
            rtol=1e-6,
        )
    ),
}

# -------------------------------------------------
# 8. KL metrics using train-defined bins
# -------------------------------------------------
max_spread = max(
    float(train_spread.max()),
    float(spread.max()),
    float((val[..., 0] - val[..., 2]).max())
)

spread_edges = np.linspace(
    0, max_spread + TICK, 81
)

return_limit = max(
    float(np.quantile(
        np.abs(train_r.ravel()), .999
    )) * 1.2,
    1e-7
)

return_edges = np.linspace(
    -return_limit, return_limit, 82
)

def kl(real, fake, edges):
    a = np.clip(real.ravel(), edges[0], edges[-1])
    b = np.clip(fake.ravel(), edges[0], edges[-1])

    p = np.histogram(a, bins=edges)[0].astype(float)
    q = np.histogram(b, bins=edges)[0].astype(float)

    p += 1e-8
    q += 1e-8
    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))

spread_kl = kl(
    val[..., 0] - val[..., 2],
    spread,
    spread_edges,
)

# Compare steps 1..23 because the integrated
# sequence starts from an explicit initial price.
return_kl = kl(
    val_r[:, 1:],
    reconstructed_returns[:, 1:],
    return_edges,
)

# -------------------------------------------------
# 9. Random-pair depth SSIM (diagnostic only)
# -------------------------------------------------
def heatmap(x):
    vol = np.maximum(
        np.expm1(x[..., 1:40:2]),
        0
    )
    asks_v = vol[..., 0::2]
    bids_v = vol[..., 1::2]

    surface = np.concatenate(
        [bids_v[..., ::-1], asks_v],
        axis=-1
    )
    return np.log1p(surface).transpose(0, 2, 1)

real_heat = heatmap(val)
fake_heat = heatmap(book)

scale = max(
    float(np.quantile(real_heat, .99)),
    1e-6
)

real_heat = np.clip(real_heat / scale, 0, 1)
fake_heat = np.clip(fake_heat / scale, 0, 1)

pairing = np.random.default_rng(
    SEED
).permutation(N)

ssim = float(np.mean([
    structural_similarity(
        real_heat[i],
        fake_heat[pairing[i]],
        data_range=1.0,
        win_size=7,
    )
    for i in range(N)
]))

# -------------------------------------------------
# 10. Save full 41-feature output
# -------------------------------------------------
features = np.concatenate(
    [book, reconstructed_returns[..., None]],
    axis=-1
)

np.savez_compressed(
    out / "generated_lob.npz",
    features=features.astype(np.float32),
    relative_orderbook=book.astype(np.float32),
    log_returns=reconstructed_returns.astype(np.float32),
    midprice=mid_abs.astype(np.float64),
    volumes=volume.astype(np.float32),
)

results = {
    "model": "V5.21 Hybrid TimeGAN",
    "shape": list(features.shape),
    "market_constraints": checks,
    "spread_kl_validation": spread_kl,
    "return_kl_validation": return_kl,
    "random_pair_depth_ssim_diagnostic": ssim,
    "mean_generated_volume": float(volume.mean()),
    "mean_real_validation_volume": float(
        np.maximum(
            np.expm1(val[..., 1:40:2]), 0
        ).mean()
    ),
    "return_zero_fraction": float(
        np.mean(
            np.abs(reconstructed_returns[:, 1:])
            < THRESHOLD
        )
    ),
}

print("=== V5.21 INTEGRATED LOB RESULTS ===")
print(json.dumps(results, indent=2))

with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved:", out / "generated_lob.npz")
print("V5.21 integration complete!")
