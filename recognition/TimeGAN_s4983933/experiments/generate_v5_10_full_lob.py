import json
from pathlib import Path

import numpy as np
import torch

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42
from lob_head_v5_8 import MLPConstraintAwareLOBHead

torch.manual_seed(12345)
torch.set_num_threads(4)

COUNT = 512
SEQ_LEN = 24
OUTPUT = Path("outputs_v5_10_full_lob")
OUTPUT.mkdir(parents=True, exist_ok=True)

data = prepare_dataset_v4("data")
normalizer = data["normalizer"]
book = normalizer.book_normalizer

# The anchor is taken from the training period, not test data.
initial_midprice = float(data["mid_prices"][0])

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

head = MLPConstraintAwareLOBHead(
    hidden_dim=hc["hidden_dim"],
    minimum=book.minimum,
    scale=book.scale,
    intermediate=hc["intermediate"],
    min_spread=0.01,
    min_gap=0.01,
)
head.load_state_dict(hc["head_state_dict"])
head.eval()

noise = torch.rand(
    COUNT, SEQ_LEN, ck["noise_dim"]
)

with torch.no_grad():
    hidden = model.supervisor(
        model.generator(noise)
    )

    # V5.9: structurally constrained relative LOB
    relative_book = head.forward_raw(hidden).numpy()

    # V4.3 Recovery: normalized log returns
    recovered = model.recovery(hidden).numpy()
    normalized_returns = recovered[..., 40]

# Inverse asinh + z-score transformation.
log_returns = (
    normalizer.return_scale
    * np.sinh(
        normalized_returns.astype(np.float64)
        * normalizer.return_std
        + normalizer.return_mean
    )
)

# The anchor is the mid-price BEFORE the generated
# sequence. Thus all generated returns, including
# the first, contribute to the new price path.
log_mid = (
    np.log(initial_midprice)
    + np.cumsum(log_returns, axis=1)
)
midprices = np.exp(log_mid)

# Features 0,2,... are ask/bid price offsets.
# Features 1,3,... are log1p order sizes.
absolute_book = relative_book.astype(
    np.float64, copy=True
)

absolute_book[..., 0::2] += midprices[..., None]
absolute_book[..., 1::2] = np.expm1(
    relative_book[..., 1::2]
)

# Recover ten ask/bid levels.
ask_prices = absolute_book[..., 0::4]
bid_prices = absolute_book[..., 2::4]
ask_sizes = absolute_book[..., 1::4]
bid_sizes = absolute_book[..., 3::4]

checks = {
    "finite": bool(
        np.isfinite(absolute_book).all()
        and np.isfinite(midprices).all()
        and np.isfinite(log_returns).all()
    ),
    "positive_prices": bool(
        (ask_prices > 0).all()
        and (bid_prices > 0).all()
    ),
    "positive_spread": bool(
        (ask_prices[..., 0] > bid_prices[..., 0]).all()
    ),
    "ask_ordered": bool(
        (np.diff(ask_prices, axis=-1) > 0).all()
    ),
    "bid_ordered": bool(
        (np.diff(bid_prices, axis=-1) < 0).all()
    ),
    "midprice_consistency": bool(
        (ask_prices[..., 0] > midprices).all()
        and (bid_prices[..., 0] < midprices).all()
    ),
    "nonnegative_sizes": bool(
        (ask_sizes >= 0).all()
        and (bid_sizes >= 0).all()
    ),
}

real_return_scale = normalizer.return_scale
real_return_mean = normalizer.return_mean
real_return_std = normalizer.return_std

results = {
    "model": "V5.10 Generator + V5.9 LOB Decoder",
    "return_decoder": "V4.3 Recovery",
    "count": COUNT,
    "sequence_length": SEQ_LEN,
    "initial_midprice": initial_midprice,
    "anchor_convention":
        "Initial mid-price is the pre-sequence anchor; "
        "all 24 generated returns are applied.",
    "generated_return_mean": float(log_returns.mean()),
    "generated_return_std": float(log_returns.std()),
    "generated_return_temporal_change": float(
        np.abs(np.diff(log_returns, axis=1)).mean()
    ),
    "generated_return_across_sample_std": float(
        log_returns.std(axis=0).mean()
    ),
    "generated_midprice_min": float(midprices.min()),
    "generated_midprice_max": float(midprices.max()),
    "generated_midprice_path_std_mean": float(
        midprices.std(axis=1).mean()
    ),
    "generated_spread_mean": float(
        (ask_prices[..., 0] - bid_prices[..., 0]).mean()
    ),
    "generated_spread_std": float(
        (ask_prices[..., 0] - bid_prices[..., 0]).std()
    ),
    "normalizer": {
        "return_scale": float(real_return_scale),
        "return_mean": float(real_return_mean),
        "return_std": float(real_return_std),
    },
    "checks": checks,
}

np.savez_compressed(
    OUTPUT / "generated_full_lob.npz",
    absolute_orderbook=absolute_book,
    relative_orderbook=relative_book,
    midprices=midprices,
    log_returns=log_returns,
    normalized_returns=normalized_returns,
)

with open(OUTPUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("=== V5.10 FULL LOB GENERATION ===")
print("Absolute LOB shape:", absolute_book.shape)
print("Mid-price shape:", midprices.shape)
print("Return shape:", log_returns.shape)
print(json.dumps(results, indent=2))

print("\nFirst generated mid-price path:")
print(np.round(midprices[0, :10], 6))

print("\nFirst generated top-of-book snapshot:")
print("Best ask:", float(ask_prices[0, 0, 0]))
print("Best bid:", float(bid_prices[0, 0, 0]))
print("Ask size:", float(ask_sizes[0, 0, 0]))
print("Bid size:", float(bid_sizes[0, 0, 0]))

print("\nSaved:", OUTPUT / "generated_full_lob.npz")
print("Saved:", OUTPUT / "results.json")
print("Full LOB generation complete!")
