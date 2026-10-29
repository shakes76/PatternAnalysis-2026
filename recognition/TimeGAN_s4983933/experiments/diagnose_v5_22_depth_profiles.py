import json
from pathlib import Path

import numpy as np
from dataset_v4 import prepare_dataset_v4

OUT = Path("outputs_v5_22_depth_profiles")
OUT.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

val = norm.inverse_transform(
    np.asarray(data["val"][:512], dtype=np.float64)
)

generated = np.load(
    "outputs_v5_21_integrated/generated_lob.npz"
)["relative_orderbook"].astype(np.float64)

assert val.shape == (512, 24, 41)
assert generated.shape == (512, 24, 40)

def extract_volumes(book):
    return np.maximum(
        np.expm1(book[..., 1:40:2]), 0
    )

real = extract_volumes(val)
fake = extract_volumes(generated)

# Feature order:
# Ask1, Bid1, Ask2, Bid2, ..., Ask10, Bid10

def profile(x):
    flat = x.reshape(-1, 20)

    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    p90 = np.quantile(flat, .9, axis=0)

    changes = np.abs(
        np.diff(x, axis=1)
    ).mean(axis=(0, 1))

    return {
        "mean": means,
        "std": stds,
        "p90": p90,
        "temporal_change": changes,
    }

r = profile(real)
f = profile(fake)

print("=== V5.22 DEPTH LEVEL PROFILES ===")
print(
    "Level | Real Mean | Fake Mean "
    "| Ratio | Real Change | Fake Change"
)

rows = []

for level in range(10):
    for side, index in [
        ("Ask", 2 * level),
        ("Bid", 2 * level + 1),
    ]:
        ratio = (
            f["mean"][index]
            / max(r["mean"][index], 1e-8)
        )

        row = {
            "level": level + 1,
            "side": side,
            "real_mean": float(r["mean"][index]),
            "fake_mean": float(f["mean"][index]),
            "mean_ratio": float(ratio),
            "real_std": float(r["std"][index]),
            "fake_std": float(f["std"][index]),
            "real_p90": float(r["p90"][index]),
            "fake_p90": float(f["p90"][index]),
            "real_change": float(
                r["temporal_change"][index]
            ),
            "fake_change": float(
                f["temporal_change"][index]
            ),
        }

        rows.append(row)

        print(
            f"{side}{level+1:<2} | "
            f"{row['real_mean']:9.2f} | "
            f"{row['fake_mean']:9.2f} | "
            f"{ratio:5.2f} | "
            f"{row['real_change']:11.2f} | "
            f"{row['fake_change']:11.2f}"
        )

# Compare cross-level correlation of log volumes.
def corr_matrix(x):
    z = np.log1p(x).reshape(-1, 20)
    return np.corrcoef(z, rowvar=False)

cr = corr_matrix(real)
cf = corr_matrix(fake)

mask = ~np.eye(20, dtype=bool)

def mean_off_diagonal(c):
    return float(np.mean(c[mask]))

def mean_abs_off_diagonal(c):
    return float(np.mean(np.abs(c[mask])))

def ask_bid_correlation(c):
    return float(np.mean([
        c[2*i, 2*i+1]
        for i in range(10)
    ]))

summary = {
    "real_overall_mean": float(real.mean()),
    "fake_overall_mean": float(fake.mean()),
    "real_offdiag_correlation": mean_off_diagonal(cr),
    "fake_offdiag_correlation": mean_off_diagonal(cf),
    "real_abs_offdiag_correlation": (
        mean_abs_off_diagonal(cr)
    ),
    "fake_abs_offdiag_correlation": (
        mean_abs_off_diagonal(cf)
    ),
    "real_ask_bid_correlation": (
        ask_bid_correlation(cr)
    ),
    "fake_ask_bid_correlation": (
        ask_bid_correlation(cf)
    ),
    "real_mean_temporal_change": float(
        np.abs(np.diff(real, axis=1)).mean()
    ),
    "fake_mean_temporal_change": float(
        np.abs(np.diff(fake, axis=1)).mean()
    ),
}

print("\n=== V5.22 CORRELATION SUMMARY ===")
print(json.dumps(summary, indent=2))

with open(OUT / "results.json", "w") as fp:
    json.dump(
        {"summary": summary, "levels": rows},
        fp,
        indent=2
    )

print("\nV5.22 depth diagnostic complete!")
