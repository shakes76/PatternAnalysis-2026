import json
from pathlib import Path

import numpy as np
from scipy.special import rel_entr
from dataset_v4 import prepare_dataset_v4

OUT = Path("outputs_v5_36_spread_recency")
OUT.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def spread_ticks(x):
    raw = norm.inverse_transform(
        np.asarray(x, dtype=np.float64)
    )
    return np.maximum(
        np.rint(
            (raw[..., 0] - raw[..., 2]) / 0.01
        ).astype(np.int64),
        1
    )

train = spread_ticks(data["train"])
val = spread_ticks(data["val"][:512])

q1, q2, q3, q4 = np.array_split(train, 4)

MAX_TICK = int(train.max())

def hist(x):
    # The last bin is for values above TRAIN maximum.
    values = np.clip(
        x.ravel(), 1, MAX_TICK + 1
    )
    counts = np.bincount(
        values - 1,
        minlength=MAX_TICK + 1
    ).astype(np.float64)
    return counts

def kl(target, generated):
    a = hist(target) + 1
    b = hist(generated) + 1

    a /= a.sum()
    b /= b.sum()

    return float(np.sum(rel_entr(a, b)))

def stats(x):
    delta = np.abs(np.diff(x, axis=1))
    return {
        "mean_ticks": round(float(x.mean()), 4),
        "unchanged_fraction": round(
            float((delta == 0).mean()), 5
        ),
        "mean_tick_change": round(
            float(delta.mean()), 5
        ),
    }

# Compare historical distributions against the
# future TRAIN quarter Q4. No Validation data
# is used to select the candidate.
historical = {
    "all_q1_q3": np.concatenate(
        [q1, q2, q3], axis=0
    ),
    "recent_q2_q3": np.concatenate(
        [q2, q3], axis=0
    ),
    "recent_q3": q3,
    "recent_half_q3": q3[len(q3)//2:],
}

backtest = {}

print("=== V5.36 TRAIN-ONLY BACKTEST ===")
print("TRAIN Q4 REFERENCE:", stats(q4))

for name, sample in historical.items():
    score = kl(q4, sample)
    backtest[name] = {
        "kl_q4_to_history": score,
        "reference_stats": stats(sample),
    }
    print(
        name,
        "KL =", round(score, 6),
        stats(sample)
    )

selected = min(
    backtest,
    key=lambda name: backtest[name][
        "kl_q4_to_history"
    ]
)

print("\nSELECTED ON TRAIN ONLY:", selected)

# Refit the same historical-window rule using
# all Train data. For recency windows, scale the
# selected fraction to the full Train length.
fractions = {
    "all_q1_q3": 1.0,
    "recent_q2_q3": 2/3,
    "recent_q3": 1/3,
    "recent_half_q3": 1/6,
}

fraction = fractions[selected]
count = max(
    1, int(round(len(train) * fraction))
)

reference = train[-count:]

# This is an external diagnostic only.
# Validation is not used to select or fit.
validation_results = {
    "selected_history": selected,
    "train_sequences_used": int(len(reference)),
    "reference_stats": stats(reference),
    "validation_stats": stats(val),
    "kl_validation_to_reference": kl(
        val, reference
    ),
    "kl_reference_to_validation": kl(
        reference, val
    ),
    "kl_validation_to_all_train": kl(
        val, train
    ),
}

print("\n=== V5.36 VALIDATION DIAGNOSTIC ===")
print(json.dumps(
    validation_results, indent=2
))

results = {
    "backtest": backtest,
    "selected": selected,
    "validation": validation_results,
}

with open(OUT / "results.json", "w") as fp:
    json.dump(results, fp, indent=2)

print("\nV5.36 recency evaluation complete!")
