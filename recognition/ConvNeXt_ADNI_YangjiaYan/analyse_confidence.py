"""Audit fixed predictions and select a reject rule using validation only."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

BASE = Path.home() / "data/comp3710_a3_evaluation"
OUT = Path("results/confidence")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ("cnn", "convnext")
LEVELS = ("slice", "scan", "patient")


def load_rows(model, split, level):
    path = BASE / model / split / f"{level}_predictions.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r["label"]) for r in rows])
    p = np.array([float(r["prob_ad"]) for r in rows])
    pred = (p >= 0.5).astype(int)
    saved = np.array([int(r["prediction"]) for r in rows])
    assert np.array_equal(pred, saved)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return y, p, pred


def write_csv(name, rows):
    with open(OUT / name, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def selective_metrics(y, p, threshold):
    pred = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1 - p)
    keep = (
        confidence >= threshold if threshold is not None
        else np.zeros(len(y), dtype=bool)
    )
    wrong = pred != y
    return {
        "n": len(y),
        "retained": int(keep.sum()),
        "referred": int((~keep).sum()),
        "coverage": float(keep.mean()),
        "retained_errors": int((keep & wrong).sum()),
        "retained_accuracy": (
            float((~wrong[keep]).mean()) if keep.any() else None
        ),
        "errors_referred": int((~keep & wrong).sum()),
        "all_errors": int(wrong.sum()),
        "retained_false_negatives": int(
            (keep & (y == 1) & (pred == 0)).sum()
        ),
        "retained_false_positives": int(
            (keep & (y == 0) & (pred == 1)).sum()
        ),
        "ad_patients_referred": int((~keep & (y == 1)).sum()),
        "nc_patients_referred": int((~keep & (y == 0)).sum()),
    }


# Per-class metrics at the original classification threshold of 0.5.
class_rows = []
for split in ("val", "test"):
    for model in MODELS:
        for level in LEVELS:
            y, p, pred = load_rows(model, split, level)
            precision, recall, f1, support = precision_recall_fscore_support(
                y, pred, labels=[0, 1], zero_division=0
            )
            for i, name in enumerate(("NC", "AD")):
                class_rows.append({
                    "split": split, "model": model, "level": level,
                    "class": name, "support": int(support[i]),
                    "precision": float(precision[i]),
                    "recall": float(recall[i]), "f1": float(f1[i]),
                })
write_csv("per_class_metrics.csv", class_rows)

audit_rows, bin_rows, rules = [], [], {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
colours = {"cnn": "#2878B5", "convnext": "#D95319"}

for model in MODELS:
    # Select the referral threshold using validation patients only.
    vy, vp, _ = load_rows(model, "val", "patient")
    candidates = []
    for threshold in np.linspace(0.5, 1.0, 51):
        result = selective_metrics(vy, vp, float(threshold))
        n = result["retained"]
        if n >= 20 and result["retained_errors"] / n <= 0.05:
            candidates.append((n, float(threshold)))

    threshold = (
        sorted(candidates, key=lambda x: (-x[0], x[1]))[0][1]
        if candidates else None
    )
    rules[model] = {
        "selection_split": "val",
        "classification_threshold": 0.5,
        "confidence_definition": "max(P(AD), 1-P(AD))",
        "candidate_thresholds": "0.50 to 1.00 inclusive, step 0.01",
        "max_validation_retained_error_rate": 0.05,
        "minimum_validation_retained_patients": 20,
        "selected_confidence_threshold": threshold,
        "fallback": "Refer all patients if no candidate meets the criteria",
        "val": selective_metrics(vy, vp, threshold),
    }

    for split in ("val", "test"):
        y, p, pred = load_rows(model, split, "patient")
        confidence = np.maximum(p, 1 - p)
        correct = pred == y
        ece = 0.0
        plotted_x, plotted_y = [], []

        # Ten equal-width confidence bins over [0.5, 1.0].
        for i in range(10):
            low, high = 0.5 + i * 0.05, 0.5 + (i + 1) * 0.05
            mask = (confidence >= low) & (
                (confidence < high) if i < 9 else (confidence <= 1.0)
            )
            n = int(mask.sum())
            mean_conf = float(confidence[mask].mean()) if n else None
            accuracy = float(correct[mask].mean()) if n else None
            if n:
                ece += n / len(y) * abs(mean_conf - accuracy)
                plotted_x.append(mean_conf)
                plotted_y.append(accuracy)
            bin_rows.append({
                "model": model, "split": split,
                "bin_low": low, "bin_high": high, "n": n,
                "mean_confidence": mean_conf, "accuracy": accuracy,
            })

        wrong = ~correct
        high_conf = confidence >= 0.9
        audit_rows.append({
            "model": model, "split": split, "n": len(y),
            "accuracy": float(correct.mean()),
            "brier_score": float(np.mean((p - y) ** 2)),
            "confidence_ece_10_bins": float(ece),
            "mean_confidence_correct": (
                float(confidence[correct].mean()) if correct.any() else None
            ),
            "mean_confidence_wrong": (
                float(confidence[wrong].mean()) if wrong.any() else None
            ),
            "errors": int(wrong.sum()),
            "high_confidence_patients": int(high_conf.sum()),
            "high_confidence_errors": int((wrong & high_conf).sum()),
        })

        if split == "test":
            rules[model]["test"] = selective_metrics(y, p, threshold)
            axes[0].plot(
                plotted_x, plotted_y, "o-",
                color=colours[model], label=model,
            )
            coverage, risk = [], []
            for t in np.linspace(0.5, 1.0, 101):
                r = selective_metrics(y, p, float(t))
                if r["retained"]:
                    coverage.append(r["coverage"])
                    risk.append(r["retained_errors"] / r["retained"])
            axes[1].plot(
                coverage, risk, ".-", color=colours[model], label=model
            )

axes[0].plot([0.5, 1], [0.5, 1], "k--", alpha=0.5)
axes[0].set(
    xlabel="Mean predicted-class confidence",
    ylabel="Observed accuracy",
    title="Test confidence reliability (patient level)",
    xlim=(0.5, 1.01), ylim=(0, 1.03),
)
axes[1].set(
    xlabel="Retained fraction (coverage)",
    ylabel="Error rate among retained patients",
    title="Test risk–coverage curves (descriptive)",
    xlim=(0, 1.03), ylim=(0, 1.03),
)
for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend()
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"confidence_analysis.{ext}", dpi=300)
plt.close(fig)

write_csv("patient_confidence_audit.csv", audit_rows)
write_csv("confidence_bins.csv", bin_rows)
with open(OUT / "referral_rules.json", "w") as f:
    json.dump(rules, f, indent=2, allow_nan=False)

print("===== Patient-level test class metrics =====")
for row in class_rows:
    if row["split"] == "test" and row["level"] == "patient":
        print(json.dumps(row))

print("\n===== Patient confidence audit =====")
print(json.dumps(audit_rows, indent=2))
print("\n===== Validation-selected referral rules =====")
print(json.dumps(rules, indent=2))
print("\nSaved to:", OUT.resolve())
