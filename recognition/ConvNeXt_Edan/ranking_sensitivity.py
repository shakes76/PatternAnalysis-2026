"""Validation-selected rejection policies and risk-coverage sensitivity."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path("outputs")
DEST = ROOT / "ranking_sensitivity"
DEST.mkdir(parents=True, exist_ok=True)

THRESHOLDS = {"baseline": 0.78, "convnext": 0.45}
TARGETS = [0.60, 0.70, 0.80]
METHODS = ["raw", "side_normalised"]


def load(model, split):
    path = ROOT / model / split / "patient_predictions.csv"
    df = pd.read_csv(
        path,
        usecols=["patient_id", "true_label", "probability_ad"],
        dtype={"patient_id": str},
    )

    if df.patient_id.isna().any() or df.patient_id.duplicated().any():
        raise ValueError(f"Invalid patient IDs: {path}")
    if not df.true_label.isin([0, 1]).all():
        raise ValueError(f"Invalid class labels: {path}")
    if not np.isfinite(df.probability_ad).all():
        raise ValueError(f"Invalid probabilities: {path}")
    if not df.probability_ad.between(0, 1).all():
        raise ValueError(f"Probabilities outside [0,1]: {path}")

    return df


def confidence(p, threshold, method):
    distance = np.abs(p - threshold)

    if method == "raw":
        return distance

    # Scale the NC and AD sides separately to [0, 1].
    denominator = np.where(p < threshold, threshold, 1 - threshold)
    return distance / denominator


def evaluate(df, threshold, method, cutoff):
    p = df.probability_ad.to_numpy(dtype=float)
    y = df.true_label.to_numpy(dtype=int)

    pred = (p >= threshold).astype(int)
    accepted = confidence(p, threshold, method) >= cutoff

    ya = y[accepted]
    pa = pred[accepted]

    tn = int(((ya == 0) & (pa == 0)).sum())
    fp = int(((ya == 0) & (pa == 1)).sum())
    fn = int(((ya == 1) & (pa == 0)).sum())
    tp = int(((ya == 1) & (pa == 1)).sum())

    n = len(ya)
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan

    return {
        "accepted": n,
        "rejected": len(y) - n,
        "coverage": n / len(y),
        "accuracy": (tp + tn) / n if n else np.nan,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2
            if np.isfinite(sensitivity) and np.isfinite(specificity)
            else np.nan
        ),
        "macro_f1": (
            f1_score(
                ya, pa, labels=[0, 1],
                average="macro", zero_division=0
            )
            if n else np.nan
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "FP": fp,
        "FN": fn,
    }


def risk_curve(df, threshold, method):
    p = df.probability_ad.to_numpy(dtype=float)
    y = df.true_label.to_numpy(dtype=int)

    scores = confidence(p, threshold, method)
    errors = ((p >= threshold).astype(int) != y).astype(int)

    order = np.argsort(-scores, kind="stable")
    n = np.arange(1, len(y) + 1)

    return n / len(y), np.cumsum(errors[order]) / n


def main():
    rows = []

    for model, threshold in THRESHOLDS.items():
        val = load(model, "val")
        test = load(model, "test")

        if set(val.patient_id) & set(test.patient_id):
            raise ValueError(f"Patient leakage detected: {model}")

        fig, ax = plt.subplots(figsize=(8, 5.5))

        for method in METHODS:
            val_scores = confidence(
                val.probability_ad.to_numpy(dtype=float),
                threshold, method
            )

            for target in TARGETS:
                # Select cutoff using validation only.
                cutoff = float(np.quantile(val_scores, 1 - target))

                val_result = evaluate(val, threshold, method, cutoff)
                test_result = evaluate(test, threshold, method, cutoff)

                rows.append({
                    "model": model,
                    "method": method,
                    "target_coverage": target,
                    "validation_cutoff": cutoff,
                    "validation_coverage": val_result["coverage"],
                    **test_result,
                })

            # Test curve is retrospective/descriptive, not for selection.
            coverage, risk = risk_curve(test, threshold, method)
            ax.plot(
                coverage * 100,
                risk * 100,
                label=method.replace("_", " ").title(),
            )

        ax.set(
            title=f"{model}: Test risk–coverage sensitivity",
            xlabel="Automated coverage (%)",
            ylabel="Error rate among accepted patients (%)",
        )
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            DEST / f"{model}_ranking_comparison.png", dpi=160
        )
        plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(DEST / "summary.csv", index=False)

    columns = [
        "model", "method", "target_coverage",
        "validation_coverage", "coverage",
        "accepted", "accuracy", "balanced_accuracy",
        "macro_f1", "sensitivity", "specificity",
        "FP", "FN",
    ]

    print("\nREJECTION RANKING SENSITIVITY — HELD-OUT TEST")
    print("Cutoffs selected using validation data only.\n")
    print(
        summary[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print("\nSaved aggregate results to:", DEST)


if __name__ == "__main__":
    main()
