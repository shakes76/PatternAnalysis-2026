"""
COMP3710: Patient-level statistical uncertainty analysis.

Uses previously frozen validation-selected thresholds.
Stratified paired bootstrap: sample patients, not MRI slices.
Test labels are never used to tune models or thresholds.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("outputs")
DEST = ROOT / "statistical_analysis"
DEST.mkdir(parents=True, exist_ok=True)

THRESHOLDS = {
    "baseline": 0.78,
    "convnext": 0.45,
}

N_BOOTSTRAPS = 5000
SEED = 42

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "auroc",
    "NC_precision",
    "NC_recall",
    "NC_f1",
    "AD_precision",
    "AD_recall",
    "AD_f1",
]


def load_predictions(model):
    path = ROOT / model / "test" / "patient_predictions.csv"

    df = pd.read_csv(
        path,
        usecols=["patient_id", "true_label", "probability_ad"],
        dtype={"patient_id": str},
    )

    if df.patient_id.isna().any() or df.patient_id.duplicated().any():
        raise ValueError(f"Invalid patient IDs in {path}")

    if not df.true_label.isin([0, 1]).all():
        raise ValueError(f"Invalid labels in {path}")

    if not np.isfinite(df.probability_ad.to_numpy()).all():
        raise ValueError(f"Non-finite probabilities in {path}")

    if not df.probability_ad.between(0, 1).all():
        raise ValueError(f"Invalid probability range in {path}")

    return df.sort_values("patient_id").reset_index(drop=True)


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def f1(precision, recall):
    if not np.isfinite(precision) or not np.isfinite(recall):
        return np.nan
    return safe_div(2 * precision * recall, precision + recall)


def calculate_metrics(y, probability, threshold):
    pred = (probability >= threshold).astype(int)

    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())

    nc_precision = safe_div(tn, tn + fn)
    nc_recall = safe_div(tn, tn + fp)
    ad_precision = safe_div(tp, tp + fp)
    ad_recall = safe_div(tp, tp + fn)

    nc_f1 = f1(nc_precision, nc_recall)
    ad_f1 = f1(ad_precision, ad_recall)

    return {
        "accuracy": (tn + tp) / len(y),
        "balanced_accuracy": (nc_recall + ad_recall) / 2,
        "macro_f1": (nc_f1 + ad_f1) / 2,
        "auroc": roc_auc_score(y, probability),
        "NC_precision": nc_precision,
        "NC_recall": nc_recall,
        "NC_f1": nc_f1,
        "AD_precision": ad_precision,
        "AD_recall": ad_recall,
        "AD_f1": ad_f1,
    }


def main():
    frames = {
        model: load_predictions(model)
        for model in THRESHOLDS
    }

    baseline = frames["baseline"]
    convnext = frames["convnext"]

    if not baseline[["patient_id", "true_label"]].equals(
        convnext[["patient_id", "true_label"]]
    ):
        raise ValueError(
            "Models do not have identical patients and labels."
        )

    y = baseline.true_label.to_numpy(dtype=int)

    probabilities = {
        model: df.probability_ad.to_numpy(dtype=float)
        for model, df in frames.items()
    }

    nc_indices = np.flatnonzero(y == 0)
    ad_indices = np.flatnonzero(y == 1)

    print(
        f"Test patients: {len(y)} "
        f"(NC={len(nc_indices)}, AD={len(ad_indices)})"
    )

    rng = np.random.default_rng(SEED)

    observed = {
        model: calculate_metrics(
            y, probabilities[model], threshold
        )
        for model, threshold in THRESHOLDS.items()
    }

    bootstrap = {
        model: {metric: [] for metric in METRICS}
        for model in THRESHOLDS
    }

    for _ in range(N_BOOTSTRAPS):
        # Preserve the original NC/AD patient counts.
        indices = np.concatenate([
            rng.choice(
                nc_indices, size=len(nc_indices), replace=True
            ),
            rng.choice(
                ad_indices, size=len(ad_indices), replace=True
            ),
        ])

        sampled_labels = y[indices]

        # Identical sampled patients for both models.
        for model, threshold in THRESHOLDS.items():
            values = calculate_metrics(
                sampled_labels,
                probabilities[model][indices],
                threshold,
            )

            for metric in METRICS:
                bootstrap[model][metric].append(values[metric])

    rows = []

    for model in THRESHOLDS:
        for metric in METRICS:
            samples = np.asarray(
                bootstrap[model][metric], dtype=float
            )

            lower, upper = np.percentile(
                samples, [2.5, 97.5]
            )

            rows.append({
                "model": model,
                "metric": metric,
                "estimate": observed[model][metric],
                "ci_lower": lower,
                "ci_upper": upper,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(DEST / "bootstrap_summary.csv", index=False)

    differences = []

    for metric in METRICS:
        # Positive difference means ConvNeXt > baseline.
        observed_difference = (
            observed["convnext"][metric]
            - observed["baseline"][metric]
        )

        paired_differences = (
            np.asarray(bootstrap["convnext"][metric])
            - np.asarray(bootstrap["baseline"][metric])
        )

        lower, upper = np.percentile(
            paired_differences, [2.5, 97.5]
        )

        differences.append({
            "metric": metric,
            "convnext_minus_baseline": observed_difference,
            "ci_lower": lower,
            "ci_upper": upper,
        })

    difference_df = pd.DataFrame(differences)
    difference_df.to_csv(
        DEST / "paired_differences.csv", index=False
    )

    print("\n" + "=" * 88)
    print("PATIENT-LEVEL RESULTS — VALIDATION-SELECTED THRESHOLDS")
    print("95% stratified patient-bootstrap percentile intervals")
    print("=" * 88)

    for metric in METRICS:
        print(f"\n{metric}")

        for model in THRESHOLDS:
            result = summary[
                (summary.model == model)
                & (summary.metric == metric)
            ].iloc[0]

            print(
                f"  {model:<10} "
                f"{result['estimate']:.4f} "
                f"[{result['ci_lower']:.4f}, "
                f"{result['ci_upper']:.4f}]"
            )

    print("\n" + "=" * 88)
    print("PAIRED MODEL DIFFERENCES (CONVNEXT - BASELINE)")
    print("=" * 88)

    for _, row in difference_df.iterrows():
        print(
            f"{row['metric']:<22} "
            f"{row['convnext_minus_baseline']:+.4f} "
            f"[{row['ci_lower']:+.4f}, "
            f"{row['ci_upper']:+.4f}]"
        )

    print("\nSaved results to:", DEST)
    print("Thresholds are frozen; no test-based tuning performed.")
    print(
        "Bootstrap intervals condition on the observed "
        "NC/AD class counts."
    )


if __name__ == "__main__":
    main()
