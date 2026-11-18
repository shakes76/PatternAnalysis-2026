"""
COMP3710: Patient-level selective classification experiment.

Classification thresholds were selected previously using validation
data. Rejection margins are derived from validation predictions only.

Test labels are used for retrospective evaluation, never to select
an operating threshold or rejection margin.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


OUTPUTS = Path("outputs")
RESULTS_DIR = OUTPUTS / "reject_option"

# Previously selected using validation balanced accuracy.
THRESHOLDS = {
    "baseline": 0.78,
    "convnext": 0.45,
}

# Predefined experimental coverage target.
TARGET_COVERAGE = 0.80


def load_predictions(model, split):
    path = OUTPUTS / model / split / "patient_predictions.csv"

    df = pd.read_csv(
        path,
        usecols=["patient_id", "true_label", "probability_ad"],
        dtype={"patient_id": str},
    )

    df["true_label"] = pd.to_numeric(df["true_label"])
    df["probability_ad"] = pd.to_numeric(df["probability_ad"])

    if df["patient_id"].isna().any():
        raise ValueError(f"Missing patient IDs: {path}")

    if df["patient_id"].duplicated().any():
        raise ValueError(f"Duplicate patients: {path}")

    if not df["true_label"].isin([0, 1]).all():
        raise ValueError(f"Unexpected labels: {path}")

    probabilities = df["probability_ad"].to_numpy()

    if not np.isfinite(probabilities).all():
        raise ValueError(f"Invalid probabilities: {path}")

    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError(f"Probabilities outside [0, 1]: {path}")

    return df


def evaluate(df, threshold, rejection_margin):
    y = df["true_label"].to_numpy(dtype=int)
    p = df["probability_ad"].to_numpy(dtype=float)

    prediction = (p >= threshold).astype(int)
    distance = np.abs(p - threshold)

    # Reject predictions closest to the decision threshold.
    accepted = distance >= rejection_margin

    ya = y[accepted]
    pa = prediction[accepted]

    tn = int(((ya == 0) & (pa == 0)).sum())
    fp = int(((ya == 0) & (pa == 1)).sum())
    fn = int(((ya == 1) & (pa == 0)).sum())
    tp = int(((ya == 1) & (pa == 1)).sum())

    n = len(y)
    n_accepted = int(accepted.sum())
    n_nc = int((y == 0).sum())
    n_ad = int((y == 1).sum())

    accuracy = (
        float((ya == pa).mean())
        if n_accepted else float("nan")
    )

    sensitivity = (
        tp / (tp + fn) if tp + fn else float("nan")
    )

    specificity = (
        tn / (tn + fp) if tn + fp else float("nan")
    )

    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else float("nan")
    )

    macro_f1 = (
        f1_score(
            ya, pa,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
        if n_accepted else float("nan")
    )

    return {
        "threshold": threshold,
        "rejection_margin": rejection_margin,
        "n_total": n,
        "n_accepted": n_accepted,
        "n_rejected": n - n_accepted,
        "coverage": n_accepted / n,
        "accuracy": accuracy,
        "error_rate": 1 - accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "NC_coverage": (
            ((y == 0) & accepted).sum() / n_nc
            if n_nc else float("nan")
        ),
        "AD_coverage": (
            ((y == 1) & accepted).sum() / n_ad
            if n_ad else float("nan")
        ),
    }


def select_margin(validation_df, threshold):
    """
    Use the validation probability distribution only.

    The 20th percentile of distance to the decision boundary
    rejects approximately 20% of validation patients.
    """
    probabilities = validation_df["probability_ad"].to_numpy()
    distances = np.abs(probabilities - threshold)

    return float(
        np.quantile(distances, 1 - TARGET_COVERAGE)
    )


def risk_coverage(df, threshold):
    y = df["true_label"].to_numpy(dtype=int)
    p = df["probability_ad"].to_numpy(dtype=float)

    prediction = (p >= threshold).astype(int)
    errors = (prediction != y).astype(int)

    distance = np.abs(p - threshold)

    # Highest decision-margin patients first.
    order = np.argsort(-distance, kind="stable")
    ordered_errors = errors[order]

    k = np.arange(1, len(y) + 1)
    coverage = k / len(y)
    risk = np.cumsum(ordered_errors) / k

    return coverage, risk


def plot_risk_coverage(model, val_df, test_df, threshold, margin):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for split, df, linestyle in [
        ("Validation", val_df, "--"),
        ("Test (descriptive only)", test_df, "-"),
    ]:
        coverage, risk = risk_coverage(df, threshold)

        ax.plot(
            coverage * 100,
            risk * 100,
            linestyle=linestyle,
            label=split,
        )

    selected = evaluate(test_df, threshold, margin)

    ax.scatter(
        selected["coverage"] * 100,
        selected["error_rate"] * 100,
        marker="o",
        s=100,
        zorder=5,
        label="Frozen policy on test",
    )

    ax.set_xlabel("Automated coverage (%)")
    ax.set_ylabel("Error rate among accepted patients (%)")
    ax.set_title(f"{model}: Risk–coverage analysis")
    ax.set_xlim(0, 102)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / f"{model}_risk_coverage.png",
        dpi=160,
    )
    plt.close(fig)


def plot_margin_correctness(model, df, threshold):
    y = df["true_label"].to_numpy(dtype=int)
    p = df["probability_ad"].to_numpy(dtype=float)

    prediction = (p >= threshold).astype(int)
    correct = prediction == y
    margin = np.abs(p - threshold)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.hist(
        margin[correct],
        bins=15,
        alpha=0.65,
        label=f"Correct (n={correct.sum()})",
    )

    ax.hist(
        margin[~correct],
        bins=15,
        alpha=0.65,
        label=f"Incorrect (n={(~correct).sum()})",
    )

    ax.set_xlabel("Absolute distance from classification threshold")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"{model}: Test decision margin vs correctness")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / f"{model}_margin_correctness.png",
        dpi=160,
    )
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_predictions = {
        model: {
            split: load_predictions(model, split)
            for split in ("val", "test")
        }
        for model in THRESHOLDS
    }

    # Check for patient overlap and ensure identical evaluation cohorts.
    for model, frames in all_predictions.items():
        val_ids = set(frames["val"]["patient_id"])
        test_ids = set(frames["test"]["patient_id"])

        if val_ids & test_ids:
            raise ValueError(f"Validation/test patient overlap: {model}")

    for split in ("val", "test"):
        a = all_predictions["baseline"][split]
        b = all_predictions["convnext"][split]

        labels_a = set(zip(a["patient_id"], a["true_label"]))
        labels_b = set(zip(b["patient_id"], b["true_label"]))

        if labels_a != labels_b:
            raise ValueError(f"Model cohorts differ: {split}")

    rows = []

    for model, threshold in THRESHOLDS.items():
        frames = all_predictions[model]
        val_df = frames["val"]
        test_df = frames["test"]

        # Rejection policy selected exclusively from validation.
        margin = select_margin(val_df, threshold)

        print(f"\n{model.upper()}")
        print(f"Classification threshold: {threshold:.2f}")
        print(f"Validation-selected rejection margin: {margin:.4f}")
        print(
            f"Accept NC if P(AD) <= {threshold - margin:.4f}; "
            f"accept AD if P(AD) >= {threshold + margin:.4f}"
        )

        for split, df in [("val", val_df), ("test", test_df)]:
            for policy, rejection_margin in [
                ("No rejection", 0.0),
                ("80% validation coverage", margin),
            ]:
                result = evaluate(df, threshold, rejection_margin)

                rows.append({
                    "model": model,
                    "split": split,
                    "policy": policy,
                    **result,
                })

        plot_risk_coverage(
            model, val_df, test_df, threshold, margin
        )

        plot_margin_correctness(model, test_df, threshold)

    summary = pd.DataFrame(rows)

    summary.to_csv(
        RESULTS_DIR / "reject_option_summary.csv",
        index=False,
    )

    columns = [
        "model",
        "split",
        "policy",
        "coverage",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "sensitivity",
        "specificity",
        "FP",
        "FN",
    ]

    print("\n" + "=" * 110)
    print("SELECTIVE CLASSIFICATION RESULTS")
    print("=" * 110)

    print(
        summary[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nClass-specific coverage:")
    print(
        summary[
            ["model", "split", "policy", "NC_coverage", "AD_coverage"]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nResults saved to:", RESULTS_DIR)
    print("Test results are retrospective, not policy-selection data.")


if __name__ == "__main__":
    main()
