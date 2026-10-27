"""Reproducible patient-level bootstrap analysis for COMP3710 ADNI results.

This script compares saved patient-level predictions from the baseline CNN
and ConvNeXt-Tiny without retraining either model.

It reports:
- point estimates for accuracy, macro-F1, AD recall and AUROC;
- 95% stratified patient-bootstrap confidence intervals;
- paired ConvNeXt-minus-baseline bootstrap differences;
- aggregate CSV/JSON summaries and a paired-difference plot.

The bootstrap unit is the patient, not the MRI slice, because slices from the
same patient are correlated.
"""

import argparse
import csv
import json
import random
from pathlib import Path


METRICS = ("accuracy", "macro_f1", "ad_recall", "auroc")


def load_predictions(path: Path):
    """Load one patient_predictions.csv and validate required fields."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No prediction rows found in {path}")

    required = {"patient_id", "label", "prediction", "p_ad"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    result = {}
    for row in rows:
        patient_id = row["patient_id"]
        if patient_id in result:
            raise ValueError(f"Duplicate patient_id in {path}: {patient_id}")

        label = int(row["label"])
        prediction = int(row["prediction"])
        p_ad = float(row["p_ad"])

        if label not in (0, 1) or prediction not in (0, 1):
            raise ValueError(f"Invalid binary label/prediction for {patient_id}")
        if not 0.0 <= p_ad <= 1.0:
            raise ValueError(f"Invalid P(AD) for {patient_id}: {p_ad}")

        result[patient_id] = {
            "patient_id": patient_id,
            "label": label,
            "prediction": prediction,
            "p_ad": p_ad,
        }

    return result


def roc_auc(rows):
    """Binary AUROC using the Mann-Whitney rank formulation with tie handling."""
    ranked = sorted((float(row["p_ad"]), int(row["label"])) for row in rows)

    positives = sum(label for _, label in ranked)
    negatives = len(ranked) - positives

    if positives == 0 or negatives == 0:
        return None

    positive_rank_sum = 0.0
    i = 0

    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1

        # Ranks are one-based. All tied observations receive the average rank.
        average_rank = ((i + 1) + j) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in ranked[i:j]
        )
        i = j

    return (
        positive_rank_sum
        - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def classification_metrics(rows):
    """Compute patient-level accuracy, macro-F1, AD recall and AUROC."""
    if not rows:
        raise ValueError("Cannot compute metrics from an empty sample")

    # Rows are true class; columns are predicted class.
    confusion = [[0, 0], [0, 0]]

    for row in rows:
        confusion[int(row["label"])][int(row["prediction"])] += 1

    total = len(rows)
    accuracy = (confusion[0][0] + confusion[1][1]) / total

    f1_values = []
    for class_id in (0, 1):
        tp = confusion[class_id][class_id]
        fp = confusion[1 - class_id][class_id]
        fn = confusion[class_id][1 - class_id]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)

    ad_total = confusion[1][0] + confusion[1][1]
    ad_recall = confusion[1][1] / ad_total if ad_total else None

    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / 2.0,
        "ad_recall": ad_recall,
        "auroc": roc_auc(rows),
        "confusion_matrix": confusion,
    }


def percentile(values, probability):
    """Linear-interpolated percentile matching the original project analysis."""
    if not values:
        raise ValueError("Cannot compute percentile from an empty list")

    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return (
        values[lower] * (1.0 - fraction)
        + values[upper] * fraction
    )


def stratified_bootstrap(rows, replicates, seed):
    """Bootstrap one model while preserving observed NC/AD patient counts."""
    nc_rows = [row for row in rows if int(row["label"]) == 0]
    ad_rows = [row for row in rows if int(row["label"]) == 1]

    if not nc_rows or not ad_rows:
        raise ValueError("Both NC and AD patients are required")

    rng = random.Random(seed)
    samples = {metric: [] for metric in METRICS}

    for _ in range(replicates):
        bootstrap_rows = (
            [rng.choice(nc_rows) for _ in range(len(nc_rows))]
            + [rng.choice(ad_rows) for _ in range(len(ad_rows))]
        )

        result = classification_metrics(bootstrap_rows)

        for metric in METRICS:
            value = result[metric]
            if value is not None:
                samples[metric].append(value)

    return {
        metric: {
            "lower_95": percentile(values, 0.025),
            "upper_95": percentile(values, 0.975),
        }
        for metric, values in samples.items()
    }


def paired_bootstrap(baseline, convnext, replicates, seed):
    """Paired bootstrap using identical resampled patient IDs for both models."""
    baseline_ids = set(baseline)
    convnext_ids = set(convnext)

    if baseline_ids != convnext_ids:
        missing_baseline = sorted(convnext_ids - baseline_ids)
        missing_convnext = sorted(baseline_ids - convnext_ids)
        raise ValueError(
            "Model prediction files contain different patient IDs. "
            f"Missing baseline={missing_baseline[:5]}, "
            f"missing ConvNeXt={missing_convnext[:5]}"
        )

    patient_ids = sorted(baseline_ids)

    for patient_id in patient_ids:
        if baseline[patient_id]["label"] != convnext[patient_id]["label"]:
            raise ValueError(f"Label mismatch for patient {patient_id}")

    nc_ids = [
        patient_id
        for patient_id in patient_ids
        if baseline[patient_id]["label"] == 0
    ]
    ad_ids = [
        patient_id
        for patient_id in patient_ids
        if baseline[patient_id]["label"] == 1
    ]

    baseline_rows = [baseline[patient_id] for patient_id in patient_ids]
    convnext_rows = [convnext[patient_id] for patient_id in patient_ids]

    baseline_point = classification_metrics(baseline_rows)
    convnext_point = classification_metrics(convnext_rows)

    point_delta = {
        metric: convnext_point[metric] - baseline_point[metric]
        for metric in METRICS
    }

    rng = random.Random(seed)
    delta_samples = {metric: [] for metric in METRICS}

    for _ in range(replicates):
        sampled_ids = (
            [rng.choice(nc_ids) for _ in range(len(nc_ids))]
            + [rng.choice(ad_ids) for _ in range(len(ad_ids))]
        )

        baseline_sample = [
            baseline[patient_id] for patient_id in sampled_ids
        ]
        convnext_sample = [
            convnext[patient_id] for patient_id in sampled_ids
        ]

        baseline_metrics = classification_metrics(baseline_sample)
        convnext_metrics = classification_metrics(convnext_sample)

        for metric in METRICS:
            delta_samples[metric].append(
                convnext_metrics[metric] - baseline_metrics[metric]
            )

    intervals = {}
    for metric in METRICS:
        values = delta_samples[metric]
        intervals[metric] = {
            "point_delta": point_delta[metric],
            "lower_95": percentile(values, 0.025),
            "upper_95": percentile(values, 0.975),
            "bootstrap_fraction_positive": (
                sum(value > 0 for value in values) / len(values)
            ),
        }

    return {
        "patient_count": len(patient_ids),
        "class_counts": {
            "NC": len(nc_ids),
            "AD": len(ad_ids),
        },
        "baseline_point": baseline_point,
        "convnext_point": convnext_point,
        "differences": intervals,
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, paired_results):
    """Save aggregate paired bootstrap differences with 95% intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Accuracy", "Macro-F1", "AD recall", "AUROC"]
    keys = ["accuracy", "macro_f1", "ad_recall", "auroc"]

    values = [
        paired_results["differences"][key]["point_delta"]
        for key in keys
    ]
    lows = [
        paired_results["differences"][key]["lower_95"]
        for key in keys
    ]
    highs = [
        paired_results["differences"][key]["upper_95"]
        for key in keys
    ]

    lower_error = [
        value - low for value, low in zip(values, lows)
    ]
    upper_error = [
        high - value for value, high in zip(values, highs)
    ]

    x_positions = list(range(len(keys)))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axhline(0.0, linewidth=1)

    ax.errorbar(
        x_positions,
        values,
        yerr=[lower_error, upper_error],
        fmt="o",
        capsize=6,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Difference (ConvNeXt - Baseline)")
    ax.set_title("Paired patient-level bootstrap comparison")
    ax.grid(axis="y", alpha=0.25)

    for index, value in enumerate(values):
        ax.annotate(
            f"{value:+.4f}",
            (index, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.text(
        0.5,
        -0.19,
        "95% paired stratified patient bootstrap intervals; "
        f"{paired_results['replicates']:,} replicates; "
        f"n={paired_results['patient_count']} patients",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-predictions",
        required=True,
        type=Path,
        help="Baseline patient_predictions.csv",
    )
    parser.add_argument(
        "--convnext-predictions",
        required=True,
        type=Path,
        help="ConvNeXt patient_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for aggregate bootstrap outputs",
    )
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.replicates < 100:
        parser.error("--replicates must be at least 100")

    baseline = load_predictions(
        args.baseline_predictions.expanduser().resolve()
    )
    convnext = load_predictions(
        args.convnext_predictions.expanduser().resolve()
    )

    baseline_ids = sorted(baseline)
    convnext_ids = sorted(convnext)

    if baseline_ids != convnext_ids:
        raise ValueError(
            "Baseline and ConvNeXt files must contain identical patients"
        )

    baseline_rows = [baseline[patient_id] for patient_id in baseline_ids]
    convnext_rows = [convnext[patient_id] for patient_id in convnext_ids]

    baseline_point = classification_metrics(baseline_rows)
    convnext_point = classification_metrics(convnext_rows)

    baseline_ci = stratified_bootstrap(
        baseline_rows, args.replicates, args.seed
    )
    convnext_ci = stratified_bootstrap(
        convnext_rows, args.replicates, args.seed
    )

    paired = paired_bootstrap(
        baseline,
        convnext,
        args.replicates,
        args.seed,
    )
    paired["replicates"] = args.replicates
    paired["seed"] = args.seed

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_ci_rows = []
    for model_name, point, intervals in (
        ("baseline", baseline_point, baseline_ci),
        ("convnext", convnext_point, convnext_ci),
    ):
        for metric in METRICS:
            patient_ci_rows.append({
                "model": model_name,
                "metric": metric,
                "estimate": point[metric],
                "lower_95": intervals[metric]["lower_95"],
                "upper_95": intervals[metric]["upper_95"],
            })

    write_csv(
        output_dir / "patient_metric_ci.csv",
        patient_ci_rows,
        ["model", "metric", "estimate", "lower_95", "upper_95"],
    )

    paired_rows = []
    for metric in METRICS:
        result = paired["differences"][metric]
        paired_rows.append({
            "metric": metric,
            "point_delta": result["point_delta"],
            "lower_95": result["lower_95"],
            "upper_95": result["upper_95"],
            "bootstrap_fraction_positive": (
                result["bootstrap_fraction_positive"]
            ),
        })

    write_csv(
        output_dir / "paired_metric_delta.csv",
        paired_rows,
        [
            "metric",
            "point_delta",
            "lower_95",
            "upper_95",
            "bootstrap_fraction_positive",
        ],
    )

    summary = {
        "method": {
            "bootstrap_unit": "patient",
            "stratified_by_label": True,
            "paired_model_comparison": True,
            "replicates": args.replicates,
            "seed": args.seed,
            "confidence_level": 0.95,
        },
        "patient_count": len(baseline_ids),
        "baseline": {
            "point_estimates": baseline_point,
            "confidence_intervals": baseline_ci,
        },
        "convnext": {
            "point_estimates": convnext_point,
            "confidence_intervals": convnext_ci,
        },
        "paired_convnext_minus_baseline": paired["differences"],
    }

    (output_dir / "bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    save_plot(
        output_dir / "paired_bootstrap_delta.png",
        paired,
    )

    print("PATIENT-LEVEL BOOTSTRAP ANALYSIS")
    print("--------------------------------")
    print(
        f"Patients: {paired['patient_count']} "
        f"(NC={paired['class_counts']['NC']}, "
        f"AD={paired['class_counts']['AD']})"
    )
    print(f"Replicates: {args.replicates}")
    print(f"Seed: {args.seed}")

    for model_name, point, intervals in (
        ("Baseline", baseline_point, baseline_ci),
        ("ConvNeXt", convnext_point, convnext_ci),
    ):
        print(f"\n{model_name}")
        for metric in METRICS:
            print(
                f"{metric:10s}: {point[metric]:.4f} "
                f"[{intervals[metric]['lower_95']:.4f}, "
                f"{intervals[metric]['upper_95']:.4f}]"
            )

    print("\nPaired delta = ConvNeXt - Baseline")
    for metric in METRICS:
        result = paired["differences"][metric]
        print(
            f"{metric:10s}: "
            f"{result['point_delta']:+.4f} "
            f"[{result['lower_95']:+.4f}, "
            f"{result['upper_95']:+.4f}]"
        )

    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}") from error
