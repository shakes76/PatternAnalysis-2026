"""Regression test for the recorded patient-level bootstrap results."""

import argparse
from pathlib import Path

from bootstrap_analysis import (
    classification_metrics,
    load_predictions,
    paired_bootstrap,
    stratified_bootstrap,
)

REPLICATES = 10000
SEED = 42


def check(name, actual, expected):
    if round(float(actual), 4) != round(float(expected), 4):
        raise AssertionError(
            f"{name}: expected {expected:.4f}, got {actual:.4f}"
        )


def check_ci(name, result, lower, upper):
    check(f"{name} lower", result["lower_95"], lower)
    check(f"{name} upper", result["upper_95"], upper)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True, type=Path)
    parser.add_argument("--convnext-predictions", required=True, type=Path)
    args = parser.parse_args()

    baseline = load_predictions(
        args.baseline_predictions.expanduser().resolve()
    )
    convnext = load_predictions(
        args.convnext_predictions.expanduser().resolve()
    )

    if set(baseline) != set(convnext):
        raise AssertionError("Patient IDs differ between models")

    ids = sorted(baseline)

    if len(ids) != 102:
        raise AssertionError(f"Expected 102 patients, found {len(ids)}")

    nc = sum(baseline[x]["label"] == 0 for x in ids)
    ad = sum(baseline[x]["label"] == 1 for x in ids)

    if (nc, ad) != (69, 33):
        raise AssertionError(
            f"Expected NC=69, AD=33; got NC={nc}, AD={ad}"
        )

    baseline_rows = [baseline[x] for x in ids]
    convnext_rows = [convnext[x] for x in ids]

    b = classification_metrics(baseline_rows)
    c = classification_metrics(convnext_rows)

    if b["confusion_matrix"] != [[54, 15], [7, 26]]:
        raise AssertionError("Unexpected baseline confusion matrix")

    if c["confusion_matrix"] != [[55, 14], [3, 30]]:
        raise AssertionError("Unexpected ConvNeXt confusion matrix")

    expected_b = {
        "accuracy": 0.7843,
        "macro_f1": 0.7667,
        "ad_recall": 0.7879,
        "auroc": 0.9043,
    }

    expected_c = {
        "accuracy": 0.8333,
        "macro_f1": 0.8227,
        "ad_recall": 0.9091,
        "auroc": 0.9372,
    }

    for metric, value in expected_b.items():
        check(f"Baseline {metric}", b[metric], value)

    for metric, value in expected_c.items():
        check(f"ConvNeXt {metric}", c[metric], value)

    print("PASS: patient composition and point estimates")

    b_ci = stratified_bootstrap(
        baseline_rows, REPLICATES, SEED
    )
    c_ci = stratified_bootstrap(
        convnext_rows, REPLICATES, SEED
    )

    expected_b_ci = {
        "accuracy": (0.7059, 0.8627),
        "macro_f1": (0.6819, 0.8478),
        "ad_recall": (0.6364, 0.9091),
        "auroc": (0.8428, 0.9543),
    }

    expected_c_ci = {
        "accuracy": (0.7549, 0.9020),
        "macro_f1": (0.7479, 0.8927),
        "ad_recall": (0.7879, 1.0000),
        "auroc": (0.8889, 0.9745),
    }

    for metric, (lo, hi) in expected_b_ci.items():
        check_ci(f"Baseline {metric}", b_ci[metric], lo, hi)

    for metric, (lo, hi) in expected_c_ci.items():
        check_ci(f"ConvNeXt {metric}", c_ci[metric], lo, hi)

    print("PASS: single-model bootstrap confidence intervals")

    paired = paired_bootstrap(
        baseline,
        convnext,
        REPLICATES,
        SEED,
    )

    expected_paired = {
        "accuracy": (0.0490, -0.0196, 0.1176),
        "macro_f1": (0.0559, -0.0134, 0.1309),
        "ad_recall": (0.1212, 0.0303, 0.2424),
        "auroc": (0.0329, -0.0031, 0.0755),
    }

    for metric, (delta, lo, hi) in expected_paired.items():
        result = paired["differences"][metric]
        check(f"Paired {metric} delta", result["point_delta"], delta)
        check_ci(f"Paired {metric}", result, lo, hi)

    print("PASS: paired bootstrap differences and confidence intervals")
    print()
    print("PASS: all recorded bootstrap results are reproducible")
    print(
        "Configuration: "
        "10000 replicates, seed=42, patients=102 (NC=69, AD=33)"
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        raise SystemExit(f"TEST FAILED: {exc}") from exc
