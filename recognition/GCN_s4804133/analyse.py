"""Report classification metrics from saved model predictions.

Use the test split only for final reporting; use validation results to make choices.
"""

import argparse
from pathlib import Path

import numpy as np

from dataset import CLASS_NAMES, load_graph
from utils import classification_metrics, expected_calibration_error


def main() -> None:
    """Load each model's saved predictions and print metrics for the chosen split."""
    parser = argparse.ArgumentParser(description=__doc__)
    # validation = used during development; test is the final report by default
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--tag",
        default="five-seed",
        help="prediction checkpoint tag used in the saved file names",
    )
    args = parser.parse_args()

    labels = load_graph().y.numpy()  # labels align with the saved node indices
    mask_name = f"{args.split}_mask"

    for model_name in ("mlp", "gcn"):
        # filenames match predict.py's default output naming
        prediction_path = Path("outputs") / f"predictions_{model_name}_seed42_{args.tag}.npz"
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Missing {prediction_path}; run predict.py for this model first."
            )

        with np.load(prediction_path) as saved:
            # only score nodes from the requested split
            mask = saved[mask_name]
            probabilities = saved["probabilities"][mask]

        y_true = labels[mask]
        # probabilities for AUROC, PR-AUC, and calibration metrics.
        metrics = classification_metrics(y_true, probabilities, CLASS_NAMES)
        ece = expected_calibration_error(y_true, probabilities)

        # ECE measures the gap between confidence and accuracy across 15 bins
        print(f"\n{model_name.upper()} — {args.split} set")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Macro one-vs-rest AUROC: {metrics['macro_auroc_ovr']:.4f}")
        print(f"Macro PR-AUC: {metrics['macro_pr_auc']:.4f}")
        print(f"ECE (15 bins): {ece:.4f}")
        print("Class                 Precision  Recall  F1      AUROC   PR-AUC  Support")
        for class_name, scores in metrics["per_class"].items():
            # each row reports one class against all remaining classes.
            print(
                f"{class_name:20s} "
                f"{scores['precision']:9.4f} "
                f"{scores['recall']:7.4f} "
                f"{scores['f1']:7.4f} "
                f"{scores['auroc_ovr']:7.4f} "
                f"{scores['pr_auc']:7.4f} "
                f"{scores['support']:8d}"
            )


if __name__ == "__main__":
    main()
