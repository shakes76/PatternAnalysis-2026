"""
Example usage of the trained GFNet: load the saved model, run it on the test set,
and print a summary of its performance.

Prints a table of slice-level and patient-level metrics, the patient-level
confusion matrix, and a few example patient predictions. Slice-level metrics
treat each of the 9,000 test slices separately; patient-level metrics average
the AD probability over each patient's 20 slices and predict AD if the mean is
at least 0.5 (threshold).

Usage (from the project folder; uses a GPU if available, otherwise the CPU):
    python predict.py
"""

from collections import Counter

import torch

from dataset import get_datasets, get_loaders
from evaluate import load_model
from train import DATA_ROOT, PROJECT_DIR
from utils import evaluate

# Settings
CHECKPOINT_RUN = (
    PROJECT_DIR / "runs" / "gfnet_20261027_144653"
)  # folder containing best.pt
BATCH_SIZE = 64
NUM_WORKERS = 1
N_EXAMPLES = 8  # example patient predictions to print

METRICS = [  # (key from utils.classification_metrics, display name)
    ("acc", "Accuracy"),
    ("auroc", "AUROC"),
    ("macro_f1", "Macro-F1"),
    ("recall_ad", "Sensitivity (AD recall)"),
    ("recall_nc", "Specificity (NC recall)"),
    ("precision_ad", "Precision AD"),
    ("precision_nc", "Precision NC"),
]


def confusion(rows):
    """Counts of true/false positives and negatives, with AD as the positive class."""
    c = Counter((r["label"], int(r["prob_ad"] >= 0.5)) for r in rows)
    return {"TP": c[(1, 1)], "FN": c[(1, 0)], "TN": c[(0, 0)], "FP": c[(0, 1)]}


def main():
    """Load the checkpoint, run test-set inference, and print the summary."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(CHECKPOINT_RUN, device)
    print(
        f"Model: {ckpt['model']} from {CHECKPOINT_RUN.name} (epoch {ckpt['epoch']}) on {device}"
    )

    datasets, _ = get_datasets(DATA_ROOT, num_workers=NUM_WORKERS)  # no augmentation
    test_loader = get_loaders(datasets, BATCH_SIZE, NUM_WORKERS)["test"]
    print(f"Running inference on {len(datasets['test']):,} test slices...")
    result = evaluate(model, test_loader, device)
    slices, patients = result["slice_rows"], result["patient_rows"]

    # Summary table
    print(f"\n{'Metric':<26}{'Slice level':>13}{'Patient level':>15}")
    print("-" * 54)
    print(f"{'Count':<26}{len(slices):>13,}{len(patients):>15,}")
    for key, name in METRICS:
        print(f"{name:<26}{result['slice'][key]:>13.3f}{result['patient'][key]:>15.3f}")
    print(f"{'Cross-entropy loss':<26}{result['loss']:>13.3f}{'-':>15}")

    # Confusion matrix at patient level
    c = confusion(patients)
    print("\nPatient-level confusion matrix (AD = positive, threshold 0.5)")
    print(f"{'':<12}{'Pred NC':>9}{'Pred AD':>9}")
    print(f"{'True NC':<12}{c['TN']:>9}{c['FP']:>9}")
    print(f"{'True AD':<12}{c['FN']:>9}{c['TP']:>9}")

    # A few example patient predictions
    print(f"\nExample patient predictions (first {N_EXAMPLES} by patient ID)")
    print(
        f"{'Patient':<10}{'True':>6}{'P(AD)':>8}{'Predicted':>11}{'Slice std':>11}  Correct"
    )
    for r in patients[:N_EXAMPLES]:
        true = "AD" if r["label"] else "NC"
        pred = "AD" if r["prob_ad"] >= 0.5 else "NC"
        print(
            f"{r['pid']:<10}{true:>6}{r['prob_ad']:>8.3f}{pred:>11}{r['slice_prob_std']:>11.3f}  "
            f"{'yes' if true == pred else 'no'}"
        )


if __name__ == "__main__":
    main()
