"""Compare matched predictions with a paired patient bootstrap.

The command accepts baseline and Siamese prediction CSVs with matching
image and patient order. Each draw resamples whole patients, stratified by
melanoma presence, and scores both models on that draw. Print estimates and
95% percentile intervals as a Markdown table.

Patient resampling follows Field and Welsh (2007),
https://doi.org/10.1111/j.1467-9868.2007.00593.x.

Author: Alex Anderson (47472442), COMP3710 Pattern Recognition, 2026.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def read_predictions(path: Path) -> dict[str, np.ndarray]:
    """Read saved image predictions into arrays in CSV row order.

    Parameters
    ----------
    path : pathlib.Path
        Prediction CSV written by the prediction command.

    Returns
    -------
    dict of str to numpy.ndarray
        Image names, patient identifiers, integer labels, probabilities,
        reference scores and decision strings. The CSV actual column is
        returned as label; all arrays have one entry per image.
    """
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        "image_name": np.array([r["image_name"] for r in rows]),
        "patient_id": np.array([r["patient_id"] for r in rows]),
        "label": np.array([int(r["actual"]) for r in rows]),
        "probability": np.array([float(r["probability"]) for r in rows]),
        "reference_score": np.array([float(r["reference_score"]) for r in rows]),
        "decision": np.array([r["decision"] for r in rows]),
    }


def summarise(run: dict[str, np.ndarray], index: np.ndarray) -> np.ndarray:
    """Calculate classification and review measures for selected images.

    Parameters
    ----------
    run : dict of str to numpy.ndarray
        Aligned prediction arrays returned by `read_predictions`.
    index : numpy.ndarray of int
        Image indices to include. Repeated indices represent bootstrap
        draws that select the same patient more than once.

    Returns
    -------
    numpy.ndarray of shape (7,)
        AUROC, melanoma AP, classifier recall, precision, accuracy, review
        workload and melanoma capture, in that order. A melanoma decision
        is classifier-positive; melanoma and review decisions both count
        towards specialist review.
    """
    labels, decision = run["label"][index], run["decision"][index]
    flagged, reviewed = decision == "melanoma", decision != "benign"
    return np.array([
        roc_auc_score(labels, run["probability"][index]),
        average_precision_score(labels, run["probability"][index]),
        flagged[labels == 1].mean(),
        labels[flagged].mean(),
        (flagged == labels).mean(),
        reviewed.mean(),
        reviewed[labels == 1].mean(),
    ])


def interval(values: np.ndarray) -> str:
    """Format the central 95% percentile interval of bootstrap values.

    Parameters
    ----------
    values : numpy.ndarray
        Bootstrap estimates. NaN values are ignored.

    Returns
    -------
    str
        Lower and upper bounds in brackets, rounded to four decimals.
    """
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return f"[{low:.4f}, {high:.4f}]"


NAMES = (
    "AUROC", "Melanoma AP", "Decision recall", "Decision precision", "Decision accuracy",
    "Review workload", "Review capture",
)


def main() -> None:
    """Print paired patient-bootstrap intervals for two prediction CSVs.

    Read baseline and Siamese paths and an optional draw count from the
    command line. Both files must list the same images in the same order.
    Resample patients with seed 42 and print a Markdown comparison table.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("siamese", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    baseline, siamese = read_predictions(args.baseline), read_predictions(args.siamese)

    images_by_patient: dict[str, list[int]] = defaultdict(list)
    for i, patient in enumerate(baseline["patient_id"]):
        images_by_patient[patient].append(i)
    has_melanoma = {p: baseline["label"][i].any() for p, i in images_by_patient.items()}
    melanoma_patients = [p for p in images_by_patient if has_melanoma[p]]
    benign_patients = [p for p in images_by_patient if not has_melanoma[p]]
    strata = [np.array(melanoma_patients), np.array(benign_patients)]

    rng = np.random.default_rng(42)
    draws = []
    for _ in range(args.draws):
        patients = np.concatenate([rng.choice(group, len(group)) for group in strata])
        index = np.concatenate([images_by_patient[p] for p in patients])
        draws.append(np.stack([summarise(baseline, index), summarise(siamese, index)]))
    draws = np.array(draws)  # [draw, model, metric]
    everyone = np.arange(len(baseline["label"]))
    estimates = np.stack([summarise(baseline, everyone), summarise(siamese, everyone)])

    print("| Measure | Baseline | Siamese | Siamese - baseline |")
    print("| --- | ---: | ---: | ---: |")
    for m, name in enumerate(NAMES):
        cells = [f"{estimates[k, m]:.4f} {interval(draws[:, k, m])}" for k in (0, 1)]
        difference = f"{estimates[1, m] - estimates[0, m]:+.4f} {interval(draws[:, 1, m] - draws[:, 0, m])}"
        print(f"| {name} | {cells[0]} | {cells[1]} | {difference} |")


if __name__ == "__main__":
    main()
