"""
Fit the AD decision threshold on validation, then check it on test.

Why this file exists. The README recommends moving the AD decision threshold
below 0.50, and the number it quoted was picked by looking at test
predictions. Both code reviewers said that this is selection on the test set,
and they were right. This script fits the threshold on the validation split
only, and then reports what that same threshold does on test. The test figure
is a check on a decision already made, not the decision itself.

What the threshold means. The model outputs P(AD) for each slice. The default
rule calls a slice AD when P(AD) is at least 0.50. That default assumes a
missed AD case and a false alarm cost the same, which is wrong for screening:
a missed case goes home undiagnosed, a false alarm gets a specialist's second
look.

How the threshold is chosen. I maximise Youden's J, which is sensitivity plus
specificity minus one. J treats both classes equally regardless of how many
of each there are, and it is the point on the ROC curve furthest above the
chance line, so it pairs with the AUROC already reported. I also print the
accuracy-optimal threshold for comparison, because the two usually disagree
and that disagreement is the whole point.

Usage, from the project folder:

    python analysis/fit_threshold.py \\
        --validation-npz figures/validation_predictions_gfnet.npz \\
        --test-npz outputs_gfnet_seed42_strong_p8_recipe_mix/test_predictions_gfnet.npz \\
        --label "GFNet, recommended"
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# AD is label 1, so column 1 of the probabilities holds P(AD).
AD_LABEL = 1


def load_split(path):
    """Read one predictions file and hand back what the sweep needs."""
    data = np.load(path)
    return (data["probabilities"][:, AD_LABEL],
            data["labels"],
            data["patient_ids"])


def to_patients(ad_scores, labels, patient_ids):
    """
    Collapse each patient's slices to one score by averaging.

    Same rule as predict.py and evaluate_calibration.py use, so the numbers
    here line up with the rest of the report.
    """
    patients = sorted(set(patient_ids.tolist()))
    mean_scores = []
    patient_labels = []
    for patient in patients:
        mask = patient_ids == patient
        mean_scores.append(ad_scores[mask].mean())
        patient_labels.append(labels[mask][0])
    return np.array(mean_scores), np.array(patient_labels)


def score_threshold(ad_scores, labels, threshold):
    """
    Work out what one threshold does. Everything by hand, as elsewhere.

    Calling a case AD when P(AD) >= threshold means a lower threshold catches
    more AD and raises more false alarms.
    """
    predicted_ad = ad_scores >= threshold
    actually_ad = labels == AD_LABEL

    true_positives = int((predicted_ad & actually_ad).sum())
    false_positives = int((predicted_ad & ~actually_ad).sum())
    false_negatives = int((~predicted_ad & actually_ad).sum())
    true_negatives = int((~predicted_ad & ~actually_ad).sum())

    total = len(labels)
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": (true_positives + true_negatives) / total,
        "ad_recall": sensitivity,
        "nc_recall": specificity,
        "youden_j": sensitivity + specificity - 1.0,
        "ad_missed": false_negatives,
        "false_alarms": false_positives,
    }


def sweep(ad_scores, labels, step=0.01):
    """Score every threshold from 0.01 to 0.99."""
    return [score_threshold(ad_scores, labels, t)
            for t in np.arange(step, 1.0, step)]


def show(title, result):
    print("  %-34s threshold %.2f   accuracy %.4f   AD recall %.4f   "
          "NC recall %.4f   AD missed %d"
          % (title, result["threshold"], result["accuracy"],
             result["ad_recall"], result["nc_recall"], result["ad_missed"]))


def run_level(level_name, validation, test):
    """Fit on validation at one level of aggregation, then check on test."""
    validation_scores, validation_labels = validation
    test_scores, test_labels = test

    results = sweep(validation_scores, validation_labels)
    best_j = max(results, key=lambda r: r["youden_j"])
    best_accuracy = max(results, key=lambda r: r["accuracy"])

    print()
    print("%s" % level_name.upper())
    print()
    print("  fitted on validation only")
    show("best Youden J", best_j)
    show("best accuracy", best_accuracy)
    show("default 0.50", score_threshold(validation_scores, validation_labels, 0.50))

    print()
    print("  the fitted threshold applied to test")
    chosen = score_threshold(test_scores, test_labels, best_j["threshold"])
    default = score_threshold(test_scores, test_labels, 0.50)
    show("fitted %.2f" % best_j["threshold"], chosen)
    show("default 0.50", default)

    print()
    print("  change from moving the threshold: accuracy %+.4f, AD recall %+.4f, "
          "%+d AD cases missed"
          % (chosen["accuracy"] - default["accuracy"],
             chosen["ad_recall"] - default["ad_recall"],
             chosen["ad_missed"] - default["ad_missed"]))
    return best_j["threshold"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-npz", required=True,
                        help="Predictions on the validation split. The threshold is fitted here.")
    parser.add_argument("--test-npz", required=True,
                        help="Predictions on the test split. Used only to check the fitted threshold.")
    parser.add_argument("--label", default="model", help="Name to print in the header.")
    arguments = parser.parse_args()

    validation_scores, validation_labels, validation_patients = load_split(arguments.validation_npz)
    test_scores, test_labels, test_patients = load_split(arguments.test_npz)

    print("=" * 78)
    print("OPERATING THRESHOLD FOR %s" % arguments.label)
    print("=" * 78)
    print("  validation: %d slices, %d patients"
          % (len(validation_labels), len(set(validation_patients.tolist()))))
    print("  test:       %d slices, %d patients"
          % (len(test_labels), len(set(test_patients.tolist()))))

    run_level("slice level",
              (validation_scores, validation_labels),
              (test_scores, test_labels))

    run_level("patient level",
              to_patients(validation_scores, validation_labels, validation_patients),
              to_patients(test_scores, test_labels, test_patients))

    print()
    print("The threshold comes from validation. The test rows are a check on it,")
    print("not a choice between options.")


if __name__ == "__main__":
    main()
