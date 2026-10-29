"""
evaluate_calibration.py

The actual investigation of the open research dilemma for this project.

The assignment asks two questions I need to answer with evidence:

  1. Does the model's confidence score correlate with whether it is correct, or
     is it confidently wrong on the hard cases?
  2. Can I set a confidence threshold that safely automates the easy cases and
     sends the uncertain ones to a human specialist? And does my advanced model
     earn its extra cost over the baseline on this measure?

This script reads the test_predictions_<model>.npz files that train.py writes,
so it needs no GPU and no retraining. That was the point of saving per-image
probabilities during training.

Usage:
    python evaluate_calibration.py \
        --predictions outputs_baseline_seed42/test_predictions_baseline.npz \
                      outputs_gfnet_seed42/test_predictions_gfnet.npz \
        --names baseline gfnet \
        --output-dir figures

Optional, for temperature scaling (see the note in fit_temperature below):
    --val-predictions outputs_gfnet_seed42/val_predictions_gfnet.npz

What it produces:
    calibration_reliability.png     reliability diagram, one panel per model
    calibration_confidence.png      confidence histograms, correct vs wrong
    reject_option_curves.png        what happens as the threshold moves
    calibration_results.json        every number, for the README tables
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# AD is the positive class (label 1). A false negative is therefore a patient
# with Alzheimer's that the model called normal, which is the expensive mistake.
AD_LABEL = 1
NC_LABEL = 0


def load_predictions(path):
    """Read one npz written by train.py."""
    data = np.load(path)
    return {
        "probabilities": data["probabilities"],   # (n, 2), rows sum to 1
        "labels": data["labels"],                 # (n,) 0 or 1
        "patient_ids": data["patient_ids"],       # (n,) strings
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def compute_calibration_bins(probabilities, labels, num_bins=15):
    """
    Group predictions by how confident they were, and in each group compare the
    average confidence against the actual accuracy.

    The idea: if a model says "90% sure" on a thousand cases, it should be right
    about nine hundred times. If it is only right six hundred times, it is
    overconfident, and its confidence score cannot be used to decide what to
    automate.

    Confidence here is the probability assigned to the predicted class, so it
    always lies between 0.5 and 1.0 for a two-class problem - the predicted
    class is by definition the one with the larger probability.

    Returns a list of per-bin dictionaries, skipping empty bins.
    """
    predicted_labels = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    was_correct = (predicted_labels == labels).astype(np.float64)

    # Bin edges across the range confidence can actually take.
    bin_edges = np.linspace(0.5, 1.0, num_bins + 1)

    bins = []
    for index in range(num_bins):
        low = bin_edges[index]
        high = bin_edges[index + 1]

        # Include the right edge on the final bin so confidence == 1.0 is counted.
        if index == num_bins - 1:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences >= low) & (confidences < high)

        count = int(in_bin.sum())
        if count == 0:
            continue

        bins.append({
            "low": float(low),
            "high": float(high),
            "count": count,
            "mean_confidence": float(confidences[in_bin].mean()),
            "accuracy": float(was_correct[in_bin].mean()),
        })

    return bins


def compute_expected_calibration_error(probabilities, labels, num_bins=15):
    """
    Expected Calibration Error: the average gap between confidence and accuracy,
    weighted by how many predictions fall in each bin.

        ECE = sum over bins of (count / total) * |accuracy - mean confidence|

    A perfectly calibrated model scores 0. The weighting matters - a wild gap in
    a bin holding three predictions should not count the same as a small gap in
    a bin holding three thousand.

    One honest limitation: ECE depends on the number of bins. More bins catch
    finer miscalibration but each bin holds fewer samples and gets noisier. I
    report at 15 bins and keep it the same for both models so the comparison is
    fair, which is the part that actually matters here.
    """
    bins = compute_calibration_bins(probabilities, labels, num_bins)
    total = len(labels)

    error = 0.0
    for single_bin in bins:
        weight = single_bin["count"] / total
        error += weight * abs(single_bin["accuracy"] - single_bin["mean_confidence"])

    return float(error)


def compute_maximum_calibration_error(probabilities, labels, num_bins=15):
    """
    The worst single bin's gap, ignoring how many samples it holds.

    Worth reporting alongside ECE because ECE can hide a severe local problem.
    If one confidence band is badly wrong but holds few samples, ECE barely
    moves - but if that band is the one I plan to automate, it matters a lot.
    """
    bins = compute_calibration_bins(probabilities, labels, num_bins)
    if not bins:
        return 0.0
    return float(max(abs(b["accuracy"] - b["mean_confidence"]) for b in bins))


def fit_temperature(validation_probabilities, validation_labels):
    """
    Fit a single temperature that recalibrates the model's confidence.

    How it works. Dividing the logits by a temperature T before the softmax
    makes the output distribution flatter (T > 1) or sharper (T < 1). One scalar
    tuned on held-out data often fixes most of a network's overconfidence, and
    it changes no predictions at all - argmax is unaffected - so accuracy stays
    identical while confidence becomes more honest.

    Why it must be fitted on VALIDATION, not test. The temperature is a fitted
    parameter. Fitting it on test data and then reporting test calibration would
    be tuning on the test set, which is exactly the kind of leakage the rest of
    this project is careful to avoid.

    I only have probabilities saved, not logits, so I recover the logits by
    taking a log. For two classes that is exact up to an additive constant,
    which the softmax ignores, so the fit is unaffected.

    I search a grid rather than running gradient descent. One parameter over a
    bounded range does not need an optimiser, and a grid is easier to explain
    and impossible to get subtly wrong.
    """
    # Recover logits up to an irrelevant constant.
    safe_probabilities = np.clip(validation_probabilities, 1e-12, 1.0)
    logits = np.log(safe_probabilities)

    best_temperature = 1.0
    best_loss = float("inf")

    for temperature in np.arange(0.5, 5.01, 0.01):
        scaled = logits / temperature

        # Softmax, computed with the max subtracted for numerical stability.
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)

        # Negative log likelihood of the true labels.
        true_class_probabilities = probabilities[np.arange(len(validation_labels)), validation_labels]
        loss = float(-np.log(np.clip(true_class_probabilities, 1e-12, 1.0)).mean())

        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)

    return best_temperature, best_loss


def apply_temperature(probabilities, temperature):
    """Recalibrate probabilities with an already-fitted temperature."""
    safe_probabilities = np.clip(probabilities, 1e-12, 1.0)
    logits = np.log(safe_probabilities) / temperature

    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Reject option
# ---------------------------------------------------------------------------

def sweep_reject_option(probabilities, labels, thresholds=None):
    """
    Work out what happens as I vary the confidence threshold for automation.

    The rule: if the model's confidence is at or above the threshold, accept its
    answer automatically. If it is below, refer the case to a human specialist.

    For each threshold I record:

      referral_rate        - fraction sent to a human. This is the cost. A
                             system referring 80% of cases saves nobody any time.
      retained_accuracy    - accuracy on the cases handled automatically. This
                             is the headline benefit.
      auto_cleared_false_negatives
                           - the number of Alzheimer's patients the model
                             confidently called normal and nobody reviewed.

    That last one is the number that should actually drive the decision, and it
    is the reason overall accuracy is the wrong metric here. The three outcomes
    do not cost the same:

      - A false positive that got referred costs a specialist a few minutes.
      - A false positive that was auto-flagged costs unnecessary follow-up and
        patient worry.
      - A false negative that was auto-cleared sends someone home undiagnosed,
        and nobody ever checks.

    Accuracy treats all of those identically. The operating point should instead
    minimise auto-cleared false negatives subject to a referral rate the clinic
    can actually absorb.
    """
    if thresholds is None:
        thresholds = np.arange(0.50, 1.000, 0.005)

    predicted_labels = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    results = []

    for threshold in thresholds:
        automated = confidences >= threshold
        referred = ~automated

        automated_count = int(automated.sum())
        referral_rate = float(referred.mean())

        if automated_count == 0:
            # Everything referred. Retained accuracy is undefined, not 1.0, and
            # recording it as 1.0 would make the plot lie at the right-hand end.
            results.append({
                "threshold": float(threshold),
                "referral_rate": referral_rate,
                "automated_count": 0,
                "retained_accuracy": None,
                "auto_cleared_false_negatives": 0,
                "auto_flagged_false_positives": 0,
            })
            continue

        correct_and_automated = (predicted_labels == labels) & automated
        retained_accuracy = float(correct_and_automated.sum() / automated_count)

        # Auto-cleared false negative: truly AD, predicted NC, confidence high
        # enough that no human ever looked at it.
        auto_cleared_false_negatives = int(
            ((labels == AD_LABEL) & (predicted_labels == NC_LABEL) & automated).sum()
        )

        # Auto-flagged false positive: truly NC, predicted AD, automated.
        auto_flagged_false_positives = int(
            ((labels == NC_LABEL) & (predicted_labels == AD_LABEL) & automated).sum()
        )

        results.append({
            "threshold": float(threshold),
            "referral_rate": referral_rate,
            "automated_count": automated_count,
            "retained_accuracy": retained_accuracy,
            "auto_cleared_false_negatives": auto_cleared_false_negatives,
            "auto_flagged_false_positives": auto_flagged_false_positives,
        })

    return results


def recommend_operating_point(sweep, target_retained_accuracy=0.95,
                              maximum_referral_rate=0.30):
    """
    Pick a threshold to recommend to the project manager.

    The rule I apply: among thresholds that reach the target accuracy on the
    automated cases AND stay within the referral budget, choose the one that
    refers the fewest cases. If nothing satisfies both, say so plainly rather
    than quietly returning the least bad option - a system that cannot meet its
    own acceptance criteria is a finding, not a failure to be hidden.

    The two numbers are deliberately arguments rather than constants, because
    they are a policy choice made by the clinic, not a property of the model. In
    the report I should state what I assumed and why.
    """
    feasible = [
        row for row in sweep
        if row["retained_accuracy"] is not None
        and row["retained_accuracy"] >= target_retained_accuracy
        and row["referral_rate"] <= maximum_referral_rate
    ]

    if not feasible:
        return {
            "feasible": False,
            "target_retained_accuracy": target_retained_accuracy,
            "maximum_referral_rate": maximum_referral_rate,
            "note": (
                "No threshold reaches %.2f accuracy on automated cases while "
                "referring no more than %.0f%% of cases."
                % (target_retained_accuracy, 100 * maximum_referral_rate)
            ),
        }

    best = min(feasible, key=lambda row: row["referral_rate"])
    return {
        "feasible": True,
        "target_retained_accuracy": target_retained_accuracy,
        "maximum_referral_rate": maximum_referral_rate,
        **best,
    }


# ---------------------------------------------------------------------------
# Patient level
# ---------------------------------------------------------------------------

def aggregate_to_patients(probabilities, labels, patient_ids):
    """
    Combine each patient's 20 slice predictions into one decision per patient.

    Why this belongs in the report. A clinician does not get told "slice 107 of
    this scan looks like Alzheimer's" - they get told something about the
    patient. So the clinically meaningful unit of evaluation is the patient, not
    the slice, and every number computed per slice is answering a slightly
    different question from the one that matters.

    I average the probabilities across a patient's slices rather than taking a
    majority vote, because averaging keeps the confidence information that the
    whole reject-option analysis depends on. A majority vote would throw away
    exactly the signal I am trying to study.

    I also expect patient-level accuracy to come out higher than slice-level,
    because averaging twenty noisy predictions cancels some of the noise. If it
    does not, that is worth investigating - it would suggest the errors are
    systematic for particular patients rather than random across slices, which
    is a completely different failure mode.
    """
    unique_patients = sorted(set(patient_ids.tolist()))

    patient_probabilities = []
    patient_labels = []
    slice_counts = []

    for patient in unique_patients:
        mask = patient_ids == patient

        patient_probabilities.append(probabilities[mask].mean(axis=0))

        # Every slice of one patient carries the same label, so the first is
        # enough - but check, because a mismatch would mean the data is wrong.
        patient_label_set = set(labels[mask].tolist())
        assert len(patient_label_set) == 1, (
            "patient %s has more than one label: %s" % (patient, patient_label_set)
        )
        patient_labels.append(labels[mask][0])
        slice_counts.append(int(mask.sum()))

    return (
        np.array(patient_probabilities),
        np.array(patient_labels),
        np.array(unique_patients),
        np.array(slice_counts),
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_reliability_diagrams(model_results, output_dir, num_bins=15):
    """
    Reliability diagram: mean confidence on the x axis, actual accuracy on the y.

    A perfectly calibrated model sits on the diagonal. Points below the diagonal
    mean overconfidence - the model claims more certainty than it earns, which
    is the failure mode the assignment asks about.
    """
    model_names = list(model_results.keys())
    figure, axes = plt.subplots(1, len(model_names),
                                figsize=(5 * len(model_names), 4.6), squeeze=False)

    for column, name in enumerate(model_names):
        result = model_results[name]
        axis = axes[0][column]

        bins = result["calibration_bins"]
        confidences = [b["mean_confidence"] for b in bins]
        accuracies = [b["accuracy"] for b in bins]
        counts = [b["count"] for b in bins]

        axis.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1,
                  label="perfect calibration")

        # Marker area scaled by how many predictions the bin holds, so a bin
        # with three samples is visibly less important than one with three
        # thousand.
        total = sum(counts)
        sizes = [20 + 400 * (c / total) for c in counts]
        axis.scatter(confidences, accuracies, s=sizes, alpha=0.7,
                     color="tab:blue", edgecolors="black", linewidths=0.5,
                     label="bins (area = count)")
        axis.plot(confidences, accuracies, color="tab:blue", alpha=0.4, linewidth=1)

        axis.set_xlim(0.45, 1.02)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("mean confidence")
        axis.set_ylabel("accuracy")
        axis.set_title("%s\nECE %.4f   max gap %.4f"
                       % (name, result["ece"], result["max_calibration_error"]))
        axis.legend(fontsize=8, loc="lower right")
        axis.grid(alpha=0.3)

    figure.suptitle("Reliability: does stated confidence match actual accuracy?",
                    fontsize=12)
    figure.tight_layout()

    path = os.path.join(output_dir, "calibration_reliability.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("wrote %s" % path)


def plot_confidence_histograms(model_results, output_dir):
    """
    Confidence distribution split by whether the prediction was right or wrong.

    What I want to see: wrong predictions concentrated at low confidence. That
    would mean the model knows when it is struggling, and a threshold can
    separate the two populations.

    What would be bad: the two distributions overlapping heavily at high
    confidence. That means the model is confidently wrong, no threshold can
    separate the cases, and the reject-option idea does not work for this model.
    """
    model_names = list(model_results.keys())
    figure, axes = plt.subplots(1, len(model_names),
                                figsize=(5 * len(model_names), 4.2), squeeze=False)

    for column, name in enumerate(model_names):
        result = model_results[name]
        axis = axes[0][column]

        axis.hist(result["correct_confidences"], bins=25, range=(0.5, 1.0),
                  alpha=0.6, label="correct (n=%d)" % len(result["correct_confidences"]),
                  color="tab:green", density=True)
        axis.hist(result["wrong_confidences"], bins=25, range=(0.5, 1.0),
                  alpha=0.6, label="wrong (n=%d)" % len(result["wrong_confidences"]),
                  color="tab:red", density=True)

        axis.set_xlabel("confidence in predicted class")
        axis.set_ylabel("density")
        axis.set_title(name)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)

    figure.suptitle("Is the model confidently wrong, or does it know when it is unsure?",
                    fontsize=12)
    figure.tight_layout()

    path = os.path.join(output_dir, "calibration_confidence.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("wrote %s" % path)


def plot_reject_option_curves(model_results, output_dir):
    """
    Three panels showing the cost and benefit of referring uncertain cases.

    Left: accuracy on the automated cases against the referral rate. This is the
    curve the recommendation comes from - how much human review buys how much
    reliability.

    Middle: auto-cleared false negatives against the referral rate. The harm
    curve, and the one that should actually set the operating point.

    Right: how many cases get automated at each threshold, for context.
    """
    figure, (accuracy_axis, harm_axis, volume_axis) = plt.subplots(1, 3, figsize=(15, 4.4))

    for name, result in model_results.items():
        sweep = result["reject_sweep"]

        usable = [row for row in sweep if row["retained_accuracy"] is not None]
        referral_rates = [row["referral_rate"] for row in usable]
        accuracies = [row["retained_accuracy"] for row in usable]
        false_negatives = [row["auto_cleared_false_negatives"] for row in usable]
        thresholds = [row["threshold"] for row in usable]

        accuracy_axis.plot(referral_rates, accuracies, label=name, linewidth=1.6)
        harm_axis.plot(referral_rates, false_negatives, label=name, linewidth=1.6)
        volume_axis.plot(thresholds, [row["automated_count"] for row in usable],
                         label=name, linewidth=1.6)

    accuracy_axis.axhline(0.95, color="black", linestyle="--", linewidth=1,
                          label="0.95 target")
    accuracy_axis.set_xlabel("fraction referred to a human")
    accuracy_axis.set_ylabel("accuracy on automated cases")
    accuracy_axis.set_title("Benefit: reliability bought by review")
    accuracy_axis.legend(fontsize=8)
    accuracy_axis.grid(alpha=0.3)

    harm_axis.set_xlabel("fraction referred to a human")
    harm_axis.set_ylabel("auto-cleared false negatives")
    harm_axis.set_title("Harm: missed AD cases nobody reviewed")
    harm_axis.legend(fontsize=8)
    harm_axis.grid(alpha=0.3)

    volume_axis.set_xlabel("confidence threshold")
    volume_axis.set_ylabel("cases handled automatically")
    volume_axis.set_title("Volume automated")
    volume_axis.legend(fontsize=8)
    volume_axis.grid(alpha=0.3)

    figure.suptitle("Reject option: what human review buys, and what it prevents",
                    fontsize=12)
    figure.tight_layout()

    path = os.path.join(output_dir, "reject_option_curves.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("wrote %s" % path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyse_one_model(name, predictions, validation_predictions=None, num_bins=15):
    """Run every analysis for a single model and return a results dictionary."""
    probabilities = predictions["probabilities"]
    labels = predictions["labels"]
    patient_ids = predictions["patient_ids"]

    predicted_labels = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    was_correct = predicted_labels == labels

    result = {
        "name": name,
        "slice_level": {
            "count": int(len(labels)),
            "accuracy": float(was_correct.mean()),
        },
        "ece": compute_expected_calibration_error(probabilities, labels, num_bins),
        "max_calibration_error": compute_maximum_calibration_error(probabilities, labels, num_bins),
        "calibration_bins": compute_calibration_bins(probabilities, labels, num_bins),
        "mean_confidence_when_correct": float(confidences[was_correct].mean()),
        "mean_confidence_when_wrong": float(confidences[~was_correct].mean()) if (~was_correct).any() else None,
        "correct_confidences": confidences[was_correct],
        "wrong_confidences": confidences[~was_correct],
        "reject_sweep": sweep_reject_option(probabilities, labels),
    }

    result["recommended_operating_point"] = recommend_operating_point(result["reject_sweep"])

    # Patient level.
    (patient_probabilities, patient_labels,
     unique_patients, slice_counts) = aggregate_to_patients(probabilities, labels, patient_ids)

    patient_predicted = patient_probabilities.argmax(axis=1)
    result["patient_level"] = {
        "count": int(len(patient_labels)),
        "slices_per_patient_min": int(slice_counts.min()),
        "slices_per_patient_max": int(slice_counts.max()),
        "accuracy": float((patient_predicted == patient_labels).mean()),
        "ece": compute_expected_calibration_error(patient_probabilities, patient_labels, num_bins=10),
    }

    # Temperature scaling, only if validation predictions were supplied.
    if validation_predictions is not None:
        temperature, validation_loss = fit_temperature(
            validation_predictions["probabilities"],
            validation_predictions["labels"],
        )
        recalibrated = apply_temperature(probabilities, temperature)

        result["temperature_scaling"] = {
            "temperature": temperature,
            "validation_nll": validation_loss,
            "ece_before": result["ece"],
            "ece_after": compute_expected_calibration_error(recalibrated, labels, num_bins),
            "accuracy_before": float(was_correct.mean()),
            "accuracy_after": float((recalibrated.argmax(axis=1) == labels).mean()),
        }
    else:
        result["temperature_scaling"] = None

    return result


def print_model_summary(result):
    """Readable summary to the terminal, so I can see the findings immediately."""
    name = result["name"]
    print()
    print("=" * 72)
    print("MODEL: %s" % name)
    print("=" * 72)

    print("Slice level:   %d slices, accuracy %.4f"
          % (result["slice_level"]["count"], result["slice_level"]["accuracy"]))
    print("Patient level: %d patients, accuracy %.4f  (%d-%d slices each)"
          % (result["patient_level"]["count"], result["patient_level"]["accuracy"],
             result["patient_level"]["slices_per_patient_min"],
             result["patient_level"]["slices_per_patient_max"]))

    print()
    print("Calibration:")
    print("  expected calibration error: %.4f" % result["ece"])
    print("  worst single bin gap:       %.4f" % result["max_calibration_error"])
    print("  mean confidence when right: %.4f" % result["mean_confidence_when_correct"])
    if result["mean_confidence_when_wrong"] is not None:
        print("  mean confidence when wrong: %.4f" % result["mean_confidence_when_wrong"])
        separation = result["mean_confidence_when_correct"] - result["mean_confidence_when_wrong"]
        print("  separation:                 %.4f" % separation)
        if separation < 0.02:
            print("    -> barely any separation. The model is about as confident when")
            print("       wrong as when right, so a confidence threshold cannot")
            print("       reliably pick out the cases a human should see.")

    if result["temperature_scaling"]:
        scaling = result["temperature_scaling"]
        print()
        print("Temperature scaling (fitted on validation):")
        print("  temperature: %.2f %s" % (
            scaling["temperature"],
            "(>1 means it was overconfident)" if scaling["temperature"] > 1 else "(<1 means it was underconfident)",
        ))
        print("  ECE %.4f -> %.4f" % (scaling["ece_before"], scaling["ece_after"]))
        print("  accuracy %.4f -> %.4f (should be unchanged)"
              % (scaling["accuracy_before"], scaling["accuracy_after"]))

    print()
    print("Recommended operating point:")
    point = result["recommended_operating_point"]
    if point["feasible"]:
        print("  threshold %.3f" % point["threshold"])
        print("  refers %.1f%% of cases to a human" % (100 * point["referral_rate"]))
        print("  accuracy on the %d automated cases: %.4f"
              % (point["automated_count"], point["retained_accuracy"]))
        print("  auto-cleared false negatives (missed AD, unreviewed): %d"
              % point["auto_cleared_false_negatives"])
        print("  auto-flagged false positives: %d" % point["auto_flagged_false_positives"])
    else:
        print("  NOT ACHIEVABLE. %s" % point["note"])


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calibration and reject-option analysis from saved predictions."
    )
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="One or more test_predictions_*.npz files.")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Labels for each predictions file. Defaults to the filenames.")
    parser.add_argument("--val-predictions", nargs="+", default=None,
                        help="Matching validation prediction files, for temperature scaling.")
    parser.add_argument("--output-dir", default="figures",
                        help="Where to write figures and the results JSON.")
    parser.add_argument("--num-bins", type=int, default=15,
                        help="Bins for the calibration computation.")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    os.makedirs(arguments.output_dir, exist_ok=True)

    names = arguments.names
    if names is None:
        names = [
            os.path.basename(p).replace("test_predictions_", "").replace(".npz", "")
            for p in arguments.predictions
        ]

    if len(names) != len(arguments.predictions):
        raise ValueError("--names must have the same number of entries as --predictions")

    validation_paths = arguments.val_predictions or [None] * len(arguments.predictions)
    if len(validation_paths) != len(arguments.predictions):
        raise ValueError("--val-predictions must match --predictions in length")

    model_results = {}

    for name, predictions_path, validation_path in zip(names, arguments.predictions, validation_paths):
        print("Loading %s from %s" % (name, predictions_path))
        predictions = load_predictions(predictions_path)

        validation_predictions = None
        if validation_path:
            print("  with validation predictions from %s" % validation_path)
            validation_predictions = load_predictions(validation_path)

        model_results[name] = analyse_one_model(
            name, predictions, validation_predictions, arguments.num_bins
        )

    for name in names:
        print_model_summary(model_results[name])

    # Figures.
    print()
    plot_reliability_diagrams(model_results, arguments.output_dir, arguments.num_bins)
    plot_confidence_histograms(model_results, arguments.output_dir)
    plot_reject_option_curves(model_results, arguments.output_dir)

    # JSON, with the raw confidence arrays stripped out since they are large and
    # already summarised by the figures.
    serialisable = {}
    for name, result in model_results.items():
        trimmed = {k: v for k, v in result.items()
                   if k not in ("correct_confidences", "wrong_confidences")}
        serialisable[name] = trimmed

    json_path = os.path.join(arguments.output_dir, "calibration_results.json")
    with open(json_path, "w") as handle:
        json.dump(serialisable, handle, indent=2)
    print("wrote %s" % json_path)


if __name__ == "__main__":
    main()
