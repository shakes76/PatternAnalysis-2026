"""Compute slice and scan classification metrics without external dependencies.

Labels are NC=0 and AD=1. A scan prediction is the arithmetic mean of its
slice-level AD probabilities. Scans from one longitudinal patient retain their
individual labels; this module never converts scan diagnoses to a patient label.
"""

import math


def _binary_label(value):
    """Reject label encodings that could silently invert the AD/NC mapping."""
    if type(value) is not int or value not in (0, 1):
        raise ValueError("Labels must be integers with NC=0 and AD=1.")
    return value


def _probability(value, name="Probability"):
    """Require a finite numeric probability rather than coercing text or NaN."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number between 0 and 1.")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1.")
    return value


def _identifier(value, name):
    """Keep identifiers nonempty and consistent across aggregation records."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def aggregate_scans(slice_predictions, expected_slices=20):
    """Average AD probabilities for each complete scan, sorted by image ID.

    Every scan must contain exactly ``expected_slices`` unique slice indices
    and paths, one patient ID, and one label. A repeated path anywhere in the
    input is rejected so one source image cannot contribute twice. The fixed
    scan decision threshold is 0.5, including a tie as AD.
    """
    if type(expected_slices) is not int or expected_slices <= 0:
        raise ValueError("expected_slices must be a positive integer.")
    scans = {}
    seen_paths = set()
    required = ("patient_id", "image_id", "slice_index", "relative_path", "label", "probability")
    for row_number, row in enumerate(slice_predictions, start=1):
        if not isinstance(row, dict) or any(key not in row for key in required):
            raise ValueError(f"Slice prediction {row_number} is missing required fields.")
        patient_id = _identifier(row["patient_id"], "patient_id")
        image_id = _identifier(row["image_id"], "image_id")
        path = _identifier(row["relative_path"], "relative_path")
        label = _binary_label(row["label"])
        probability = _probability(row["probability"])
        slice_index = row["slice_index"]
        if type(slice_index) is not int or slice_index < 0:
            raise ValueError("slice_index must be a nonnegative integer.")
        if path in seen_paths:
            raise ValueError(f"Repeated slice path: {path}")
        seen_paths.add(path)
        scan = scans.setdefault(image_id, {
            "patient_id": patient_id, "label": label, "indices": set(), "probabilities": [],
        })
        if scan["patient_id"] != patient_id or scan["label"] != label:
            raise ValueError(f"Inconsistent patient or label within scan {image_id}.")
        if slice_index in scan["indices"]:
            raise ValueError(f"Repeated slice index {slice_index} in scan {image_id}.")
        scan["indices"].add(slice_index)
        scan["probabilities"].append(probability)
    if not scans:
        raise ValueError("At least one slice prediction is required.")

    results = []
    for image_id, scan in sorted(scans.items()):
        count = len(scan["probabilities"])
        if count != expected_slices:
            raise ValueError(f"Scan {image_id} has {count} slices; expected {expected_slices}.")
        probability = math.fsum(scan["probabilities"]) / count
        results.append({
            "patient_id": scan["patient_id"], "image_id": image_id,
            "label": scan["label"], "probability": probability,
            "prediction": int(probability >= 0.5), "num_slices": count,
        })
    return results


def _auroc(labels, probabilities):
    """Use average ranks for tied scores, with O(n log n) sorting cost."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranked = sorted(zip(probabilities, labels))
    positive_rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        # This tied block occupies one-based ranks start + 1 through end.
        mean_rank = (start + 1 + end) / 2.0
        positive_rank_sum += mean_rank * sum(label for _, label in ranked[start:end])
        start = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_metrics(labels, probabilities, threshold=0.5):
    """Return JSON-compatible metrics for NC=0 and AD=1 probabilities.

    The confusion matrix has true labels on rows and predicted labels on
    columns, both ordered [NC, AD]. Undefined precision, recall, or F1 is zero.
    AUROC and balanced accuracy are ``None`` if either true class is absent.
    Log loss clamps endpoint probabilities to [1e-15, 1 - 1e-15] solely for
    numerical stability; decisions and AUROC use the original probabilities.
    """
    threshold = _probability(threshold, "Threshold")
    labels = [_binary_label(label) for label in labels]
    probabilities = [_probability(probability) for probability in probabilities]
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal, nonzero lengths.")
    confusion = [[0, 0], [0, 0]]
    losses = []
    for label, probability in zip(labels, probabilities):
        confusion[label][int(probability >= threshold)] += 1
        clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
        losses.append(-math.log(clipped) if label else -math.log1p(-clipped))

    per_class = {}
    for label, name in enumerate(("NC", "AD")):
        true_positive = confusion[label][label]
        support = sum(confusion[label])
        predicted = sum(row[label] for row in confusion)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    both_classes = all(values["support"] for values in per_class.values())
    return {
        "n_samples": len(labels),
        "accuracy": (confusion[0][0] + confusion[1][1]) / len(labels),
        "balanced_accuracy": (per_class["NC"]["recall"] + per_class["AD"]["recall"]) / 2
        if both_classes else None,
        "macro_f1": (per_class["NC"]["f1"] + per_class["AD"]["f1"]) / 2,
        "auroc": _auroc(labels, probabilities),
        "log_loss": math.fsum(losses) / len(losses),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
