"""Confidence, risk-coverage and patient-level summaries of fixed predictions.

ECE uses the predicted-class confidence max(p, 1 - p) in equal-width bins.
The Brier score uses the AD probability. A referral threshold is only applied
when one has been fitted on the calibration patients.
"""

import math

from evaluation.metrics import binary_metrics


def confidence_metrics(labels: list[int], probabilities: list[float], bins: int = 15,
                       reject_threshold: float | None = None) -> dict:
    """Reliability bins, ECE, Brier score and the empirical risk-coverage curve."""
    observations = [(max(p, 1 - p), int(int(p >= 0.5) == y)) for y, p in zip(labels, probabilities)]
    buckets = [[] for _ in range(bins)]
    for confidence, correct in observations:
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, correct))
    reliability, histogram, ece = [], [], 0.0
    for index, bucket in enumerate(buckets):
        mean = math.fsum(c for c, _ in bucket) / len(bucket) if bucket else None
        accuracy = sum(ok for _, ok in bucket) / len(bucket) if bucket else None
        if bucket:
            ece += len(bucket) / len(observations) * abs(mean - accuracy)
        reliability.append({"bin": index, "lower": index / bins, "upper": (index + 1) / bins,
                            "count": len(bucket), "mean_confidence": mean, "accuracy": accuracy})
        histogram.append({"bin": index, "correct": sum(ok for _, ok in bucket),
                          "incorrect": sum(1 - ok for _, ok in bucket)})

    # Accept cases from most to least confident; tied confidences enter together.
    curve = [{"confidence_threshold": None, "accepted": 0, "coverage": 0.0,
              "accepted_accuracy": None, "risk": None}]
    groups = {}
    for confidence, correct in observations:
        groups.setdefault(confidence, []).append(correct)
    accepted = correct_total = 0
    for confidence, group in sorted(groups.items(), reverse=True):
        accepted += len(group)
        correct_total += sum(group)
        curve.append({"confidence_threshold": confidence, "accepted": accepted,
                      "coverage": accepted / len(observations),
                      "accepted_accuracy": correct_total / accepted,
                      "risk": 1 - correct_total / accepted})

    result = {"bins": bins, "ece_predicted_class": ece,
              "brier_ad": math.fsum((p - y) ** 2 for y, p in zip(labels, probabilities)) / len(labels),
              "reliability_bins": reliability, "confidence_histogram": histogram, "risk_coverage": curve}
    if reject_threshold is not None:
        kept = [ok for confidence, ok in observations if confidence >= reject_threshold]
        result["referral"] = {"confidence_threshold": reject_threshold, "accepted": len(kept),
                              "referred": len(observations) - len(kept),
                              "coverage": len(kept) / len(observations),
                              "accepted_accuracy": sum(kept) / len(kept) if kept else None}
    return result


def aggregate_patients(slices: list[dict]) -> list[dict]:
    """Average all slice AD probabilities of each patient (all of their scans).

    In the course data every patient has one diagnosis. Should a patient's
    scans carry different diagnoses, each diagnosis forms its own record, so
    no scan is ever relabelled.
    """
    groups = {}
    for row in slices:
        groups.setdefault((row["patient_id"], row["label"]), []).append(row)
    patients = []
    for (patient, label), rows in sorted(groups.items()):
        probability = math.fsum(r["probability"] for r in rows) / len(rows)
        patients.append({"patient_id": patient, "label": label, "probability": probability,
                         "prediction": int(probability >= 0.5), "num_slices": len(rows),
                         "num_scans": len({r["image_id"] for r in rows})})
    return patients


def prediction_report(slices: list[dict], scans: list[dict], bins: int = 15,
                      reject_threshold: float | None = None) -> tuple[dict, list[dict]]:
    """Patient metrics plus confidence summaries for the slice, scan and patient units."""
    patients = aggregate_patients(slices)
    units = {"slice": slices, "scan": scans, "patient": patients}
    confidence = {unit: confidence_metrics([r["label"] for r in rows], [r["probability"] for r in rows],
                                           bins, reject_threshold if unit == "patient" else None)
                  for unit, rows in units.items()}
    report = {"confidence": confidence,
              "patient_metrics": binary_metrics([r["label"] for r in patients],
                                                [r["probability"] for r in patients]),
              "accuracy_target": {"minimum_accuracy": 0.8, "units": ["slice", "scan", "patient"]}}
    return report, patients


def acquisition_breakdown(scans: list[dict], patients: list[dict], years: dict[str, int],
                          shared_start: int) -> dict | None:
    """Scan and patient metrics split by acquisition period (docs/ACQUISITION_CONFOUND.md).

    ``years`` maps image ID to acquisition year. A patient belongs to a period
    only if all of their scans do. When patients carry a ``decision`` (AD, NC
    or REFER) the referral counts and the accuracy of automated decisions are
    reported too. Returns None when no scan has a known year.
    """
    if not any(scan["image_id"] in years for scan in scans):
        return None
    period = {scan["image_id"]: ("unknown" if scan["image_id"] not in years else
                                 f"from_{shared_start}" if years[scan["image_id"]] >= shared_start
                                 else f"before_{shared_start}") for scan in scans}
    patient_periods = {}
    for scan in scans:
        patient_periods.setdefault((scan["patient_id"], scan["label"]), set()).add(period[scan["image_id"]])
    result = {}
    for name in sorted(set(period.values())):
        group_scans = [s for s in scans if period[s["image_id"]] == name]
        group_patients = [p for p in patients if patient_periods.get((p["patient_id"], p["label"])) == {name}]
        entry = {"scans": binary_metrics([s["label"] for s in group_scans], [s["probability"] for s in group_scans]),
                 "scan_counts": {"AD": sum(s["label"] == 1 for s in group_scans),
                                 "NC": sum(s["label"] == 0 for s in group_scans)}}
        if group_patients and len({p["label"] for p in group_patients}) == 2:
            entry["patients"] = binary_metrics([p["label"] for p in group_patients],
                                               [p["probability"] for p in group_patients])
        if group_patients and "decision" in group_patients[0]:
            automated = [p for p in group_patients if p["decision"] != "REFER"]
            entry["referral"] = {
                "patients": len(group_patients), "referred": len(group_patients) - len(automated),
                "automated_accuracy": (sum((p["decision"] == "AD") == (p["label"] == 1) for p in automated)
                                       / len(automated)) if automated else None,
                "missed_ad_automated": sum(p["decision"] == "NC" and p["label"] == 1 for p in automated)}
        result[name] = entry
    return result
