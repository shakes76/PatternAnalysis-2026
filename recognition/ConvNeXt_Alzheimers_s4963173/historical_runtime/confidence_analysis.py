"""Patient-level confidence analysis for the COMP3710 research prototype.

Uses cached validation predictions, not MRI images. Temperature scaling is applied
AFTER slice -> scan -> patient aggregation. It is not a re-trained classifier.
Cross-fitting mitigates calibration/policy fitting optimism but cannot undo the
prior use of this same validation cohort to select model checkpoints.

Code/draft assistance: ChatGPT. Review, test and disclose actual assistance.
Reference for temperature scaling: Guo et al., On Calibration of Modern Neural
Networks, ICML 2017, https://arxiv.org/abs/1706.04599 . The bounds, folds and
referral constraints below are project choices, not clinical standards.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

TEMPERATURE_BOUNDS = (0.5, 10.0)
N_FOLDS = 5
FOLD_SEED = 42
N_BINS = 5
CONFIDENCE_GRID = tuple([round(0.50 + 0.05 * i, 2) for i in range(10)] + [0.975, 0.99])
MIN_ACCEPTED = 20
MIN_COVERAGE = 0.25
MAX_EMPIRICAL_ERROR = 0.10
MAX_AD_AUTOCLEARED_FRACTION = 0.10
CAUTION = (
    "Development evidence only. The checkpoints were already selected using these "
    "validation patients. Cross-fitting only separates fitting of the temperature "
    "and referral rule from their diagnostic evaluation; it does not make the "
    "underlying model-selection process independent. Empirical referral constraints "
    "are not clinically validated safety guarantees. Final-test performance is unknown."
)


class ConfidenceError(ValueError):
    """Input error; messages deliberately omit individual patient identifiers."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfidenceError(message)


def arrays(labels: Sequence[int], probabilities: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels)
    try:
        p = np.asarray(probabilities, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ConfidenceError("Probabilities must be numeric.") from exc
    require(y.ndim == p.ndim == 1 and y.shape == p.shape and len(y) > 0,
            "Labels and probabilities must be nonempty, equal-length vectors.")
    require(np.isin(y, (0, 1)).all(), "Labels must be NC=0 or AD=1.")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
            "Probabilities must be finite and within [0,1].")
    return y.astype(np.int64), p


def log_odds(probabilities: Sequence[float]) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    require(p.ndim == 1 and p.size > 0 and np.isfinite(p).all()
            and ((p >= 0) & (p <= 1)).all(), "Invalid probability vector.")
    # Same input clipping convention as the preceding experiment's log loss.
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))
    ex = np.exp(z[~positive])
    result[~positive] = ex / (1 + ex)
    return result


def scaled_probabilities(probabilities: Sequence[float], temperature: float) -> np.ndarray:
    require(math.isfinite(temperature) and temperature > 0, "Temperature must be positive and finite.")
    return sigmoid(log_odds(probabilities) / temperature)


def fit_temperature(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    """Deterministic bounded NLL minimization, convex in inverse temperature.

    Bisection finds a zero of mean(z*(sigmoid(beta*z)-y)); its derivative is
    nonnegative. Endpoints and beta=1 are also compared explicitly. No new
    dependencies, model training, intercept fitting or diagnostic threshold tuning.
    """
    y, p = arrays(labels, probabilities)
    require(len(np.unique(y)) == 2, "Temperature fitting requires both diagnostic classes.")
    z = log_odds(p)
    lo, hi = 1 / TEMPERATURE_BOUNDS[1], 1 / TEMPERATURE_BOUNDS[0]
    left, right = lo, hi
    for _ in range(100):
        middle = (left + right) / 2
        derivative = float(np.mean(z * (sigmoid(middle * z) - y)))
        if derivative > 0:
            right = middle
        else:
            left = middle
    def nll(beta: float) -> float:
        return float(np.mean(np.logaddexp(0, beta * z) - y * beta * z))
    candidates = [1.0, lo, hi, (left + right) / 2]
    best_loss = min(nll(beta) for beta in candidates)
    # Prefer identity in a flat numerical objective, then the smallest change.
    eligible = [beta for beta in candidates if nll(beta) <= best_loss + 1e-12]
    beta = min(eligible, key=lambda b: abs(math.log(b)))
    temperature = 1 / beta
    return {
        "temperature": temperature, "n_fit_patients": len(y),
        "temperature_bounds": list(TEMPERATURE_BOUNDS),
        "at_bound": bool(np.isclose(temperature, TEMPERATURE_BOUNDS).any()),
        "raw_fit_nll": nll(1.0), "temperature_fit_nll": nll(beta),
        "method": "bounded patient log-odds scaling, deterministic NLL minimization",
    }


def probability_scores(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    y, p = arrays(labels, probabilities)
    bounded = np.clip(p, 1e-15, 1 - 1e-15)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for i in range(N_BINS):
        low, high = i / N_BINS, (i + 1) / N_BINS
        member = (p >= low) & ((p < high) if i < N_BINS - 1 else (p <= high))
        count = int(member.sum())
        avg = float(p[member].mean()) if count else None
        rate = float(y[member].mean()) if count else None
        if count:
            ece += count / len(y) * abs(avg - rate)
        rows.append({"lower": low, "upper": high, "n": count,
                     "mean_predicted_ad_probability": avg, "observed_ad_fraction": rate})
    return {"n": len(y),
            "nll": float(-np.mean(y * np.log(bounded) + (1 - y) * np.log1p(-bounded))),
            "brier": float(np.mean((p - y) ** 2)),
            "ece_ad_probability_5_equal_width_bins": float(ece),
            "reliability_bins": rows,
            "note": "Probability calibration ECE for p(AD), not top-label confidence ECE; small-sample/bin sensitive."}


def wilson_interval(successes: int, total: int) -> dict[str, Any]:
    require(isinstance(total, (int, np.integer)) and 0 <= successes <= total,
            "Invalid binomial counts.")
    if total == 0:
        return {"count": 0, "denominator": 0, "estimate": None, "lower": None, "upper": None}
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return {"count": int(successes), "denominator": int(total), "estimate": p,
            "lower": max(0.0, center - half), "upper": min(1.0, center + half)}


def confidence_diagnostics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    y, p = arrays(labels, probabilities)
    prediction = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1 - p)
    error = prediction != y
    def mean_or_none(mask: np.ndarray) -> float | None:
        return float(confidence[mask].mean()) if mask.any() else None
    return {
        "n_correct": int((~error).sum()), "n_wrong": int(error.sum()),
        "mean_confidence_correct": mean_or_none(~error),
        "mean_confidence_wrong": mean_or_none(error),
        "high_confidence_threshold": 0.90,
        "n_high_confidence": int((confidence >= 0.90).sum()),
        "n_wrong_high_confidence": int((error & (confidence >= 0.90)).sum()),
        "error_detection_auroc": float(roc_auc_score(error.astype(int), 1 - confidence))
            if len(np.unique(error)) == 2 else None,
        "error_detection_note": "Error is positive class; larger 1-confidence means more likely wrong. Chance ranking is 0.5."}


def triage_metrics(labels: Sequence[int], probabilities: Sequence[float],
                   accepted: Sequence[bool]) -> dict[str, Any]:
    y, p = arrays(labels, probabilities)
    a = np.asarray(accepted)
    require(a.shape == y.shape and a.dtype == np.bool_, "Acceptance mask must be boolean and match patients.")
    prediction = (p >= 0.5).astype(int)
    wrong = prediction != y
    n_ad = int((y == 1).sum())
    tn = int((a & (y == 0) & (prediction == 0)).sum())
    fp = int((a & (y == 0) & (prediction == 1)).sum())
    fn = int((a & (y == 1) & (prediction == 0)).sum())
    tp = int((a & (y == 1) & (prediction == 1)).sum())
    n = int(a.sum())
    error_interval = wilson_interval(int((wrong & a).sum()), n)
    return {"n_total": len(y), "n_accepted": n, "n_referred": int((~a).sum()),
            "coverage": n / len(y), "referral_fraction": float((~a).mean()),
            "accepted_accuracy": 1 - error_interval["estimate"] if n else None,
            "accepted_error_rate": error_interval["estimate"],
            "accepted_error_wilson95": error_interval,
            "accepted_confusion_matrix_rows_true_columns_predicted_NC_AD": [[tn, fp], [fn, tp]],
            "n_ad_total": n_ad, "n_ad_referred": int(((~a) & (y == 1)).sum()),
            "n_ad_automatically_positive": tp, "n_ad_automatically_cleared_as_nc": fn,
            "ad_automatically_cleared_fraction_of_all_ad": fn / n_ad if n_ad else None,
            "ad_automatically_cleared_wilson95": wilson_interval(fn, n_ad),
            "automatic_ad_detection_fraction_of_all_ad": tp / n_ad if n_ad else None,
            "recall_among_accepted_ad": tp / (tp + fn) if tp + fn else None,
            "ad_not_automatically_cleared_fraction": 1 - fn / n_ad if n_ad else None,
            "interpretation": "Referred cases have NO automated diagnosis here. AD not automatically cleared is not system recall; no human reviewer performance is assumed.",
            "interval_note": "Descriptive 95% Wilson intervals; do not adjust for policy selection, model selection or cross-fit dependence."}


def apply_referral(probabilities: Sequence[float], threshold: float | None) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    require(p.ndim == 1 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
            "Invalid probabilities for referral.")
    if threshold is None:
        return np.zeros(len(p), dtype=bool)
    require(math.isfinite(threshold) and 0.5 <= threshold <= 1, "Invalid confidence cutoff.")
    return np.maximum(p, 1 - p) >= threshold


def select_referral(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    y, p = arrays(labels, probabilities)
    require(len(np.unique(y)) == 2, "Referral fitting requires both classes.")
    required_n = max(MIN_ACCEPTED, math.ceil(MIN_COVERAGE * len(y)))
    rows = []
    eligible = []
    for threshold in CONFIDENCE_GRID:
        metrics = triage_metrics(y, p, apply_referral(p, threshold))
        passes = (metrics["n_accepted"] >= required_n
                  and metrics["accepted_error_rate"] <= MAX_EMPIRICAL_ERROR + 1e-12
                  and metrics["ad_automatically_cleared_fraction_of_all_ad"] <= MAX_AD_AUTOCLEARED_FRACTION + 1e-12)
        rows.append({"threshold": threshold, "meets_empirical_constraints": bool(passes), **metrics})
        if passes:
            eligible.append((threshold, metrics))
    threshold, selected = min(eligible, key=lambda item: (-item[1]["n_accepted"], item[0])) if eligible else (
        None, triage_metrics(y, p, np.zeros(len(y), dtype=bool)))
    return {"confidence_threshold": threshold,
            "status": "CANDIDATE_AUTOMATION_RULE" if eligible else "REFER_ALL_NO_FEASIBLE_EMPIRICAL_RULE",
            "selection": "Maximum coverage satisfying all empirical constraints; smallest cutoff on a coverage tie. If none, refer everyone.",
            "constraints": {"minimum_accepted": required_n, "minimum_fraction": MIN_COVERAGE,
                "maximum_empirical_accepted_error": MAX_EMPIRICAL_ERROR,
                "maximum_empirical_fraction_of_all_ad_automatically_cleared": MAX_AD_AUTOCLEARED_FRACTION},
            "candidate_grid": list(CONFIDENCE_GRID), "development_fit_triage": selected,
            "candidate_metrics": rows}


def fold_assignments(subjects: Sequence[str], labels: Sequence[int]) -> np.ndarray:
    y = np.asarray(labels)
    require(len(subjects) == len(y) and len(set(subjects)) == len(subjects)
            and all(isinstance(s, str) and s for s in subjects), "Subject IDs must be nonempty and unique.")
    require(y.ndim == 1 and np.isin(y, (0, 1)).all(), "Invalid fold labels.")
    result = np.empty(len(y), dtype=int)
    for label in (0, 1):
        indices = [i for i in range(len(y)) if y[i] == label]
        require(len(indices) >= N_FOLDS, "At least five patients per class are required for cross-fitting.")
        indices.sort(key=lambda i: (hashlib.sha256(
            f"COMP3710-step9|seed={FOLD_SEED}|class={label}|{subjects[i]}".encode()).hexdigest(), subjects[i]))
        for rank, index in enumerate(indices):
            result[index] = rank % N_FOLDS
    return result


def crossfit(subjects: Sequence[str], labels: Sequence[int],
             probabilities: Sequence[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    y, p = arrays(labels, probabilities)
    folds = fold_assignments(subjects, y)
    oof = np.empty(len(y), dtype=float)
    accepted = np.zeros(len(y), dtype=bool)
    reports = []
    for fold in range(N_FOLDS):
        held = folds == fold
        fit = ~held
        temp = fit_temperature(y[fit], p[fit])
        policy = select_referral(y[fit], scaled_probabilities(p[fit], temp["temperature"]))
        oof[held] = scaled_probabilities(p[held], temp["temperature"])
        accepted[held] = apply_referral(oof[held], policy["confidence_threshold"])
        reports.append({"fold": fold, "fit_n": int(fit.sum()), "held_n": int(held.sum()),
            "fit_class_counts_nc_ad": [int((y[fit] == c).sum()) for c in (0, 1)],
            "held_class_counts_nc_ad": [int((y[held] == c).sum()) for c in (0, 1)],
            "temperature": temp["temperature"], "temperature_at_bound": temp["at_bound"],
            "confidence_threshold": policy["confidence_threshold"], "policy_status": policy["status"],
            "held_probability_scores": probability_scores(y[held], oof[held]),
            "held_triage": triage_metrics(y[held], oof[held], accepted[held])})
    require(np.array_equal(p >= 0.5, oof >= 0.5), "Temperature unexpectedly changed 0.5 class decisions.")
    summary = {"n_folds": N_FOLDS, "fold_seed": FOLD_SEED,
        "pooled_probability_scores": probability_scores(y, oof),
        "pooled_triage": triage_metrics(y, oof, accepted), "fold_results": reports,
        "ranking_note": "Different fold temperatures need not preserve pooled probability ordering; pooled cross-fit AUROC is deliberately not reported.",
        "caution": CAUTION}
    private = {"fold_index": folds.tolist(), "oof_probabilities": oof.tolist(), "oof_accepted": accepted.tolist()}
    return summary, private


def analyse_patients(subjects: Sequence[str], labels: Sequence[int],
                     probabilities: Sequence[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    y, p = arrays(labels, probabilities)
    # Validate IDs and folds before any fitting.
    fold_assignments(subjects, y)
    raw_for_calibration = scaled_probabilities(p, 1.0)
    temp = fit_temperature(y, p)
    calibrated = scaled_probabilities(p, temp["temperature"])
    require(np.array_equal(p >= 0.5, calibrated >= 0.5), "Temperature changed class decisions.")
    full_policy = select_referral(y, calibrated)
    cf, cf_private = crossfit(subjects, y, p)
    risk = []
    for threshold in sorted(set([0.5 + i / 100 for i in range(51)] + list(CONFIDENCE_GRID))):
        m = triage_metrics(y, p, apply_referral(p, threshold))
        risk.append({"threshold": threshold, "coverage": m["coverage"],
                     "n_accepted": m["n_accepted"], "accepted_error_rate": m["accepted_error_rate"]})
    summary = {
        "patient_n": len(y), "class_counts_nc_ad": [int((y == i).sum()) for i in (0, 1)],
        "diagnosis_threshold": 0.5,
        "all_patient_accuracy_at_0_5": float(((p >= 0.5) == y).mean()),
        "raw_patient_auroc": float(roc_auc_score(y, p)),
        "raw_probability_scores": probability_scores(y, raw_for_calibration),
        "raw_confidence_diagnostics": confidence_diagnostics(y, p),
        "full_validation_temperature_fit": temp,
        "full_validation_fit_probability_scores": probability_scores(y, calibrated),
        "full_validation_fit_confidence_diagnostics": confidence_diagnostics(y, calibrated),
        "crossfit_diagnostic": cf, "operating_rule": full_policy,
        "raw_risk_coverage_curve": risk,
        "fixed_temperature_note": "One positive temperature preserves class decisions at 0.5 and confidence ordering (apart from clipping/roundoff). It cannot improve the underlying risk-coverage ordering or create new diagnostic signal.",
        "caution": CAUTION}
    private = {"patients": [{"subject_id": s, "label": int(y[i]), "raw_p_ad": float(p[i]),
                "full_fit_calibrated_p_ad": float(calibrated[i]), "fold_index": cf_private["fold_index"][i],
                "oof_p_ad": cf_private["oof_probabilities"][i], "oof_accepted": cf_private["oof_accepted"][i],
                "full_fit_accepted": bool(apply_referral(calibrated, full_policy["confidence_threshold"])[i])}
                for i, s in enumerate(subjects)]}
    return summary, private
