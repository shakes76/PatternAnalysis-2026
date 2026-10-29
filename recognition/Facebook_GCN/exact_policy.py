"""Exact empirical class-threshold search and label-free audit routing.

Yuhan Liao - 49839991. These are empirical fitting rules, not risk guarantees.
"""
import math
import numpy as np
import torch

from dataset import CLASS_NAMES


def validate_probabilities(probabilities):
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES) or len(probabilities) == 0:
        raise ValueError("Nonempty four-class probabilities required")
    if not probabilities.is_floating_point() or not torch.isfinite(probabilities).all():
        raise ValueError("Finite floating probabilities required")
    if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
        raise ValueError("Probabilities must lie in [0, 1]")
    if not torch.allclose(probabilities.sum(1), torch.ones_like(probabilities.sum(1)), atol=1e-5, rtol=1e-5):
        raise ValueError("Probability rows must sum to one")


def fit_exact_class_thresholds(probabilities, labels, target=.98, minimum=50):
    """Search every distinct accepted set, keeping whole confidence-tie blocks."""
    validate_probabilities(probabilities)
    if not math.isfinite(target) or not 0 < target <= 1 or minimum < 1 or int(minimum) != minimum:
        raise ValueError("Valid accuracy target and positive integer minimum required")
    if labels.ndim != 1 or len(labels) != len(probabilities) or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("Aligned integer labels required")
    if bool((labels < 0).any()) or bool((labels >= len(CLASS_NAMES)).any()):
        raise ValueError("Labels outside four-class range")
    confidence, prediction = probabilities.detach().cpu().max(1)
    confidence, prediction, truth = confidence.numpy(), prediction.numpy(), labels.detach().cpu().numpy()
    thresholds, records = [], []
    for label, name in enumerate(CLASS_NAMES):
        group = prediction == label
        c, correct = confidence[group], (truth[group] == label)
        required = max(int(minimum), math.ceil(.05 * len(c)))
        threshold, count, accuracy = 1.01, 0, None
        if len(c):
            order = np.argsort(-c, kind="stable")
            scores = c[order]
            hits = np.cumsum(correct[order], dtype=np.int64)
            ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
            counts = ends + 1
            valid = (counts >= required) & (hits[ends] + 1e-12 >= target * counts)
            if valid.any():
                index = int(ends[np.flatnonzero(valid)[-1]])
                count = index + 1
                threshold = 0.0 if count == len(c) else float(scores[index])
                accuracy = float(hits[index] / count)
        thresholds.append(threshold)
        records.append({"predicted_class": name, "policy_cases": int(len(c)), "minimum_accepted": required,
                        "accepted": count, "threshold": threshold, "accepted_accuracy": accuracy,
                        "feasible_on_fitting_data": bool(count), "search": "distinct score boundaries, complete ties"})
    return thresholds, records


def exact_decisions(probabilities, thresholds):
    """Apply saved thresholds without reference labels or known-error IDs."""
    validate_probabilities(probabilities)
    if len(thresholds) != len(CLASS_NAMES) or any(not math.isfinite(t) or not 0 <= t <= 1.01 for t in thresholds):
        raise ValueError("Four finite thresholds in [0, 1.01] required")
    confidence, prediction = probabilities.max(1)
    return confidence >= probabilities.new_tensor(thresholds)[prediction]


def random_audit_mask(accepted, fraction=0., seed=20261026):
    """Sample eligible cases for additional review without inspecting correctness."""
    if accepted.ndim != 1 or accepted.dtype != torch.bool or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Boolean acceptance vector and fraction in [0, 1] required")
    eligible = torch.where(accepted.cpu())[0].numpy()
    count = math.ceil(fraction * len(eligible))
    result = torch.zeros_like(accepted, device="cpu")
    if count:
        chosen = np.random.default_rng(seed).choice(eligible, count, replace=False)
        result[torch.from_numpy(chosen)] = True
    return result.to(accepted.device)
