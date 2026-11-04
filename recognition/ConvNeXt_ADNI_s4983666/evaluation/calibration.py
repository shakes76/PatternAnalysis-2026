"""Platt scaling and a referral (reject-option) threshold.

Both are fitted on the calibration patients only, after the checkpoint is
frozen, and then applied unchanged to the test patients.

Platt scaling: J. Platt, "Probabilistic Outputs for Support Vector Machines
and Comparisons to Regularized Likelihood Methods", 1999; for neural networks
see Guo et al., "On Calibration of Modern Neural Networks", ICML 2017,
https://arxiv.org/abs/1706.04599 . Each AD logit z becomes
p = sigmoid(slope * z + intercept). The slope rescales over- or under-confident
logits (like temperature scaling, slope = 1 / T); the intercept moves the
decision boundary, which corrects a systematic bias towards one class. Both
parameters are fitted on calibration slices by minimising the unweighted log loss.

Referral rule: a patient is referred to a specialist when the calibrated
confidence max(p, 1 - p) is below tau. tau is the smallest threshold whose
accepted calibration patients reach the target accuracy, i.e. the rule
automates as many patients as possible at that accuracy.
"""

import math

import torch


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def fit_platt(logits: list[float], labels: list[int]) -> tuple[float, float]:
    """Return (slope, intercept) minimising the log loss of sigmoid(slope * logit + intercept)."""
    if not logits or len(logits) != len(labels) or len(set(labels)) != 2:
        raise ValueError("Platt scaling needs matching logits and labels from both classes.")
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    slope = torch.ones((), dtype=torch.float64, requires_grad=True)       # Start from the identity map.
    intercept = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([slope, intercept], lr=0.1, max_iter=500, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(slope * z + intercept, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    result = float(slope), float(intercept)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Platt scaling failed to converge.")
    return result


def apply_platt(slices: list[dict], slope: float, intercept: float) -> list[dict]:
    """Copy slice rows with probability = sigmoid(slope * logit + intercept)."""
    return [{**row, "probability": _sigmoid(slope * row["logit"] + intercept)} for row in slices]


def choose_referral_threshold(labels: list[int], probabilities: list[float],
                              target_accuracy: float) -> dict:
    """Pick the lowest confidence threshold whose accepted cases reach the target accuracy.

    Returns ``threshold=None`` (refer everyone) if no threshold reaches it.
    """
    cases = sorted(((max(p, 1 - p), int(int(p >= 0.5) == y)) for y, p in zip(labels, probabilities)),
                   reverse=True)
    best = {"threshold": None, "coverage": 0.0, "accepted_accuracy": None}
    accepted = correct = 0
    for position, (confidence, is_correct) in enumerate(cases):
        accepted += 1
        correct += is_correct
        tied_with_next = position + 1 < len(cases) and cases[position + 1][0] == confidence
        if not tied_with_next and correct / accepted >= target_accuracy:
            best = {"threshold": confidence, "coverage": accepted / len(cases),
                    "accepted_accuracy": correct / accepted}
    return {**best, "target_accuracy": target_accuracy, "n_cases": len(cases)}


def referral_decision(probability: float, threshold: float | None) -> str:
    """Return "AD", "NC" or "REFER" for one calibrated probability."""
    if threshold is None or max(probability, 1 - probability) < threshold:
        return "REFER"
    return "AD" if probability >= 0.5 else "NC"
