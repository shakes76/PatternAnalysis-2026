"""calibrate.py: Calibration and reject option: temperature scaling, ECE and a frozen threshold.

Author: 49250523 (COMP3710 Pattern Recognition project, 2026)

Every fit in this file uses validation logits only. The fitted temperature and
threshold are then applied, unchanged, to the test logits (acceptance criteria
AC2-AC4 in the README).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

ECE_BINS = 15  # Guo et al. (2017), Section 5
TARGET_ACCURACY = 0.97  # AC3


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one temperature ``T`` that minimises the NLL of ``softmax(logits / T)``.

    Temperature scaling (Guo et al., 2017) divides every logit by the same
    ``T > 0``. It changes confidence but not the predicted class, so accuracy
    and macro-F1 stay the same. ``T`` is optimised in log space so it stays
    positive.

    Args:
        logits: Validation logits, shape ``(num_nodes, num_classes)``.
        labels: Validation labels, shape ``(num_nodes,)``.

    Returns:
        The fitted temperature.
    """
    logits = logits.detach()
    log_t = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    return log_t.exp().item()


def ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = ECE_BINS) -> float:
    """Return the top-label expected calibration error.

    Pages are put into ``n_bins`` equal-width bins by confidence (highest class
    probability). ECE is the size-weighted mean gap between each bin's accuracy
    and its mean confidence.

    Args:
        probs: Class probabilities, shape ``(num_nodes, num_classes)``.
        labels: True labels, shape ``(num_nodes,)``.
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE, from 0 (perfect) to 1.
    """
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            gap = (correct[in_bin].mean() - conf[in_bin].mean()).abs()
            total += in_bin.float().mean().item() * gap.item()
    return total


def choose_threshold(
    probs: torch.Tensor, labels: torch.Tensor, target: float = TARGET_ACCURACY
) -> float | None:
    """Choose the lowest confidence threshold that keeps accuracy at ``target``.

    A lower threshold auto-labels more pages, so the lowest threshold that still
    meets the target gives the highest coverage (the AC3 rule). Pages with equal
    confidence are kept or rejected together.

    Args:
        probs: Validation probabilities, shape ``(num_nodes, num_classes)``.
        labels: Validation labels, shape ``(num_nodes,)``.
        target: Minimum accuracy on the auto-labelled pages.

    Returns:
        The threshold, or ``None`` if no threshold reaches the target.
    """
    conf, pred = probs.max(dim=1)
    conf, order = conf.sort(descending=True)
    correct = (pred[order] == labels[order]).float()
    accuracy = correct.cumsum(0) / torch.arange(1, len(conf) + 1)
    # Only cut at the end of a group of equal confidences.
    group_end = torch.ones_like(conf, dtype=torch.bool)
    group_end[:-1] = conf[:-1] != conf[1:]
    ok = (accuracy >= target) & group_end
    if not ok.any():
        return None
    last = int(ok.nonzero().max())
    return conf[last].item()


def reject_report(
    probs: torch.Tensor, labels: torch.Tensor, threshold: float, num_classes: int
) -> dict[str, object]:
    """Apply a frozen threshold: auto-label pages at or above it, refer the rest.

    Args:
        probs: Class probabilities, shape ``(num_nodes, num_classes)``.
        labels: True labels, shape ``(num_nodes,)``.
        threshold: Confidence threshold chosen on validation.
        num_classes: Number of classes, for the per-class coverage.

    Returns:
        Overall coverage and accuracy of the auto-labelled pages, plus the
        coverage of each true class (does the rule refer one class far more?).
    """
    conf, pred = probs.max(dim=1)
    keep = conf >= threshold
    per_class = [keep[labels == c].float().mean().item() for c in range(num_classes)]
    return {
        "threshold": threshold,
        "coverage": keep.float().mean().item(),
        "accuracy": (pred[keep] == labels[keep]).float().mean().item() if keep.any() else 0.0,
        "coverage_per_class": per_class,
    }
