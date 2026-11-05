"""
confidence.py
Author: Derek McKinney (dm-uq), SN: 45506053

Contains utilities for confidence-based selective prediction and reject-option
analysis.

References: None

"""

# IMPORTS

import numpy as np
import torch
from torch import Tensor


# CONSTANTS


def extract_confidence_data(
        logits: Tensor,
        labels: Tensor,
        mask: Tensor,
) -> dict[str, np.ndarray]:
    """
    Extract prediction confidence and correctness for selected nodes.

    Confidence is measured using the maximum softmax response for each
    prediction.

    Args:
        logits: Model output logits of shape `(num_nodes, num_classes)`.
        labels: Ground-truth class labels of shape `(num_nodes,)`.
        mask: Boolean mask identifying the nodes to analyse.

    Returns:
        Dictionary containing confidence, predicted class, true class,
        and prediction correctness for each selected node.

    """
    # select outputs and labels using provided split mask
    selected_logits = logits[mask]
    targets = labels[mask]

    # convert logits to softmax responses using higher analysis precision
    probabilities = torch.softmax(
        selected_logits.double(),
        dim=1,
    )

    # get predicted classes and maximum softmax responses
    confidence, predictions = probabilities.max(dim=1)

    # record the correct predictions
    correct = predictions.eq(targets)

    return {
        "confidence": confidence.detach().cpu().numpy(),
        "prediction": predictions.detach().cpu().numpy(),
        "target": targets.detach().cpu().numpy(),
        "correct": correct.detach().cpu().numpy(),
    }


def select_reject_threshold(
        confidence: np.ndarray,
        correct: np.ndarray,
        max_risk: float = 0.01,
) -> float:
    """
    Select a global confidence threshold subject to a maximum empirical risk.

    Candidate thresholds are evaluated on validation predictions. The selected
    threshold maximises automated coverage while keeping the observed error
    rate among accepted predictions at or below `max_risk`.

    Args:
        confidence: Maximum softmax response for each prediction.
        correct: Boolean array indicating prediction correctness.
        max_risk: Maximum permitted empirical error rate among accepted
            predictions.

    Returns:
        Confidence threshold giving maximum coverage while satisfying the
        specified empirical risk constraint.

    Raises:
        ValueError: If `max_risk` is outside the valid interval, or no
            candidate threshold satisfies the risk constraint.

    """
    if not 0.0 <= max_risk < 1.0:
        raise ValueError("max_risk must be in the interval [0, 1).")

    # evaluate observed confidence values as candidate thresholds
    thresholds = np.unique(confidence)

    best_threshold = np.nan
    best_coverage = -1.0

    for threshold in thresholds:
        # automatically accept predictions at or above this threshold
        accepted = confidence >= threshold

        if not accepted.any():
            continue

        # measure empirical risk and coverage among accepted predictions
        risk = (~correct[accepted]).mean()
        coverage = accepted.mean()

        # retain the valid threshold providing the greatest coverage
        if risk <= max_risk and coverage > best_coverage:
            best_threshold = threshold
            best_coverage = coverage

    if np.isnan(best_threshold):
        raise ValueError(
            "No confidence threshold satisfies the specified risk constraint."
        )

    return float(best_threshold)


def evaluate_reject_rule(
        confidence: np.ndarray,
        correct: np.ndarray,
        threshold: float,
) -> dict[str, float | int]:
    """
    Evaluate a fixed confidence-based reject rule.

    Predictions with confidence greater than or equal to `threshold` are
    automatically accepted. Lower-confidence predictions are rejected, to be
    subject to 'manual review'.

    Args:
        confidence: Maximum softmax response for each prediction.
        correct: Boolean array indicating prediction correctness.
        threshold: Fixed confidence threshold used for automatic acceptance.

    Returns:
        Dictionary containing accepted and rejected counts, automated
        coverage, selective accuracy, and empirical risk.

    Raises:
        ValueError: If no predictions are accepted at the supplied threshold.

    """
    # apply the fixed reject-option decision rule
    accepted = confidence >= threshold

    accepted_count = int(accepted.sum())
    total_count = len(confidence)

    if accepted_count == 0:
        raise ValueError(
            "Reject rule accepts no predictions at the supplied threshold."
        )

    rejected_count = total_count - accepted_count

    # get classification error counts for accepted and rejected cases
    accepted_errors = int((~correct[accepted]).sum())
    rejected_errors = int((~correct[~accepted]).sum())

    # calculate 'coverage'
    coverage = accepted_count / total_count
    # calculate performance for accepted predictions under given threshold
    selective_accuracy = float(correct[accepted].mean())
    risk = 1.0 - selective_accuracy

    return {
        "accepted": accepted_count,
        "rejected": rejected_count,
        "accepted_errors": accepted_errors,
        "rejected_errors": rejected_errors,
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "risk": risk,
    }


def print_reject_results(
        results: dict[str, float | int],
        model_name: str,
) -> None:
    """
    Print formatted reject-option evaluation results.

    Args:
        results: Reject-option evaluation results containing accepted and
            rejected counts, error counts, coverage, selective accuracy,
            and empirical risk.
        model_name: Display name for the evaluated model.

    """
    print(f"{model_name} Reject-Option Results")
    result_width = 38  # fixed width
    print("-" * result_width)  # title separator

    print(f"{'Accepted:':<22}{results['accepted']:>10,}")
    print(f"{'Rejected:':<22}{results['rejected']:>10,}")
    print(f"{'Accepted errors:':<22}{results['accepted_errors']:>10,}")
    print(f"{'Rejected errors:':<22}{results['rejected_errors']:>10,}")
    print(f"{'Coverage:':<22}{results['coverage']:>10.2%}")
    print(
        f"{'Accepted accuracy:':<22}"
        f"{results['selective_accuracy']:>10.2%}"
    )
    print(f"{'Accepted risk:':<22}{results['risk']:>10.3%}")
    print()


def build_failure_data(
        mask: Tensor,
        mlp_data: dict[str, np.ndarray],
        gcn_data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Combine model confidence results with original graph node indices, for
    Failure Case Autopsy.

    Args:
        mask: Boolean mask identifying the evaluated graph split.
        mlp_data: Confidence results for the MLP.
        gcn_data: Confidence results for the GCN.

    Returns:
        Evaluation data for each node, including its original graph index,
        target, model predictions, confidence scores, and correctness.

    """
    # get original graph node indices
    node_indices = (
        mask
        .nonzero(as_tuple=False)
        .flatten()
        .cpu()
        .numpy()
    )

    return {
        "node_index": node_indices,
        "target": mlp_data["target"],
        "mlp_prediction": mlp_data["prediction"],
        "mlp_confidence": mlp_data["confidence"],
        "mlp_correct": mlp_data["correct"],
        "gcn_prediction": gcn_data["prediction"],
        "gcn_confidence": gcn_data["confidence"],
        "gcn_correct": gcn_data["correct"],
    }
