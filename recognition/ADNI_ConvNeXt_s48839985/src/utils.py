import time
import torch

from functools import wraps
from inspect import signature

from torch import nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from dataset import NC_LABEL, AD_LABEL

#######################
# RESOURCE RECORDINGS
#######################


def output_cuda_resource_usage(func):
    """
    Wrapper utility to output VRAM and runtime of supplied function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        # Retrieve device regardless of whether it was passed
        # positionally or as a keyword argument.
        bound_args = signature(func).bind(*args, **kwargs)
        device = bound_args.arguments.get("device")

        if device is None or device.type != "cuda":
            return func(*args, **kwargs)

        # Reseting GPU memory stats
        torch.cuda.reset_peak_memory_stats(device)

        # Starting runtime measurement
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()

        try:
            return func(*args, **kwargs)

        finally:
            torch.cuda.synchronize(device)

            runtime = time.perf_counter() - start_time

            print(f"Resource usage details for: {func.__name__}")
            print(f"\nTraining runtime: {runtime:.2f} seconds")

            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)

            print(f"Peak GPU memory allocated: {peak_allocated:.2f} MB")
            print(f"Peak GPU memory reserved:  {peak_reserved:.2f} MB")

    return wrapper


#####################
# CALCULATE METRICS
#####################


def calculate_classification_metrics(y_true, y_pred, y_prob):
    """
    Calculating relevant classificaiton metrics based on values provided
    """

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[NC_LABEL, AD_LABEL],
        average=None,
        zero_division=0,
    )

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[NC_LABEL, AD_LABEL],
    ).ravel()

    if len(np.unique(y_true)) == 2:
        auroc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        auroc = float("nan")
        pr_auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "confusion_matrix": (tn, fp, fn, tp),
    }


def calculate_confidence_metrics(y_true, y_pred, y_prob, confidence_threshold):
    """
    Calculate classification coverage and accuracy for high-confidence
    predictions.
    """

    # Calculate confidence in the predicted class.
    confidence = np.where(
        y_pred == AD_LABEL,
        y_prob,
        1 - y_prob,
    )

    # Identify predictions meeting the confidence threshold.
    accepted = confidence >= confidence_threshold
    coverage = np.mean(accepted)

    if np.any(accepted):
        high_confidence_accuracy = accuracy_score(
            y_true[accepted],
            y_pred[accepted],
        )
    else:
        high_confidence_accuracy = float("nan")

    return {
        "coverage": coverage,
        "high_confidence_accuracy": high_confidence_accuracy,
    }


def calculate_computational_metrics(
    model: nn.Module,
    runtime: float,
    total_slices: int,
    device: torch.device,
):
    """
    Calculate model complexity, inference performance and GPU memory usage.
    """

    # Model complexity.
    parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Inference performance.
    latency = runtime / total_slices
    throughput = total_slices / runtime

    # GPU memory usage.
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)

        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
    else:
        peak_allocated = float("nan")
        peak_reserved = float("nan")

    return {
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "runtime": runtime,
        "latency": latency,
        "throughput": throughput,
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
    }
