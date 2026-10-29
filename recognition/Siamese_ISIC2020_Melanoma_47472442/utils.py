"""Share helpers for devices, scoring, calibration and metrics.

Author: Alex Anderson (47472442), COMP3710 Pattern Recognition, 2026.
"""

import random
import resource
import sys
from math import ceil
from typing import Any, cast

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Sampler

from .dataset import ISICDataset, Sample
from .modules import MelanomaModel


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy and PyTorch random generators.

    Parameters
    ----------
    seed : int
        Seed shared by the three generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    """Choose the first available device: CUDA, Apple MPS, then CPU.

    Returns
    -------
    torch.device
        Device used for training or prediction.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def image_loader(
    samples: list[Sample], config: dict[str, Any], train: bool = False, sampler: Sampler[int] | None = None
) -> DataLoader[tuple[torch.Tensor, int]]:
    """Build an image loader with workers that persist across epochs.

    Parameters
    ----------
    samples : list of Sample
        Labelled image records.
    config : dict
        Settings containing image_size, batch_size and workers. The worker
        count must be positive because persistent workers are enabled.
    train : bool, optional
        Enable training augmentation if True. Default is False.
    sampler : torch.utils.data.Sampler or None, optional
        Sample indices in this order. Default is None, which preserves the
        input order without shuffling.

    Returns
    -------
    torch.utils.data.DataLoader
        Batches of normalised RGB image tensors and integer class labels.
    """
    dataset = ISICDataset(samples, config["image_size"], train)
    return DataLoader(
        dataset, config["batch_size"], sampler=sampler, num_workers=config["workers"], persistent_workers=True
    )


@torch.no_grad()
def score_images(
    model: MelanomaModel, loader: DataLoader[tuple[torch.Tensor, int]], device: torch.device
) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """Score images without gradients and return results in loader order.

    Parameters
    ----------
    model : MelanomaModel
        Model on the selected device. It is left in evaluation mode.
    loader : torch.utils.data.DataLoader
        Batches of image tensors and integer labels.
    device : torch.device
        Device used for the forward passes.

    Returns
    -------
    margins : numpy.ndarray of shape (images,)
        Melanoma logit minus benign logit, before calibration.
    embeddings : torch.Tensor of shape (images, 512)
        Unit-length image embeddings stored on the CPU.
    labels : numpy.ndarray of shape (images,)
        Class labels: 0 for benign and 1 for melanoma.
    """
    model.eval()
    margins, embeddings, labels = [], [], []
    for images, targets in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits, embedding = model(images.to(device))
        logits = logits.float()
        margins.append((logits[:, 1] - logits[:, 0]).cpu())
        embeddings.append(embedding.float().cpu())
        labels.append(targets)
    return torch.cat(margins).numpy(), torch.cat(embeddings), torch.cat(labels).numpy()


def fit_platt(margins: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit a logistic mapping from logit margins to probabilities.

    Parameters
    ----------
    margins : numpy.ndarray of shape (images,)
        Melanoma logit minus benign logit for calibration images.
    labels : numpy.ndarray of shape (images,)
        Calibration labels: 0 for benign and 1 for melanoma.

    Returns
    -------
    tuple of float
        Slope and intercept for ``p = sigmoid(a * margin + b)``.
    """
    model = LogisticRegression(C=1e6).fit(margins[:, None], labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def melanoma_probability(margins: np.ndarray, platt: tuple[float, float]) -> np.ndarray:
    """Apply the fitted logistic mapping to melanoma logit margins.

    Parameters
    ----------
    margins : numpy.ndarray of shape (images,)
        Melanoma logit minus benign logit.
    platt : tuple of float
        Slope and intercept returned by `fit_platt`.

    Returns
    -------
    numpy.ndarray of shape (images,)
        Calibrated melanoma probabilities.
    """
    return 1 / (1 + np.exp(-(platt[0] * margins + platt[1])))


def recall_threshold(labels: np.ndarray, scores: np.ndarray, recall: float) -> float:
    """Choose the highest observed cutoff meeting the recall target.

    Parameters
    ----------
    labels : numpy.ndarray of shape (images,)
        Class labels with at least one melanoma, labelled 1.
    scores : numpy.ndarray of shape (images,)
        Scores where higher values indicate melanoma.
    recall : float
        Required melanoma recall, greater than 0 and at most 1.

    Returns
    -------
    float
        Cutoff meeting the target on these images when scores at or above
        it are flagged. Ties can produce higher recall than requested.
    """
    melanoma_scores = np.sort(scores[labels == 1])[::-1]
    return float(melanoma_scores[ceil(recall * len(melanoma_scores)) - 1])


def specificity_threshold(labels: np.ndarray, scores: np.ndarray, specificity: float) -> float:
    """Choose a cutoff from the benign score distribution.

    Parameters
    ----------
    labels : numpy.ndarray of shape (images,)
        Class labels with at least one benign image, labelled 0.
    scores : numpy.ndarray of shape (images,)
        Scores where higher values indicate melanoma.
    specificity : float
        Target benign specificity, between 0 and 1.

    Returns
    -------
    float
        Benign score quantile at the requested specificity. Ties and
        quantile interpolation can affect the achieved specificity.
    """
    return float(np.quantile(scores[labels == 0], specificity))


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Summarise binary classification at a fixed score cutoff.

    Parameters
    ----------
    labels : numpy.ndarray of shape (images,)
        Class labels: 0 for benign and 1 for melanoma.
    scores : numpy.ndarray of shape (images,)
        Scores where higher values indicate melanoma.
    threshold : float
        Classify scores at or above this value as melanoma.

    Returns
    -------
    dict of str to float or int
        Cutoff, accuracy, macro-F1, both class precisions and recalls,
        melanoma F1, flagged fraction and TN/FP/FN/TP counts. Undefined
        precision, recall or F1 values are reported as zero.
    """
    predicted = (scores >= threshold).astype(int)
    # average=None returns class-wise arrays; sklearn's annotations omit this case and numeric zero_division.
    precision, recall, f1, _ = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        precision_recall_fscore_support(labels, predicted, labels=[0, 1], average=None, zero_division=cast(Any, 0)),
    )
    (tn, fp), (fn, tp) = confusion_matrix(labels, predicted, labels=[0, 1])
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1.mean()),
        "benign_precision": float(precision[0]),
        "benign_recall": float(recall[0]),
        "melanoma_precision": float(precision[1]),
        "melanoma_recall": float(recall[1]),
        "melanoma_f1": float(f1[1]),
        "flagged_fraction": float(predicted.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def peak_process_memory_gib() -> float:
    """Return peak resident memory for the current process on macOS.

    Returns
    -------
    float
        Peak resident memory in GiB on macOS, excluding loader workers.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 if sys.platform == "darwin" else 1024) / 2**30


def accelerator_memory_gib(device: torch.device) -> float:
    """Read accelerator memory usage in GiB.

    Parameters
    ----------
    device : torch.device
        Accelerator whose allocator statistics will be read.

    Returns
    -------
    float
        Peak reserved CUDA memory, current MPS driver allocation or zero
        for CPU. MPS has no peak counter, so training polls this value after
        every optimiser step and keeps the maximum.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_reserved() / 2**30
    return torch.mps.driver_allocated_memory() / 2**30 if device.type == "mps" else 0.0
