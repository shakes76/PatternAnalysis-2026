"""
Scoring: box matching, detection metrics, routing outcomes, uncertainty and timing.

Box matching and IoU are implemented here. Routing is scored in `policy.py`, which depends
on this module; the remaining functions are placeholders for the final evaluation.
"""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from yolo_lesion_47041585.dataset import NormalizedBox
from yolo_lesion_47041585.predict import Detection


@dataclass(frozen=True)
class MatchResult:
    """The outcome of matching predictions to ground truth.

    Attributes:
        matches (tuple[tuple[Detection, Detection, float], ...]): (prediction, truth,
            IoU) for every class-correct match.
        false_positives (tuple[Detection, ...]): Predictions with no match, including
            duplicates and wrong-class boxes.
        misses (tuple[Detection, ...]): Ground-truth boxes with no match.
    """

    matches: tuple[tuple[Detection, Detection, float], ...]
    false_positives: tuple[Detection, ...]
    misses: tuple[Detection, ...]


def box_iou(a: NormalizedBox, b: NormalizedBox) -> float:
    """IoU of two normalised centre-format boxes.

    Normalising by the same image width and height scales the intersection and the union
    by the same factor, so the result equals the pixel IoU.

    Args:
        a (NormalizedBox): First box.
        b (NormalizedBox): Second box.

    Returns:
        float: Intersection over union in [0, 1].
    """
    # compute box edge coordinates
    ax1, ax2 = a.x_center - a.width / 2, a.x_center + a.width / 2
    ay1, ay2 = a.y_center - a.height / 2, a.y_center + a.height / 2
    bx1, bx2 = b.x_center - b.width / 2, b.x_center + b.width / 2
    by1, by2 = b.y_center - b.height / 2, b.y_center + b.height / 2

    # compute overlap along each axis
    overlap_w = min(ax2, bx2) - max(ax1, bx1)
    overlap_h = min(ay2, by2) - max(ay1, by1)

    # if there is no overlap, return 0.0
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0

    # compute IoU
    intersection = overlap_w * overlap_h
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union


def match_detections(
    predictions: Sequence[Detection],
    truths: Sequence[Detection],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """One-to-one, class-aware matching of predictions to ground truth by IoU.

    Boxes are only compared within the same image. Predictions are taken in order of
    decreasing confidence and each claims the unmatched same-class ground truth it
    overlaps most, provided the IoU is at least `iou_threshold`. A ground-truth box is
    matched at most once, so duplicates become false positives, and a wrong-class box
    is both a false positive and a miss.

    Args:
        predictions (Sequence[Detection]): Predicted boxes for any number of images.
        truths (Sequence[Detection]): Ground-truth boxes. Their confidence is ignored.
        iou_threshold (float): Minimum IoU for a match.

    Returns:
        MatchResult: The matches, unmatched predictions and unmatched ground truth.
    """
    # group ground-truth boxes by image for faster lookup
    truths_by_image: dict[str, list[Detection]] = defaultdict(list)
    for truth in truths:
        truths_by_image[truth.image_id].append(truth)

    claimed: set[int] = set()  # indices into `truths` already matched
    index_of = {id(truth): i for i, truth in enumerate(truths)}

    matches: list[tuple[Detection, Detection, float]] = []
    false_positives: list[Detection] = []
    # go through predictions from most to least confident
    for pred in sorted(predictions, key=lambda d: d.confidence, reverse=True):
        best: Detection | None = None
        best_iou = 0.0
        for truth in truths_by_image.get(pred.image_id, ()):
            # if the classes don't match or the truth is already claimed, skip it
            if truth.class_id != pred.class_id or index_of[id(truth)] in claimed:
                continue
            iou = box_iou(pred.box, truth.box)
            # if IoU is above the threshold and it's the best so far, remember it
            if iou >= iou_threshold and iou > best_iou:
                best, best_iou = truth, iou
        # if there is no match, it's a false positive, otherwise claim the truth and
        # record the match
        if best is None:
            false_positives.append(pred)
        else:
            claimed.add(index_of[id(best)])
            matches.append((pred, best, best_iou))

    # any unmatched truths are misses
    misses = tuple(t for i, t in enumerate(truths) if i not in claimed)

    return MatchResult(tuple(matches), tuple(false_positives), misses)


@dataclass(frozen=True)
class BootstrapResult:
    """The statistics of a patient-cluster bootstrap.

    Attributes:
        draws (dict[str, list[float]]): For each named statistic, its value in every valid
            resample. The lists are aligned, so draw `i` of one statistic and draw `i` of
            another come from the same resample, which is what paired differences need.
        n_valid (int): Resamples in which every statistic was defined.
        n_repeats (int): Resamples attempted. The rest had an undefined statistic, such as
            a class that no sampled patient had, and were dropped, never counted as zero.
    """

    draws: dict[str, list[float]]
    n_valid: int
    n_repeats: int


def bootstrap_by_patient(
    groups: Sequence[Sequence[int]],
    statistic: Callable[[list[int]], Mapping[str, float | None]],
    repeats: int = 2000,
    seed: int = 47041585,
) -> BootstrapResult:
    """Resample whole patients with replacement and recompute statistics each time.

    Images of one patient are correlated, so the unit of resampling is the patient, and
    a patient drawn twice contributes all their images twice. The same resamples are used
    for every statistic, so differences between models are paired.

    Args:
        groups (Sequence[Sequence[int]]): One entry per patient, holding the indices of
            that patient's images.
        statistic (Callable[[list[int]], Mapping[str, float | None]]): Given the indices
            of a resample (with repeats), returns named statistics, None if undefined.
        repeats (int): How many resamples to draw.
        seed (int): Seed of the random generator, fixed so the intervals reproduce.

    Returns:
        BootstrapResult: The values of every statistic in every valid resample.

    Raises:
        ValueError: If there are no patients.
    """
    if not groups:
        raise ValueError("Cannot resample no patients.")
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    n_valid = 0
    for _ in range(repeats):
        picked: list[int] = rng.integers(0, len(groups), size=len(groups)).tolist()
        indices = [i for patient in picked for i in groups[patient]]
        values = statistic(indices)
        if any(v is None for v in values.values()):
            continue
        n_valid += 1
        for name, value in values.items():
            draws.setdefault(name, []).append(float(value))  # pyright: ignore[reportArgumentType]
    return BootstrapResult(draws, n_valid, repeats)


def percentile_interval(
    values: Sequence[float], level: float = 0.95
) -> tuple[float, float]:
    """The central `level` interval of bootstrap draws (2.5th to 97.5th percentile)."""
    if not values:
        raise ValueError("No draws to summarise.")
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)
