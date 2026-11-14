"""
Validation scoring from saved detections: per-image arrays and the threshold sweep.

Reads cached detection files and YOLO label files, so no model is needed. Everything here
uses the lenient routing rule with the benign threshold at its floor (so any benign box can clear an
image). An optional clear guard, `tau_c`, stops an image from clearing while its best
malignant confidence is at or above `tau_c`.

The sweep is a vectorised version of `policy.search_policy`; `check_against_policy` compares
the two so a mistake in either shows up as a disagreement.
"""

import csv
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from yolo_lesion_47041585.dataset import NormalizedBox
from yolo_lesion_47041585.helpers.policy import (
    THRESHOLD_GRID,
    TOLERANCE,
    PolicySpec,
    build_evidence,
    evaluate_policy,
)
from yolo_lesion_47041585.predict import Detection, load_run, predict_cases

THRESHOLDS = np.array(THRESHOLD_GRID)
CACHE_HEADER = ["image_id", "x_center", "y_center", "width", "height", "class_id", "confidence"]
MAX_BENIGN_ESCALATION = 0.30  # the plan's workload limit: benign images routed urgent or monitor


def read_targets(labels_dir: Path) -> list[Detection]:
    """Each image's annotated box from YOLO label files (exactly one per file)."""
    targets: list[Detection] = []
    for path in sorted(labels_dir.glob("*.txt")):
        lines = path.read_text().splitlines()
        if len(lines) != 1:
            raise ValueError(f"{path.name} has {len(lines)} boxes, expected exactly one")
        class_id, x, y, w, h = lines[0].split()
        box = NormalizedBox(float(x), float(y), float(w), float(h))
        targets.append(Detection(path.stem, box, int(class_id), 1.0))
    return targets


def read_detections(path: Path) -> list[Detection]:
    """Load a cached detection file."""
    with path.open(newline="") as file:
        return [
            Detection(
                r["image_id"],
                NormalizedBox(
                    float(r["x_center"]), float(r["y_center"]),
                    float(r["width"]), float(r["height"]),
                ),
                int(r["class_id"]),
                float(r["confidence"]),
            )
            for r in csv.DictReader(file)
        ]  # fmt: skip


def write_detections(path: Path, detections: Sequence[Detection]) -> None:
    """Save detections in the cache format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CACHE_HEADER)
        for d in detections:
            b = d.box
            writer.writerow([d.image_id, b.x_center, b.y_center, b.width, b.height, d.class_id, d.confidence])


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks so large checkpoints stay out of memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def cached_predictions(
    checkpoint: Path, images: Sequence[Path], cache_dir: Path, device: str | None
) -> list[Detection]:
    """Predict on the images, or reload the detections saved for this exact checkpoint.

    The cache file name holds the run name and the checkpoint's hash, but not the images, so
    use a separate `cache_dir` for each split.
    """
    cached = cache_dir / f"{checkpoint.parents[1].name}-{file_sha256(checkpoint)[:12]}.csv"
    if cached.is_file():
        return read_detections(cached)

    model, settings = load_run(checkpoint)
    detections = predict_cases(model, images, settings, device)
    write_detections(cached, detections)
    return detections


def patient_groups(manifest: Path, image_ids: Sequence[str]) -> list[list[int]]:
    """Indices of the images of each validation patient, for patient-level resampling."""
    with manifest.open(newline="") as file:
        patient_of = {r["isic_id"]: r["patient_id"] for r in csv.DictReader(file)}
    groups: dict[str, list[int]] = {}
    for index, image_id in enumerate(image_ids):
        groups.setdefault(patient_of[image_id], []).append(index)
    return list(groups.values())


@dataclass(frozen=True)
class RunArrays:
    """One run reduced to per-image arrays, in the order of the targets.

    Attributes:
        name (str): The run, such as "A0-s47041585".
        score (NDArray[np.float64]): Highest malignant-class confidence, 0 if none.
        has_benign (NDArray[np.bool_]): Whether the image has any benign detection.
        target (NDArray[np.float64]): For each image, the highest confidence of a
            class-correct box overlapping the target at IoU 0.5 or more, else 0. For a
            malignant image this decides whether the lesion is detected at a threshold.
    """

    name: str
    score: NDArray[np.float64]
    has_benign: NDArray[np.bool_]
    target: NDArray[np.float64]


def arrays_for(name: str, predictions: Sequence[Detection], targets: Sequence[Detection]) -> RunArrays:
    evidence = build_evidence(predictions, targets)
    return RunArrays(
        name=name,
        score=np.array([e.inputs.max_malignant_conf for e in evidence]),
        has_benign=np.array([e.inputs.n_benign > 0 for e in evidence]),
        target=np.array([e.candidates[0][0] if e.candidates else 0.0 for e in evidence]),
    )


def auc(run: RunArrays, is_malignant: NDArray[np.bool_], index: NDArray[np.int_]) -> float:
    """AUC of the malignant score, malignant against benign images, ties counting half."""
    positive = run.score[index][is_malignant[index]]
    negative = run.score[index][~is_malignant[index]]
    wins = (positive[:, None] > negative[None, :]).sum() + 0.5 * (positive[:, None] == negative[None, :]).sum()
    return float(wins / (positive.size * negative.size))


@dataclass(frozen=True)
class Sweep:
    """Rates at every malignant threshold of the grid (one entry per threshold).

    Benign and malignant rates each use all images of that class as the denominator.

    Attributes:
        sensitivity (NDArray[np.float64]): Malignant lesions with a correctly placed
            malignant box at or above the threshold.
        malignant_urgent / malignant_monitor / malignant_clear: Malignant images by route.
        benign_urgent / benign_monitor / benign_clear: Benign images by route. The urgent
            rate is the biopsy burden; urgent plus monitor is the plan's benign escalation.
    """

    sensitivity: NDArray[np.float64]
    malignant_urgent: NDArray[np.float64]
    malignant_monitor: NDArray[np.float64]
    malignant_clear: NDArray[np.float64]
    benign_urgent: NDArray[np.float64]
    benign_monitor: NDArray[np.float64]
    benign_clear: NDArray[np.float64]

    @property
    def benign_escalation(self) -> NDArray[np.float64]:
        return self.benign_urgent + self.benign_monitor


def sweep(
    run: RunArrays,
    is_malignant: NDArray[np.bool_],
    index: NDArray[np.int_],
    clear_guard: float | None = None,
) -> Sweep:
    """Route every image at every malignant threshold under the lenient rule, benign threshold at its floor."""
    mal = is_malignant[index]
    guard = np.inf if clear_guard is None else clear_guard

    def rates(selector: NDArray[np.bool_]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        score = run.score[index][selector][:, None]
        can_clear = run.has_benign[index][selector][:, None] & (score < guard)
        urgent = score >= THRESHOLDS[None, :]
        clear = ~urgent & can_clear
        monitor = ~urgent & ~can_clear
        return urgent.mean(axis=0), monitor.mean(axis=0), clear.mean(axis=0)

    m_urgent, m_monitor, m_clear = rates(mal)
    b_urgent, b_monitor, b_clear = rates(~mal)
    return Sweep(
        sensitivity=(run.target[index][mal][:, None] >= THRESHOLDS[None, :]).mean(axis=0),
        malignant_urgent=m_urgent, malignant_monitor=m_monitor, malignant_clear=m_clear,
        benign_urgent=b_urgent, benign_monitor=b_monitor, benign_clear=b_clear,
    )  # fmt: skip


def s30(result: Sweep) -> float:
    """Best sensitivity among thresholds that escalate at most 30% of benign images."""
    ok = result.benign_escalation <= MAX_BENIGN_ESCALATION + TOLERANCE
    return float(result.sensitivity[ok].max()) if ok.any() else 0.0


def check_against_policy(
    run: RunArrays,
    predictions: Sequence[Detection],
    targets: Sequence[Detection],
    is_malignant: NDArray[np.bool_],
    clear_guard: float | None = None,
    step: int = 5,
) -> float:
    """Largest disagreement between the sweep and `evaluate_policy`, at every `step`-th threshold."""
    result = sweep(run, is_malignant, np.arange(len(targets)), clear_guard)
    worst = 0.0
    for i in range(0, len(THRESHOLDS), step):
        spec = PolicySpec(float(THRESHOLDS[i]), THRESHOLDS[0].item(), "lenient", clear_guard)
        p = evaluate_policy(predictions, targets, spec).point
        pairs = (
            (result.sensitivity[i], p.m), (result.benign_urgent[i], p.benign_urgent),
            (result.benign_monitor[i], p.benign_monitor), (result.benign_clear[i], p.benign_clear),
            (result.malignant_urgent[i], p.malignant_urgent), (result.malignant_monitor[i], p.malignant_monitor),
            (result.malignant_clear[i], p.malignant_clear),
        )  # fmt: skip
        worst = max(worst, *(abs(float(a) - b) for a, b in pairs))
    return worst
