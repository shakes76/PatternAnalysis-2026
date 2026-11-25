"""Runtime subject validation, acquisition parsing and optional sampling."""

import re
from collections import Counter, defaultdict

import numpy as np
import torch

ACQUISITION_DATE = re.compile(r"_raw_((?:19|20)\d{2})\d{4}")


SUBJECT_PATTERN = re.compile(r"(?<!\d)\d{3}_S_\d{4}(?!\d)")
PATH_FIELDS = ("raw", "masked", "c1", "c2", "c3", "c4", "c5")


def resolve_subject(metadata, scan, label):
    if scan not in metadata:
        raise ValueError("A prepared scan has no metadata mapping")
    record = metadata[scan]
    mapped = []
    for field in PATH_FIELDS:
        value = record.get(field)
        if not isinstance(value, str):
            raise ValueError("Metadata must contain all source-path fields")
        tokens = set(SUBJECT_PATTERN.findall(value))
        if len(tokens) != 1:
            raise ValueError("Source path has missing or ambiguous subject tokens")
        mapped.append(next(iter(tokens)))
    if len(set(mapped)) != 1:
        raise ValueError("Metadata source paths disagree on subject identity")
    if label not in (0, 1) or record.get("label") != {0: 0, 1: 2}[label]:
        raise ValueError("Prepared class disagrees with metadata diagnostic code")
    return mapped[0]


def acquisition_group(record):
    raw = str(record.get("raw", "")) if record else ""
    match = ACQUISITION_DATE.search(raw)
    era = "unknown" if not match else ("pre-2017" if int(match.group(1)) < 2017 else "2017+")
    sequence = "unknown" if not raw else ("accelerated" if "Accel" in raw else "non-accelerated")
    return era, sequence


EXPECTED_DRAW_MASS = {
    (0, "2017+"): 0.5,
    (1, "2017+"): 0.25,
    (1, "pre-2017"): 0.25,
}
CLASS_NAMES = {0: "NC", 1: "AD"}


def era_balanced_sample_weights(dataset, metadata):
    """Keep NC/AD draw mass equal and split AD draws equally by acquisition era.

    Subjects receive equal mass within each class-era group, then their slices
    share that mass. Sequence is not balanced separately.
    """
    from dataset import extract_patient_id

    labels = np.asarray(dataset.labels)
    if (len(labels) != len(dataset.file_paths)
            or len(labels) != len(dataset.patient_ids)
            or set(labels.tolist()) != {0, 1}):
        raise ValueError("Era balancing requires matching paths, scan IDs and NC/AD labels")
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Era balancing requires nonempty scan metadata")

    scan_subjects = {}
    subject_labels = {}
    sample_groups = []
    subjects_by_group = defaultdict(set)
    scans_by_group = defaultdict(set)
    slices_by_subject_group = Counter()

    for path, label_value, subject_value in zip(
        dataset.file_paths, labels, dataset.patient_ids
    ):
        scan = extract_patient_id(path)
        label = int(label_value)
        record = metadata.get(scan)
        era, _ = acquisition_group(record)
        if era == "unknown":
            raise ValueError("Training scan is missing a parseable acquisition year")
        group = (label, era)
        if group not in EXPECTED_DRAW_MASS:
            raise ValueError("Training data contains an acquisition group outside the sampling design")
        if scan not in scan_subjects:
            scan_subjects[scan] = resolve_subject(metadata, scan, label)
        subject = scan_subjects[scan]
        if str(subject_value) != subject:
            raise ValueError("Dataset subject identity disagrees with metadata")
        previous_label = subject_labels.setdefault(subject, label)
        if previous_label != label:
            raise ValueError("A subject has conflicting labels in the training data")

        subject_group = (label, era, subject)
        sample_groups.append(subject_group)
        subjects_by_group[group].add(subject)
        scans_by_group[group].add(scan)
        slices_by_subject_group[subject_group] += 1

    if set(subjects_by_group) != set(EXPECTED_DRAW_MASS):
        raise ValueError("Training split must contain NC later-era and AD early- and later-era groups")

    weights = np.empty(len(labels), dtype=np.float64)
    for index, (label, era, subject) in enumerate(sample_groups):
        group = (label, era)
        weights[index] = (
            EXPECTED_DRAW_MASS[group]
            / len(subjects_by_group[group])
            / slices_by_subject_group[(label, era, subject)]
        )

    report = {
        "policy": (
            "Equal total NC and AD draw mass. Split AD draw mass equally between "
            "pre-2017 and 2017+ groups. Give subjects equal mass within each "
            "class-era group and divide each subject's mass across its slices."
        ),
        "sequence_balanced": False,
        "groups": [
            {
                "class": CLASS_NAMES[label],
                "era": era,
                "subjects": len(subjects_by_group[(label, era)]),
                "scans": len(scans_by_group[(label, era)]),
                "slices": sum(
                    count for (group_label, group_era, _), count
                    in slices_by_subject_group.items()
                    if (group_label, group_era) == (label, era)
                ),
                "expected_draw_mass": EXPECTED_DRAW_MASS[(label, era)],
            }
            for label, era in EXPECTED_DRAW_MASS
        ],
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
    }
    return torch.as_tensor(weights, dtype=torch.double), report
