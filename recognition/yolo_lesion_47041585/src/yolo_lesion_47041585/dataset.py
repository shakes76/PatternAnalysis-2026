"""
ISIC2018 data pipeline: cohort, pair audit, patient split and YOLO export.

Reusable dataset logic is contained here. The stages run in order (build_cohort,
audit_pairs, build_independence_groups, make_grouped_splits, validate_split_manifest,
export_yolo_dataset) and are driven by audit.py, which also decides the
split seed and mask threshold. Detailed reasoning of the data decisions can be found in
the project's README.
"""

import csv
import hashlib
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

# type aliases for the diagnoses
Diagnosis = Literal["Benign", "Malignant"]
ClassId = Literal[0, 1]
CLASS_MAPPING: Mapping[Diagnosis, ClassId] = {
    "Benign": 0,
    "Malignant": 1,
}

# split names matching the keys YOLO dataset YAML files use
Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")
DEFAULT_SPLIT_RATIOS: Mapping[Split, float] = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True)
class CohortRecord:
    """An included image with its mask, identifiers and class.

    Attributes:
        isic_id (str): ISIC image identifier.
        source_set (str): Name of the original set the image came from.
        image_path (Path): Path to the image file.
        mask_path (Path): Path to the segmentation mask file.
        patient_id (str): Patient identifier.
        lesion_id (str | None): Lesion identifier, or None if the metadata has none.
        diagnosis (Diagnosis): The record's diagnosis.
        class_id (ClassId): 0 for Benign, 1 for Malignant.
    """

    isic_id: str
    source_set: str
    image_path: Path
    mask_path: Path
    patient_id: str
    lesion_id: str | None
    diagnosis: Diagnosis
    class_id: ClassId


@dataclass(frozen=True)
class ExcludedRecord:
    """A rejected metadata row and all of its exclusion reasons.

    Attributes:
        metadata_row (int): Zero-based index of the row in the raw CSV, excluding the
            header.
        isic_id (str | None): The row's ISIC ID, or None if it has none.
        reasons (tuple[str, ...]): Every rule the row failed.
    """

    metadata_row: int
    isic_id: str | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CohortResult:
    """The included and excluded records, and the raw metadata they came from.

    Attributes:
        records (tuple[CohortRecord, ...]): Included records, ordered by ISIC ID.
        exclusions (tuple[ExcludedRecord, ...]): Rejected rows, in metadata order.
        raw_metadata_sha256 (str): SHA-256 of the raw metadata file's bytes.
        input_row_count (int): Number of data rows in the raw metadata.
    """

    records: tuple[CohortRecord, ...]
    exclusions: tuple[ExcludedRecord, ...]
    raw_metadata_sha256: str
    input_row_count: int


@dataclass(frozen=True)
class NormalizedBox:
    """A bounding box derived from a binary mask, in centre format.

    Attributes:
        x_center (float): Horizontal centre, as a fraction of the image width.
        y_center (float): Vertical centre, as a fraction of the image height.
        width (float): Box width, as a fraction of the image width.
        height (float): Box height, as a fraction of the image height.
    """

    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class AuditedRecord:
    """A cohort record with its image size, bounding box and file hashes.

    Attributes:
        record (CohortRecord): The audited cohort record.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.
        box (NormalizedBox): Box around the mask's foreground.
        image_sha256 (str): SHA-256 of the image file's bytes.
        mask_sha256 (str): SHA-256 of the mask file's bytes.
    """

    record: CohortRecord
    image_width: int
    image_height: int
    box: NormalizedBox
    image_sha256: str
    mask_sha256: str


@dataclass(frozen=True)
class SplitRecord:
    """An audited record assigned to a split, with its independence group.

    Attributes:
        audited (AuditedRecord): The audited record.
        group_id (str): ID of the independence group the record belongs to.
        split (Split): The split the record is assigned to.
    """

    audited: AuditedRecord
    group_id: str
    split: Split


@dataclass(frozen=True)
class SplitManifest:
    """Every record's split assignment.

    Attributes:
        records (tuple[SplitRecord, ...]): One entry per record, ordered by ISIC ID.
    """

    records: tuple[SplitRecord, ...]


def _index_files(
    root: Path,
    pattern: str,
    id_suffix: str = "",
) -> dict[str, Path]:
    """Index files in a directory tree by their ISIC identifier.

    Args:
        root (Path): Root directory to search for files.
        pattern (str): Glob pattern to match files.
        id_suffix (str): Suffix to remove from the file stem to obtain the
            ISIC ID (default: "").

    Returns:
        dict[str, Path]: Mapping of ISIC IDs to file paths.

    Raises:
        ValueError: If multiple files are found for the same ISIC ID.
    """
    paths_by_id: dict[str, Path] = {}

    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue

        isic_id = path.stem.removesuffix(id_suffix)

        if isic_id in paths_by_id:
            raise ValueError(
                f"Multiple files for {isic_id}: {paths_by_id[isic_id]} and {path}"
            )

        paths_by_id[isic_id] = path

    return paths_by_id


def build_cohort(
    raw_metadata: Path,
    image_roots: Mapping[str, Path],
    mask_roots: Mapping[str, Path],
) -> CohortResult:
    """Build the cohort of included records from the raw metadata.

    Applies the inclusion and exclusion rules, and pairs each retained record with
    its image and mask. Only Benign (0) and Malignant (1) records with patient IDs
    are retained.

    Args:
        raw_metadata (Path): Path to the raw metadata CSV file.
        image_roots (Mapping[str, Path]): Mapping of source set names to image root
            directories
        mask_roots (Mapping[str, Path]): Mapping of source set names to mask root
            directories

    Returns:
        CohortResult: Included records, exclusions with reasons, and a hash and row
            count of the raw metadata.

    Raises:
        ValueError: If the metadata path is not a file, required columns are missing,
            roots are invalid, or an image ID appears in multiple sources.
    """
    # ensure the raw metadata file and image/mask roots exist
    if not raw_metadata.is_file():
        raise ValueError(f"Raw metadata file not found: {raw_metadata}")

    if not image_roots:
        raise ValueError("No image roots provided")

    if set(image_roots) != set(mask_roots):
        raise ValueError("Image and mask roots must have identical source-set keys")

    for root in image_roots.values():
        if not root.is_dir():
            raise ValueError(f"Image root directory not found: {root}")

    for root in mask_roots.values():
        if not root.is_dir():
            raise ValueError(f"Mask root directory not found: {root}")

    required_columns = {"isic_id", "lesion_id", "patient_id", "diagnosis_1"}

    # utf-8-sig drops any byte-order mark. every field is read as a string
    with raw_metadata.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing_columns = required_columns - set(reader.fieldnames or ())

        if missing_columns:
            raise ValueError(
                f"Raw metadata file is missing required columns: {', '.join(sorted(missing_columns))}"
            )

        rows = list(reader)

    # hash the raw metadata for provenance
    metadata_hash = hashlib.sha256(raw_metadata.read_bytes()).hexdigest()

    def clean(value: str | None) -> str | None:
        # strip whitespace and treat empty strings as missing
        return (value or "").strip() or None

    images_by_source = {
        source: _index_files(root, "*.jpg") for source, root in image_roots.items()
    }
    masks_by_source = {
        source: _index_files(root, "*_segmentation.png", "_segmentation")
        for source, root in mask_roots.items()
    }

    id_counts = Counter(clean(row["isic_id"]) for row in rows)
    duplicate_ids = {
        isic_id for isic_id, count in id_counts.items() if isic_id and count > 1
    }

    records: list[CohortRecord] = []
    exclusions: list[ExcludedRecord] = []

    for metadata_row, row in enumerate(rows):
        reasons: list[str] = []
        source_set: str | None = None
        image_path: Path | None = None
        mask_path: Path | None = None

        isic_id = clean(row["isic_id"])
        patient_id = clean(row["patient_id"])
        lesion_id = clean(row["lesion_id"])
        diagnosis = clean(row["diagnosis_1"])

        # check for missing or invalid fields
        if isic_id is None:
            reasons.append("missing_isic_id")

        elif isic_id in duplicate_ids:
            reasons.append("duplicate_metadata_id")

        if diagnosis not in CLASS_MAPPING:
            reasons.append("unsupported_diagnosis")

        if patient_id is None:
            reasons.append("missing_patient_id")

        if isic_id is not None:
            # find all sources containing the image for this ISIC ID
            matching_sources = [
                source
                for source, images in images_by_source.items()
                if isic_id in images
            ]

            if not matching_sources:
                reasons.append("missing_image")

            elif len(matching_sources) > 1:
                raise ValueError(
                    f"ISIC ID {isic_id} appears in multiple image "
                    f"sources: {sorted(matching_sources)}"
                )

            else:
                source_set = matching_sources[0]
                image_path = images_by_source[source_set][isic_id]
                mask_path = masks_by_source[source_set].get(isic_id)

                if mask_path is None:
                    reasons.append("missing_mask")

        # append to the exclusion list if any reasons were found
        if reasons:
            exclusions.append(
                ExcludedRecord(
                    metadata_row=metadata_row,
                    isic_id=isic_id,
                    reasons=tuple(reasons),
                )
            )
            continue

        # all inclusion checks have passed.
        assert isic_id is not None
        assert patient_id is not None
        assert source_set is not None
        assert image_path is not None
        assert mask_path is not None
        assert diagnosis in ("Benign", "Malignant")

        # append to record list
        records.append(
            CohortRecord(
                isic_id=isic_id,
                source_set=source_set,
                image_path=image_path,
                mask_path=mask_path,
                patient_id=patient_id,
                lesion_id=lesion_id,
                diagnosis=diagnosis,
                class_id=CLASS_MAPPING[diagnosis],
            )
        )

    return CohortResult(
        records=tuple(sorted(records, key=lambda record: record.isic_id)),
        exclusions=tuple(exclusions),
        raw_metadata_sha256=metadata_hash,
        input_row_count=len(rows),
    )


def mask_to_box(mask: NDArray[np.bool_]) -> NormalizedBox:
    """Convert a binary mask to a normalized bounding box.

    Args:
        mask (NDArray[np.bool_]): A 2D binary mask where True indicates the lesion.

    Returns:
        NormalizedBox: The bounding box in normalized coordinates
            (x_center, y_center, width, height).

    Raises:
        ValueError: If the mask is not a non-empty 2D boolean array, or has no
            foreground pixels.
    """
    if mask.dtype != np.bool_:
        raise ValueError(f"Mask must be boolean, got dtype={mask.dtype}")

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape={mask.shape}")

    if mask.size == 0:
        raise ValueError(f"Mask has a zero-sized axis, got shape={mask.shape}")

    image_height, image_width = mask.shape
    ys, xs = np.nonzero(mask)

    if xs.size == 0:
        raise ValueError("Mask has no foreground pixels")

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    # calculate normalized center coordinates
    x_center = (x_min + x_max + 1) / (2 * image_width)
    y_center = (y_min + y_max + 1) / (2 * image_height)

    # calculate normalized width and height
    width = (x_max - x_min + 1) / image_width
    height = (y_max - y_min + 1) / image_height

    return NormalizedBox(
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )


def _audit_pair(record: CohortRecord, threshold: int) -> AuditedRecord:
    """Audit one image/mask pair and derive its bounding box.

    Args:
        record (CohortRecord): The pair to audit.
        threshold (int): Mask intensity at or above which a pixel is foreground.

    Returns:
        AuditedRecord: The pair with its image size, box and file hashes.

    Raises:
        ValueError: On the first blocking problem found with the pair.
    """
    # read each file once to avoid repeated I/O
    try:
        image_bytes = record.image_path.read_bytes()
        mask_bytes = record.mask_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{record.isic_id}: could not read files: {exc}") from exc

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            # image.load()
            image_width, image_height = image.size

        with Image.open(BytesIO(mask_bytes)) as mask_image:
            mask_image.load()
            mask_mode = mask_image.mode
            mask = np.array(mask_image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        # Pillow's message for unreadable data embeds a memory address
        detail = (
            "not a recognisable image file"
            if isinstance(exc, UnidentifiedImageError)
            else str(exc)
        )
        raise ValueError(f"{record.isic_id}: could not decode files: {detail}") from exc

    # check if 8-bit greycale 2D mask
    if mask_mode != "L" or mask.ndim != 2:
        raise ValueError(
            f"{record.isic_id}: expected 2D greyscale mask, got mode={mask_mode}, "
            f"shape={mask.shape}"
        )

    # check if image and mask dimensions match
    mask_height, mask_width = mask.shape
    if (image_width, image_height) != (mask_width, mask_height):
        raise ValueError(
            f"{record.isic_id}: image is {image_width}×{image_height} but mask is "
            f"{mask_width}×{mask_height}"
        )

    # check if mask contains any foreground pixels
    foreground = mask >= threshold
    if not foreground.any():
        raise ValueError(f"{record.isic_id}: mask contains no foreground pixels")

    return AuditedRecord(
        record=record,
        image_width=image_width,
        image_height=image_height,
        box=mask_to_box(foreground),
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
    )


def audit_pairs(
    records: Sequence[CohortRecord], threshold: int = 128
) -> tuple[AuditedRecord, ...]:
    """Audit image/mask pairs for consistency and generate bounding boxes.

    Stops at the first pair that fails a check. Unexpected errors (e.g. programming
    bugs) are not caught and will propagate.

    Args:
        records (Sequence[CohortRecord]): A sequence of CohortRecord objects.
        threshold (int): Pixel intensity threshold for foreground in the mask, default
            value chosen due to interpolated 2x upscaling of the original 8-bit masks
            (default: 128).

    Returns:
        tuple[AuditedRecord, ...]: The audited records, ordered by ISIC ID.

    Raises:
        ValueError: If the threshold is not in the range [1, 255], or if a pair can't
            be read, has an unexpected mask format, has mismatched dimensions or has
            an empty mask.
    """
    if not 1 <= threshold <= 255:
        raise ValueError(f"Threshold must be in [1, 255], got threshold={threshold}")

    return tuple(
        _audit_pair(record, threshold)
        for record in sorted(records, key=lambda record: record.isic_id)
    )


def build_independence_groups(records: Sequence[AuditedRecord]) -> dict[str, str]:
    """Group records that must stay together in a single split.

    A group is all of one patient's records, identified by the smallest ISIC ID
    among them. Lesion IDs are not used to link patients: a lesion ID recorded
    under more than one patient is contradictory, so it raises for review.

    Args:
        records (Sequence[AuditedRecord]): Audited records to group.

    Returns:
        dict[str, str]: A group ID for every ISIC ID.

    Raises:
        ValueError: If an ISIC ID appears more than once, or a lesion ID is recorded
            under more than one patient.
    """
    group_by_id: dict[str, str] = {}
    first_id_by_patient: dict[str, str] = {}
    patients_by_lesion: dict[str, set[str]] = {}

    # in ISIC ID order, so a patient's first record is its smallest ID
    for audited in sorted(records, key=lambda a: a.record.isic_id):
        record = audited.record

        if record.isic_id in group_by_id:
            raise ValueError(f"Duplicate ISIC ID: {record.isic_id}")

        group_id = first_id_by_patient.setdefault(record.patient_id, record.isic_id)
        group_by_id[record.isic_id] = group_id

        if record.lesion_id is not None:
            patients_by_lesion.setdefault(record.lesion_id, set()).add(
                record.patient_id
            )

    conflicts = {
        lesion_id: sorted(patients)
        for lesion_id, patients in sorted(patients_by_lesion.items())
        if len(patients) > 1
    }
    if conflicts:
        raise ValueError(f"Lesion IDs under more than one patient: {conflicts}")

    return group_by_id


def make_grouped_splits(
    records: Sequence[AuditedRecord],
    groups: Mapping[str, str],
    seed: int,
    ratios: Mapping[Split, float] = DEFAULT_SPLIT_RATIOS,
) -> SplitManifest:
    """Assign whole independence groups to train, val and test splits.

    Groups are placed largest first, each into the split whose image, benign and
    malignant counts it brings closest to their targets. Groups are never divided,
    so group separation takes precedence over exact ratios. Model performance is
    never used.

    Args:
        records (Sequence[AuditedRecord]): Audited records to split.
        groups (Mapping[str, str]): Group ID for every ISIC ID.
        seed (int): Seed for tie-breaking between equally sized groups.
        ratios (Mapping[Split, float]): Target fraction of images per split, which
            must sum to 1 (default: 70/15/15).

    Returns:
        SplitManifest: One split assignment per record, ordered by ISIC ID.

    Raises:
        ValueError: If the ratios are invalid, an ISIC ID is duplicated or has no
            group, the records lack a class, or a split ends up without both classes.
    """
    if set(ratios) != set(SPLITS) or any(r <= 0 for r in ratios.values()):
        raise ValueError(f"Ratios must be positive and cover {SPLITS}: {dict(ratios)}")

    if abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError(f"Ratios must sum to 1, got {sum(ratios.values())}")

    ids = [audited.record.isic_id for audited in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ISIC IDs in records")

    if missing := sorted(set(ids) - set(groups)):
        raise ValueError(f"Records without a group: {missing[:5]}")

    # count images, benign and malignant records per group
    group_counts: dict[str, np.ndarray] = {}
    for audited in records:
        counts = group_counts.setdefault(groups[audited.record.isic_id], np.zeros(3))
        counts += (1, audited.record.class_id == 0, audited.record.class_id == 1)

    # check that both classes are present in the dataset
    # creates empty totals array first to avoid dtype issues
    totals = sum(group_counts.values(), start=np.zeros(3))
    if totals[1] == 0 or totals[2] == 0:
        raise ValueError(f"Records must include both classes, got {totals[1:]}")
    targets = {split: ratios[split] * totals for split in SPLITS}
    current = {split: np.zeros(3) for split in SPLITS}

    # relative squared deviation from target counts for a split
    def deviation(split: Split, counts: np.ndarray) -> float:
        # squared deviation from target, relative so the rarer malignant class counts
        # as much as the images total
        return float((((counts - targets[split]) / targets[split]) ** 2).sum())

    # largest groups first. ties are broken by the seed
    order = sorted(group_counts)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda group_id: -group_counts[group_id][0])

    split_by_group: dict[str, Split] = {}
    for group_id in order:
        # compare how much each split's deviation would change if this group is added
        split_by_group[group_id] = min(
            SPLITS,
            key=lambda split: (
                deviation(split, current[split] + group_counts[group_id])
                - deviation(split, current[split])
            ),
        )
        current[split_by_group[group_id]] += group_counts[group_id]

    # ensure that each split has at least one record from each class
    for split in SPLITS:
        if current[split][1] == 0 or current[split][2] == 0:
            raise ValueError(f"Split '{split}' is missing a class: {current[split]}")

    return SplitManifest(
        records=tuple(
            SplitRecord(
                audited=audited,
                group_id=groups[audited.record.isic_id],
                split=split_by_group[groups[audited.record.isic_id]],
            )
            for audited in sorted(records, key=lambda a: a.record.isic_id)
        ),
    )


def validate_split_manifest(
    manifest: SplitManifest,
    expected_records: Sequence[AuditedRecord],
) -> tuple[str, ...]:
    """Check a split manifest against the audited cohort it should cover.

    The expected records are a separate input because a manifest cannot prove its
    own completeness. Every violation is collected, not just the first.

    Args:
        manifest (SplitManifest): The split assignment to check.
        expected_records (Sequence[AuditedRecord]): The audited cohort, which the
            manifest must cover exactly once each and without altering.

    Returns:
        tuple[str, ...]: A description of each violation. Empty if the manifest is valid.

    Raises:
        ValueError: If the expected records contain a duplicate ISIC ID.
    """
    expected = {audited.record.isic_id: audited for audited in expected_records}
    if len(expected) != len(expected_records):
        raise ValueError("Duplicate ISIC IDs in expected records")

    violations: list[str] = []

    # every expected record appears exactly once, and nothing else does
    assigned = Counter(r.audited.record.isic_id for r in manifest.records)
    for isic_id in sorted(set(expected) - set(assigned)):
        violations.append(f"{isic_id}: missing from the manifest")
    for isic_id in sorted(set(assigned) - set(expected)):
        violations.append(f"{isic_id}: not in the audited cohort")
    for isic_id, times in sorted(assigned.items()):
        if times > 1:
            violations.append(f"{isic_id}: assigned {times} times")

    for entry in manifest.records:
        if entry.split not in SPLITS:
            violations.append(
                f"{entry.audited.record.isic_id}: invalid split '{entry.split}'"
            )

        isic_id = entry.audited.record.isic_id

        # the manifest must not alter paths, class, hashes or box
        if isic_id in expected and entry.audited != expected[isic_id]:
            violations.append(f"{isic_id}: differs from its audited record")

    def check_not_shared(kind: str, keyed: Iterable[tuple[str, Split]]) -> None:
        splits_by_key: dict[str, set[Split]] = {}
        for key, split in keyed:
            splits_by_key.setdefault(key, set()).add(split)
        for key, splits in sorted(splits_by_key.items()):
            if len(splits) > 1:
                violations.append(f"{kind} {key} appears in splits {sorted(splits)}")

    # nothing that links two records may be split between them
    check_not_shared(
        "patient", ((e.audited.record.patient_id, e.split) for e in manifest.records)
    )
    check_not_shared(
        "lesion",
        (
            (e.audited.record.lesion_id, e.split)
            for e in manifest.records
            if e.audited.record.lesion_id is not None
        ),
    )
    check_not_shared("group", ((e.group_id, e.split) for e in manifest.records))
    check_not_shared(
        "image hash", ((e.audited.image_sha256, e.split) for e in manifest.records)
    )

    for split in SPLITS:
        classes = {
            e.audited.record.class_id for e in manifest.records if e.split == split
        }
        if classes != {0, 1}:
            violations.append(
                f"split {split} is missing a class: has {sorted(classes)}"
            )

    return tuple(violations)


def export_yolo_dataset(manifest: SplitManifest, destination: Path) -> Path:
    """Export a split manifest as a YOLO detection dataset.

    Images are symlinked, not copied, into `images/<split>/` and each gets a label
    file in `labels/<split>/`, since YOLO finds a label by swapping the last
    `/images/` in the image path for `/labels/`. Nothing is duplicated or
    oversampled. The symlinks point at this machine's source images, so generate
    the export where it will be used.

    Args:
        manifest (SplitManifest): The split assignment to export. The caller should
            also have checked it covers the audited cohort, which the manifest
            cannot prove itself.
        destination (Path): Directory to create the export in. Must be new or empty.

    Returns:
        Path: The dataset YAML. It has no `path:` key, so YOLO resolves the splits
            from the YAML file's own folder and the export can be moved.

    Raises:
        ValueError: If the manifest fails validation against itself.
        FileExistsError: If the destination already holds files.
    """
    if violations := validate_split_manifest(
        manifest, [e.audited for e in manifest.records]
    ):
        raise ValueError(f"Split manifest is invalid: {violations[:3]}")

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Export destination is not empty: {destination}")

    for entry in manifest.records:
        record, box = entry.audited.record, entry.audited.box

        # create the split directories for images and labels
        image_dir = destination / "images" / entry.split
        label_dir = destination / "labels" / entry.split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        # create a symlink to the image in the split's image directory
        image_link = image_dir / f"{record.isic_id}{record.image_path.suffix}"
        image_link.symlink_to(record.image_path.resolve())

        # YOLO label row: class_id x_center y_center width height
        (label_dir / f"{record.isic_id}.txt").write_text(
            f"{record.class_id} {box.x_center:.6f} {box.y_center:.6f} "
            f"{box.width:.6f} {box.height:.6f}\n"
        )

    names = "\n".join(
        f"  {class_id}: {diagnosis}"
        for diagnosis, class_id in sorted(CLASS_MAPPING.items(), key=lambda i: i[1])
    )
    yaml_path = destination / "dataset.yaml"
    yaml_path.write_text(
        f"train: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n"
    )
    return yaml_path


# --- Training loader and sampling ---------------------------------------------


def read_train_classes(manifest_csv: Path) -> dict[str, int]:
    """Read the class of every training image from a saved split manifest.

    Only the `isic_id`, `split` and `class_id` columns are used, so the manifest file
    is enough: no image or mask paths are needed. Validation and test images are
    left out, which keeps them out of any sampler built from the result.

    Args:
        manifest_csv (Path): The `split_manifest.csv` written by the audit.

    Returns:
        dict[str, int]: ISIC ID to class ID for each training image.

    Raises:
        ValueError: If the manifest has no training images or a class ID is not 0 or 1.
    """
    with manifest_csv.open(newline="") as file:
        classes = {
            row["isic_id"]: int(row["class_id"])
            for row in csv.DictReader(file)
            if row["split"] == "train"
        }

    if not classes:
        raise ValueError(f"No training images in {manifest_csv}")

    if not set(classes.values()) <= set(CLASS_MAPPING.values()):
        raise ValueError(f"Unexpected class IDs in {manifest_csv}")

    return classes


def build_sampling_weights(
    class_ids: Sequence[int], target_fraction: float | None
) -> NDArray[np.float64]:
    """Per-image draw weights giving the requested expected malignant fraction.

    The malignant images share `target_fraction` of the probability equally and the
    benign images share the rest, so the expected malignant share of each epoch's draws
    is the target.

    Args:
        class_ids (Sequence[int]): The class of each training image, in the order the
            weights should be returned.
        target_fraction (float | None): Expected malignant share of each epoch, or None
            for natural sampling (uniform weights).

    Returns:
        NDArray[np.float64]: One weight per image, summing to 1.

    Raises:
        ValueError: If there are no images, the target is outside (0, 1), or one class
            is missing when a target is set.
    """
    if not class_ids:
        raise ValueError("Cannot build sampling weights for no images.")

    # if no target fraction is specified, return uniform weights (1/n)
    if target_fraction is None:
        return np.full(len(class_ids), 1 / len(class_ids), dtype=np.float64)

    if not 0 < target_fraction < 1:
        raise ValueError(f"Target fraction must be in (0, 1), got {target_fraction}")

    # check that both classes are present
    malignant = np.array([c == CLASS_MAPPING["Malignant"] for c in class_ids])
    if not malignant.any() or malignant.all():
        raise ValueError("Both classes are needed to sample to a target fraction.")

    # compute weights so the expected malignant fraction is the target
    weights = np.empty(len(class_ids), dtype=np.float64)
    weights[malignant] = target_fraction / malignant.sum()
    weights[~malignant] = (1 - target_fraction) / (~malignant).sum()

    return weights


class EpochSampler:
    """Fixed-length weighted sampler reseeded each epoch from the seed and epoch.

    Epoch length stays equal to the training split size so every condition sees the same
    number of optimizer steps. Draws are with replacement. Every iteration is the next
    epoch, counting from zero, so the draws depend only on the seed and how many epochs
    came before. Ultralytics never calls `set_epoch` on a single GPU and its loader
    prefetches indices for the following epoch early, so the sampler tracks the epoch
    itself instead of being told.

    Args:
        weights (NDArray[np.float64]): Draw probability of each image, summing to 1.
        length (int): Number of indices drawn per epoch.
        seed (int): Base seed, combined with the epoch for each epoch's draws.

    Attributes:
        history (list[NDArray[np.int64]]): The indices drawn in each epoch so far, so a
            run can report what it actually trained on.

    Raises:
        ValueError: If the weights are not a 1D non-negative array summing to 1, or the
            length is not positive.
    """

    def __init__(self, weights: NDArray[np.float64], length: int, seed: int) -> None:
        if weights.ndim != 1 or (weights < 0).any() or not np.isclose(weights.sum(), 1):
            raise ValueError("Weights must be a 1D non-negative array summing to 1.")
        if length < 1:
            raise ValueError(f"Length must be positive, got {length}")

        self._weights = weights.copy()
        self._length = length
        self._seed = seed
        self._epoch = 0
        self.history: list[NDArray[np.int64]] = []

    def __iter__(self) -> Iterator[int]:
        # use a new random generator for each epoch to keep draws independent
        rng = np.random.default_rng([self._seed, self._epoch])
        # iterate over draws for this epoch then increment epoch counter for next call
        self._epoch += 1
        draws = rng.choice(len(self._weights), size=self._length, p=self._weights)
        self.history.append(draws)

        return iter(draws.tolist())

    def __len__(self) -> int:
        return self._length
