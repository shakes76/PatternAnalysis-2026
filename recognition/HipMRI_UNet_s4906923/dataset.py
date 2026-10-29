"""Load paired HipMRI scans, check their geometry and sample training patches.

Patients stay in one split throughout the experiment. Images and masks share
the same spatial transformations. Arrays use XYZ voxel order until conversion
to channel/depth/height/width for the model.
Source MRI and mask files are never modified.
"""
from __future__ import annotations
# Patient-level file discovery. Listing files does not load MRI or mask arrays.


import json
import re
from dataclasses import dataclass
from pathlib import Path


_NAME = re.compile(r"^(?P<patient>[A-Z][0-9]{3})_Week(?P<week>[0-9]+)_(?P<kind>LFOV|SEMANTIC)\.nii\.gz$")
DEFAULT_SPLIT = Path(__file__).resolve().parent / "config" / "patient_split.json"


@dataclass(frozen=True)
class Case:
    patient: str
    week: int
    split: str
    mri: Path
    mask: Path

    @property
    def identifier(self) -> str:
        return f"{self.patient}_Week{self.week}"


def read_split(path: str | Path = DEFAULT_SPLIT) -> dict:
    """Read the fixed patient lists and reject overlap or unexpected cohort sizes."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = ("train", "validation", "test")
    expected = (26, 6, 6)
    seen: set[str] = set()
    for group, count in zip(groups, expected):
        names = record[group]
        if len(names) != count or len(set(names)) != count:
            raise ValueError(f"{group} must contain {count} distinct patients")
        if any(re.fullmatch(r"[A-Z][0-9]{3}", name) is None for name in names):
            raise ValueError(f"Invalid patient ID in {group}")
        if seen.intersection(names):
            raise ValueError(f"Patient appears in multiple splits: {seen.intersection(names)}")
        seen.update(names)
    if len(seen) != 38 or record["expected_volumes"] != {
        "train": 143, "validation": 38, "test": 30
    }:
        raise ValueError("Approved cohort counts changed")
    return record


def discover(root: str | Path, split_path: str | Path = DEFAULT_SPLIT) -> dict[str, tuple[Case, ...]]:
    """Pair all 211 MRI/mask cases by patient and visit, then assign their splits."""
    root = Path(root)
    record = read_split(split_path)
    reverse = {name: group for group in ("train", "validation", "test") for name in record[group]}
    indexed: dict[str, dict[tuple[str, int], Path]] = {}
    for role, subdir, suffix in (
        ("mri", "semantic_MRs", "LFOV"),
        ("mask", "semantic_labels_only", "SEMANTIC"),
    ):
        folder = root / subdir
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        matches: dict[tuple[str, int], Path] = {}
        for file in folder.glob("*.nii.gz"):
            match = _NAME.fullmatch(file.name)
            if match is None or match["kind"] != suffix:
                raise ValueError(f"Unexpected {role} file: {file.name}")
            key = (match["patient"], int(match["week"]))
            if key in matches:
                raise ValueError(f"Duplicate {role} case: {key}")
            matches[key] = file
        indexed[role] = matches
    if indexed["mri"].keys() != indexed["mask"].keys():
        raise ValueError("MRI/mask case sets differ")
    cases: dict[str, list[Case]] = {group: [] for group in ("train", "validation", "test")}
    for patient, week in sorted(indexed["mri"]):
        if patient not in reverse:
            raise ValueError(f"Patient is outside approved split: {patient}")
        group = reverse[patient]
        cases[group].append(Case(patient, week, group, indexed["mri"][(patient, week)], indexed["mask"][(patient, week)]))
    for group, expected_count in record["expected_volumes"].items():
        if len(cases[group]) != expected_count:
            raise ValueError(f"{group} has {len(cases[group])}, expected {expected_count} volumes")
    return {group: tuple(items) for group, items in cases.items()}

# Paired geometry-aware NIfTI handling and shared downsampling.
#
# All arrays use NIfTI voxel order (X, Y, Z) until the model input adapter
# explicitly transposes them. This module never modifies source files.


from dataclasses import dataclass

import numpy as np

NUM_CLASSES = 6


@dataclass(frozen=True)
class Volume:
    case: Case
    image: np.ndarray
    mask: np.ndarray
    affine: np.ndarray
    spacing_mm: tuple[float, float, float]


def load_case(case: Case) -> Volume:
    """Decode one approved pair, refusing mismatched grids and invalid labels."""
    import nibabel as nib

    mri = nib.load(str(case.mri))
    seg = nib.load(str(case.mask))
    if mri.ndim != 3 or seg.ndim != 3 or mri.shape != seg.shape:
        raise ValueError(f"{case.identifier}: MRI/mask must share a 3D shape")
    if not np.allclose(mri.affine, seg.affine, rtol=0, atol=1e-4):
        raise ValueError(f"{case.identifier}: MRI/mask affines differ")
    if not np.allclose(mri.header.get_zooms()[:3], seg.header.get_zooms()[:3], rtol=0, atol=1e-5):
        raise ValueError(f"{case.identifier}: MRI/mask voxel spacing differs")
    image_unit = mri.header.get_xyzt_units()[0]
    mask_unit = seg.header.get_xyzt_units()[0]
    if image_unit != "mm" or mask_unit not in ("mm", "unknown"):
        raise ValueError(f"{case.identifier}: unsupported spatial units {image_unit}, {mask_unit}")
    affine = np.asarray(mri.affine, dtype=np.float64)
    if not np.isfinite(affine).all() or np.linalg.matrix_rank(affine[:3, :3]) != 3:
        raise ValueError(f"{case.identifier}: invalid affine")
    image = mri.get_fdata(dtype=np.float32, caching="unchanged")
    raw_mask = np.asarray(seg.dataobj)
    if not np.isfinite(image).all() or not np.isfinite(raw_mask).all():
        raise ValueError(f"{case.identifier}: non-finite image or mask")
    if np.any(raw_mask != np.floor(raw_mask)) or np.any((raw_mask < 0) | (raw_mask >= NUM_CLASSES)):
        raise ValueError(f"{case.identifier}: mask labels must be integer 0..5")
    return Volume(case, image, raw_mask.astype(np.uint8), affine,
                  tuple(float(v) for v in mri.header.get_zooms()[:3]))


def downsample(volume: Volume, factor: tuple[int, int, int] = (2, 2, 2)) -> Volume:
    """Sample MRI/mask at identical voxel-edge-aligned physical locations.

    Output centre j maps to source centre (j + 1/2)*factor - 1/2. The affine
    is transformed by that same map. Lost source labels trigger a failure.
    """
    from scipy.ndimage import map_coordinates

    if len(factor) != 3 or any(type(v) is not int or v < 1 for v in factor):
        raise ValueError("factor must contain three positive integers")
    source_shape = np.asarray(volume.image.shape, dtype=np.int64)
    target_shape = tuple(int(np.ceil(n / f)) for n, f in zip(source_shape, factor))
    axes = [(np.arange(size, dtype=np.float64) + 0.5) * f - 0.5
            for size, f in zip(target_shape, factor)]
    grid = np.meshgrid(*axes, indexing="ij", sparse=False)
    coords = np.asarray(grid)
    image = map_coordinates(volume.image, coords, order=1, mode="nearest").astype(np.float32)
    mask = map_coordinates(volume.mask, coords, order=0, mode="grid-constant", cval=0).astype(np.uint8)
    before_counts = np.bincount(volume.mask.ravel(), minlength=NUM_CLASSES)
    after_counts = np.bincount(mask.ravel(), minlength=NUM_CLASSES)
    lost = np.flatnonzero((before_counts > 0) & (after_counts == 0))
    if lost.size:
        raise ValueError(f"{volume.case.identifier}: downsampling removed classes {lost.tolist()}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag(factor)
    transform[:3, 3] = (np.asarray(factor, dtype=np.float64) - 1) / 2
    affine = volume.affine @ transform
    spacing = tuple(float(v * f) for v, f in zip(volume.spacing_mm, factor))
    return Volume(volume.case, image, mask, affine, spacing)


def zscore_nonzero(volume: Volume) -> Volume:
    """Normalize each MRI using image-only nonzero voxels; retain outside zeros."""
    chosen = volume.image != 0
    if not np.any(chosen):
        raise ValueError(f"{volume.case.identifier}: all-zero MRI")
    values = volume.image[chosen].astype(np.float64)
    mean, scale = float(values.mean()), float(values.std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{volume.case.identifier}: constant or invalid MRI")
    image = np.zeros_like(volume.image, dtype=np.float32)
    image[chosen] = ((values - mean) / scale).astype(np.float32)
    return Volume(volume.case, image, volume.mask, volume.affine, volume.spacing_mm)


def one_hot(mask: np.ndarray) -> np.ndarray:
    """Return channel-first six-class float32 mask in the same spatial grid."""
    if mask.ndim not in (2, 3) or mask.dtype.kind not in "iu":
        raise ValueError("mask must be a 2D/3D integer array")
    if np.any((mask < 0) | (mask >= NUM_CLASSES)):
        raise ValueError("mask contains IDs outside 0..5")
    return np.moveaxis(np.eye(NUM_CLASSES, dtype=np.float32)[mask], -1, 0)


def to_3d_model_arrays(volume: Volume) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Convert XYZ to DHW, padding only high ends to a multiple of 16.

    The original downsampled XYZ shape is returned for exact inversion after
    prediction. Padded mask voxels are background and must be excluded from
    final metrics by removing padding first.
    """
    shape_xyz = tuple(int(n) for n in volume.image.shape)
    image = np.transpose(volume.image, (2, 1, 0))
    mask = np.transpose(volume.mask, (2, 1, 0))
    padding = tuple((0, (-n) % 16) for n in image.shape)
    return (np.pad(image, padding).astype(np.float32, copy=False),
            np.pad(mask, padding).astype(np.int64, copy=False), shape_xyz)


def from_3d_model_prediction(prediction_dhw: np.ndarray, shape_xyz: tuple[int, int, int]) -> np.ndarray:
    """Remove high-end padding and return a label volume in the input XYZ grid."""
    if prediction_dhw.ndim != 3 or len(shape_xyz) != 3:
        raise ValueError("Expected DHW prediction and XYZ original shape")
    shape_dhw = shape_xyz[::-1]
    if any(size > available or size < 1 for size, available in zip(shape_dhw, prediction_dhw.shape)):
        raise ValueError("Prediction is too small for the recorded original shape")
    cropped = prediction_dhw[tuple(slice(0, size) for size in shape_dhw)]
    return np.transpose(cropped, (2, 1, 0))

# Train-only patient-balanced patch selection and measured class exposure.
#
# Coordinates remain XYZ until ``crop_for_3d_model`` converts them to DHW.
# No source MRI or mask is modified. Call the exposure recorder on the exact
# mask delivered with each model input.


import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


CLASS_NAMES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
TARGET_IDS = {"Bladder": 3, "Rectum": 4, "Prostate": 5}
PATCH_XYZ = (64, 64, 32)
UNIFORM_POLICY = ("uniform",) * 10
ORGAN_AWARE_POLICY = ("uniform",) * 4 + ("Bladder",) * 2 + ("Rectum",) * 2 + ("Prostate",) * 2


def _rng(*parts: object) -> np.random.Generator:
    key = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "little"))


def epoch_cases(cases: tuple[Case, ...], epoch: int, seed: int) -> tuple[Case, ...]:
    """Choose one visit per training patient, cycling through a shuffled visit list."""
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    grouped: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        if case.split != "train":
            raise ValueError("Sampler accepts Train cases only")
        grouped[case.patient].append(case)
    if not grouped:
        raise ValueError("No Train cases")
    selected = []
    for patient, visits in sorted(grouped.items()):
        visits.sort(key=lambda case: case.week)
        cycle, offset = divmod(epoch, len(visits))
        order = _rng(seed, patient, cycle, "visits").permutation(len(visits))
        selected.append(visits[int(order[offset])])
    order = _rng(seed, epoch, "patients").permutation(len(selected))
    return tuple(selected[int(i)] for i in order)


def crop_origin(mask: np.ndarray, patch_xyz: tuple[int, int, int],
                target: str, *, seed: int, epoch: int, case_id: str,
                draw: int) -> tuple[int, int, int]:
    """Choose a crop origin uniformly, or choose one that includes the target organ."""
    if mask.ndim != 3 or mask.dtype.kind not in "iu":
        raise ValueError("Expected integer XYZ mask")
    if len(patch_xyz) != 3 or any(size < 1 or size > n
                                  for size, n in zip(patch_xyz, mask.shape)):
        raise ValueError("Patch shape does not fit mask")
    rng = _rng(seed, epoch, case_id, draw, target)
    if target == "uniform":
        return tuple(int(rng.integers(n - size + 1))
                     for n, size in zip(mask.shape, patch_xyz))
    if target not in TARGET_IDS:
        raise ValueError(f"Unknown target: {target}")
    locations = np.argwhere(mask == TARGET_IDS[target])
    if len(locations) == 0:
        raise ValueError(f"{case_id}: target {target} absent after preprocessing")
    voxel = locations[int(rng.integers(len(locations)))]
    starts = []
    for n, size, pos in zip(mask.shape, patch_xyz, voxel):
        low = max(0, int(pos) - size + 1)
        high = min(int(pos), n - size)
        starts.append(int(rng.integers(low, high + 1)))
    return tuple(starts)


def crop_for_3d_model(volume: Volume, origin: tuple[int, int, int],
                      patch_xyz: tuple[int, int, int] = PATCH_XYZ) -> tuple[np.ndarray, np.ndarray]:
    """Crop normalized/downsampled paired arrays into C,D,H,W and D,H,W."""
    if volume.image.shape != volume.mask.shape or len(origin) != 3:
        raise ValueError("MRI/mask grid or origin mismatch")
    if any(start < 0 or size < 1 or start + size > n
           for start, size, n in zip(origin, patch_xyz, volume.image.shape)):
        raise ValueError("Crop falls outside the shared grid")
    slices = tuple(slice(start, start + size) for start, size in zip(origin, patch_xyz))
    image = np.transpose(volume.image[slices], (2, 1, 0))[None].astype(np.float32, copy=False)
    mask = np.transpose(volume.mask[slices], (2, 1, 0)).astype(np.int64, copy=False)
    if not np.isfinite(image).all() or np.any((mask < 0) | (mask >= NUM_CLASSES)):
        raise ValueError("Nonfinite MRI or illegal label in sampled patch")
    return image, mask


@dataclass
class ExposureRecorder:
    """Record class counts and organ presence in the patches actually used for training."""

    patches: int = 0

    def __post_init__(self) -> None:
        self.hits: Counter = Counter()
        self.voxels: Counter = Counter()
        self.by_patient: dict[str, Counter] = defaultdict(Counter)

    def add(self, mask_dhw: np.ndarray, patient: str) -> dict[str, int]:
        if mask_dhw.ndim != 3 or mask_dhw.dtype.kind not in "iu":
            raise ValueError("Expected integer DHW mask after all transforms")
        if np.any((mask_dhw < 0) | (mask_dhw >= NUM_CLASSES)):
            raise ValueError("Illegal class in actual patch")
        counts = np.bincount(mask_dhw.ravel(), minlength=NUM_CLASSES)
        self.patches += 1
        self.by_patient[patient]["patches"] += 1
        for index, name in enumerate(CLASS_NAMES):
            amount = int(counts[index])
            self.voxels[name] += amount
            self.hits[name] += int(amount > 0)
            self.by_patient[patient][f"{name}_hits"] += int(amount > 0)
        return {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)}

    def summary(self) -> dict:
        total = sum(self.voxels.values())
        return {
            "patches": self.patches,
            "per_class": {name: {"patch_hits": self.hits[name],
                                 "patch_hit_rate": self.hits[name] / self.patches if self.patches else None,
                                 "patch_voxels": self.voxels[name],
                                 "voxel_share": self.voxels[name] / total if total else None}
                          for name in CLASS_NAMES},
            "per_patient": {patient: dict(counts) for patient, counts in sorted(self.by_patient.items())},
        }
