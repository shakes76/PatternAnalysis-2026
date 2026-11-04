"""Opt-in, scan-consistent foreground intensity mapping and integer crop/pad.

A thresholded foreground is a conservative head proxy, not a brain mask. No
labels or cross-patient intensity reference are used. Percentiles are computed
from the supplied complete scan, as a fixed per-input inference operation.
Only automatic crop dimensions are fitted across cases, using training scans.
Sources are never edited. Crop mode never rescales anatomy or interpolates.

Pillow histogram, point and crop API:
https://pillow.readthedocs.io/en/stable/reference/Image.html
"""

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


PREPROCESSING_NAMES = ("none", "scan_intensity", "scan_crop", "scan_intensity_crop")
LEGACY_NORMALIZATION = "(grayscale_uint8 / 255 - 0.5) / 0.5"
SCAN_NORMALIZATION = "(scan_preprocessed_grayscale_uint8 / 255 - 0.5) / 0.5"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Serialize the exact label-free algorithm and resolved crop dimensions."""

    name: str = "none"
    foreground_threshold: int = 16
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    crop_margin: int = 8
    crop_height: int = 0
    crop_width: int = 0

    def __post_init__(self) -> None:
        """Reject ambiguous dimensions, invalid thresholds and nonfinite settings."""
        if self.name not in PREPROCESSING_NAMES:
            raise ValueError("Unsupported preprocessing profile.")
        if type(self.foreground_threshold) is not int or not 0 <= self.foreground_threshold < 255:
            raise ValueError("Foreground threshold must be an integer in [0, 254].")
        if any(type(v) not in (int, float) or not math.isfinite(v)
               for v in (self.lower_percentile, self.upper_percentile)):
            raise ValueError("Foreground percentiles must be finite numbers.")
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("Foreground percentiles must satisfy 0 <= lower < upper <= 100.")
        if any(type(v) is not int or v < 0 for v in (self.crop_margin, self.crop_height, self.crop_width)):
            raise ValueError("Crop margin/dimensions must be nonnegative integers.")
        if bool(self.crop_height) != bool(self.crop_width):
            raise ValueError("Specify both crop dimensions, or leave both zero for training-only fitting.")
        if not self.crops and (self.crop_height or self.crop_width):
            raise ValueError("Explicit crop dimensions require a crop preprocessing profile.")
        object.__setattr__(self, "lower_percentile", float(self.lower_percentile))
        object.__setattr__(self, "upper_percentile", float(self.upper_percentile))

    @property
    def crops(self) -> bool:
        """Whether geometry uses a scan-union bounding box and fixed crop window."""
        return self.name in ("scan_crop", "scan_intensity_crop")

    @property
    def maps_intensity(self) -> bool:
        """Whether the scan's foreground histogram determines the brightness map."""
        return self.name in ("scan_intensity", "scan_intensity_crop")

    def to_dict(self) -> dict[str, Any]:
        """Record constants and algorithm semantics, not just a short profile name."""
        return {**asdict(self), "algorithm": "scan_foreground_percentile_crop_v1",
                "foreground": "uint8_greater_than_threshold_head_proxy_not_brain_mask",
                "statistics_scope": "current_complete_scan_only_no_labels_no_population_fit",
                "percentile": "nearest_rank_count_weighted_native_pixels",
                "intensity_mapping": "clipped_linear_round_uint8_background_zero",
                "degenerate_intensity": "identity_foreground_background_zero",
                "geometry": "native_scan_union_bbox_integer_center_crop_zero_pad_no_rescale",
                "crop_dimension_fit": "training_only_max_union_extent_plus_margin_rounded_to_32",
                "application": "same_deterministic_rule_all_roles_before_train_augmentation"}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PreprocessingConfig":
        """Reject unknown or changed algorithm versions and malformed settings."""
        if type(value) is not dict:
            raise ValueError("Preprocessing configuration must be a dictionary.")
        fields = cls.__dataclass_fields__
        if any(key not in value for key in fields):
            raise ValueError("Missing preprocessing configuration fields.")
        config = cls(**{key: value[key] for key in fields})
        if set(value) != set(config.to_dict()) or value != config.to_dict():
            raise ValueError("Unsupported preprocessing algorithm or serialized settings.")
        return config


def add_preprocessing_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose four isolated controls without changing historical defaults."""
    parser.add_argument("--preprocessing", choices=PREPROCESSING_NAMES, default="none")
    parser.add_argument("--foreground-threshold", type=int, default=16)
    parser.add_argument("--intensity-lower-percentile", type=float, default=1.0)
    parser.add_argument("--intensity-upper-percentile", type=float, default=99.0)
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--crop-height", type=int, default=0,
                        help="Fixed native-pixel crop height; zero fits from training scans only.")
    parser.add_argument("--crop-width", type=int, default=0,
                        help="Fixed native-pixel crop width; zero fits from training scans only.")


def preprocessing_from_args(args: argparse.Namespace) -> PreprocessingConfig:
    """Build the configuration from the parsed command-line options."""
    return PreprocessingConfig(
        name=args.preprocessing, foreground_threshold=args.foreground_threshold,
        lower_percentile=args.intensity_lower_percentile, upper_percentile=args.intensity_upper_percentile,
        crop_margin=args.crop_margin, crop_height=args.crop_height, crop_width=args.crop_width)


def checkpoint_preprocessing(config: dict[str, Any]) -> PreprocessingConfig:
    """Rebuild the exact preprocessing saved with a checkpoint."""
    saved = config.get("preprocessing_config")
    preprocessing = PreprocessingConfig() if saved is None else PreprocessingConfig.from_dict(saved)
    if preprocessing.crops and list(config["image_size"]) != [preprocessing.crop_height, preprocessing.crop_width]:
        raise ValueError("Checkpoint crop dimensions disagree with its input shape.")
    return preprocessing


def canonical_image(row: dict[str, Any], data_root: Path) -> Image.Image:
    """Read a verified source inside the root without changing its bytes."""
    root = Path(data_root).resolve()
    relative = Path(row["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Preprocessing source path must stay inside the data directory.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Preprocessing source resolves outside the data directory.")
    content = path.read_bytes()
    if row.get("file_sha256") and hashlib.sha256(content).hexdigest() != row["file_sha256"]:
        raise ValueError("Source image changed before preprocessing.")
    with Image.open(io.BytesIO(content)) as source:
        return ImageOps.exif_transpose(source).convert("L")


def foreground_bbox(image: Image.Image, threshold: int) -> tuple[int, int, int, int] | None:
    """Threshold all pixels conservatively; do not claim skull stripping."""
    return image.point([0 if x <= threshold else 255 for x in range(256)]).getbbox()


def histogram_percentile(histogram: list[int], percentile: float) -> int:
    """Return a deterministic nearest-rank quantile without floating interpolation."""
    count = sum(histogram)
    if not count:
        return 0
    rank = max(1, math.ceil(count * percentile / 100))
    cumulative = 0
    for value, frequency in enumerate(histogram):
        cumulative += frequency
        if cumulative >= rank:
            return value
    raise ValueError("Invalid foreground histogram.")


def prepare_scans(rows: list[dict[str, Any]], data_root: Path,
                  preprocessing: PreprocessingConfig) -> dict[str, dict[str, Any]]:
    """Read every supplied slice once; keep scans, patients and source bindings separate."""
    if not rows:
        raise ValueError("Preprocessing requires nonempty scan rows.")
    scans: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: r["relative_path"]):
        image = canonical_image(row, data_root)
        scan = scans.setdefault(row["image_id"], {
            "image_id": row["image_id"], "patient_id": row["patient_id"],
            "native_size": list(image.size), "bbox": None, "histogram": [0] * 256,
            "source_files": {}, "slice_indices": []})
        if scan["patient_id"] != row["patient_id"] or scan["native_size"] != list(image.size):
            raise ValueError("A scan must have one owner and one canonical native image size.")
        if row["relative_path"] in scan["source_files"] or int(row["slice_index"]) in scan["slice_indices"]:
            raise ValueError("Duplicate preprocessing source or scan slice index.")
        scan["source_files"][row["relative_path"]] = row.get("file_sha256")
        scan["slice_indices"].append(int(row["slice_index"]))
        bbox = foreground_bbox(image, preprocessing.foreground_threshold)
        if bbox is not None:
            old = scan["bbox"]
            scan["bbox"] = list(bbox) if old is None else [min(old[0], bbox[0]), min(old[1], bbox[1]),
                                                         max(old[2], bbox[2]), max(old[3], bbox[3])]
        histogram = image.histogram()
        for value in range(preprocessing.foreground_threshold + 1, 256):
            scan["histogram"][value] += histogram[value]
    for scan in scans.values():
        scan["statistics_config"] = {
            "foreground_threshold": preprocessing.foreground_threshold,
            "lower_percentile": preprocessing.lower_percentile,
            "upper_percentile": preprocessing.upper_percentile}
        scan["foreground_pixels"] = sum(scan["histogram"])
        scan["intensity_low"] = histogram_percentile(scan["histogram"], preprocessing.lower_percentile)
        scan["intensity_high"] = histogram_percentile(scan["histogram"], preprocessing.upper_percentile)
        scan["intensity_status"] = ("empty_foreground_identity" if not scan["foreground_pixels"] else
                                     "constant_foreground_identity" if scan["intensity_high"] <= scan["intensity_low"] else
                                     "scan_percentiles_available")
        scan["source_binding_sha256"] = hashlib.sha256(json.dumps(
            scan["source_files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return scans


def fit_training_crop(scans: dict[str, dict[str, Any]], preprocessing: PreprocessingConfig,
                      *, role: str, minimum_size: int = 32) -> tuple[PreprocessingConfig, dict[str, Any]]:
    """Freeze automatic geometry using training foreground extents, never held-out cases."""
    if role != "train":
        raise ValueError("Automatic crop dimensions may be fitted on training scans only.")
    if not scans:
        raise ValueError("Cannot fit crop geometry without training scans.")
    if not preprocessing.crops:
        return preprocessing, {"status": "not_required"}
    extents = [(s["bbox"][3] - s["bbox"][1], s["bbox"][2] - s["bbox"][0])
               if s["bbox"] is not None else (s["native_size"][1], s["native_size"][0])
               for s in scans.values()]
    required = [max(e[i] for e in extents) + 2 * preprocessing.crop_margin for i in (0, 1)]
    explicit = bool(preprocessing.crop_height)
    shape = [preprocessing.crop_height, preprocessing.crop_width] if explicit else [
        32 * math.ceil(max(n, minimum_size) / 32) for n in required]
    if any(n < max(needed, minimum_size) for n, needed in zip(shape, required)):
        raise ValueError(f"Crop window cannot contain training foreground and margins: requires at least {required}.")
    fitted = replace(preprocessing, crop_height=shape[0], crop_width=shape[1])
    return fitted, {"status": "explicit_checked_on_train" if explicit else "fitted_on_train",
                    "fit_role": role, "training_scans": len(scans), "required_height_width": required,
                    "resolved_height_width": shape, "uses_labels": False}


def crop_box(scan: dict[str, Any], preprocessing: PreprocessingConfig) -> tuple[int, int, int, int]:
    """Compute one integer crop per scan and refuse foreground/margin clipping."""
    height, width = preprocessing.crop_height, preprocessing.crop_width
    if not height or not width:
        raise ValueError("Crop dimensions must be fitted before constructing loaders.")
    bbox = scan["bbox"]
    center_x = scan["native_size"][0] // 2 if bbox is None else (bbox[0] + bbox[2]) // 2
    center_y = scan["native_size"][1] // 2 if bbox is None else (bbox[1] + bbox[3]) // 2
    left, top = center_x - width // 2, center_y - height // 2
    box = (left, top, left + width, top + height)
    if bbox is not None:
        margin = preprocessing.crop_margin
        if (box[0] > bbox[0] - margin or box[1] > bbox[1] - margin
                or box[2] < bbox[2] + margin or box[3] < bbox[3] + margin):
            raise ValueError(f"Scan {scan['image_id']} does not fit the frozen crop window. "
                             "No foreground was cropped; declare a larger window in a fresh experiment.")
    return box


def validate_scan_parameters(rows: list[dict[str, Any]], scans: dict[str, dict[str, Any]],
                             preprocessing: PreprocessingConfig) -> None:
    """Allow failure-figure subsets only with full-scan parameters bound to their sources."""
    for row in rows:
        scan = scans.get(row["image_id"])
        if (scan is None or scan["patient_id"] != row["patient_id"]
                or row["relative_path"] not in scan["source_files"]
                or scan["source_files"][row["relative_path"]] != row.get("file_sha256")):
            raise ValueError("Preprocessing parameters do not match the supplied scan/source identities.")
        if scan.get("statistics_config") != {
                "foreground_threshold": preprocessing.foreground_threshold,
                "lower_percentile": preprocessing.lower_percentile,
                "upper_percentile": preprocessing.upper_percentile}:
            raise ValueError("Preprocessing scan statistics were prepared with different settings.")
        if preprocessing.crops:
            crop_box(scan, preprocessing)


def apply_preprocessing(image: Image.Image, scan: dict[str, Any],
                        preprocessing: PreprocessingConfig, image_size: tuple[int, int]) -> Image.Image:
    """Apply the fixed scan brightness map, then crop/pad or historical resize."""
    if list(image.size) != scan["native_size"]:
        raise ValueError("Source dimensions changed after scan preprocessing preparation.")
    if preprocessing.maps_intensity:
        low, high = scan["intensity_low"], scan["intensity_high"]
        lut = [0 if value <= preprocessing.foreground_threshold else
               value if high <= low else round(255 * max(0.0, min(1.0, (value - low) / (high - low))))
               for value in range(256)]
        image = image.point(lut)
    if preprocessing.crops:
        if image_size != (preprocessing.crop_height, preprocessing.crop_width):
            raise ValueError("Loader image dimensions disagree with the frozen crop window.")
        return image.crop(crop_box(scan, preprocessing))
    target = (image_size[1], image_size[0])
    return image.resize(target, Image.Resampling.BILINEAR) if image.size != target else image
