"""Retrospective source-input quality evidence; no model execution or data edits.

All 211 pairs receive header-only inspection. Pixel inspection is restricted to
explicitly named cases from an allowed 14-case set; TEST pixels are restricted
to M015 with explicit acknowledgement. The two additional TRAIN controls were
chosen after initial contrast summaries and are exploratory. Reconstructing the declared
normalization/support policy is NOT an independent preprocessing validation.
NIfTI free-text fields are never exported, only hashes and whitelisted hints.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.cache_identity import (arrays_identity, atomic_json, file_sha256, full_manifest,
                                 load_verified_book, read_arrays, safe_relative, validate_arrays)

EXPECTED_MANIFEST = "c47fe880d353629348786b852e3f699291af1a4d5628b2980bf42a74635d0424"
EXPECTED_IDENTITY = "eacb84bf20626a6ec9d7f6d36489a81e2d4e4af9fbe4ade1ce8f508a2022e08a"
EXPECTED_SOURCEBOOK = "a3cf4ed4eb21c9d79df9a50b53308df797a30cb112655f4ee85d809ad7916e54"
ALLOWED_CASES = {f"M015/Week{i}" for i in range(8)} | {
    "B006/Week0", "K019/Week1", "B040/Week0", "K018/Week0",
    "R016/Week7", "T014/Week3"}
EXPLORATORY_CONTROLS = {"R016/Week7", "T014/Week3"}
QUANTILES = (.01, .05, .25, .50, .75, .95, .99)


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def matrix_json(matrix):
    return None if matrix is None else [[finite_float(v) for v in row] for row in np.asarray(matrix)]


def describe(values, clip_bounds=None):
    """Population statistics; empty ROIs stay null, not artificial zero scores."""
    values = np.asarray(values).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite intensities cannot be summarized.")
    result = {"n": int(values.size), "mean": None, "std": None,
              "quantiles": {f"q{round(q * 100):02d}": None for q in QUANTILES},
              "zero_count": 0, "zero_fraction": None, "clip_bounds": clip_bounds,
              "at_or_below_lower_count": None, "at_or_above_upper_count": None,
              "at_or_beyond_clip_fraction": None}
    if not values.size:
        return result
    result.update(mean=float(np.mean(values, dtype=np.float64)),
                  std=float(np.std(values, dtype=np.float64)),
                  quantiles={f"q{round(q * 100):02d}": float(v)
                             for q, v in zip(QUANTILES, np.quantile(values, QUANTILES, method="linear"))},
                  zero_count=int(np.count_nonzero(values == 0)))
    result["zero_fraction"] = result["zero_count"] / result["n"]
    if clip_bounds is not None:
        lower, upper = clip_bounds
        low, high = int(np.count_nonzero(values <= lower)), int(np.count_nonzero(values >= upper))
        result.update(at_or_below_lower_count=low, at_or_above_upper_count=high,
                      at_or_beyond_clip_fraction=(low + high) / result["n"])
    return result


def class_statistics(image, mask, clip_bounds=None):
    if image.shape != mask.shape:
        raise ValueError("Intensity and class-index grids differ.")
    return {str(label): describe(image[mask == label], clip_bounds) for label in range(6)}


def class_contrasts(statistics):
    """Descriptive within-scan median contrasts, with no diagnostic threshold."""
    body = statistics["1"]["quantiles"]
    median = body["q50"]
    iqr = None if median is None else body["q75"] - body["q25"]
    result = {"body_iqr": iqr}
    for label, name in (("3", "bladder"), ("5", "prostate")):
        organ = statistics[label]["quantiles"]["q50"]
        gap = organ - median if organ is not None and median is not None else None
        result[f"{name}_minus_body_median"] = gap
        result[f"{name}_minus_body_median_div_body_iqr"] = gap / iqr if gap is not None and iqr else None
    return result


def normalize_nonzero(image):
    """Reproduce the declared policy for attribution of its information effects."""
    image = np.asarray(image)
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite native image.")
    support = image != 0
    mean = float(np.mean(image[support], dtype=np.float64)) if support.any() else 0.0
    std = float(np.std(image[support], dtype=np.float64)) if support.any() else 0.0
    normalized = np.zeros(image.shape, dtype=np.float32)
    if support.any():
        normalized[support] = np.clip((image[support].astype(np.float64) - mean) / max(std, 1e-8), -5, 5)
    return normalized, support, {"nonzero_count": int(support.sum()), "mean": mean, "std": std,
                                 "degenerate_nonzero_std": std < 1e-8}


def map_labels(mask):
    mask = np.asarray(mask)
    if (not np.isfinite(mask).all() or np.any(mask != np.rint(mask))
            or not set(np.unique(mask).tolist()).issubset(set(range(6)))):
        raise ValueError("Native mask must contain only finite integer IDs 0..5.")
    return mask.astype(np.uint8)


def support_effect(before, support, mask):
    if before.shape != support.shape or before.shape != mask.shape:
        raise ValueError("Support effect grids differ.")
    support = np.asarray(support, dtype=bool)
    after = np.array(before, copy=True)
    after[~support] = 0
    target = mask == 5
    changed = before != after
    return after, {"target_voxels": int(target.sum()),
                   "target_outside_resampled_nonzero_support": int((target & ~support).sum()),
                   "forced_zero_nonzero_voxels_all": int(changed.sum()),
                   "forced_zero_nonzero_voxels_target": int((changed & target).sum()),
                   "target_before_forcing": describe(before[target], [-5, 5]),
                   "target_after_forcing": describe(after[target], [-5, 5])}


def select_cases(records, names, acknowledge_test=False):
    if not names or len(set(names)) != len(names):
        raise ValueError("Supply distinct explicit --case PATIENT/VISIT values.")
    indexed = {f"{r['patient']}/{r['visit']}": r for r in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate patient/visit in manifest.")
    selected = []
    for name in names:
        if name not in ALLOWED_CASES or name not in indexed:
            raise ValueError(f"Case is outside the allowed 14-case diagnostic set: {name}")
        record = indexed[name]
        if name in EXPLORATORY_CONTROLS and record["split"] != "train":
            raise ValueError("Exploratory contrast controls must retain their TRAIN assignments.")
        if record["split"] == "test" and (record["patient"] != "M015" or not acknowledge_test):
            raise ValueError("TEST pixels require M015 and --ack-retrospective-m015-test.")
        if record["patient"] == "M015" and record["split"] != "test":
            raise ValueError("M015 must retain its frozen TEST assignment.")
        selected.append((name, record))
    return selected


def text_field_evidence(header, field):
    raw = np.asarray(header[field]).tobytes()
    text = raw.decode("utf-8", errors="replace").upper()
    patterns = {name: rf"(?<![A-Z0-9]){token}(?![A-Z0-9])" for name, token in
                (("T1", "T1W?"), ("T2", "T2W?"), ("DWI", "DWI"),
                 ("ADC", "ADC"), ("STIR", "STIR"), ("FLAIR", "FLAIR"))}
    return {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw),
            "sequence_hints": [name for name, pattern in patterns.items() if re.search(pattern, text)],
            "free_text_exported": False}


def inspect_header(path, frozen, nib):
    volume = nib.load(str(path))  # Header/proxy access only; no array conversion.
    qform, qcode = volume.get_qform(coded=True)
    sform, scode = volume.get_sform(coded=True)
    affine = np.asarray(volume.affine)
    shape = list(volume.shape)
    frozen_affine = np.asarray(frozen["affine_xyz"])
    form_delta = None
    if qform is not None and sform is not None:
        form_delta = finite_float(np.max(np.abs(qform - sform)))
    return {"shape_xyz": shape, "data_dtype": volume.get_data_dtype().str,
            "endianness": volume.header.endianness,
            "proxy_slope": finite_float(volume.dataobj.slope),
            "proxy_inter": finite_float(volume.dataobj.inter),
            "zooms": [finite_float(z) for z in volume.header.get_zooms()],
            "units": list(volume.header.get_xyzt_units()),
            "orientation": list(nib.aff2axcodes(affine)), "affine_xyz": matrix_json(affine),
            "affine_chosen_source": "sform" if scode else "qform" if qcode else "fallback",
            "qform_code": int(qcode), "sform_code": int(scode),
            "qform_xyz": matrix_json(qform), "sform_xyz": matrix_json(sform),
            "qform_sform_max_abs_difference": form_delta,
            "geometry_matches_frozen_record": shape == frozen["shape_xyz"] and bool(
                np.allclose(affine, frozen_affine, rtol=1e-5, atol=1e-5)),
            "text_fields": {field: text_field_evidence(volume.header, field)
                            for field in ("descrip", "aux_file", "intent_name")}}


def verify_inputs(manifest_path, identity_path, sourcebook_path):
    actual = {"manifest": file_sha256(manifest_path), "cache_identity": file_sha256(identity_path),
              "training_sources": file_sha256(sourcebook_path)}
    if actual != {"manifest": EXPECTED_MANIFEST, "cache_identity": EXPECTED_IDENTITY,
                  "training_sources": EXPECTED_SOURCEBOOK}:
        raise ValueError("Input SHA-256 differs from this frozen v1 diagnostic's pinned identities.")
    manifest = full_manifest(manifest_path)
    if (len(manifest["records"]) != 211 or manifest["num_classes"] != 6
            or manifest["label_ids"] != list(range(6))
            or manifest["raw_label_to_index"] != {str(i): i for i in range(6)}):
        raise ValueError("Expected all 211 frozen pairs with identity-mapped labels 0..5.")
    sourcebook = json.loads(sourcebook_path.read_text())
    if sourcebook["manifest_sha256"] != actual["manifest"]:
        raise ValueError("Sourcebook manifest differs.")
    for name, wanted in sourcebook["source_file_sha256"].items():
        if file_sha256(safe_relative(PROJECT_ROOT, name)) != wanted:
            raise ValueError(f"Frozen training source differs: {name}")
    book = load_verified_book(identity_path, manifest_path, manifest)
    return manifest, book, actual


def inspect_case(name, record, manifest, book, data_root, manifest_root, nib, zoom):
    image_path = safe_relative(data_root, record["source_image"])
    mask_path = safe_relative(data_root, record["source_mask"])
    image_volume, mask_volume = nib.load(str(image_path)), nib.load(str(mask_path))
    if image_volume.shape != mask_volume.shape or not np.allclose(image_volume.affine, mask_volume.affine, rtol=1e-5, atol=1e-5):
        raise ValueError("Native image/mask geometry differs.")
    raw_image = image_volume.get_fdata(dtype=np.float32, caching="unchanged")
    raw_mask = map_labels(np.asanyarray(mask_volume.dataobj))
    raw_stats = class_statistics(raw_image, raw_mask)
    native_target_zero = int(((raw_mask == 5) & (raw_image == 0)).sum())
    canonical_image, canonical_mask = nib.as_closest_canonical(image_volume), nib.as_closest_canonical(mask_volume)
    if canonical_image.shape != canonical_mask.shape or not np.allclose(canonical_image.affine, canonical_mask.affine, rtol=1e-5, atol=1e-5):
        raise ValueError("Canonical image/mask geometry differs.")
    source = canonical_image.get_fdata(dtype=np.float32, caching="unchanged")
    labels = map_labels(np.asanyarray(canonical_mask.dataobj))
    normalized, active, normalization = normalize_nonzero(source)
    normalized_stats = class_statistics(normalized, labels, [-5, 5])
    unclipped = np.zeros(source.shape, dtype=np.float32)
    unclipped[active] = ((source[active].astype(np.float64) - normalization["mean"])
                         / max(normalization["std"], 1e-8))
    unclipped_stats = class_statistics(unclipped, labels, [-5, 5])
    del unclipped
    cache_path = safe_relative(manifest_root, record["cache_path"])
    arrays = read_arrays(cache_path)
    validate_arrays(arrays, record, manifest)
    identities = arrays_identity(arrays)
    if identities != book["records"][record["cache_path"]]["arrays"]:
        raise ValueError("Decoded cache identity differs from the frozen book.")
    shape_xyz = np.asarray(manifest["target_shape_dhw"][::-1])
    ratio = shape_xyz / np.asarray(source.shape)
    options = {"grid_mode": True, "mode": "nearest", "prefilter": False}
    before_xyz = zoom(normalized, ratio, order=1, **options)
    active_xyz = zoom(active.astype(np.uint8), ratio, order=0, **options).astype(bool)
    mask_xyz = zoom(labels, ratio, order=0, **options)
    before, support_dhw = before_xyz.transpose(2, 1, 0), active_xyz.transpose(2, 1, 0)
    reconstructed, effects = support_effect(before, support_dhw, arrays["mask"])
    stages = {"native_scaled_before_normalization": raw_stats,
              "canonical_normalized_unclipped": unclipped_stats,
              "canonical_normalized_clipped": normalized_stats,
              "cached_linear_before_support_forcing": class_statistics(before, arrays["mask"], [-5, 5]),
              "cached_after_support_forcing": class_statistics(arrays["image"], arrays["mask"], [-5, 5])}
    target_affine = np.asarray(record["resampled_affine_xyz"])
    native_volume = int((raw_mask == 5).sum()) * abs(float(np.linalg.det(mask_volume.affine[:3, :3])))
    cache_volume = int((arrays["mask"] == 5).sum()) * abs(float(np.linalg.det(target_affine[:3, :3])))
    spatial_unit = image_volume.header.get_xyzt_units()[0]
    return {"case": name, "split": record["split"],
            "selection_basis": "exploratory TRAIN contrast control selected after initial descriptive statistics"
                               if name in EXPLORATORY_CONTROLS else "original twelve-case D0 diagnostic set",
            "decoded_cache_hashes_verified": identities,
            "source_file_sha256": {"image": file_sha256(image_path), "mask": file_sha256(mask_path)},
            "intensity_stages": stages,
            "within_stage_class_contrasts": {stage: class_contrasts(stats) for stage, stats in stages.items()},
            "normalization": normalization,
            "support_policy": {"native_target_zero_support_voxels": native_target_zero, **effects},
            "policy_reconstruction": {"independent_validation": False,
                                      "mask_equal_frozen": bool(np.array_equal(mask_xyz.transpose(2, 1, 0), arrays["mask"])),
                                      "after_support_image_max_abs_error": float(np.max(np.abs(reconstructed - arrays["image"])))},
            "target_volume": {"source_spatial_unit": spatial_unit,
                              "native_affine_units_cubed": native_volume,
                              "cache_affine_units_cubed": cache_volume,
                              "native_mm3_if_unit_confirmed": native_volume if spatial_unit == "mm" else None,
                              "cache_mm3_if_unit_confirmed": cache_volume if spatial_unit == "mm" else None,
                              "cache_to_native_ratio": cache_volume / native_volume if native_volume else None}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-identity", type=Path, required=True)
    parser.add_argument("--training-sources", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, dest="cases", metavar="PATIENT/VISIT",
                        help="Repeat explicit cases from the 14-case allowed set; R016/Week7 and T014/Week3 are exploratory TRAIN controls.")
    parser.add_argument("--ack-retrospective-m015-test", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().absolute()
    if os.path.lexists(output):
        parser.error("Use a new output directory; no existing artifact is overwritten.")
    manifest_path, identity_path, sourcebook_path = (p.expanduser().resolve(strict=True)
        for p in (args.manifest, args.cache_identity, args.training_sources))
    data_root = args.data_root.expanduser().resolve(strict=True)
    manifest, book, hashes = verify_inputs(manifest_path, identity_path, sourcebook_path)
    selected = select_cases(manifest["records"], args.cases, args.ack_retrospective_m015_test)
    import nibabel as nib
    import scipy
    from scipy.ndimage import zoom
    report = {"schema_version": 1, "diagnostic_only": True, "no_model_execution": True,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": hashes,
              "diagnostic_source_sha256": file_sha256(Path(__file__)),
              "versions": {"numpy": np.__version__, "nibabel": nib.__version__, "scipy": scipy.__version__},
              "selection": args.cases, "test_pixel_acknowledged": args.ack_retrospective_m015_test,
              "exploratory_control_selection": {
                  "cases": sorted(EXPLORATORY_CONTROLS),
                  "selected_after_initial_statistics": True,
                  "basis": "Two TRAIN cases with cache contrasts closest to M015 in the initial 143-TRAIN descriptive comparison.",
                  "interpretation": "Exploratory selected controls, not a prospectively sampled independent confirmation cohort."},
              "header_scope": "All 211 image/mask pairs, including TEST headers; pixel reads only for explicit cases, TEST pixels only M015.",
              "limits": ["Selected after v1 TEST was seen; no new accuracy score, model selection or causal claim.",
                         "Uses the declared zscore/clip/zoom/support policy; NOT an independent validation of that policy or anatomical GT.",
                         "Raw means scaled NIfTI intensities before normalization, not unscaled storage integers.",
                         "Raw clip fractions are null: [-5,5] thresholds apply only to normalized/cached stages.",
                         "No free-text header fields, images, masks, checkpoints or predictions are exported.",
                         "Zero normalized values need not denote background; support uses raw intensity != 0."],
              "headers": [], "cases": [], "issues": []}
    for record in manifest["records"]:
        item = {"patient": record["patient"], "visit": record["visit"], "split": record["split"]}
        for role in ("image", "mask"):
            try:
                item[role] = inspect_header(safe_relative(data_root, record[f"source_{role}"]), record[f"original_{role}"], nib)
                if not item[role]["geometry_matches_frozen_record"]:
                    report["issues"].append(f"{record['patient']}/{record['visit']} {role}: source geometry differs")
            except Exception as error:
                item[role] = {"header_error_type": type(error).__name__}
                report["issues"].append(f"{record['patient']}/{record['visit']} {role}: header inspection failed")
        report["headers"].append(item)
    for name, record in selected:
        try:
            result = inspect_case(name, record, manifest, book, data_root, manifest_path.parent, nib, zoom)
            report["cases"].append(result)
            reconstruction = result["policy_reconstruction"]
            if not reconstruction["mask_equal_frozen"] or reconstruction["after_support_image_max_abs_error"] > 1e-5:
                report["issues"].append(f"{name}: declared policy reconstruction differs")
        except Exception as error:
            report["cases"].append({"case": name, "status": "failed", "error_type": type(error).__name__, "error": str(error)})
            report["issues"].append(f"{name}: pixel inspection failed")
        print(f"Source-input diagnostic: {name}", flush=True)
    if any(file_sha256(path) != hashes[key] for path, key in ((manifest_path, "manifest"), (identity_path, "cache_identity"), (sourcebook_path, "training_sources"))):
        raise RuntimeError("Protected input metadata changed during diagnosis.")
    report["status"] = "completed_with_issues" if report["issues"] else "completed"
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "source_input_diagnostic.json", report)
    print(f"Evidence: {output / 'source_input_diagnostic.json'}")
    return int(bool(report["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
