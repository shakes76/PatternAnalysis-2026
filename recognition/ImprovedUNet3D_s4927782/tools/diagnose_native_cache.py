"""Retrospective native/cache alignment diagnostic; changes no model decisions.

Uses nibabel.processing.resample_from_to independently of prepare_cache.py.
Matching the preprocessing implementation is not proof of anatomical truth.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

DEFAULT_CASES = [f"M015/Week{i}" for i in range(8)] + ["B006/Week0", "K019/Week1", "B040/Week0", "K018/Week0"]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))
            or Path(relative).is_absolute()):
        raise ValueError(f"Unsafe relative path: {relative!r}")
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError:
        raise ValueError("Path escapes its declared root.") from None
    return result


def zscore_nonzero(image):
    """Independent implementation of the declared per-case intensity policy."""
    if not np.isfinite(image).all():
        raise ValueError("Native image contains nonfinite values.")
    support = image != 0
    values = image[support].astype(np.float64)
    mean, std = (float(values.mean()), float(values.std())) if values.size else (0.0, 0.0)
    result = np.zeros(image.shape, dtype=np.float32)
    if values.size:
        result[support] = np.minimum(np.maximum((values - mean) / max(std, 1e-8), -5.0), 5.0).astype(np.float32)
    return result, support, {"nonzero_count": int(values.size), "mean": mean, "standard_deviation": std}


def centroid_xyz(mask_xyz, affine):
    points = np.argwhere(mask_xyz)
    if not len(points):
        return None
    voxel = points.mean(axis=0)
    world = (np.asarray(affine) @ np.r_[voxel, 1.0])[:3]
    return {"voxel_xyz": voxel.tolist(), "ras_xyz_mm": world.tolist()}


def project_centroid(native_mask_xyz, native_affine, cache_mask_dhw, cache_affine):
    native = centroid_xyz(native_mask_xyz, native_affine)
    cached = centroid_xyz(cache_mask_dhw.transpose(2, 1, 0), cache_affine)
    result = {"native": native, "cached": cached, "native_projected_to_cache_voxel_xyz": None,
              "cache_minus_native_centroid_ras_mm": None, "centroid_distance_mm": None}
    if native is not None:
        projected = np.linalg.inv(cache_affine) @ np.r_[native["ras_xyz_mm"], 1.0]
        result["native_projected_to_cache_voxel_xyz"] = projected[:3].tolist()
    if native is not None and cached is not None:
        delta = np.asarray(cached["ras_xyz_mm"]) - native["ras_xyz_mm"]
        result["cache_minus_native_centroid_ras_mm"] = delta.tolist()
        result["centroid_distance_mm"] = float(np.linalg.norm(delta))
    return result


def compare_arrays(independent_image, independent_mask, cached_image, cached_mask, num_classes, target_index):
    if independent_image.shape != cached_image.shape or independent_mask.shape != cached_mask.shape:
        raise ValueError("Independent and cached array shapes differ.")
    if not np.isfinite(independent_image).all() or not np.isfinite(cached_image).all():
        raise ValueError("An image contains nonfinite values.")
    error = np.abs(independent_image.astype(np.float64) - cached_image.astype(np.float64))
    mismatch = independent_mask != cached_mask
    return {"mask_exact_match": bool(not mismatch.any()), "mask_different_voxels": int(mismatch.sum()),
            "mask_different_fraction": float(mismatch.mean()),
            "target_mask_different_voxels": int(np.logical_xor(independent_mask == target_index, cached_mask == target_index).sum()),
            "independent_class_counts": np.bincount(independent_mask.ravel(), minlength=num_classes).tolist(),
            "cached_class_counts": np.bincount(cached_mask.ravel(), minlength=num_classes).tolist(),
            "image_allclose": bool(np.allclose(independent_image, cached_image, rtol=1e-5, atol=1e-5)),
            "image_rtol": 1e-5, "image_atol": 1e-5, "image_max_abs_error": float(error.max()),
            "image_mean_abs_error": float(error.mean()), "image_p99_abs_error": float(np.percentile(error, 99)),
            "independent_image_finite": True, "cached_image_finite": True}


def stabilized_resample(normalized, raw_mask, support, transform, shape_xyz, labels):
    """Secondary numerical tie diagnostic, never a replacement cache/protocol."""
    from scipy.ndimage import affine_transform
    stable = np.round(transform, decimals=12)
    kwargs = {"matrix": stable[:3, :3], "offset": stable[:3, 3], "output_shape": shape_xyz,
              "mode": "nearest", "prefilter": False}
    image = affine_transform(normalized, order=1, **kwargs)
    active = affine_transform(support.astype(np.uint8), order=0, **kwargs).astype(bool)
    image[~active] = 0
    raw = affine_transform(raw_mask.astype(np.int32), order=0, **kwargs)
    mask = np.zeros(shape_xyz, dtype=np.uint8)
    for index, label in enumerate(labels):
        mask[raw == label] = index
    return image, mask, stable


def select_records(manifest, cases):
    if manifest.get("partial") or manifest.get("development_only"):
        raise ValueError("Use the full frozen manifest.")
    mapping = {str(raw): index for index, raw in enumerate(manifest["label_ids"])}
    if manifest.get("raw_label_to_index") != mapping or "5" not in mapping:
        raise ValueError("Manifest must explicitly map raw class 5 to a cached class index.")
    splits = manifest["patient_splits"]
    seen_patients = set()
    for name in ("train", "validation", "test"):
        patients = splits[name]
        if len(set(patients)) != len(patients) or seen_patients.intersection(patients):
            raise ValueError("Manifest patient partitions overlap or contain duplicates.")
        seen_patients.update(patients)
    records = {}
    for record in manifest["records"]:
        key = f"{record['patient']}/{record['visit']}"
        if key in records or record["patient"] not in splits[record["split"]]:
            raise ValueError("Duplicate case or invalid patient split assignment.")
        records[key] = record
    if len(set(cases)) != len(cases):
        raise ValueError("Duplicate --case selections are not allowed.")
    if any(case not in records for case in cases):
        raise ValueError(f"Cases absent from manifest: {[case for case in cases if case not in records]}")
    return [(case, records[case]) for case in cases]


def plot_alignment(native_image_xyz, native_mask_xyz, native_affine, cache_image_xyz,
                   cache_mask_xyz, independent_image_xyz, independent_mask_xyz,
                   target_affine, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 3, figsize=(13, 11))
    spacing_native = np.linalg.norm(native_affine[:3, :3], axis=0)
    spacing_target = np.linalg.norm(target_affine[:3, :3], axis=0)
    native_centre = np.argwhere(native_mask_xyz).mean(axis=0) if native_mask_xyz.any() else (np.asarray(native_image_xyz.shape) - 1) / 2
    chosen = []
    for row, axis in enumerate((0, 1, 2)):
        sums = native_mask_xyz.sum(axis=tuple(i for i in range(3) if i != axis))
        native_index = int(sums.argmax()) if sums.max() else native_image_xyz.shape[axis] // 2
        point = native_centre.copy(); point[axis] = native_index
        projected = np.linalg.inv(target_affine) @ native_affine @ np.r_[point, 1.0]
        cache_index = int(np.clip(np.rint(projected[axis]), 0, cache_image_xyz.shape[axis] - 1))
        chosen.append({"axis_xyz": axis, "native_index": native_index, "cache_index": cache_index,
                       "native_plane_world_point_ras_mm": (native_affine @ np.r_[point, 1])[:3].tolist(),
                       "cache_plane_index_rounding_error_voxels": float(cache_index - projected[axis])})
        for column, image, mask, index, spacing in [
            (0, native_image_xyz, native_mask_xyz, native_index, spacing_native),
            (1, cache_image_xyz, cache_mask_xyz, cache_index, spacing_target),
            (2, independent_image_xyz, independent_mask_xyz, cache_index, spacing_target),
        ]:
            panel = np.take(image, index, axis=axis).T
            contour = np.take(mask, index, axis=axis).T
            remaining = [i for i in range(3) if i != axis]
            extent = (0, panel.shape[1] * spacing[remaining[0]], 0, panel.shape[0] * spacing[remaining[1]])
            axes[row, column].imshow(panel, cmap="gray", vmin=-3, vmax=3, origin="lower", extent=extent)
            if contour.any() and not contour.all():
                x = (np.arange(panel.shape[1]) + 0.5) * spacing[remaining[0]]
                y = (np.arange(panel.shape[0]) + 0.5) * spacing[remaining[1]]
                axes[row, column].contour(x, y, contour.astype(float), levels=[0.5], colors=["lime"], linewidths=0.9)
            columns = ["Native RAS: z-score only", "Existing cache", "Independent unrounded affine resample"]
            planes = ["Sagittal X", "Coronal Y", "Axial Z"]
            axes[row, column].set_title(f"{columns[column]} | {planes[row]}={index}", fontsize=9)
            axes[row, column].set_xlabel("mm along grid axis", fontsize=8)
            axes[row, column].set_ylabel("mm along grid axis", fontsize=8)
    figure.suptitle(title + "\nGreen: recorded prostate GT; fixed native-GT maximum-area planes. Independent/manual QA, not anatomical confirmation.", fontsize=10)
    figure.tight_layout(rect=(0, 0.01, 1, 0.93))
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return chosen


def diagnose_case(case, record, manifest, data_root, manifest_root, output):
    import nibabel as nib
    from nibabel.processing import resample_from_to
    image_path = safe_path(data_root, record["source_image"])
    mask_path = safe_path(data_root, record["source_mask"])
    cache_path = safe_path(manifest_root, record["cache_path"])
    original_image, original_mask = nib.load(str(image_path)), nib.load(str(mask_path))
    if len(original_image.shape) != 3 or original_image.shape != original_mask.shape:
        raise ValueError(f"{case}: native shape mismatch.")
    if (not np.isfinite(original_image.affine).all() or not np.isfinite(original_mask.affine).all()
            or not np.allclose(original_image.affine, original_mask.affine, rtol=1e-5, atol=1e-5)):
        raise ValueError(f"{case}: native affine mismatch/nonfinite geometry.")
    raw_image = original_image.get_fdata(dtype=np.float32, caching="unchanged")
    raw_mask = np.asanyarray(original_mask.dataobj)
    if not np.isfinite(raw_image).all() or not np.isfinite(raw_mask).all() or np.any(raw_mask != np.rint(raw_mask)):
        raise ValueError(f"{case}: native image/mask is nonfinite or mask is fractional.")
    labels = manifest["label_ids"]; raw_to_index = {raw: index for index, raw in enumerate(labels)}
    if not set(int(v) for v in np.unique(raw_mask)).issubset(raw_to_index):
        raise ValueError(f"{case}: undeclared native labels.")
    canonical_image = nib.as_closest_canonical(original_image)
    canonical_mask = nib.as_closest_canonical(original_mask)
    if canonical_image.shape != canonical_mask.shape or not np.allclose(canonical_image.affine, canonical_mask.affine, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{case}: canonical image/mask geometry mismatch.")
    source_image = canonical_image.get_fdata(dtype=np.float32, caching="unchanged")
    source_raw_mask = np.asanyarray(canonical_mask.dataobj)
    normalized, support, normalization = zscore_nonzero(source_image)
    affine = np.asarray(record["resampled_affine_xyz"], dtype=np.float64)
    shape_xyz = tuple(reversed(manifest["target_shape_dhw"]))
    target = (shape_xyz, affine)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Invalid frozen target affine.")
    independent_image = np.asarray(resample_from_to(nib.Nifti1Image(normalized, canonical_image.affine), target, order=1, mode="nearest", cval=0).dataobj, dtype=np.float32)
    independent_support = np.asarray(resample_from_to(nib.Nifti1Image(support.astype(np.uint8), canonical_image.affine), target, order=0, mode="nearest", cval=0).dataobj).astype(bool)
    independent_image[~independent_support] = 0
    # Resample raw IDs, then map IDs; nearest-neighbour never blends classes.
    resampled_raw = np.asarray(resample_from_to(nib.Nifti1Image(source_raw_mask.astype(np.int32), canonical_mask.affine), target, order=0, mode="nearest", cval=0).dataobj)
    independent_mask = np.empty(shape_xyz, dtype=np.uint8)
    for raw, index in raw_to_index.items():
        independent_mask[resampled_raw == raw] = index
    if not set(int(v) for v in np.unique(resampled_raw)).issubset(raw_to_index):
        raise ValueError("Independent resampler introduced an undeclared label.")
    with np.load(cache_path, allow_pickle=False) as cache:
        cached_image, cached_mask, spacing = cache["image"], cache["mask"], cache["spacing"]
    if cached_image.dtype != np.float32 or cached_mask.dtype != np.uint8 or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0) or int(cached_mask.max()) >= len(labels):
        raise ValueError("Invalid cache types/spacing.")
    target_index = raw_to_index[5]
    comparison = compare_arrays(independent_image.transpose(2, 1, 0), independent_mask.transpose(2, 1, 0), cached_image, cached_mask, len(labels), target_index)
    centroid = project_centroid(raw_mask == 5, original_mask.affine, cached_mask == target_index, affine)
    target_to_source = np.linalg.inv(canonical_image.affine) @ affine
    stable_image, stable_mask, stable_transform = stabilized_resample(normalized, source_raw_mask, support, target_to_source, shape_xyz, labels)
    stable_comparison = compare_arrays(stable_image.transpose(2, 1, 0), stable_mask.transpose(2, 1, 0), cached_image, cached_mask, len(labels), target_index)
    class5_native_count, class5_cached_count = int((raw_mask == 5).sum()), int((cached_mask == target_index).sum())
    figure_name = re.sub(r"[^A-Za-z0-9_.-]", "_", case) + ".png"
    planes = plot_alignment(normalized, source_raw_mask == 5, canonical_image.affine,
                            cached_image.transpose(2, 1, 0), (cached_mask == target_index).transpose(2, 1, 0),
                            independent_image, independent_mask == target_index, affine, output / figure_name, case)
    def geometry_matches(volume, frozen):
        return (isinstance(frozen, dict) and list(volume.shape) == frozen.get("shape_xyz")
                and np.asarray(frozen.get("affine_xyz")).shape == (4, 4)
                and bool(np.allclose(volume.affine, frozen["affine_xyz"], rtol=1e-5, atol=1e-5)))

    return {"case": case, "split": record["split"], "retrospective": True,
            "source_sha256": {"native_image": file_hash(image_path), "native_mask": file_hash(mask_path), "existing_cache_npz": file_hash(cache_path)},
            "native_shape_xyz": list(original_image.shape), "native_affine_xyz": original_image.affine.tolist(),
            "native_orientation": list(nib.aff2axcodes(original_image.affine)),
            "canonical_shape_xyz": list(canonical_image.shape), "canonical_affine_xyz": canonical_image.affine.tolist(),
            "target_shape_dhw": list(cached_image.shape), "target_affine_xyz": affine.tolist(),
            "cached_spacing_dhw_mm": spacing.tolist(), "normalization": normalization,
            "output_index_to_canonical_input_index": target_to_source.tolist(),
            "rounded_12_decimal_sampling_transform": stable_transform.tolist(),
            "rounding_max_abs_coefficient_change": float(np.abs(stable_transform - target_to_source).max()),
            "sampling_geometry_note": "Matrices map target indices to canonical source indices. Rounded coefficients are only a secondary half-voxel-tie diagnostic.",
            "raw_class5_voxels": class5_native_count, "cached_target_voxels": class5_cached_count,
            "raw_class5_volume_mm3": class5_native_count * float(abs(np.linalg.det(original_mask.affine[:3, :3]))),
            "cached_target_volume_mm3": class5_cached_count * float(abs(np.linalg.det(affine[:3, :3]))),
            "class5_centroid_projection": centroid, "comparison": comparison,
            "secondary_rounded_coefficient_comparison": stable_comparison,
            "invariants": {"native_image_mask_shape_match": True, "native_image_mask_affine_match": True,
                           "native_image_mask_finite": True,
                           "native_image_geometry_matches_frozen_record": geometry_matches(original_image, record.get("original_image")),
                           "native_mask_geometry_matches_frozen_record": geometry_matches(original_mask, record.get("original_mask")),
                           "canonical_image_geometry_matches_frozen_record": geometry_matches(canonical_image, record.get("canonical_image")),
                           "canonical_mask_geometry_matches_frozen_record": geometry_matches(canonical_mask, record.get("canonical_mask")),
                           "canonical_orientation_is_ras": nib.aff2axcodes(canonical_image.affine) == ("R", "A", "S"),
                           "cache_spacing_matches_affine": bool(np.allclose(np.linalg.norm(affine[:3, :3], axis=0)[::-1], spacing, rtol=1e-5, atol=1e-6)),
                           "cached_counts_match_manifest": np.bincount(cached_mask.ravel(), minlength=len(labels)).tolist() == record["class_voxel_counts"]},
            "figure": figure_name, "figure_planes": planes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="cases", help="Repeat PATIENT/VISIT; default fixed M015 eight visits plus two TRAIN/two VAL controls.")
    args = parser.parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    data_root = args.data_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error("Use a fresh output directory; diagnostic outputs are never overwritten.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = select_records(manifest, args.cases or DEFAULT_CASES)
    for _, record in selected:
        for key in ("source_image", "source_mask"):
            if not safe_path(data_root, record[key]).is_file():
                parser.error(f"Missing source file: {record[key]}")
        if not safe_path(manifest_path.parent, record["cache_path"]).is_file():
            parser.error(f"Missing cache: {record['cache_path']}")
    import nibabel
    import scipy
    manifest_digest = file_hash(manifest_path)
    output.mkdir(parents=True)
    report = {"schema_version": 1, "status": "retrospective_implementation_QA_not_anatomical_confirmation",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": manifest_digest,
              "diagnostic_source_sha256": file_hash(Path(__file__)), "versions": {"numpy": np.__version__, "nibabel": nibabel.__version__, "scipy": scipy.__version__},
              "selection": [case for case, _ in selected], "selection_uses_predictions": False,
              "independent_method": "nibabel.processing.resample_from_to on canonical RAS source; no prepare_cache.py import/call",
              "boundary_and_tie_policy": "Explicit mode=nearest rather than resample_from_to's default constant; image order=1, label/support order=0. Canonicalize before nearest interpolation to match the declared tie orientation at half-voxel sample centres. Primary comparison is unmodified nibabel resample_from_to. A separate scipy affine_transform comparison rounds target-to-source coefficients to 12 decimals solely to diagnose floating-point half-voxel ties. Neither changes the frozen affine or cache. Mismatches are reported, not automatically labelled preprocessing bugs or anatomical errors.",
              "hypothesis_limits": ["Agreement verifies consistency of two implementations of the declared preprocessing, not anatomical correctness of native labels.",
                                    "A raw-mask centroid projected into the cache need not equal the decimated mask centroid; nearest-neighbour sampling changes boundary occupancy.",
                                    "Hard invariants exclude resampler exact-equality checks and centroid differences; a false exact comparison can result from nearest-neighbour ties rather than physical misregistration.",
                                    "Native GT maximum-area planes are fixed independently of predictions. The nearest cached plane can differ by up to half a cached index; inspect recorded rounding offsets.",
                                    "M015 was selected after observing failures; these are retrospective diagnostics, not a new held-out evaluation or proof of an acquisition/domain shift.",
                                    "No model, predictions, frozen protocol, cache arrays or manifest are modified."], "cases": []}
    for case, record in selected:
        report["cases"].append(diagnose_case(case, record, manifest, data_root, manifest_path.parent, output))
        print(f"Diagnosed {case}", flush=True)
    if file_hash(manifest_path) != manifest_digest:
        raise RuntimeError("Manifest changed during diagnosis.")
    report["all_reported_invariants_pass"] = all(all(case["invariants"].values()) for case in report["cases"])
    report["resampler_summary"] = {"num_cases": len(report["cases"]),
        "primary_exact_mask_cases": sum(case["comparison"]["mask_exact_match"] for case in report["cases"]),
        "primary_image_allclose_cases": sum(case["comparison"]["image_allclose"] for case in report["cases"]),
        "secondary_rounded_exact_mask_cases": sum(case["secondary_rounded_coefficient_comparison"]["mask_exact_match"] for case in report["cases"]),
        "secondary_rounded_image_allclose_cases": sum(case["secondary_rounded_coefficient_comparison"]["image_allclose"] for case in report["cases"])}
    (output / "native_cache_diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Retrospective diagnostic report: {output / 'native_cache_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
