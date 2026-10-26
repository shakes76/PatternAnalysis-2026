"""Produce TRAIN-only spatial QA for source-confirmed or candidate labels.

Official names and provenance are read from attached manifest metadata. Without
that metadata, candidate names remain pending review. Physical measurements use
the cached grid and do not establish clinical segmentation accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dataset import load_manifest


CANDIDATE_MAPPING = {0: "background", 1: "body", 2: "bone", 3: "bladder", 4: "rectum", 5: "prostate"}


def natural_key(text):
    return [(0, int(part)) if part.isdigit() else (1, part.casefold())
            for part in re.split(r"(\d+)", str(text))]


def measure_case(image, mask, spacing, affine, num_classes):
    """Measure class indices without treating the candidate names as truth."""
    voxel_mm3 = float(np.prod(spacing))
    classes = []
    for label in range(num_classes):
        indices = np.argwhere(mask == label)
        count = int(len(indices))
        item = {"class_index": label, "voxel_count": count,
                "volume_mm3": count * voxel_mm3, "volume_ml": count * voxel_mm3 / 1000,
                "centroid_ras_xyz_mm": None, "bbox_ras_xyz_mm": None,
                "bbox_dhw_inclusive": None, "image_intensity_mean": None,
                "image_intensity_p10_p50_p90": None}
        if count:
            centre_xyz = indices.mean(axis=0)[::-1]
            item["centroid_ras_xyz_mm"] = (affine @ np.r_[centre_xyz, 1])[:3].tolist()
            minimum, maximum = indices.min(axis=0), indices.max(axis=0)
            item["bbox_dhw_inclusive"] = {"minimum": minimum.tolist(), "maximum": maximum.tolist()}
            # Physical bounding box includes full edge-voxel extents, not only centres.
            xyz_min, xyz_max = minimum[::-1] - 0.5, maximum[::-1] + 0.5
            corners = np.asarray(list(itertools.product(*zip(xyz_min, xyz_max))))
            physical = np.c_[corners, np.ones(8)] @ affine.T
            item["bbox_ras_xyz_mm"] = {"minimum": physical[:, :3].min(axis=0).tolist(),
                                        "maximum": physical[:, :3].max(axis=0).tolist()}
            values = image[mask == label]
            item["image_intensity_mean"] = float(values.mean(dtype=np.float64))
            item["image_intensity_p10_p50_p90"] = np.percentile(values, [10, 50, 90]).tolist()
        classes.append(item)

    # Shared faces give an explicit adjacency measure, independent of organ names.
    shared_faces, shared_area = {}, {}
    for axis in range(3):
        before, after = [slice(None)] * 3, [slice(None)] * 3
        before[axis], after[axis] = slice(None, -1), slice(1, None)
        a, b = mask[tuple(before)].astype(np.int64), mask[tuple(after)].astype(np.int64)
        different = a != b
        codes = np.minimum(a[different], b[different]) * num_classes + np.maximum(a[different], b[different])
        values, counts = np.unique(codes, return_counts=True)
        face_mm2 = float(np.prod(np.delete(spacing, axis)))
        for code, count in zip(values, counts):
            pair = (int(code // num_classes), int(code % num_classes))
            shared_faces[pair] = shared_faces.get(pair, 0) + int(count)
            shared_area[pair] = shared_area.get(pair, 0.0) + int(count) * face_mm2
    relations = []
    for first, second in itertools.combinations(range(1, num_classes), 2):
        left, right = classes[first], classes[second]
        relation = {"first_class_index": first, "second_class_index": second,
                    "centroid_delta_second_minus_first_ras_xyz_mm": None,
                    "centroid_distance_mm": None, "axis_aligned_bbox_gap_mm": None,
                    "shared_voxel_faces": shared_faces.get((first, second), 0),
                    "shared_face_area_mm2": shared_area.get((first, second), 0.0)}
        if left["voxel_count"] and right["voxel_count"]:
            delta = np.asarray(right["centroid_ras_xyz_mm"]) - left["centroid_ras_xyz_mm"]
            relation["centroid_delta_second_minus_first_ras_xyz_mm"] = delta.tolist()
            relation["centroid_distance_mm"] = float(np.linalg.norm(delta))
            lo = np.maximum(left["bbox_ras_xyz_mm"]["minimum"], right["bbox_ras_xyz_mm"]["minimum"])
            hi = np.minimum(left["bbox_ras_xyz_mm"]["maximum"], right["bbox_ras_xyz_mm"]["maximum"])
            relation["axis_aligned_bbox_gap_mm"] = float(np.linalg.norm(np.maximum(lo - hi, 0)))
        relations.append(relation)

    evidence = {}
    evidence["class1_largest_nonbackground_by_volume"] = bool(classes[1]["voxel_count"] >= max(item["voxel_count"] for item in classes[2:]))
    evidence["class1_bbox_contains_other_foreground_centroids_fraction"] = None
    evidence["class2_left_right_balance_about_class1_centroid"] = None
    if classes[1]["voxel_count"]:
        bounds = classes[1]["bbox_ras_xyz_mm"]
        other_centres = [item["centroid_ras_xyz_mm"] for item in classes[2:] if item["voxel_count"]]
        if other_centres:
            evidence["class1_bbox_contains_other_foreground_centroids_fraction"] = float(np.mean([
                np.all(np.asarray(point) >= bounds["minimum"]) and np.all(np.asarray(point) <= bounds["maximum"])
                for point in other_centres]))
        if classes[2]["voxel_count"]:
            bone_indices = np.argwhere(mask == 2)[:, ::-1]
            physical_x = bone_indices @ affine[0, :3] + affine[0, 3]
            midline = classes[1]["centroid_ras_xyz_mm"][0]
            left, right = int((physical_x < midline).sum()), int((physical_x >= midline).sum())
            evidence["class2_left_right_balance_about_class1_centroid"] = min(left, right) / max(left, right) if max(left, right) else None
            evidence["class2_voxels_left_right_of_class1_centroid"] = [left, right]
    for key, first, second, axis in [
        ("class3_superior_to_class5_margin_mm", 3, 5, 2),
        ("class5_anterior_to_class4_margin_mm", 5, 4, 1),
        ("class3_anterior_to_class4_margin_mm", 3, 4, 1),
    ]:
        evidence[key] = (classes[first]["centroid_ras_xyz_mm"][axis] - classes[second]["centroid_ras_xyz_mm"][axis]
                         if classes[first]["voxel_count"] and classes[second]["voxel_count"] else None)
    evidence["class5_smaller_than_class3_by_volume"] = (
        classes[5]["volume_mm3"] < classes[3]["volume_mm3"] if classes[5]["voxel_count"] and classes[3]["voxel_count"] else None)
    return {"voxel_volume_mm3": voxel_mm3, "per_class": classes,
            "pairwise_relations": relations, "candidate_evidence_observations": evidence}


def plot_case(image, mask, spacing, measurements, destination, patient, visit, num_classes,
              label_names=None, official=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    body = measurements["per_class"][1]["bbox_dhw_inclusive"]
    mid_w = int(round((body["minimum"][2] + body["maximum"][2]) / 2)) if body else image.shape[2] // 2
    nonzero = image[image != 0]
    low, high = np.percentile(nonzero, [1, 99]) if nonzero.size else (0, 1)
    if high <= low:
        high = low + 1
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    colours = {3: "deepskyblue", 4: "orange", 5: "lime"}
    depths = {}
    for column, label in enumerate((3, 4, 5)):
        counts = (mask == label).sum(axis=(1, 2))
        depth = int(counts.argmax()) if counts.max() else image.shape[0] // 2
        depths[str(label)] = depth
        for row, source, labels in [(0, image[depth], mask[depth]), (1, image[:, :, mid_w], mask[:, :, mid_w])]:
            axis = axes[row, column]
            extent = (0, image.shape[2] * spacing[2], 0, image.shape[1] * spacing[1]) if row == 0 else (0, image.shape[1] * spacing[1], 0, image.shape[0] * spacing[0])
            axis.imshow(source, cmap="gray", vmin=low, vmax=high, origin="lower", extent=extent)
            for context in (3, 4, 5):
                overlay = np.ma.masked_where(labels != context, np.ones(labels.shape))
                axis.imshow(overlay, cmap=ListedColormap([colours[context]]), alpha=0.6 if context == label else 0.18,
                            vmin=0, vmax=1, origin="lower", extent=extent, interpolation="nearest")
            plane = f"Axial D={depth}, largest class-{label} area" if row == 0 else f"Estimated midsagittal W={mid_w}"
            present = bool((labels == label).any())
            name = label_names.get(str(label), f"Class {label}") if official else f"Class {label}"
            axis.set_title(f"Class {label}: {name} | {plane}" + (" | absent in plane" if not present else ""), fontsize=9)
            axis.set_xlabel("R+ mm from grid edge" if row == 0 else "A+ mm from grid edge", fontsize=8)
            axis.set_ylabel("A+ mm from grid edge" if row == 0 else "S+ mm from grid edge", fontsize=8)
    interpretation = "Official source-confirmed names; empirical spatial QA" if official else "class IDs only; candidate names unconfirmed"
    figure.suptitle(f"TRAIN {patient} / {visit} — {interpretation}\nClass 3 blue; class 4 orange; class 5 green. Same intensity window; cached canonical grid.", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return {"estimated_midsagittal_w_index": mid_w, "representative_axial_d_indices": depths,
            "midline_method": "midpoint of class-1 bounding box in W, an estimate rather than a registered anatomical midline"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train", help="Only training data may inform this interpretation.")
    parser.add_argument("--count", type=int, default=5, help="Distinct training patients, first available visit per patient.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be positive.")
    path = args.manifest.expanduser().resolve()
    manifest = load_manifest(path)
    semantics = manifest.get("label_semantics", {})
    official = semantics.get("status") == "official_source_confirmed"
    label_names = semantics.get("class_index_to_name", {}) if official else {}
    if official and (not isinstance(label_names, dict) or any(str(index) not in label_names for index in range(manifest["num_classes"]))):
        parser.error("Source-confirmed metadata must name every cached class index.")
    if manifest.get("partial") or manifest.get("development_only"):
        parser.error("Use a complete manifest, not a partial development cache.")
    if manifest["num_classes"] < 6:
        parser.error("This six-class candidate review requires class indices 0 through 5.")
    records = sorted((record for record in manifest["records"] if record["split"] == "train"),
                     key=lambda record: (natural_key(record["patient"]), natural_key(record["visit"])))
    selected, seen = [], set()
    for record in records:
        if record["patient"] not in seen:
            selected.append(record)
            seen.add(record["patient"])
        if len(selected) >= args.count:
            break
    if not selected:
        parser.error("No training patients are available.")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "label_semantics_review.json").exists():
        parser.error("Review output exists; select a fresh output directory.")
    result = {"status": "official_source_confirmed_with_empirical_review" if official else "inference_pending_review",
              "primary_documentation_confirmed": official,
              "mapping_key_space": "cached class index; raw IDs are separately listed in label_ids",
              "label_ids": manifest["label_ids"], "split": "train",
              "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "requested_distinct_patients": args.count, "reviewed_distinct_patients": len(selected),
              "selection": "First naturally sorted visit of first naturally sorted distinct training patients; deterministic, not random.",
              "anatomical_review_checks" if official else "candidate_hypotheses": [
                  "Class 1 may be the large enclosing soft-tissue/body region; inspect volume, bbox containment and images.",
                  "Class 2 may contain bilateral bony anatomy; inspect left/right balance and images (balance alone cannot identify bone).",
                  "Class 3 may be fluid-filled bladder; inspect intensity and location superior to class 5.",
                  "Class 4 may be rectum; inspect location posterior to class 5 and sagittal morphology.",
                  "Class 5 may be prostate; inspect location inferior to class 3 and anterior to class 4.",
              ],
              "limitations": [
                  "Official label names come from the attached source; these empirical measurements do not establish clinical accuracy." if official else "All anatomical names are provisional inference; no primary label-key document is asserted.",
                  "Positive RAS Y means anterior and positive RAS Z superior. Centroids/bboxes alone cannot identify an organ.",
                  "Measurements use the downsampled cache, not native geometry. Anatomical size, distension and contour policy vary.",
                  "Shared-face contact is grid-dependent; bbox gap is an axis-aligned lower bound, not a surface distance.",
                  "Five deterministic patients are a review sample, not proof of label semantics or clinical validity.",
              ], "patients": []}
    if official:
        result.update({"class_index_to_name": label_names, "raw_id_to_name": semantics["raw_id_to_name"],
                       **{field: semantics[field] for field in ("source_url", "source_sha256", "dataset_doi", "attribution", "acknowledgement")}})
        result["anatomical_review_checks"] = [
            "Check the documented Body label against enclosing soft-tissue extent.",
            "Check the documented Bones label for bilateral bony anatomy.",
            "Review documented Bladder, Rectum and Prostate labels in axial and sagittal images; spatial relations vary with anatomy and distension.",
        ]
    else:
        result["candidate_mapping"] = CANDIDATE_MAPPING
    for record in selected:
        with np.load(path.parent / record["cache_path"], allow_pickle=False) as cache:
            image, mask, spacing = cache["image"], cache["mask"], cache["spacing"].astype(np.float64)
        affine = np.asarray(record["resampled_affine_xyz"], dtype=np.float64)
        if image.shape != tuple(manifest["target_shape_dhw"]) or mask.shape != image.shape or mask.max() >= manifest["num_classes"]:
            raise ValueError("Cached image/mask shape or label range does not match the manifest.")
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.allclose(np.linalg.norm(affine[:3, :3], axis=0)[::-1], spacing, rtol=1e-4):
            raise ValueError("Cache spacing and resampled XYZ affine are inconsistent.")
        measured = measure_case(image, mask, spacing, affine, manifest["num_classes"])
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{record['patient']}__{record['visit']}") + ".png"
        plane_info = plot_case(image, mask, spacing, measured, output / filename, record["patient"], record["visit"], manifest["num_classes"], label_names, official)
        result["patients"].append({"patient": record["patient"], "visit": record["visit"],
                                   "figure": filename, "spacing_dhw_mm": spacing.tolist(),
                                   "resampled_affine_xyz": affine.tolist(), **plane_info, **measured})
    observations = [item["candidate_evidence_observations"] for item in result["patients"]]
    result["observed_relation_summary"] = {}
    for key in ("class3_superior_to_class5_margin_mm", "class5_anterior_to_class4_margin_mm", "class3_anterior_to_class4_margin_mm"):
        values = [item[key] for item in observations if item[key] is not None]
        result["observed_relation_summary"][key] = {"n_measured": len(values),
            "median_mm": float(np.median(values)) if values else None,
            "fraction_positive": float(np.mean(np.asarray(values) > 0)) if values else None}
    (output / "label_semantics_review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    names_description = ("Official class names: " + "; ".join(f"{index}: {name}" for index, name in label_names.items()) + ".") if official else "Candidate names: class 0 background; 1 body; 2 bone; 3 bladder; 4 rectum; 5 prostate. These remain hypotheses until the images and measurements are reviewed."
    lines = ["# Training-label interpretation review", "", f"Status: **{result['status']}**. Primary documentation confirmed: **{str(official).lower()}**.", "",
             names_description, "",
             "Read the JSON for per-class physical volumes, RAS centroids/bounds, intensity statistics, pairwise adjacency and observed directional margins. All measurements refer to the shared downsampled grid.", "",
             "| Patient | Visit | Class 3 mL | Class 4 mL | Class 5 mL | Figure |", "|---|---|---:|---:|---:|---|"]
    for item in result["patients"]:
        volumes = [item["per_class"][label]["volume_ml"] for label in (3, 4, 5)]
        lines.append(f"| {item['patient']} | {item['visit']} | {volumes[0]:.2f} | {volumes[1]:.2f} | {volumes[2]:.2f} | [{item['figure']}]({item['figure']}) |")
    if official:
        lines += ["", f"Official source: [{semantics['source_url']}]({semantics['source_url']})", f"Source SHA-256: `{semantics['source_sha256']}`", f"Dataset DOI: {semantics['dataset_doi']}", semantics["attribution"], semantics["acknowledgement"]]
    lines += ["", "A positive superior/anterior margin supports only the measured spatial relation. These checks do not establish clinical segmentation accuracy; anatomy, distension and preprocessing affect the measurements."]
    (output / "label_semantics_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"TRAIN-only label review ({result['status']}): {len(selected)} distinct patients; {output / 'label_semantics_review.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
