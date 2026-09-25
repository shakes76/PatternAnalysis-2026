#!/usr/bin/env python3
"""Create and verify patient-isolated ADNI manifests without modifying images.

Requires Python >= 3.9 and Pillow. See DATA_PROTOCOL.md for training safeguards
that these checks cannot enforce. A patient may have several scans, and each
scan contains several JPEG slices; the patient is the unit of every split.
Checks cover supplied identifiers and exact image duplicates, not approximate
duplicates, upstream preprocessing, or future model-training behavior.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile

VERSION = 1
SOURCE_FIELDS = [
    "relative_path", "original_split", "folder_label", "label", "metadata_label",
    "patient_id", "image_id", "slice_index", "width", "height", "mode",
    "file_bytes", "file_sha256", "pixel_sha256",
]
FIELDS = SOURCE_FIELDS + ["patient_stratum", "partition", "fold"]
PATIENT_FIELDS = ["patient_id", "stratum", "labels", "num_scans", "num_images", "partition", "fold"]
# Each pair is (original metadata label, binary training label): AD 2 -> 1,
# NC 0 -> 0. Keep both values in the manifests so the mapping remains auditable.
SOURCE_LABELS = {"AD": (2, 1), "NC": (0, 0)}
PATIENT_PATTERN = re.compile(r"ADNI_(\d{3}_S_\d+)_")
IMAGE_PATTERN = re.compile(r"_I(\d+)\.nii(?:\.gz)?$")


class AuditError(ValueError):
    """Input data or manifests violate the declared protocol."""


def require(condition, message):
    """Stop the audit when an input or split violates a required condition."""
    if not condition:
        raise AuditError(message)


def digest(data):
    """Return a SHA-256 fingerprint for an exact sequence of bytes."""
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    """Serialize a value consistently for reproducible fingerprints."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def read_json(path):
    """Read JSON, accepting an optional UTF-8 byte-order mark."""
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path, value):
    """Write an indented, UTF-8 audit record."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def profiles_for(rows):
    """Summarize each patient's scan labels without changing any scan's label.

    A longitudinal patient can have both NC and AD scans. Such patients use
    the mixed stratum and still stay together during every partition.
    """
    profiles = {}
    for row in rows:
        profile = profiles.setdefault(row["patient_id"], {"labels": set(), "scans": set(), "images": 0})
        profile["labels"].add(int(row["label"]))
        profile["scans"].add(row["image_id"])
        profile["images"] += 1
    for profile in profiles.values():
        profile["stratum"] = {frozenset({0}): "NC_only", frozenset({1}): "AD_only",
                              frozenset({0, 1}): "mixed"}[frozenset(profile["labels"])]
    return profiles


def inventory(data_root, expected_slices):
    """Audit source identities, labels, scan completeness, and exact duplicates.

    Original train/test folders provide provenance only; their patient overlap
    is measured here and corrected by the new patient-based assignments.
    """
    try:
        import PIL
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise AuditError("Pillow is missing. Install Pillow in the current Python environment before running.") from exc

    metadata_path = data_root / "meta_data_with_label.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes.decode("utf-8-sig"))
    require(isinstance(metadata, dict), "The metadata root must be a dictionary.")
    rows, errors, ignored = [], [], []
    seen_slices = set()
    identities = {}
    pixel_owners = {}
    within_patient_duplicates = 0

    for split in ("train", "test"):
        for folder_label, (metadata_label, label) in SOURCE_LABELS.items():
            folder = data_root / "AD_NC" / split / folder_label
            require(folder.is_dir(), f"Missing source directory: {folder}")
            files = sorted(path for path in folder.rglob("*") if path.is_file())
            jpeg_files = [path for path in files if path.suffix.lower() in (".jpeg", ".jpg")]
            ignored.extend(str(path.relative_to(data_root)) for path in files
                           if path.suffix.lower() not in (".jpeg", ".jpg"))
            require(jpeg_files, f"No JPEG images found in directory: {folder}")
            for path in jpeg_files:
                relative = path.relative_to(data_root).as_posix()
                try:
                    match = re.fullmatch(r"(\d+)_(\d+)", path.stem)
                    require(match is not None, f"Cannot parse image filename: {relative}")
                    image_id, slice_text = match.groups()
                    slice_index = int(slice_text)
                    record = metadata.get(image_id)
                    require(isinstance(record, dict), f"JSON has no valid scan record for {image_id}: {relative}")
                    require(type(record.get("label")) is int and record["label"] == metadata_label,
                            f"Folder/JSON label mismatch: {relative}, JSON label={record.get('label')!r}")
                    raw = record.get("raw")
                    require(isinstance(raw, str), f"Scan {image_id} has no raw path")
                    patient_match = PATIENT_PATTERN.search(raw)
                    source_match = IMAGE_PATTERN.search(raw)
                    require(patient_match is not None and source_match is not None,
                            f"Cannot parse patient/scan identifiers from metadata path: {raw}")
                    require(source_match.group(1) == image_id, f"JSON key and path scan identifier disagree: {image_id}")
                    patient_id = patient_match.group(1)
                    # Available derivative paths must refer to the same patient
                    # and scan as raw; missing optional paths are not required.
                    for field in ("c1", "c2", "c3", "c4", "c5", "masked"):
                        value = record.get(field)
                        if value is None:
                            continue
                        require(isinstance(value, str), f"{image_id}: {field} is not a path string")
                        pm, im = PATIENT_PATTERN.search(value), IMAGE_PATTERN.search(value)
                        require(pm is not None and im is not None and pm.group(1) == patient_id
                                and im.group(1) == image_id, f"{image_id}: {field} and raw identify different patients or scans")
                    identity = (patient_id, label)
                    require(image_id not in identities or identities[image_id] == identity,
                            f"One scan maps to different patients or labels: {image_id}")
                    identities[image_id] = identity
                    slice_key = (image_id, slice_index)
                    # Different filenames must not disguise the same scan/slice.
                    require(slice_key not in seen_slices, f"Duplicate scan/slice identifier: {slice_key}")
                    seen_slices.add(slice_key)

                    content = path.read_bytes()
                    with Image.open(io.BytesIO(content)) as original:
                        require(original.format == "JPEG", f"File extension does not match the actual image format: {relative}")
                        original.load()  # Detect truncated/undecodable JPEGs, not just header errors.
                        mode = original.mode
                        # Compare decoded pixels as well as file bytes: JPEG
                        # encodings or metadata can differ for identical pixels.
                        # This normalization supports hashing, not preprocessing
                        # for model input, and does not detect near-duplicates.
                        canonical = ImageOps.exif_transpose(original).convert("RGB")
                        width, height = canonical.size
                        pixel_hash = digest(json_bytes([width, height, "RGB"]) + b"\0" + canonical.tobytes())
                    previous = pixel_owners.get(pixel_hash)
                    if previous is not None:
                        previous_patient, previous_label, previous_path = previous
                        require(previous_label == label,
                                f"Identical pixels have different labels: {previous_path} <-> {relative}")
                        require(previous_patient == patient_id,
                                f"Identical pixels occur across patients; review before splitting: {previous_path} <-> {relative}")
                        # Same-patient, same-label copies are recorded; keeping
                        # that patient together prevents them crossing roles.
                        within_patient_duplicates += 1
                    else:
                        pixel_owners[pixel_hash] = (patient_id, label, relative)
                    row = dict(zip(SOURCE_FIELDS, [relative, split, folder_label, label, metadata_label,
                               patient_id, image_id, slice_index, width, height, mode, len(content),
                               digest(content), pixel_hash]))
                    rows.append({key: str(value) for key, value in row.items()})
                except (AuditError, OSError, ValueError) as exc:
                    errors.append(f"{relative}: {exc}")
    if errors:
        raise AuditError(f"Data audit failed with {len(errors)} issues; first 12:\n" + "\n".join(errors[:12]))

    scans = Counter(row["image_id"] for row in rows)
    # image_id identifies a scan, whereas each row is one JPEG slice. The
    # expected count checks completeness; it does not infer slice anatomy.
    incomplete = {key: count for key, count in scans.items() if count != expected_slices}
    require(not incomplete, f"Expected {expected_slices} slices per scan; invalid examples: {list(incomplete.items())[:10]}")
    rows.sort(key=lambda row: row["relative_path"])
    profiles = profiles_for(rows)
    old_patients = {split: {r["patient_id"] for r in rows if r["original_split"] == split}
                    for split in ("train", "test")}
    old_scans = {split: {r["image_id"] for r in rows if r["original_split"] == split}
                 for split in ("train", "test")}
    summary = {
        "images": len(rows), "image_records": len(scans), "patients": len(profiles),
        "patient_strata": dict(Counter(p["stratum"] for p in profiles.values())),
        "mixed_label_patients": sorted(p for p, profile in profiles.items() if profile["stratum"] == "mixed"),
        "original_patient_overlap": len(old_patients["train"] & old_patients["test"]),
        "original_image_overlap": len(old_scans["train"] & old_scans["test"]),
        "within_patient_same_label_duplicate_pixels": within_patient_duplicates,
        "ignored_non_jpeg_files": ignored,
        "metadata_sha256": digest(metadata_bytes), "source_fingerprint": digest(json_bytes(rows)),
        "python_version": sys.version.split()[0], "pillow_version": PIL.__version__,
        "pixel_hash_rule": "SHA256(dimensions + RGB decoded pixels after EXIF transpose); exact equality only",
    }
    return rows, summary


def stratified_take(patient_ids, count, profiles, rng):
    """Take a fixed patient count with largest-remainder stratum allocation.

    Strata are patient label histories, not slice counts. This is deliberately
    independent of scikit-learn's sample-weighted StratifiedGroupKFold.
    """
    patient_ids = sorted(patient_ids)
    require(0 < count < len(patient_ids), "The patient subset is too small for the requested independent holdout.")
    buckets = defaultdict(list)
    for patient_id in patient_ids:
        buckets[profiles[patient_id]["stratum"]].append(patient_id)
    exact = {key: len(value) * count / len(patient_ids) for key, value in buckets.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remainder = count - sum(quotas.values())
    for key in sorted(buckets, key=lambda key: (-(exact[key] - quotas[key]), key))[:remainder]:
        quotas[key] += 1
    chosen = set()
    for key in sorted(buckets):
        rng.shuffle(buckets[key])
        chosen.update(buckets[key][:quotas[key]])
    return chosen, set(patient_ids) - chosen


def make_plan(rows, config):
    """Reserve test/calibration patients, then assign development CV roles.

    Patient-history strata balance patient counts, not slice counts. This is
    a custom deterministic splitter, not sklearn's StratifiedGroupKFold.
    """
    profiles = profiles_for(rows)
    rng = random.Random(config["seed"])
    patients = set(profiles)
    test_count = math.floor(len(patients) * config["test_fraction"] + 0.5)
    calibration_count = math.floor(len(patients) * config["calibration_fraction"] + 0.5)
    # Both held-out counts are fractions of all patients. Neither group enters
    # CV: calibration is reserved for thresholds after model selection, and
    # test is reserved for the final frozen pipeline's evaluation.
    test, remaining = stratified_take(patients, test_count, profiles, rng)
    calibration, development = stratified_take(remaining, calibration_count, profiles, rng)
    folds = config["folds"]
    require(len(development) >= folds, "The development set has fewer patients than folds.")
    assignment = {patient: ("test", "") for patient in test}
    assignment.update({patient: ("calibration", "") for patient in calibration})
    loads = [0] * folds
    for stratum in sorted({p["stratum"] for p in profiles.values()}):
        bucket = sorted(p for p in development if profiles[p]["stratum"] == stratum)
        rng.shuffle(bucket)
        stratum_loads = [0] * folds
        for patient in bucket:
            best = min((stratum_loads[i], loads[i]) for i in range(folds))
            choices = [i for i in range(folds) if (stratum_loads[i], loads[i]) == best]
            fold = rng.choice(choices)
            assignment[patient] = ("development", str(fold + 1))
            stratum_loads[fold] += 1
            loads[fold] += 1
    early_stops = {}
    for fold in range(1, folds + 1):
        # The outer validation patients never select a stopping epoch. Draw a
        # separate early-stop group from this fold's training side instead.
        training_pool = {p for p in development if assignment[p][1] != str(fold)}
        count = math.floor(len(training_pool) * config["early_stop_fraction"] + 0.5)
        early, _ = stratified_take(training_pool, count, profiles, random.Random(config["seed"] + 1000 + fold))
        early_stops[str(fold)] = sorted(early)
    return assignment, early_stops


def patient_rows(rows):
    """Build a patient-level summary of assigned roles and scan/image counts."""
    profiles = profiles_for(rows)
    assignment = {row["patient_id"]: (row["partition"], row["fold"]) for row in rows}
    return [dict(zip(PATIENT_FIELDS, [patient, profiles[patient]["stratum"],
                 "|".join(map(str, sorted(profiles[patient]["labels"]))),
                 str(len(profiles[patient]["scans"])), str(profiles[patient]["images"]),
                 *assignment[patient]])) for patient in sorted(profiles)]


def partition_summary(rows):
    """Report patient, scan, slice, and label counts for one manifest."""
    profiles = profiles_for(rows)
    scans = {r["image_id"]: r for r in rows}
    return {"images": len(rows), "image_records": len(scans), "patients": len(profiles),
            "images_by_label": dict(Counter(r["folder_label"] for r in rows)),
            "image_records_by_label": dict(Counter(r["folder_label"] for r in scans.values())),
            "patient_strata": dict(Counter(p["stratum"] for p in profiles.values()))}


def artifacts_for(rows, config, early_stops):
    """Expand patient assignments into the complete set of CSV manifests.

    A development patient's fold identifies their one outer-validation turn.
    Their role can change to train or early_stop in another independently
    trained fold; isolation is required within each fold, not across all folds.
    """
    artifacts = {"all.csv": rows, "patients.csv": patient_rows(rows)}
    for partition in ("development", "calibration", "test"):
        artifacts[f"{partition}.csv"] = [r for r in rows if r["partition"] == partition]
    development = artifacts["development.csv"]
    for fold in range(1, config["folds"] + 1):
        early = set(early_stops[str(fold)])
        pool = [r for r in development if r["fold"] != str(fold)]
        prefix = f"fold_{fold:02d}"
        artifacts[f"{prefix}/train.csv"] = [r for r in pool if r["patient_id"] not in early]
        artifacts[f"{prefix}/early_stop.csv"] = [r for r in pool if r["patient_id"] in early]
        artifacts[f"{prefix}/val.csv"] = [r for r in development if r["fold"] == str(fold)]
    return artifacts


def check_boundaries(artifacts, config):
    """Check within-fold isolation, held-out isolation, and full CV coverage.

    These checks use manifest identifiers and exact content fingerprints.
    They cannot establish that upstream patient identifiers are correct.
    """
    def check(roles):
        """Require both classes and pairwise disjoint identities/content."""
        for name in roles:
            rows = artifacts[name]
            require(rows, f"Empty partition: {name}")
            require({r["label"] for r in rows} == {"0", "1"},
                    f"{name} does not contain both diagnoses, so the required classification metrics cannot be reliably computed. Check the patient count and prespecified fractions.")
        for index, first in enumerate(roles):
            for second in roles[index + 1:]:
                for field in ("patient_id", "image_id", "relative_path", "file_sha256", "pixel_sha256"):
                    overlap = {r[field] for r in artifacts[first]} & {r[field] for r in artifacts[second]}
                    require(not overlap, f"Overlap across partitions: {first} <-> {second}, {field}: {list(overlap)[:3]}")

    check(["development.csv", "calibration.csv", "test.csv"])
    all_val_paths = []
    development_paths = {r["relative_path"] for r in artifacts["development.csv"]}
    for fold in range(1, config["folds"] + 1):
        prefix = f"fold_{fold:02d}"
        roles = [f"{prefix}/{name}.csv" for name in ("train", "early_stop", "val")]
        check(roles + ["calibration.csv", "test.csv"])
        union = {r["relative_path"] for role in roles for r in artifacts[role]}
        require(union == development_paths, f"Fold {fold} does not exactly cover the complete development set.")
        all_val_paths.extend(r["relative_path"] for r in artifacts[f"{prefix}/val.csv"])
    require(set(all_val_paths) == development_paths and len(all_val_paths) == len(development_paths),
            "Every development image must appear in outer validation exactly once.")


def write_csv(path, rows, fields):
    """Write a manifest with a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path, fields):
    """Read a manifest only when its columns match the declared schema."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == fields, f"CSV fields do not match the protocol: {path}")
        return list(reader)


def config_from_args(args):
    """Validate the reproducible split settings used by prepare and verify."""
    config = {key: getattr(args, key) for key in ("seed", "folds", "test_fraction",
              "calibration_fraction", "early_stop_fraction", "expected_slices")}
    require(config["folds"] >= 2, "Cross-validation requires at least 2 folds.")
    require(config["expected_slices"] > 0, "expected-slices must be greater than zero.")
    require(0 < config["test_fraction"] < 1 and 0 < config["calibration_fraction"] < 1
            and config["test_fraction"] + config["calibration_fraction"] < 1,
            "Test and calibration fractions must be positive and sum to less than 1.")
    require(0 < config["early_stop_fraction"] < 0.5, "early-stop-fraction must be between 0 and 0.5.")
    return config


def prepare(args):
    """Audit sources, create a new split, and publish only complete outputs."""
    root, output = args.data_root.resolve(), args.output.resolve()
    require(output != root and root not in output.parents, "The output directory must be outside the source dataset directory.")
    require(not output.exists(), f"Output already exists; refusing to overwrite the frozen split: {output}. Use verify to check it.")
    config = config_from_args(args)
    print("Checking metadata and decoding all JPEG images...", flush=True)
    source_rows, audit = inventory(root, config["expected_slices"])
    profiles = profiles_for(source_rows)
    assignment, early_stops = make_plan(source_rows, config)
    rows = [dict(row, patient_stratum=profiles[row["patient_id"]]["stratum"],
                 partition=assignment[row["patient_id"]][0], fold=assignment[row["patient_id"]][1])
            for row in source_rows]
    artifacts = artifacts_for(rows, config, early_stops)
    check_boundaries(artifacts, config)
    report = {
        "protocol_version": VERSION, "config": config, "data_root_at_creation": str(root),
        "source_audit": audit, "early_stop_patients": early_stops,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "stratification": "Patient histories: AD_only / NC_only / mixed; per-scan labels unchanged.",
        "partitions": {name: partition_summary(value) for name, value in artifacts.items()
                       if name not in ("all.csv", "patients.csv")},
        "checks": {"patient_image_path_and_exact_hash_boundaries": "passed",
                   "outer_validation_exactly_once_per_development_image": "passed"},
        "limitations": ["Checks use the patient identifiers supplied by the dataset.",
                        "Exact decoded duplicates are checked; near-duplicates are not exhaustively detected.",
                        "Upstream preprocessing provenance and future training code require separate review.",
                        "Fold models must be initialized independently; CSVs alone cannot enforce this."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary sibling directory first. Failed checks or writes
    # must not leave a partially populated directory that appears ready to use.
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent)))
    try:
        for name, values in artifacts.items():
            write_csv(stage / name, values, PATIENT_FIELDS if name == "patients.csv" else FIELDS)
        write_json(stage / "report.json", report)
        checksums = {p.relative_to(stage).as_posix(): digest(p.read_bytes())
                     for p in sorted(stage.rglob("*")) if p.is_file()}
        write_json(stage / "COMPLETED.json", {"protocol_version": VERSION, "sha256": checksums})
        require(not output.exists(), f"The output directory was created during this run; refusing to overwrite: {output}")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"Patients shared by the original train/test folders: {audit['original_patient_overlap']}")
    print(f"Audited dataset: {audit['patients']} patients / {audit['image_records']} scans / {audit['images']} images")
    for name in ("development.csv", "calibration.csv", "test.csv"):
        value = report["partitions"][name]
        print(f"{name}: {value['patients']} patients, {value['images']} images")
    print(f"Created {config['folds']} folds; all patient, scan, and exact-duplicate boundary checks passed.")
    print(f"Output: {output}\nRun verify before training; training must read the generated manifests.")


def verify(args):
    """Independently reread sources and reconstruct the recorded split.

    Checksums reveal changed outputs, but are not proof of correct identities.
    Therefore, do not trust patient IDs or assignments merely because they
    appear in a CSV: rebuild them from source metadata and decoded images.
    """
    root, output = args.data_root.resolve(), args.output.resolve()
    seal = read_json(output / "COMPLETED.json")
    require(seal.get("protocol_version") == VERSION, "Unsupported manifest protocol version.")
    checksums = seal.get("sha256")
    require(isinstance(checksums, dict), "The completion marker is missing file checksums.")
    actual_files = {p.relative_to(output).as_posix() for p in output.rglob("*")
                    if p.is_file() and p != output / "COMPLETED.json"}
    require(actual_files == set(checksums), "The output file set has changed or is incomplete.")
    for relative, expected in checksums.items():
        path = output / relative
        require(path.resolve().is_relative_to(output), "The completion marker contains a path outside the output directory.")
        require(digest(path.read_bytes()) == expected, f"Output file was modified: {relative}")
    report = read_json(output / "report.json")
    require(report.get("protocol_version") == VERSION, "Report protocol version mismatch.")
    config = config_from_args(argparse.Namespace(**report["config"]))
    print("Rereading source data, metadata, and decoded pixels to independently verify the manifests...", flush=True)
    source_rows, audit = inventory(root, config["expected_slices"])
    for field in ("source_fingerprint", "metadata_sha256"):
        require(audit[field] == report["source_audit"][field], f"Source data or metadata has changed: {field}")
    assignment, early_stops = make_plan(source_rows, config)
    # Reproduce the frozen seed/protocol, then compare every manifest row.
    # Editing a patient ID and updating its checksum cannot bypass this check.
    require(early_stops == report["early_stop_patients"], "Early-stopping patient lists disagree with the fixed seed.")
    profiles = profiles_for(source_rows)
    expected_rows = [dict(row, patient_stratum=profiles[row["patient_id"]]["stratum"],
                         partition=assignment[row["patient_id"]][0], fold=assignment[row["patient_id"]][1])
                     for row in source_rows]
    expected_artifacts = artifacts_for(expected_rows, config, early_stops)
    require(set(checksums) == set(expected_artifacts) | {"report.json"}, "The manifest file set does not match the protocol.")
    actual_artifacts = {}
    for name, expected in expected_artifacts.items():
        actual = read_csv(output / name, PATIENT_FIELDS if name == "patients.csv" else FIELDS)
        require(actual == expected, f"Manifest disagrees with source identities, the fixed split, or complete coverage: {name}")
        actual_artifacts[name] = actual
    check_boundaries(actual_artifacts, config)
    for name, expected in expected_artifacts.items():
        if name not in ("all.csv", "patients.csv"):
            require(report["partitions"][name] == partition_summary(expected), f"Reported statistics disagree with the manifest: {name}")
    print("PASS: source data matches; no patient, scan, path, or exact-duplicate overlap across required boundaries; every fold is complete and each development image is validated exactly once.")
    print("This result does not replace review of the training pipeline, near-duplicates, or upstream preprocessing.")


def main(argv=None):
    """Expose prepare/verify commands and return a nonzero status on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--data-root", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if command == "prepare":
            child.add_argument("--seed", type=int, default=3710)
            child.add_argument("--folds", type=int, default=5)
            child.add_argument("--test-fraction", type=float, default=0.20)
            child.add_argument("--calibration-fraction", type=float, default=0.10)
            child.add_argument("--early-stop-fraction", type=float, default=0.10)
            child.add_argument("--expected-slices", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        {"prepare": prepare, "verify": verify}[args.command](args)
    except (AuditError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
