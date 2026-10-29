"""Read-only JPEG integrity checks and reproducible one-scan-per-patient selection.

Pillow is required. No medical quality assessment or model predictions are used.
Outputs contain patient identifiers: keep them outside version control.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess

from PIL import Image, ImageFile, __version__ as pillow_version

FIELDS = ("file_path", "original_split", "image_id", "patient_id", "scan_date", "label")
CHECK_FIELDS = ("image_id", "patient_id", "label", "slice_count",
                "unique_slice_count", "readable_count", "qualified", "reasons",
                "dimensions", "modes", "channels", "slice_indices",
                "patient_qualified_scan_count")
SELECT_FIELDS = ("patient_id", "selected_image_id", "label", "candidate_scan_count")
ALGORITHM = ("Sort patient_id lexicographically; sort each patient's qualified "
             "image_id candidates as integers; initialize random.Random(seed) once; "
             "call choice once per patient, in that order.")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_samples(path):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Input must have exactly the six B2 manifest columns.")
        rows = list(reader)
    if not rows or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Empty or malformed input manifest.")
    return rows


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_id"]].append(row)
    return groups


def check_scans(rows):
    """Open every listed path; verify JPEG structure and fully load decoded pixels."""
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    groups = grouped(rows)
    path_counts = Counter(r["file_path"] for r in rows)
    patient_labels = defaultdict(set)
    for r in rows:
        patient_labels[r["patient_id"]].add(r["label"])
    checks = []
    for image_id, group in sorted(groups.items()):
        reasons, indices, dimensions, modes, channels = [], [], set(), set(), set()
        patients = {r["patient_id"] for r in group}
        labels = {r["label"] for r in group}
        patient = next(iter(patients)) if len(patients) == 1 else ""
        label = next(iter(labels)) if len(labels) == 1 else ""
        if not re.fullmatch(r"[1-9]\d*", image_id):
            reasons.append("invalid_image_id")
        if len(patients) != 1 or not re.fullmatch(r"\d{3}_S_\d{4}", patient):
            reasons.append("missing_invalid_or_conflicting_patient")
        if len(labels) != 1 or label not in ("AD", "NC"):
            reasons.append("missing_invalid_or_conflicting_label")
        if any(len(patient_labels[p]) != 1 or not patient_labels[p] <= {"AD", "NC"}
               for p in patients):
            reasons.append("patient_label_conflict_across_scans")
        if len(group) != 20:
            reasons.append(f"slice_count:{len(group)}:expected_20")
        readable = 0
        for row in group:
            path = Path(row["file_path"])
            match = re.fullmatch(r"(\d+)_(\d+)\.(?:jpeg|jpg)", path.name, re.I)
            if not match or match.group(1) != image_id:
                reasons.append(f"filename_image_id_mismatch:{path}")
            else:
                indices.append(int(match.group(2)))
            if path_counts[row["file_path"]] != 1:
                reasons.append(f"duplicate_path:{path}")
            if row["scan_date"]:
                reasons.append(f"unexpected_nonblank_scan_date:{path}")
            if row["original_split"] not in ("train", "test"):
                reasons.append(f"invalid_original_split:{path}")
            try:
                with Image.open(path) as image:
                    if image.format != "JPEG":
                        raise ValueError("File is not JPEG")
                    image.verify()
                # verify() alone does not decode pixels; reopen and load all pixels.
                with Image.open(path) as image:
                    image.load()
                    dimensions.add(tuple(image.size))
                    modes.add(image.mode)
                    channels.add(len(image.getbands()))
                readable += 1
            except Exception as exc:
                reasons.append(f"unreadable:{path}:{type(exc).__name__}:{exc}")
        if len(set(indices)) != 20 or len(indices) != 20:
            reasons.append(f"unique_slice_indices:{len(set(indices))}:expected_20")
        checks.append(dict(image_id=image_id, patient_id=patient, label=label,
                           slice_count=len(group), unique_slice_count=len(set(indices)),
                           readable_count=readable, qualified=not reasons,
                           reasons=sorted(set(reasons)),
                           dimensions=sorted(dimensions), modes=sorted(modes),
                           channels=sorted(channels), slice_indices=sorted(set(indices))))
    counts = Counter(c["patient_id"] for c in checks if c["qualified"])
    for c in checks:
        c["patient_qualified_scan_count"] = counts[c["patient_id"]]
    return checks


def choose_scans(rows, checks, seed):
    """Rebuild candidates from input rows on each call; never sample individual slices."""
    by_id = grouped(rows)
    qualified = {c["image_id"] for c in checks if c["qualified"]}
    candidates = defaultdict(list)
    patients = {r["patient_id"] for r in rows}
    for image_id in qualified:
        group = by_id[image_id]
        candidates[group[0]["patient_id"]].append(image_id)
    if "" in patients or any(not candidates[p] for p in patients):
        raise ValueError("At least one patient has no qualified scan; discuss before selecting.")
    rng = random.Random(seed)
    selected, samples = [], []
    for patient in sorted(patients):
        pool = sorted(candidates[patient], key=int)
        image_id = rng.choice(pool)
        group = by_id[image_id]
        selected.append(dict(patient_id=patient, selected_image_id=image_id,
                             label=group[0]["label"], candidate_scan_count=len(pool)))
        samples.extend(sorted(group, key=lambda r: int(Path(r["file_path"]).stem.split("_")[-1])))
    return selected, samples


def write_csv(path, fields, rows):
    with Path(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict))
                             else v for k, v in row.items()})
    Path(path).chmod(0o600)


def code_version():
    here = Path(__file__).resolve()
    def git(*args):
        result = subprocess.run(["git", "-C", str(here.parent), *args],
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    except FileNotFoundError:
        commit = status = None
    return dict(base_commit=commit, working_tree_dirty=None if status is None else bool(status),
                script_sha256=sha256(here), committed_script=False)


def run(input_path, output_dir, seed=42, version=None):
    input_path = Path(input_path).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Output directory already exists; use a new versioned directory.")
    rows = read_samples(input_path)
    # Avoid writing into any input-image directory or its descendants.
    for parent in {Path(r["file_path"]).resolve().parent for r in rows}:
        if output == parent or parent in output.parents:
            raise ValueError("Output must be outside source image directories.")
    input_hash = sha256(input_path)
    checks = check_scans(rows)
    if sha256(input_path) != input_hash:
        raise RuntimeError("Input manifest changed during inspection.")
    output.mkdir(parents=True, mode=0o700)
    write_csv(output / "scan_checks.csv", CHECK_FIELDS, checks)
    patients = {r["patient_id"] for r in rows}
    counts = Counter(c["patient_id"] for c in checks if c["qualified"])
    unavailable = sorted(p for p in patients if not counts[p])
    qualified = [c for c in checks if c["qualified"]]
    summary = dict(
        status="blocked" if unavailable else "checking",
        generated_at=datetime.now(timezone.utc).isoformat(),
        seed=seed, algorithm=ALGORITHM,
        python_version=platform.python_version(), pillow_version=pillow_version,
        code_version=version or code_version(),
        input_manifest=str(input_path), input_manifest_sha256=input_hash,
        scan_checks_sha256=sha256(output / "scan_checks.csv"),
        input_slice_count=len(rows), checked_scan_count=len(checks),
        qualified_scan_count=len(qualified), excluded_scan_count=len(checks)-len(qualified),
        input_patient_count=len(patients), patients_without_qualified_scans=len(unavailable),
        patients_without_qualified_scan_ids=unavailable,
        qualified_scan_count_per_patient=dict(sorted((p, counts[p]) for p in patients)),
        observed_dimensions=sorted({tuple(d) for c in checks for d in c["dimensions"]}),
        observed_modes=sorted({m for c in checks for m in c["modes"]}),
        observed_channels=sorted({n for c in checks for n in c["channels"]}),
        scans_with_multiple_dimensions=sum(len(c["dimensions"]) > 1 for c in checks),
        dimensions_are_exclusion_criteria=False,
        scan_date_policy="Unverified acquisition dates remain blank.",
        scope="File integrity only; not medical image quality assessment.",
        outputs={"scan_checks": str(output / "scan_checks.csv"),
                 "summary": str(output / "selection_summary.json")})
    if not unavailable:
        selected, samples = choose_scans(rows, checks, seed)
        again, again_samples = choose_scans(rows, checks, seed)
        shuffled = list(rows)
        random.Random(2026).shuffle(shuffled)
        after_shuffle, shuffled_samples = choose_scans(shuffled, checks, seed)
        mapping = lambda records: {r["patient_id"]: r["selected_image_id"] for r in records}
        source_records = {tuple(r[k] for k in FIELDS) for r in rows}
        chosen_groups = grouped(samples)
        patient_groups = defaultdict(set)
        for r in samples:
            patient_groups[r["patient_id"]].add(r["image_id"])
        validation = dict(
            repeat_mapping_identical=mapping(selected) == mapping(again),
            shuffled_input_mapping_identical=mapping(selected) == mapping(after_shuffle),
            repeat_samples_identical=samples == again_samples,
            shuffled_input_samples_identical=samples == shuffled_samples,
            no_duplicate_paths=len({r["file_path"] for r in samples}) == len(samples),
            one_scan_per_patient=all(len(v) == 1 for v in patient_groups.values()),
            all_patients_retained=set(patient_groups) == patients and len(selected) == len(patients),
            one_patient_per_scan=all(len({r["patient_id"] for r in g}) == 1 for g in chosen_groups.values()),
            twenty_unique_slices_per_scan=all(
                len(g) == 20 and len({int(Path(r["file_path"]).stem.split("_")[-1]) for r in g}) == 20
                for g in chosen_groups.values()),
            source_subset=all(tuple(r[k] for k in FIELDS) in source_records for r in samples),
            all_selected_scans_qualified=set(chosen_groups) <= {c["image_id"] for c in qualified},
            scan_dates_blank=all(not r["scan_date"] for r in samples))
        if not all(validation.values()):
            raise RuntimeError(f"Selection validation failed: {validation}")
        write_csv(output / "selected_scans.csv", SELECT_FIELDS, selected)
        write_csv(output / "selected_samples.csv", FIELDS, samples)
        summary.update(status="passed", selected_patient_count=len(selected),
                       selected_slice_count=len(samples),
                       selected_patient_class_counts=dict(Counter(r["label"] for r in selected)),
                       selected_slice_class_counts=dict(Counter(r["label"] for r in samples)),
                       duplicate_path_count=len(samples)-len({r["file_path"] for r in samples}),
                       validation=validation)
        for name in ("selected_scans", "selected_samples"):
            summary["outputs"][name] = str(output / (name + ".csv"))
            summary[name + "_sha256"] = sha256(output / (name + ".csv"))
    with (output / "selection_summary.json").open("x") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    (output / "selection_summary.json").chmod(0o600)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-version-json", type=Path,
                        help="Local git provenance for a standalone cluster script copy.")
    args = parser.parse_args()
    version = json.loads(args.code_version_json.read_text()) if args.code_version_json else None
    if version and version["script_sha256"] != sha256(__file__):
        raise SystemExit("Code provenance hash does not match this script.")
    result = run(args.input, args.output_dir, args.seed, version)
    print(json.dumps({k: v for k, v in result.items() if k in
                     ("status", "checked_scan_count", "qualified_scan_count",
                      "excluded_scan_count", "patients_without_qualified_scans",
                      "selected_patient_count", "selected_slice_count")}), flush=True)
    raise SystemExit(2 if result["status"] == "blocked" else 0)
