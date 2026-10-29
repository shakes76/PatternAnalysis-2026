"""B6: strict, reproducible stratified patient splits using manifests only."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import random
import re
import subprocess

FIELDS = ("file_path", "original_split", "image_id", "patient_id", "scan_date", "label")
PATIENT_FIELDS = ("patient_id", "selected_image_id", "label", "split")
SPLITS = ("train", "validation", "test")
QUOTAS = {"AD": dict(zip(SPLITS, (133, 44, 44))),
          "NC": dict(zip(SPLITS, (275, 92, 92)))}
ALGORITHM = ("Sort patient_id strings within AD and NC; initialize independent "
             "random.Random(42) once; shuffle AD then NC once each; assign "
             "successive slices using fixed AD 133/44/44 and NC 275/92/92 quotas.")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_version(path=None):
    """Report HEAD identity, actual worktree state and exact HEAD/script equality."""
    path = Path(path or __file__).resolve()
    def git(*args):
        result = subprocess.run(["git", "-C", str(path.parent), *args],
                                capture_output=True)
        return result.stdout if result.returncode == 0 else None
    commit = dirty = committed = None
    try:
        root = git("rev-parse", "--show-toplevel")
        if root:
            root = Path(root.decode().strip())
            head = git("rev-parse", "HEAD")
            commit = head.decode().strip() if head else None
            status = git("status", "--porcelain", "--untracked-files=all")
            dirty = bool(status) if status is not None else None
            blob = git("show", "HEAD:" + path.relative_to(root).as_posix())
            committed = blob is not None and blob == path.read_bytes()
    except FileNotFoundError:
        pass
    return dict(commit=commit, working_tree_dirty=dirty, committed_script=committed,
                script_sha256=sha256(path))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build(scans, samples):
    """Validate all associations before using one independent, fixed-seed RNG."""
    require(len(scans) == 680, "Expected exactly 680 patient rows")
    by_patient, images = {}, set()
    for row in scans:
        p, image, label = row["patient_id"], row["selected_image_id"], row["label"]
        require(p not in by_patient, "duplicate patient in selected_scans")
        require(re.fullmatch(r"\d{3}_S_\d{4}", p), "invalid patient ID")
        require(re.fullmatch(r"[1-9]\d*", image), "invalid image ID")
        require(image not in images, "image ID assigned to multiple patients")
        require(label in QUOTAS, "unknown label")
        by_patient[p] = row
        images.add(image)
    require(Counter(r["label"] for r in scans) == {"AD": 221, "NC": 459},
            "Expected AD 221 / NC 459")
    grouped, paths = defaultdict(list), set()
    for row in samples:
        p = row["patient_id"]
        require(p in by_patient, "unmatched patient in selected_samples")
        parent = by_patient[p]
        require(row["image_id"] == parent["selected_image_id"], "image_id mismatch")
        require(row["label"] == parent["label"], "label mismatch")
        require(row["file_path"] not in paths, "duplicate file_path")
        paths.add(row["file_path"])
        require(not row["scan_date"], "scan_date must remain blank")
        require(row["original_split"] in ("train", "test"), "invalid original_split")
        match = re.fullmatch(r"(\d+)_(\d+)\.(?:jpeg|jpg)",
                            Path(row["file_path"]).name, re.I)
        require(match and match.group(1) == row["image_id"], "filename/image mismatch")
        grouped[p].append(row)
    for p in by_patient:
        require(len(grouped[p]) == 20, "Each patient must have exactly 20 slices")
        require(len({int(Path(r["file_path"]).stem.split("_")[-1])
                     for r in grouped[p]}) == 20, "duplicate slice index")
    rng = random.Random(42)
    mapping = {}
    for label in ("AD", "NC"):
        patients = sorted(p for p, r in by_patient.items() if r["label"] == label)
        rng.shuffle(patients)
        offset = 0
        for split in SPLITS:
            count = QUOTAS[label][split]
            mapping.update(dict.fromkeys(patients[offset:offset+count], split))
            offset += count
    patient_rows = [dict(patient_id=p, selected_image_id=by_patient[p]["selected_image_id"],
                         label=by_patient[p]["label"], split=mapping[p])
                    for p in sorted(by_patient)]
    slice_rows = sorted((dict(r, split=mapping[r["patient_id"]]) for r in samples),
                        key=lambda r: (r["patient_id"], r["image_id"], r["file_path"]))
    checks = {
        "exact_patient_set": set(mapping) == set(by_patient) and len(patient_rows) == 680,
        "exact_slice_multiset": Counter(tuple(r[k] for k in FIELDS) for r in slice_rows) ==
                                Counter(tuple(r[k] for k in FIELDS) for r in samples),
        "no_duplicate_paths": len(paths) == len(samples),
        "twenty_slices_per_patient": all(len(v) == 20 for v in grouped.values()),
        "one_split_per_patient": all(len({r["split"] for r in slice_rows
                                          if r["patient_id"] == p}) == 1 for p in mapping),
        "fixed_class_quotas": all(
            sum(r["label"] == label and r["split"] == split for r in patient_rows) == n
            for label, quotas in QUOTAS.items() for split, n in quotas.items()),
        "exact_slice_class_counts": all(
            sum(r["label"] == label and r["split"] == split for r in slice_rows) == n*20
            for label, quotas in QUOTAS.items() for split, n in quotas.items()),
        "scan_and_label_unchanged": all(
            (r["selected_image_id"], r["label"]) ==
            (by_patient[r["patient_id"]]["selected_image_id"], by_patient[r["patient_id"]]["label"])
            for r in patient_rows)}
    require(all(checks.values()), "Output integrity validation failed")
    return patient_rows, slice_rows, checks


def read_csv(path, required, exact=False):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        require(len(header) == len(set(header)) and
                (tuple(header) == tuple(required) if exact else set(required) <= set(header)),
                f"Invalid columns in {Path(path).name}")
        rows = list(reader)
    require(rows and all(None not in r and all(v is not None for v in r.values()) for r in rows),
            f"Empty or malformed rows in {Path(path).name}")
    return rows


def write_csv(path, fields, rows):
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    path.chmod(0o600)


def run(selection_dir, output_dir, provenance=None):
    source = Path(selection_dir).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve()
    require(output != source and source not in output.parents,
            "Outputs must be outside selection input directory")
    if output.exists():
        raise FileExistsError("Output directory already exists; use a new versioned location")
    names = ("selected_scans", "selected_samples")
    hashes = {name: sha256(source / (name + ".csv")) for name in names}
    evidence_path = source / "selection_summary.json"
    evidence_hash = sha256(evidence_path)
    evidence = json.loads(evidence_path.read_text())
    require(evidence.get("status") == "passed", "Selection stage has not passed")
    for name in names:
        require(hashes[name] == evidence.get(name + "_sha256"), f"{name} hash mismatch")
    scans = read_csv(source / "selected_scans.csv", PATIENT_FIELDS[:3])
    samples = read_csv(source / "selected_samples.csv", FIELDS, exact=True)
    patients, slices, validation = build(scans, samples)
    repeat = build(scans, samples)
    shuffled_scans, shuffled_samples = list(scans), list(samples)
    shuffle = random.Random(2026)
    shuffle.shuffle(shuffled_scans)
    shuffle.shuffle(shuffled_samples)
    shuffled = build(shuffled_scans, shuffled_samples)
    mapping = lambda rs: {r["patient_id"]: r["split"] for r in rs}
    validation.update(
        repeat_mapping_identical=mapping(patients) == mapping(repeat[0]),
        shuffled_inputs_mapping_identical=mapping(patients) == mapping(shuffled[0]),
        repeat_csv_records_identical=(patients, slices) == repeat[:2],
        shuffled_inputs_csv_records_identical=(patients, slices) == shuffled[:2])
    intersections = {}
    for field in ("patient_id", "image_id", "file_path"):
        sets = {split: {r[field] for r in slices if r["split"] == split} for split in SPLITS}
        intersections[field] = {a + "_vs_" + b: len(sets[a] & sets[b])
                                for a, b in combinations(SPLITS, 2)}
    validation["all_intersections_zero"] = all(
        n == 0 for pairs in intersections.values() for n in pairs.values())
    require(all(validation.values()), "Reproducibility or intersection validation failed")
    require(hashes == {name: sha256(source / (name + ".csv")) for name in names} and
            evidence_hash == sha256(evidence_path), "Inputs changed during execution")
    version = provenance or code_version()
    require(version["script_sha256"] == sha256(__file__), "Code provenance hash mismatch")
    counts = {}
    for split in SPLITS:
        ps = [r for r in patients if r["split"] == split]
        ss = [r for r in slices if r["split"] == split]
        counts[split] = dict(patients=len(ps), images=len({r["image_id"] for r in ss}),
                             slices=len(ss),
                             patient_classes=dict(Counter(r["label"] for r in ps)),
                             slice_classes=dict(Counter(r["label"] for r in ss)))
    output.mkdir(parents=True, mode=0o700)
    write_csv(output / "patient_splits.csv", PATIENT_FIELDS, patients)
    write_csv(output / "split_samples.csv", (*FIELDS, "split"), slices)
    summary = dict(
        status="passed", generated_at=datetime.now(timezone.utc).isoformat(),
        split_seed=42, ratios=dict(zip(SPLITS, (0.6, 0.2, 0.2))), class_quotas=QUOTAS,
        algorithm=ALGORITHM, algorithm_version="stratified-fixed-quotas-v1",
        python_version=platform.python_version(), code_version=version,
        input_sha256=hashes, selection_summary_sha256=evidence_hash,
        input_paths={n: str(source / (n + ".csv")) for n in names},
        output_sha256={n: sha256(output / (n + ".csv"))
                       for n in ("patient_splits", "split_samples")},
        outputs={n: str(output / (n + suffix)) for n, suffix in
                 (("patient_splits", ".csv"), ("split_samples", ".csv"), ("split_summary", ".json"))},
        counts=counts, totals=dict(patients=len(patients), images=len(patients), slices=len(slices),
                                  patient_classes=dict(Counter(r["label"] for r in patients))),
        intersections=intersections, validation=validation,
        original_split_usage="Provenance only; never used for assignment",
        shared_by_models=["CNN", "ConvNeXt"])
    with (output / "split_summary.json").open("x") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    (output / "split_summary.json").chmod(0o600)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--code-version-json", type=Path,
                        help="Verified Git provenance when running a standalone cluster copy")
    args = parser.parse_args()
    provenance = json.loads(args.code_version_json.read_text()) if args.code_version_json else None
    result = run(args.selection_dir, args.output_dir, provenance)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "validation": result["validation"]}, sort_keys=True))
