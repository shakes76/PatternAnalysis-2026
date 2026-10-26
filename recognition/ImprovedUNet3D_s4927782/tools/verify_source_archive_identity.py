"""Compare course source-container CRC32/size with a saved official ZIP catalog.

Standard library only. Reads original files as opaque byte streams, never
downloads an archive, decodes MRI/labels, evaluates a model or changes inputs.
Numeric patient suffix + visit + role is a candidate mapping, not an identity
assumption. A completed audit may contain mismatches and still exits normally.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zlib


OFFICIAL_PATTERNS = {
    "image": re.compile(r"semantic_MRs_anon/Case_(\d+)_Week(\d+)_LFOV\.nii\.gz"),
    "mask": re.compile(r"semantic_labels_anon/Case_(\d+)_Week(\d+)_SEMANTIC_LFOV\.nii\.gz"),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value, directory=False):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\0" in value:
        raise ValueError("Unsafe relative filename.")
    name = value[:-1] if directory and value.endswith("/") else value
    if Path(name).is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"Unsafe relative filename: {value!r}.")
    return name


def source_path(root, relative):
    safe_name(relative)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Source path escapes data root or is not a regular file.")
    return path


def crc_and_size(path):
    checksum = size = 0
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
            size += len(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != before.st_size:
        raise ValueError("A source file changed during its CRC32 read.")
    return {"bytes": size, "crc32": f"{checksum & 0xffffffff:08x}"}


def catalog_indexes(catalog):
    candidates, fingerprints = {}, defaultdict(list)
    names = set()
    if not isinstance(catalog.get("entries"), list) or not catalog["entries"]:
        raise ValueError("Catalog must contain entries.")
    for entry in catalog["entries"]:
        name = entry["filename"]
        directory = entry.get("is_directory", name.endswith("/"))
        safe_name(name, directory=directory)
        if name in names:
            raise ValueError("Duplicate official catalog filename.")
        names.add(name)
        if directory:
            continue
        matches = [(role, pattern.fullmatch(name)) for role, pattern in OFFICIAL_PATTERNS.items()]
        matches = [(role, match) for role, match in matches if match]
        if len(matches) != 1:
            raise ValueError(f"Unrecognized official image/mask filename: {name}.")
        role, match = matches[0]
        size, crc = entry["uncompressed_bytes"], entry["crc32"]
        if type(size) is not int or size < 0 or not isinstance(crc, str) or not re.fullmatch(r"[0-9a-fA-F]{8}", crc):
            raise ValueError("Invalid official byte count or CRC32.")
        item = {"filename": name, "bytes": size, "crc32": crc.lower()}
        key = (role, int(match[1]), int(match[2]))
        if key in candidates:
            raise ValueError("Ambiguous numeric case/visit/role in official catalog.")
        candidates[key] = item
        fingerprints[(role, size, crc.lower())].append(item)
    if not candidates:
        raise ValueError("No official source-file entries.")
    return candidates, fingerprints


def write_new_json(path, payload):
    if os.path.lexists(path):
        raise FileExistsError("Output exists; choose a new JSON path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_identity(manifest_path, catalog_path, data_root, output):
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    catalog_path = Path(catalog_path).expanduser().resolve(strict=True)
    root = Path(data_root).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError("Output exists; choose a new JSON path.")
    if not root.is_dir():
        raise ValueError("Data root must be a directory.")
    manifest = json.loads(manifest_path.read_text())
    catalog = json.loads(catalog_path.read_text())
    if manifest.get("partial") or manifest.get("development_only") or not manifest.get("records"):
        raise ValueError("Use a nonempty complete manifest, not a development subset.")
    candidates, official_fingerprints = catalog_indexes(catalog)
    pending, cases, paths, numeric_cases = [], set(), set(), {}
    # Validate every path and key before reading any source bytes or writing output.
    for record in manifest["records"]:
        patient, visit = record["patient"], record["visit"]
        patient_match = re.fullmatch(r"[A-Za-z]+(\d+)", patient)
        visit_match = re.fullmatch(r"Week(\d+)", visit)
        if not patient_match or not visit_match or (patient, visit) in cases:
            raise ValueError("Invalid/duplicate course patient or visit key.")
        cases.add((patient, visit))
        number, week = int(patient_match[1]), int(visit_match[1])
        if number in numeric_cases and numeric_cases[number] != patient:
            raise ValueError("Different course patients share an ambiguous numeric suffix.")
        numeric_cases[number] = patient
        for role in OFFICIAL_PATTERNS:
            relative = record["source_" + role]
            path = source_path(root, relative)
            if path in paths:
                raise ValueError("A course source file is assigned more than once.")
            paths.add(path)
            pending.append((record, role, number, week, relative, path))
    resolved_output = output.resolve()
    if resolved_output in paths | {manifest_path, catalog_path, Path(__file__).resolve()}:
        raise ValueError("Output must not replace an input.")
    rows, course_fingerprints = [], Counter()
    for record, role, number, week, relative, path in pending:
        actual = crc_and_size(path)
        candidate = candidates.get((role, number, week))
        fingerprint = (role, actual["bytes"], actual["crc32"])
        course_fingerprints[fingerprint] += 1
        rows.append({"patient": record["patient"], "visit": record["visit"], "split": record.get("split"),
                     "role": role, "source_path": relative, "candidate_numeric_case": number,
                     "actual": actual, "numeric_candidate": candidate,
                     "candidate_bytes_match": candidate is not None and actual["bytes"] == candidate["bytes"],
                     "candidate_crc32_match": candidate is not None and actual["crc32"] == candidate["crc32"],
                     "candidate_match": candidate is not None and actual == {k: candidate[k] for k in ("bytes", "crc32")}})
    for row in rows:
        key = (row["role"], row["actual"]["bytes"], row["actual"]["crc32"])
        matches = official_fingerprints.get(key, [])
        unique = matches[0]["filename"] if len(matches) == 1 and course_fingerprints[key] == 1 else None
        candidate_name = row["numeric_candidate"]["filename"] if row["numeric_candidate"] else None
        row.update(role_global_crc_bytes_matches=[x["filename"] for x in matches],
                   course_same_role_crc_bytes_count=course_fingerprints[key],
                   unique_role_global_join=unique,
                   global_join_status=("unique_numeric_candidate" if unique == candidate_name and unique is not None else
                                       "unique_non_candidate" if unique is not None else "ambiguous" if matches else "no_match"))
    groups = {}
    for role in OFFICIAL_PATTERNS:
        selected = [r for r in rows if r["role"] == role]
        groups[role] = {"files": len(selected), "numeric_candidate_matches": sum(r["candidate_match"] for r in selected),
                        "missing_numeric_candidates": sum(r["numeric_candidate"] is None for r in selected),
                        "global_join_status_counts": dict(Counter(r["global_join_status"] for r in selected))}
    all_match = all(row["candidate_match"] for row in rows)
    report = {"schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "completed_all_numeric_candidates_match" if all_match else "completed_with_container_identity_differences",
              "manifest_sha256": file_sha256(manifest_path), "official_catalog_sha256": file_sha256(catalog_path),
              "tool_sha256": file_sha256(__file__), "official_archive": catalog.get("archive"),
              "official_archive_url": catalog.get("official_archive_url"),
              "files_checked": len(rows), "case_visits_checked": len(cases), "all_numeric_candidates_match": all_match,
              "by_role": groups, "source_files_changed": False, "archive_downloaded": False,
              "decoded_voxels_read": False, "model_evaluated": False,
              "limits": ["Numeric suffix + visit + role supplies candidate mappings only. A global join is unique only when its CRC32/size fingerprint is unique in both the official role group and course role group.",
                         "CRC32 and file length check the .nii.gz container bytes against ZIP central-directory metadata; CRC32 is not cryptographic identity proof or clinical label validation.",
                         "Container mismatches do not establish incorrect pixels: gzip metadata, NIfTI headers or anonymization may differ. Decoded-array and geometry comparisons are then needed.",
                         "Unique non-candidate joins identify different matching container candidates; they do not by themselves prove patient identity or authorize renaming, relabelling or changing splits.",
                         "This audit reads all original image/mask files declared by the manifest, including TEST source bytes, but decodes no image/label voxels and performs no model inference or outcome selection."],
              "files": rows}
    write_new_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "catalog", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    report = verify_identity(args.manifest, args.catalog, args.data_root, args.output)
    print(f"{report['status']}: {report['files_checked']} files; {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
