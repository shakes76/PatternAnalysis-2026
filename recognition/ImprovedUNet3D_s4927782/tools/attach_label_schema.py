"""Attach a source-confirmed raw-label key to manifest metadata only.

Run before the first formal training: adding metadata changes the manifest SHA
and existing checkpoints correctly refuse a different manifest. Cache arrays,
patient partitions, and preprocessing parameters are never opened or changed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile


SOURCE_FIELDS = ("source_url", "source_sha256", "dataset_doi", "attribution", "acknowledgement")


def attach_label_schema(manifest_path, schema_path):
    """Atomically attach validated metadata; return False for a byte-stable no-op."""
    manifest_path, schema_path = Path(manifest_path), Path(schema_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(schema, dict):
        raise ValueError("Manifest and schema must both be JSON objects.")
    if schema.get("status") != "official_source_confirmed":
        raise ValueError("Schema status must be official_source_confirmed.")
    names = schema.get("raw_id_to_name")
    if not isinstance(names, dict) or not names:
        raise ValueError("Schema must provide raw_id_to_name.")
    for raw, name in names.items():
        if (not isinstance(raw, str) or not re.fullmatch(r"0|-?[1-9][0-9]*", raw)
                or not isinstance(name, str) or not name.strip()):
            raise ValueError("raw_id_to_name requires canonical integer-string IDs and nonempty names.")
    labels = manifest.get("label_ids")
    if (not isinstance(labels, list) or not labels or any(type(raw) is not int for raw in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("Manifest label_ids must be distinct integers.")
    mapping = manifest.get("raw_label_to_index")
    expected = {str(raw): index for index, raw in enumerate(labels)}
    if mapping != expected:
        raise ValueError("Manifest raw_label_to_index must match label_ids order exactly.")
    if manifest.get("num_classes", len(labels)) != len(labels):
        raise ValueError("Manifest num_classes differs from label_ids.")
    missing = [raw for raw in labels if str(raw) not in names]
    if missing:
        raise ValueError(f"Official schema is missing raw label IDs: {missing}.")
    for field in SOURCE_FIELDS:
        if not isinstance(schema.get(field), str) or not schema[field].strip():
            raise ValueError(f"Official schema requires nonempty {field}.")
    if not re.fullmatch(r"[0-9a-f]{64}", schema["source_sha256"]):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest.")
    semantics = {"status": "official_source_confirmed", "raw_id_to_name": dict(names),
                 "class_index_to_name": {str(index): names[raw] for raw, index in mapping.items()},
                 **{field: schema[field] for field in SOURCE_FIELDS}}
    existing = manifest.get("label_semantics")
    if existing == semantics:
        return False
    if isinstance(existing, dict) and str(existing.get("status", "")).startswith("official_source_confirmed"):
        raise ValueError("Manifest already has different source-confirmed semantics; refusing to replace them.")
    if existing is not None and (not isinstance(existing, dict) or existing.get("status") not in (None, "unverified", "inference_pending_review")):
        raise ValueError("Existing label semantics have an unsupported status; refusing to replace them.")
    manifest["label_semantics"] = semantics
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=manifest_path.parent,
                                         prefix=manifest_path.name + ".", suffix=".tmp", delete=False) as destination:
            temporary_path = Path(destination.name)
            json.dump(manifest, destination, indent=2, allow_nan=False)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(temporary_path, manifest_path.stat().st_mode & 0o777)
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args(argv)
    print("Run only BEFORE the first formal training: attachment changes the manifest SHA; existing checkpoints will reject it. This command edits label metadata only.", file=sys.stderr)
    try:
        changed = attach_label_schema(args.manifest.expanduser().resolve(), args.schema.expanduser().resolve())
    except (ValueError, OSError, TypeError) as error:
        parser.exit(2, f"Label schema attachment blocked: {error}\n")
    print("Attached official label metadata." if changed else "Official label metadata already matches; bytes unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
