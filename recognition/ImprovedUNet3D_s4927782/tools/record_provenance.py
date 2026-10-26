"""Verify expected source/manifest digests and write a provenance sidecar.

The expected JSON supplies training_code_commit (full 40-hex Git commit),
source_file_sha256 (project-relative paths to SHA-256), and manifest_sha256.
The remote machine needs no Git. Checkpoints are hashed as opaque bytes; they
are never deserialized. No training configuration or input file is modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_source_path(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))
            or Path(relative).is_absolute()):
        raise ValueError(f"Unsafe project-relative source path: {relative!r}.")
    actual = (root / relative).resolve(strict=True)
    try:
        actual.relative_to(root)
    except ValueError:
        raise ValueError(f"Source path escapes the project root: {relative!r}.") from None
    if not actual.is_file():
        raise ValueError(f"Source path is not a regular file: {relative!r}.")
    return actual


def record_provenance(expected_path, manifest_path, output_path, checkpoint_path=None,
                      replace=False, project_root=None):
    """Validate everything before writing. project_root is injectable for tests."""
    root = Path(project_root if project_root is not None else PROJECT_ROOT).resolve(strict=True)
    expected_path = Path(expected_path).expanduser().resolve(strict=True)
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    output_path = Path(output_path).expanduser().resolve()
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if not isinstance(expected, dict):
        raise ValueError("Expected provenance must be a JSON object.")
    commit = expected.get("training_code_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("training_code_commit must be a full lowercase 40-hex commit.")
    sources = expected.get("source_file_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source_file_sha256 must be a nonempty path-to-digest object.")
    expected_manifest_digest = expected.get("manifest_sha256")
    if not isinstance(expected_manifest_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_digest):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest.")
    protected_inputs = {expected_path, manifest_path}
    measured_sources = {}
    for relative, wanted in sorted(sources.items()):
        if not isinstance(wanted, str) or not re.fullmatch(r"[0-9a-f]{64}", wanted):
            raise ValueError(f"Invalid expected SHA-256 for {relative!r}.")
        actual = checked_source_path(root, relative)
        protected_inputs.add(actual)
        measured = sha256_file(actual)
        if measured != wanted:
            raise ValueError(f"Source digest mismatch: {relative}; expected {wanted}, measured {measured}.")
        measured_sources[relative] = measured
    if not manifest_path.is_file():
        raise ValueError("Manifest must be a regular file.")
    manifest_digest = sha256_file(manifest_path)
    if manifest_digest != expected_manifest_digest:
        raise ValueError(f"Manifest digest mismatch: expected {expected_manifest_digest}, measured {manifest_digest}.")
    checkpoint_fields = {}
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
        if not checkpoint_path.is_file():
            raise ValueError("Checkpoint must be a regular file.")
        protected_inputs.add(checkpoint_path)
        checkpoint_fields = {"checkpoint_path": str(checkpoint_path),
                             "checkpoint_sha256": sha256_file(checkpoint_path)}
    if output_path in protected_inputs:
        raise ValueError("Output must not replace a source, expected JSON, manifest or checkpoint.")
    if output_path.exists() and not replace:
        raise FileExistsError("Provenance output exists; use --replace explicitly to replace the sidecar.")
    payload = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "training_code_commit": commit, "project_root": str(root),
        "source_file_sha256": measured_sources,
        "manifest_path": str(manifest_path), "manifest_sha256": manifest_digest,
        "expected_path": str(expected_path), "expected_sha256": sha256_file(expected_path),
        **checkpoint_fields,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent,
                                         prefix=output_path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, output_path)
        else:
            # Atomic creation without replacing a concurrently created sidecar.
            os.link(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--replace", action="store_true", help="Explicitly replace an existing provenance sidecar.")
    args = parser.parse_args(argv)
    try:
        result = record_provenance(args.expected, args.manifest, args.output, args.checkpoint, args.replace)
    except (ValueError, OSError, TypeError) as error:
        parser.exit(2, f"Provenance recording blocked: {error}\n")
    print(f"Verified {len(result['source_file_sha256'])} source files and manifest; recorded {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
