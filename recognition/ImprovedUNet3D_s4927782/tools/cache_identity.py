"""Record decoded cache identities or rebuild caches for a frozen manifest.

Only array digests and descriptors are exported by record. Rebuild reproduces
the existing preprocessing, verifies its output against the identity book, and
places matching NPZ files beside the unchanged manifest. NPZ ZIP timestamps do
not affect compatibility: dtype, shape and decoded C-order bytes must match.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSOR_PATH = PROJECT_ROOT / "tools" / "prepare_cache.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dataset import load_manifest

ARRAY_NAMES = ("image", "mask", "spacing")
HASH_PROTOCOL = "sha256(canonical-json{dtype:numpy.dtype.str,shape:list}+NUL+C-order-array-bytes); JSON sort_keys=True,separators=(',',':')"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or Path(relative).is_absolute()
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError(f"Unsafe relative path: {relative!r}.")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"Path escapes its declared root: {relative!r}.") from None
    return resolved


def array_identity(array):
    array = np.asarray(array)
    if array.dtype.hasobject:
        raise ValueError("Object arrays cannot be recorded.")
    header = {"dtype": array.dtype.str, "shape": list(array.shape)}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\0")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return {**header, "sha256": digest.hexdigest()}


def arrays_identity(arrays):
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError("Cache must contain exactly image, mask and spacing arrays.")
    return {name: array_identity(arrays[name]) for name in ARRAY_NAMES}


def read_arrays(path):
    with np.load(path, allow_pickle=False) as cache:
        if set(cache.files) != set(ARRAY_NAMES):
            raise ValueError("Cache must contain exactly image, mask and spacing arrays.")
        return {name: cache[name] for name in ARRAY_NAMES}


def validate_arrays(arrays, record, manifest):
    image, mask, spacing = (arrays[name] for name in ARRAY_NAMES)
    expected_shape = tuple(manifest["target_shape_dhw"])
    if image.dtype != np.float32 or mask.dtype != np.uint8 or spacing.dtype != np.float32:
        raise ValueError("Expected float32 image/spacing and uint8 class-index mask.")
    if image.shape != expected_shape or mask.shape != expected_shape or spacing.shape != (3,):
        raise ValueError("Cache arrays do not match the frozen target grid.")
    if (not np.isfinite(image).all() or not np.isfinite(spacing).all()
            or np.any(spacing <= 0) or int(mask.max()) >= manifest["num_classes"]):
        raise ValueError("Invalid cache intensities, spacing or class indices.")
    if record.get("target_shape_dhw") != list(expected_shape):
        raise ValueError("Frozen record target shape differs from manifest.")
    if not np.array_equal(spacing, np.asarray(record.get("spacing_dhw_mm"), dtype=np.float32)):
        raise ValueError("Cache spacing differs from frozen geometry metadata.")
    affine = np.asarray(record.get("resampled_affine_xyz"), dtype=np.float64)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(np.linalg.norm(affine[:3, :3], axis=0)[::-1], spacing, rtol=1e-5, atol=1e-6)):
        raise ValueError("Frozen affine and cached spacing are inconsistent.")
    counts = np.bincount(mask.ravel(), minlength=manifest["num_classes"]).tolist()
    if record.get("class_voxel_counts") != counts:
        raise ValueError("Cache class counts differ from frozen metadata.")


def full_manifest(path):
    manifest = load_manifest(path)
    if manifest.get("partial") or manifest.get("development_only"):
        raise ValueError("Cache identity requires a full, non-development manifest.")
    return manifest


def atomic_json(path, value):
    if path.exists():
        raise FileExistsError("Identity output already exists; choose a fresh output path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic creation; never overwrite an input/book.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def record_identity(manifest_path, output_path):
    path = Path(manifest_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve()
    manifest = full_manifest(path)
    protected = {path, PREPROCESSOR_PATH.resolve()}
    caches = []
    for record in manifest["records"]:
        cache_path = safe_relative(path.parent, record["cache_path"])
        protected.add(cache_path)
        caches.append((record, cache_path))
    if output in protected:
        raise ValueError("Identity output must not replace a manifest, cache or preprocessor.")
    book = {"schema_version": 1, "hash_protocol": HASH_PROTOCOL,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "manifest_sha256": file_sha256(path),
            "preprocessor_path": "tools/prepare_cache.py", "preprocessor_sha256": file_sha256(PREPROCESSOR_PATH),
            "record_count": len(caches), "records": {}}
    for record, cache_path in caches:
        arrays = read_arrays(cache_path)
        validate_arrays(arrays, record, manifest)
        book["records"][record["cache_path"]] = {
            "patient": record["patient"], "visit": record["visit"], "split": record["split"],
            "arrays": arrays_identity(arrays)}
    atomic_json(output, book)
    return book


def load_verified_book(identity_path, manifest_path, manifest):
    book = json.loads(Path(identity_path).read_text(encoding="utf-8"))
    if (not isinstance(book, dict) or book.get("schema_version") != 1
            or book.get("hash_protocol") != HASH_PROTOCOL):
        raise ValueError("Unsupported cache identity book or array hash protocol.")
    if book.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Frozen manifest digest does not match the identity book.")
    if (book.get("preprocessor_path") != "tools/prepare_cache.py"
            or book.get("preprocessor_sha256") != file_sha256(PREPROCESSOR_PATH)):
        raise ValueError("Preprocessor digest does not match the identity book.")
    entries = book.get("records")
    expected_paths = {record["cache_path"] for record in manifest["records"]}
    if not isinstance(entries, dict) or set(entries) != expected_paths or book.get("record_count") != len(expected_paths):
        raise ValueError("Identity book does not cover exactly the frozen manifest records.")
    for record in manifest["records"]:
        safe_relative(manifest_path.parent, record["cache_path"])
        entry = entries[record["cache_path"]]
        if not isinstance(entry, dict) or any(entry.get(key) != record[key] for key in ("patient", "visit", "split")):
            raise ValueError("Identity patient/visit/split differs from the frozen manifest.")
        arrays = entry.get("arrays")
        if not isinstance(arrays, dict) or set(arrays) != set(ARRAY_NAMES):
            raise ValueError("Identity record must describe all three cache arrays.")
        for name, descriptor in arrays.items():
            if (not isinstance(descriptor, dict) or not isinstance(descriptor.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])
                    or not isinstance(descriptor.get("dtype"), str) or not isinstance(descriptor.get("shape"), list)):
                raise ValueError(f"Invalid array identity descriptor for {name}.")
    return book


def get_preprocessor():
    spec = importlib.util.spec_from_file_location("frozen_cache_preprocessor", PREPROCESSOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preprocess_pair


def compare_metadata(actual, frozen):
    # Same preprocessor and frozen source geometry should give identical JSON
    # metadata. Exact array checks remain mandatory even for matching geometry.
    for key, value in actual.items():
        if key not in frozen or frozen[key] != value:
            raise ValueError(f"Rebuilt metadata differs from frozen record: {key}.")


def atomic_cache(path, arrays, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            # A concurrent rebuild may have completed the same cache.
            if arrays_identity(read_arrays(path)) != expected:
                raise ValueError("An existing cache has different decoded array identities; refusing replacement.") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def rebuild_caches(manifest_path, identity_path, data_root, split="test", limit=None):
    if split not in ("train", "validation", "test", "all"):
        raise ValueError("Split must be train, validation, test or all.")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Limit must be a positive integer.")
    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = full_manifest(path)
    book = load_verified_book(identity_path, path, manifest)
    root = Path(data_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("data-root must be a directory.")
    selected = [record for record in manifest["records"] if split == "all" or record["split"] == split]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No records are available for the requested split.")
    # Check every selected source/output path before making any cache directory.
    paths = [(record, safe_relative(path.parent, record["cache_path"]),
              safe_relative(root, record["source_image"]), safe_relative(root, record["source_mask"])) for record in selected]
    preprocess = None
    written, reused = [], []
    for record, destination, image_path, mask_path in paths:
        expected = book["records"][record["cache_path"]]["arrays"]
        if destination.exists():
            existing = read_arrays(destination)
            validate_arrays(existing, record, manifest)
            if arrays_identity(existing) != expected:
                raise ValueError(f"Existing cache digest mismatch: {record['cache_path']}; refusing replacement.")
            reused.append(record["cache_path"])
            continue
        if preprocess is None:
            preprocess = get_preprocessor()
        arrays, metadata = preprocess(image_path, mask_path, manifest["target_shape_dhw"], manifest["label_ids"])
        compare_metadata(metadata, record)
        validate_arrays(arrays, record, manifest)
        if arrays_identity(arrays) != expected:
            raise ValueError(f"Rebuilt array digest mismatch: {record['cache_path']}; no cache written for this record.")
        atomic_cache(destination, arrays, expected)
        written.append(record["cache_path"])
    if file_sha256(path) != book["manifest_sha256"]:
        raise RuntimeError("Manifest changed concurrently; stop using this rebuild output.")
    return {"manifest_sha256": book["manifest_sha256"], "split": split, "limit": limit,
            "selected_records": len(selected), "written": written, "reused": reused,
            "manifest_rewritten": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="Export decoded array identities without exporting arrays.")
    record.add_argument("--manifest", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    rebuild = commands.add_parser("rebuild", help="Recreate exact caches beside an unchanged frozen manifest.")
    rebuild.add_argument("--manifest", type=Path, required=True)
    rebuild.add_argument("--identity", type=Path, required=True)
    rebuild.add_argument("--data-root", type=Path, required=True)
    rebuild.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    rebuild.add_argument("--limit", type=int, help="Verification only: rebuild at most N selected records.")
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            result = record_identity(args.manifest, args.output)
            print(f"Recorded {result['record_count']} decoded cache identities: {args.output.resolve()}")
        else:
            print(json.dumps(rebuild_caches(args.manifest, args.identity, args.data_root, args.split, args.limit), indent=2))
    except (ValueError, OSError, ImportError, KeyError, TypeError) as error:
        parser.exit(2, f"Cache identity operation blocked: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
