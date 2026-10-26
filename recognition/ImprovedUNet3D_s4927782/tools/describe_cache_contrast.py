"""Reproduce retrospective GT-ROI descriptors from frozen caches inside a tar.

Reads all 143 TRAIN and 38 VAL caches plus M015 Week0..7 only. No extraction,
model inference, fitted transform, threshold selection or new TEST benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

import numpy as np


MANIFEST_SHA256 = "c47fe880d353629348786b852e3f699291af1a4d5628b2980bf42a74635d0424"
IDENTITY_SHA256 = "eacb84bf20626a6ec9d7f6d36489a81e2d4e4af9fbe4ade1ce8f508a2022e08a"
SCOPE = "All 143 TRAIN and 38 VAL cases, plus eight previously inspected M015 TEST visits only. No model inference or tuning."


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_identity(array):
    header = {"dtype": array.dtype.str, "shape": list(array.shape)}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\0")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def describe(x):
    n = int(x.size)
    if not n:
        return {"n": 0}
    q = np.quantile(x, [.01, .05, .25, .5, .75, .95, .99]).tolist()
    return {"n": n, "mean": float(np.mean(x, dtype=np.float64)), "std": float(np.std(x, dtype=np.float64)),
            "quantiles": dict(zip(("q01", "q05", "q25", "q50", "q75", "q95", "q99"), q)),
            "exact_zero_fraction": float(np.count_nonzero(x == 0) / n),
            "at_or_below_minus5_fraction": float(np.count_nonzero(x <= -5) / n),
            "at_or_above_plus5_fraction": float(np.count_nonzero(x >= 5) / n)}


def describe_case(record, image, mask, spacing):
    classes = {str(c): describe(image[mask == c]) for c in range(6)}
    body, bladder, prostate = (classes[str(c)] for c in (1, 3, 5))
    contrasts = {}
    for name, organ in (("bladder", bladder), ("prostate", prostate)):
        if organ["n"] and body["n"]:
            gap = organ["quantiles"]["q50"] - body["quantiles"]["q50"]
            scale = body["quantiles"]["q75"] - body["quantiles"]["q25"]
            contrasts[name + "_median_minus_body_median"] = gap
            contrasts[name + "_body_iqr_standardized_median_gap"] = gap / scale if scale else None
    if bladder["n"] and prostate["n"]:
        contrasts["bladder_median_minus_prostate_median"] = bladder["quantiles"]["q50"] - prostate["quantiles"]["q50"]
    return {"patient": record["patient"], "visit": record["visit"], "split": record["split"],
            "decoded_identity_verified": True, "class_intensity": classes, "contrasts": contrasts,
            "spacing_dhw": spacing.tolist()}


def select_records(manifest):
    if manifest.get("partial") or manifest.get("development_only"):
        raise ValueError("Development manifests are forbidden.")
    rows = [r for r in manifest["records"] if r["split"] in ("train", "validation") or
            (r["patient"] == "M015" and r["split"] == "test")]
    keys = {(r["patient"], r["visit"]) for r in rows}
    tests = {(r["patient"], r["visit"]) for r in rows if r["split"] == "test"}
    if (len(rows) != 189 or len(keys) != 189 or Counter(r["split"] for r in rows) !=
            {"train": 143, "validation": 38, "test": 8} or tests != {("M015", f"Week{i}") for i in range(8)}):
        raise ValueError("Expected exactly the fixed 143 TRAIN / 38 VAL / eight M015 scope.")
    for r in rows:
        p = r["cache_path"]
        if not isinstance(p, str) or "\\" in p or ":" in p or Path(p).is_absolute() or any(x in ("", ".", "..") for x in p.split("/")):
            raise ValueError("Unsafe cache member path.")
    if len({r["cache_path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate selected cache member.")
    return rows


def write_new(path, data):
    if os.path.lexists(path):
        raise FileExistsError("Choose a new output JSON; existing evidence cannot be overwritten.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            json.dump(data, f, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def describe_archive(manifest_path, identity_path, archive_path, output_path, diagnostic_test_ack=False):
    if not diagnostic_test_ack:
        raise ValueError("Pass --diagnostic-test-ack for the eight retrospective M015 TEST inputs.")
    output = Path(output_path).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError("Choose a new output JSON.")
    manifest_path, identity_path, archive_path = (Path(p).expanduser().resolve(strict=True) for p in
                                                 (manifest_path, identity_path, archive_path))
    if sha256(manifest_path) != MANIFEST_SHA256 or sha256(identity_path) != IDENTITY_SHA256:
        raise ValueError("Frozen manifest/cache-identity SHA256 mismatch.")
    manifest, identity = (json.loads(p.read_text()) for p in (manifest_path, identity_path))
    if identity["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("Identity book belongs to another manifest.")
    records, rows = select_records(manifest), []
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        counts = Counter(m.name for m in members)
        for record in records:
            name = "artifacts/cache64/" + record["cache_path"]
            if counts[name] != 1:
                raise ValueError("Selected tar member is absent or duplicated.")
            member = archive.getmember(name)
            if not member.isfile() or member.size > 64 * 1024 * 1024:
                raise ValueError("Selected cache must be a bounded regular tar member, never a link.")
            with archive.extractfile(member) as stream, np.load(stream, allow_pickle=False) as saved:
                if set(saved.files) != {"image", "mask", "spacing"}:
                    raise ValueError("Unexpected cache arrays.")
                arrays = {k: saved[k] for k in ("image", "mask", "spacing")}
            expected = identity["records"][record["cache_path"]]
            if any(expected[k] != record[k] for k in ("patient", "visit", "split")):
                raise ValueError("Cache identity case assignment differs.")
            if any(array_identity(a) != expected["arrays"][k]["sha256"] for k, a in arrays.items()):
                raise ValueError("Decoded cache identity differs; no descriptor output written.")
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Nonfinite cache array.")
            rows.append(describe_case(record, arrays["image"], arrays["mask"], arrays["spacing"]))
    report = {"schema_version": 1, "diagnostic_only": True, "scope": SCOPE, "uses_GT_regions": True,
              "deployment_transform_claimed": False, "stage": "Frozen cached normalized image, not raw MRI intensities.",
              "manifest_sha256": MANIFEST_SHA256, "cache_identity_sha256": IDENTITY_SHA256,
              "cache_archive_sha256": sha256(archive_path), "script_sha256": sha256(__file__),
              "numpy_version": np.__version__, "diagnostic_test_ack": True, "tar_extracted_to_disk": False, "cases": rows}
    write_new(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "cache-identity", "cache-archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--diagnostic-test-ack", action="store_true")
    args = parser.parse_args(argv)
    report = describe_archive(args.manifest, args.cache_identity, args.cache_archive, args.output, args.diagnostic_test_ack)
    print(f"Completed fixed-scope descriptors for {len(report['cases'])} cases: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
