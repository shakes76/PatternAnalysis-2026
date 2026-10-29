"""Record actual graph checks and split provenance without training a model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dataset import (
    CLASS_NAMES, DATA_COMMIT, DATA_SHA256, DATA_URL, DEFAULT_PATH, PROJECT_DIR,
    acquire, canonical_edges, load_arrays, sha256, stratified_split,
)


def audit(path: Path, seed: int) -> tuple[dict, dict]:
    checksum = sha256(path)
    if checksum != DATA_SHA256:
        raise ValueError("Expected the pinned course archive; checksum does not match")
    x, y, edges = load_arrays(path)
    n = len(y)
    source, target = edges
    raw_codes = np.unique(source * n + target)
    canonical = canonical_edges(edges, n)
    splits = stratified_split(y, seed)
    degree = np.bincount(canonical[0], minlength=n)
    report = {
        "stage": "preliminary data audit; no model training or test-set performance",
        "source_url": DATA_URL,
        "source_commit": DATA_COMMIT,
        "sha256": checksum,
        "archive_bytes": path.stat().st_size,
        "num_nodes": n,
        "num_features": x.shape[1],
        "num_classes": len(CLASS_NAMES),
        "class_counts": {
            name: int((y == i).sum()) for i, name in enumerate(CLASS_NAMES)
        },
        "features_are_finite": bool(np.isfinite(x).all()),
        "feature_matrix_mib": x.nbytes / (1024 ** 2),
        "raw_edge_entries": edges.shape[1],
        "unique_raw_directed_edges": len(raw_codes),
        "duplicate_raw_entries": int(edges.shape[1] - len(raw_codes)),
        "raw_self_loop_entries": int((source == target).sum()),
        "unique_undirected_edges_including_self": len(np.unique(
            np.minimum(source, target) * n + np.maximum(source, target)
        )),
        "raw_reverse_edges_complete": bool(np.array_equal(
            raw_codes, np.unique(target * n + source)
        )),
        "canonical_nonself_directed_edges": canonical.shape[1],
        "canonicalization": "deduplicate, symmetrize, remove existing self-loops; GCN adds one per node",
        "isolated_nodes_without_self_loops": int((degree == 0).sum()),
        "degree_min_median_max": [int(degree.min()), float(np.median(degree)), int(degree.max())],
        "split_seed": seed,
        "split_rule": "stratified per class, floor(60%) train, floor(20%) validation, remainder test",
        "splits": {
            name: {
                "count": len(ids),
                "class_counts": {
                    label: int((y[ids] == i).sum())
                    for i, label in enumerate(CLASS_NAMES)
                },
                "index_sha256": hashlib.sha256(
                    ids.astype("<i8").tobytes()
                ).hexdigest(),
            }
            for name, ids in splits.items()
        },
        "split_exhaustive_and_disjoint": True,
        "page_name_objects_loaded": False,
        "feature_provenance": "course-provided 128-dimensional SVD features; provided transductive representation",
        "training_completed": False,
        "gpu_time_used_for_audit": False,
    }
    return report, splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "reports" / "data_audit.json")
    args = parser.parse_args()
    if args.download:
        acquire(args.input)
    report, splits = audit(args.input, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    split_path = args.input.parent.parent / f"splits_seed{args.seed}.npz"
    np.savez_compressed(split_path, **splits)
    print(json.dumps(report, indent=2))
    print(f"Saved split indices outside Git: {split_path}")


if __name__ == "__main__":
    main()
