"""Finite HipMRI v2 experiment orchestration; preserves the original project.

Reuses the frozen project model/loss, dataset validation and preprocess_pair
implementations by verified source digest. This is NOT an independently
implemented preprocessing algorithm or new model architecture. Changes are the
cache resolution, TRAIN-only sample distribution, sliding-window inference and
predeclared class-5 validation selection. Metrics remain on the new cache grid;
they are not native-resolution performance claims.

prepare creates a fresh TRAIN/VAL cache and one immutable all-record manifest.
TEST entries contain geometry/identity only and remain pending. prepare-test
later fills exactly those missing files without modifying manifest bytes.
train requires a real-data QA receipt for both models. No mode launches jobs.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import random
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    # Geometry/sampling fixtures can run in the lightweight local environment.
    # Model/cache execution requires the real measured training environment.
    torch, Dataset, DataLoader = None, object, None


DEFAULT_CONFIGURATION = {
    "target_shape_dhw": [96, 192, 192], "patch_shape_dhw": [64, 128, 128],
    "seed": 3711, "num_classes": 6, "base_channels": 16, "epochs": 30,
    "train_patches_per_scan": 2, "train_slices_per_scan": 32,
    "train_batch_size_2d": 16, "train_batch_size_3d": 1,
    "eval_slice_batch_size": 32, "foreground_sampling_probability": 0.5,
    "augmentation": "random_hw_flips", "lr": 0.001, "weight_decay": 0.0001,
    "clip_grad": 5, "amp": True, "workers": 0,
    "validation_epochs": [1, 5, 10, 15, 20, 25, 30],
    "selection": "patient_balanced_class5_dice_strict_max_earliest_tie",
    "inference_overlap": 0.5, "inference_blending": "uniform_softmax_mean",
    "class_weighting": "sqrt_inverse_frequency",
}
FROZEN_SOURCE_SHA256 = {
    "modules.py": "e752b242154ba01165e34b2323aab499aeb3dbd5346f62c8e1f301840f010fc9",
    "dataset.py": "bc3475d38dfb02a0f37a7d667b356bffac9628544d018b03ffc382cabf0a731a",
    "utils.py": "63c3111f457f531085c4658cdd03e4824172c82ac721c69983215b3115275a84",
    "tools/prepare_cache.py": "55b4f78aa555db5477f2e976b7b874d2fed00309a788d46fba112ad10ba3d13b",
}
CLASS_NAMES = ["Background", "Body", "Bones", "Bladder", "Rectum", "Prostate"]
MODELS = ("unet2d", "improved3d")
SCHEMA = "hipmri_independent_review_v2_immutable_manifest_1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_protocol(path):
    protocol = read_json(path)
    config = protocol.get("configuration", {})
    for key, expected in DEFAULT_CONFIGURATION.items():
        if key not in config or config[key] != expected:
            raise ValueError(f"Frozen v2 protocol configuration differs at {key}: expected {expected!r}.")
    if not protocol.get("experiment_id"):
        raise ValueError("Protocol must identify the finite experiment.")
    protocol["_sha256"] = sha256(path)
    return protocol


def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dependencies(project_root, protocol=None):
    """Import own frozen v1 components only after digest verification."""
    project_root = Path(project_root).expanduser().resolve(strict=True)
    required = dict(FROZEN_SOURCE_SHA256)
    if protocol:
        for filename, expected in protocol.get("required_original_sha256", {}).items():
            if filename in required and required[filename] != expected:
                raise ValueError(f"Protocol disagrees with built-in frozen source: {filename}.")
            required[filename] = expected
    for filename, expected in required.items():
        path = _safe_relative(project_root, filename)
        if sha256(path) != expected:
            raise ValueError(f"Frozen original source digest mismatch: {filename}.")
    # The frozen preprocessor imports dataset.validate_patient_splits. Its
    # dependency is the same verified source, not an independently changed copy.
    existing = sys.modules.get("dataset")
    if existing is not None and Path(existing.__file__).resolve() != project_root / "dataset.py":
        raise ValueError("An unrelated dataset module is already imported.")
    dataset = _import_file("dataset", project_root / "dataset.py")
    sys.modules["dataset"] = dataset
    dependencies = SimpleNamespace(
        modules=_import_file("v2_frozen_modules", project_root / "modules.py"),
        dataset=dataset,
        utils=_import_file("v2_frozen_utils", project_root / "utils.py"),
        prepare=_import_file("v2_frozen_prepare", project_root / "tools/prepare_cache.py"),
        source_hashes=required,
        project_root=project_root,
    )
    return dependencies


def verify_review_sources(protocol, directory):
    for filename, expected in protocol.get("required_review_sha256", {}).items():
        if sha256(_safe_relative(Path(directory), filename)) != expected:
            raise ValueError(f"Review source digest mismatch: {filename}.")


def _safe_relative(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Expected a safe relative artifact/source path.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact/source path escapes its declared root.")
    return path


def _fresh_directory(path):
    path = Path(path).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError("Use a fresh output directory; existing evidence cannot be overwritten.")
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def expected_geometry(record, target_shape):
    """Derive new-grid geometry from old audited metadata, without TEST pixels."""
    canonical = record["canonical_image"]
    old = np.asarray(canonical["shape_xyz"], dtype=np.float64)
    affine = np.asarray(canonical["affine_xyz"], dtype=np.float64)
    new = np.asarray(target_shape[::-1], dtype=np.float64)
    if old.shape != (3,) or affine.shape != (4, 4) or not np.isfinite(affine).all() or np.any(old < 1):
        raise ValueError("Original manifest is missing valid canonical geometry.")
    transform = np.eye(4)
    scale = old / new
    transform[:3, :3] = np.diag(scale)
    transform[:3, 3] = 0.5 * (scale - 1.0)
    spacing = np.linalg.norm(affine[:3, :3], axis=0) * scale
    return {
        "original_image": copy.deepcopy(record["original_image"]),
        "original_mask": copy.deepcopy(record["original_mask"]),
        "canonical_image": copy.deepcopy(canonical),
        "canonical_mask": copy.deepcopy(record["canonical_mask"]),
        "target_shape_dhw": list(target_shape), "target_shape_xyz": list(target_shape[::-1]),
        "resampled_affine_xyz": (affine @ transform).tolist(),
        "spacing_dhw_mm": spacing[::-1].tolist(),
        "metrics_grid": "downsampled_resampled_grid", "native_resolution_accuracy_claimed": False,
    }


def _check_source_geometry(expected, metadata):
    for key in ("original_image", "original_mask", "canonical_image", "canonical_mask"):
        if expected[key]["shape_xyz"] != metadata[key]["shape_xyz"] or not np.allclose(
            expected[key]["affine_xyz"], metadata[key]["affine_xyz"], rtol=1e-5, atol=1e-5
        ):
            raise ValueError(f"Raw source {key} geometry differs from immutable expected geometry.")
    if (expected["target_shape_dhw"] != metadata["target_shape_dhw"] or not np.allclose(
        expected["resampled_affine_xyz"], metadata["resampled_affine_xyz"], rtol=1e-5, atol=1e-5
    ) or not np.allclose(expected["spacing_dhw_mm"], metadata["spacing_dhw_mm"], rtol=1e-5, atol=1e-5)):
        raise ValueError("New cache geometry differs from the predeclared grid.")


def _write_cache(destination, arrays):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Cache already exists: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pending.npz")
    if temporary.exists():
        raise FileExistsError("Incomplete cache evidence already exists; inspect it and choose a fresh directory.")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination)
    return sha256(destination)


def _source_paths(raw_root, record):
    return (_safe_relative(raw_root, record["source_image"]),
            _safe_relative(raw_root, record["source_mask"]))


def prepare(manifest_path, output_dir, raw_root, protocol, protocol_path, dependencies):
    """Stage 1: create TRAIN/VAL pixels; immutable TEST placeholders only."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    source = dependencies.dataset.load_manifest(manifest_path)
    if source.get("partial") or source.get("development_only"):
        raise ValueError("Source must be the complete original patient manifest.")
    if source["num_classes"] != 6 or source["label_ids"] != list(range(6)):
        raise ValueError("V2 preserves all six original label semantics.")
    config = protocol["configuration"]
    counts = {split: sum(r["split"] == split for r in source["records"])
              for split in ("train", "validation", "test")}
    if counts != {"train": 143, "validation": 38, "test": 30}:
        raise ValueError(f"Source scan counts differ from frozen 143/38/30: {counts}.")
    source_digest, protocol_digest = sha256(manifest_path), sha256(protocol_path)
    expected_source = protocol.get("required_original_sha256", {}).get("artifacts/cache64/manifest.json")
    if expected_source and source_digest != expected_source:
        raise ValueError("Preparation source manifest differs from the frozen original manifest.")
    output = _fresh_directory(output_dir)
    receipt = {"status": "preparing_train_validation", "started_at": utc_now(),
               "source_manifest_sha256": source_digest, "protocol_sha256": protocol_digest,
               "source_hashes": dependencies.source_hashes, "v2_pipeline_sha256": sha256(__file__),
               "test_pixels_read": False, "records_completed": 0}
    write_json(output / "preparation_receipt.json", receipt)
    records = []
    try:
        for source_record in source["records"]:
            record = {key: source_record[key] for key in ("patient", "visit", "split", "source_image", "source_mask")}
            record["cache_path"] = "volumes/" + dependencies.utils.safe_case_name(record["patient"], record["visit"]) + ".npz"
            record.update(expected_geometry(source_record, config["target_shape_dhw"]))
            record["pending"] = record["split"] == "test"
            if not record["pending"]:
                image, mask = _source_paths(raw_root, record)
                arrays, metadata = dependencies.prepare.preprocess_pair(
                    image, mask, config["target_shape_dhw"], list(range(6)))
                _check_source_geometry(record, metadata)
                record.update(metadata)
                record["cache_sha256"] = _write_cache(output / record["cache_path"], arrays)
                record["source_image_sha256"], record["source_mask_sha256"] = sha256(image), sha256(mask)
                receipt["records_completed"] += 1
            records.append(record)
            print(f"Cache stage 1: {len(records)}/{len(source['records'])} {record['patient']}/{record['visit']} "
                  f"{'TEST pending: no pixels read' if record['pending'] else 'prepared'}", flush=True)
        result = {key: copy.deepcopy(source[key]) for key in
                  ("num_classes", "label_ids", "raw_label_to_index", "patient_splits", "audit_sha256")}
        result.update({"schema_version": 1, "v2_schema": SCHEMA, "records": records,
                       "label_semantics": copy.deepcopy(source.get("label_semantics", {})),
                       "class_names": CLASS_NAMES, "target_shape_dhw": config["target_shape_dhw"],
                       "partial": False, "development_only": False,
                       "metrics_grid": "downsampled_resampled_grid", "native_resolution_accuracy_claimed": False,
                       "original_manifest_sha256": source_digest, "protocol_sha256": protocol_digest,
                       "preprocessor_source_sha256": dependencies.source_hashes["tools/prepare_cache.py"],
                       "preprocessing_fit_scope": "each individual image only; no split-level fitting",
                       "pending_test_policy": "TEST entries have no pixel-derived new-grid metadata; prepare-test may fill cache files without changing this manifest.",
                       "prepared_record_count": counts["train"] + counts["validation"],
                       "source_audit_record_count": len(records)})
        write_json(output / "manifest.json", result)
        load_new_manifest(output / "manifest.json", protocol, dependencies)
        if sha256(manifest_path) != source_digest or sha256(protocol_path) != protocol_digest:
            raise RuntimeError("Source manifest or protocol changed during cache preparation.")
        receipt.update(status="train_validation_complete_test_pending", completed_at=utc_now(),
                       manifest_sha256=sha256(output / "manifest.json"))
        write_json(output / "preparation_receipt.json", receipt)
        return output / "manifest.json"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "preparation_receipt.json", receipt)
        raise


def load_new_manifest(path, protocol=None, dependencies=None):
    """Validate new manifest identity; pending TEST files are deliberately legal."""
    manifest = dependencies.dataset.load_manifest(path) if dependencies else read_json(path)
    if (manifest.get("v2_schema") != SCHEMA or manifest.get("target_shape_dhw") != [96, 192, 192]
            or manifest.get("num_classes") != 6 or manifest.get("label_ids") != list(range(6))
            or manifest.get("partial") or manifest.get("development_only")):
        raise ValueError("Expected the complete immutable v2 manifest, never the original cache.")
    if protocol and manifest.get("protocol_sha256") != protocol["_sha256"]:
        raise ValueError("New manifest is bound to a different protocol.")
    if protocol:
        expected_source = protocol.get("required_original_sha256", {}).get("artifacts/cache64/manifest.json")
        if expected_source and manifest.get("original_manifest_sha256") != expected_source:
            raise ValueError("V2 manifest does not preserve the frozen original manifest identity.")
    counts = {split: sum(r["split"] == split for r in manifest["records"])
              for split in ("train", "validation", "test")}
    if counts != {"train": 143, "validation": 38, "test": 30}:
        raise ValueError("New manifest must preserve all original 143/38/30 scan identities.")
    for record in manifest["records"]:
        if record.get("pending") != (record["split"] == "test"):
            raise ValueError("Only immutable TEST entries may be pending.")
        if record["split"] == "test":
            if "class_voxel_counts" in record or "cache_sha256" in record:
                raise ValueError("Pending TEST entries must not claim new-grid pixel counts or cache digests.")
        elif (not isinstance(record.get("class_voxel_counts"), list)
              or any(type(c) is not int or c < 0 for c in record["class_voxel_counts"])
              or sum(record["class_voxel_counts"]) != int(np.prod(manifest["target_shape_dhw"]))
              or len(record["class_voxel_counts"]) != 6
              or not re.fullmatch(r"[a-f0-9]{64}", record.get("cache_sha256", ""))):
            raise ValueError("Prepared TRAIN/VAL entry lacks new-grid counts/digest.")
    return manifest


def require_selection_freeze(path, manifest_path, protocol, model_name=None, checkpoint_digest=None):
    """Require a paired, completed, immutable selection receipt before TEST I/O."""
    if path is None:
        raise ValueError("Retrospective TEST requires --selection-freeze after both formal models finish.")
    receipt = read_json(path)
    if (receipt.get("status") != "frozen" or receipt.get("manifest_sha256") != sha256(manifest_path)
            or receipt.get("protocol_sha256") != protocol["_sha256"]):
        raise ValueError("Selection freeze is not bound to this manifest and protocol.")
    for name in MODELS:
        selected = receipt.get("models", {}).get(name, {})
        if (selected.get("completed_epochs") != 30
                or not re.fullmatch(r"[a-f0-9]{64}", selected.get("checkpoint_sha256", ""))):
            raise ValueError("Both formal models must complete 30 epochs and freeze their selected digests.")
    if model_name and receipt["models"][model_name]["checkpoint_sha256"] != checkpoint_digest:
        raise ValueError("TEST checkpoint differs from the paired frozen selection.")
    return sha256(path)


def prepare_test(manifest_path, raw_root, protocol, dependencies, selection_freeze=None):
    """Stage 2: fill predetermined TEST files; never write the manifest."""
    path = Path(manifest_path).resolve(strict=True)
    freeze_digest = require_selection_freeze(selection_freeze, path, protocol)
    digest = sha256(path)
    manifest = load_new_manifest(path, protocol, dependencies)
    destination = path.parent / "test_completion_receipt.json"
    if destination.exists():
        raise FileExistsError("TEST completion was already attempted; inspect the existing receipt.")
    receipt = {"status": "preparing_test", "manifest_sha256": digest,
               "protocol_sha256": protocol["_sha256"], "selection_freeze_sha256": freeze_digest,
               "started_at": utc_now(), "records": []}
    write_json(destination, receipt)
    try:
        for record in manifest["records"]:
            if record["split"] != "test":
                continue
            image, mask = _source_paths(raw_root, record)
            arrays, metadata = dependencies.prepare.preprocess_pair(
                image, mask, manifest["target_shape_dhw"], manifest["label_ids"])
            _check_source_geometry(record, metadata)
            digest_cache = _write_cache(_safe_relative(path.parent, record["cache_path"]), arrays)
            receipt["records"].append({"patient": record["patient"], "visit": record["visit"],
                "cache_path": record["cache_path"], "cache_sha256": digest_cache,
                "source_image_sha256": sha256(image), "source_mask_sha256": sha256(mask),
                "class_voxel_counts": metadata["class_voxel_counts"]})
            write_json(destination, receipt)
            print(f"Cache stage 2: {len(receipt['records'])}/30 TEST files", flush=True)
        if sha256(path) != digest:
            raise RuntimeError("Immutable v2 manifest changed during TEST preparation.")
        receipt.update(status="completed", completed_at=utc_now(), manifest_bytes_unchanged=True)
        write_json(destination, receipt)
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(destination, receipt)
        raise
    return destination


class CachedVolumes:
    """Small per-process LRU; validates actual new cache bytes and counts."""
    def __init__(self, manifest_path, manifest, split, max_cached=2):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split.")
        self.root = Path(manifest_path).resolve().parent
        self.manifest = manifest
        self.records = [r for r in manifest["records"] if r["split"] == split]
        self.cache, self.max_cached = OrderedDict(), max_cached
        self.completion = {}
        if split == "test":
            receipt = read_json(self.root / "test_completion_receipt.json")
            if receipt.get("status") != "completed" or receipt.get("manifest_sha256") != sha256(manifest_path):
                raise ValueError("TEST cache completion has not been independently recorded for this manifest.")
            self.completion = {r["cache_path"]: r for r in receipt["records"]}
            if set(self.completion) != {r["cache_path"] for r in self.records}:
                raise ValueError("TEST receipt does not cover precisely the immutable TEST records.")

    def __len__(self):
        return len(self.records)

    def load(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        record = self.records[index]
        identity = self.completion[record["cache_path"]] if record["pending"] else record
        path = _safe_relative(self.root, record["cache_path"])
        if sha256(path) != identity["cache_sha256"]:
            raise ValueError("New cache content differs from its immutable manifest/completion receipt.")
        with np.load(path, allow_pickle=False) as data:
            image, mask, spacing = data["image"], data["mask"], data["spacing"]
        shape = tuple(self.manifest["target_shape_dhw"])
        if (image.shape != shape or mask.shape != shape or image.dtype != np.float32 or mask.dtype != np.uint8
                or not np.isfinite(image).all() or spacing.shape != (3,) or not np.isfinite(spacing).all()
                or np.any(spacing <= 0) or int(mask.max()) >= 6
                or not np.allclose(spacing, record["spacing_dhw_mm"], rtol=1e-5, atol=1e-5)):
            raise ValueError("New cache pixels/spacing violate its manifest.")
        counts = np.bincount(mask.reshape(-1), minlength=6).tolist()
        if counts != identity["class_voxel_counts"]:
            raise ValueError("New-grid class counts differ from actual new mask pixels.")
        self.cache[index] = (image, mask, spacing)
        while len(self.cache) > self.max_cached:
            self.cache.popitem(last=False)
        return image, mask, spacing


def sample_center(mask, rng, foreground_probability):
    """TRAIN-only rule: choose class uniformly first, including absent classes.

    If a chosen class is absent, fall back to uniform space and return that fact
    for audit. Never replace the selected class with a more convenient class.
    """
    requested = bool(rng.random() < foreground_probability)
    chosen_class = int(rng.integers(1, 6)) if requested else -1
    fallback = False
    if requested:
        locations = np.flatnonzero(mask.reshape(-1) == chosen_class)
        if locations.size:
            center = np.unravel_index(int(locations[int(rng.integers(locations.size))]), mask.shape)
            return tuple(int(x) for x in center), requested, chosen_class, fallback
        fallback = True
    return tuple(int(rng.integers(size)) for size in mask.shape), requested, chosen_class, fallback


def patch_slices(center, shape, patch):
    if any(p > size or p < 1 for p, size in zip(patch, shape)):
        raise ValueError("Training patch must fit inside the cache grid.")
    starts = [int(np.clip(c - p // 2, 0, size - p)) for c, size, p in zip(center, shape, patch)]
    return tuple(slice(start, start + p) for start, p in zip(starts, patch))


class SampledTrainingDataset(Dataset):
    """Each TRAIN scan supplies 2 patches or 32 slices per deterministic epoch.

    Scan order is shuffled per epoch; samples for a scan are contiguous for I/O.
    Sampling is seeded per (epoch, example) independently of worker scheduling.
    Label-guided sample selection exists only here, never in predict_image.
    """
    def __init__(self, manifest_path, model, configuration, dependencies, manifest=None):
        if model not in MODELS:
            raise ValueError("Unknown model.")
        self.config, self.model = configuration, model
        manifest = manifest or load_new_manifest(manifest_path, dependencies=dependencies)
        self.volumes = CachedVolumes(manifest_path, manifest, "train")
        self.samples_per_scan = configuration["train_slices_per_scan"] if model == "unet2d" else configuration["train_patches_per_scan"]
        self.epoch = 1
        self.set_epoch(1)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        rng = np.random.default_rng(np.random.SeedSequence([self.config["seed"], self.epoch, 817]))
        self.order = rng.permutation(len(self.volumes))

    def __len__(self):
        return len(self.volumes) * self.samples_per_scan

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        record_index = int(self.order[index // self.samples_per_scan])
        image, mask, _ = self.volumes.load(record_index)
        rng = np.random.default_rng(np.random.SeedSequence([self.config["seed"], self.epoch, index]))
        center, requested, chosen_class, fallback = sample_center(mask, rng, self.config["foreground_sampling_probability"])
        region = (center[0], slice(None), slice(None)) if self.model == "unet2d" else patch_slices(
            center, mask.shape, self.config["patch_shape_dhw"])
        sampled_image, sampled_mask = image[region], mask[region]
        flips = []
        for axis in (-2, -1):
            if rng.random() < .5:
                sampled_image, sampled_mask = np.flip(sampled_image, axis), np.flip(sampled_mask, axis)
                flips.append(axis)
        return {"image": torch.from_numpy(np.ascontiguousarray(sampled_image)).unsqueeze(0),
                "mask": torch.from_numpy(np.ascontiguousarray(sampled_mask)).long(),
                "foreground_requested": requested, "selected_class": chosen_class,
                "foreground_missing_fallback": fallback, "record_index": record_index,
                "center_dhw": torch.tensor(center), "hw_flip_count": len(flips)}


def window_starts(size, patch, overlap=.5):
    if not 0 <= overlap < 1 or not 1 <= patch <= size:
        raise ValueError("Patch must fit and overlap must be in [0,1).")
    stride = max(1, int(round(patch * (1 - overlap))))
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def predict_image(model, image, model_name, device, configuration):
    """Return CPU hard labels from ONLY an image (1,D,H,W), never a mask.

    3D: fixed half-overlap windows, arithmetic mean of softmax probabilities,
    including every boundary voxel. Accumulators stay on CPU. 2D: all slices.
    No TTA, crop, true-label support, connected components or postprocessing.
    """
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Inference accepts image tensor (1,D,H,W) only.")
    if not image.is_floating_point() or not bool(torch.isfinite(image).all()):
        raise ValueError("Expected finite floating-point image intensities.")
    amp = bool(configuration["amp"] and device.type == "cuda")
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
        if model_name == "unet2d":
            parts = []
            for chunk in image[0].unsqueeze(1).split(configuration["eval_slice_batch_size"]):
                logits = model(chunk.to(device))["logits"]
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("Nonfinite model inference logits.")
                parts.append(logits.argmax(1).cpu())
            return torch.cat(parts)
        if model_name != "improved3d":
            raise ValueError("Unknown model.")
        shape, patch = tuple(image.shape[1:]), tuple(configuration["patch_shape_dhw"])
        starts = [window_starts(size, p, configuration["inference_overlap"]) for size, p in zip(shape, patch)]
        totals = torch.zeros((configuration["num_classes"], *shape), dtype=torch.float32, device="cpu")
        coverage = torch.zeros(shape, dtype=torch.float32, device="cpu")
        for start in itertools.product(*starts):
            region = tuple(slice(s, s + p) for s, p in zip(start, patch))
            crop = image[(slice(None), *region)].unsqueeze(0).to(device)
            logits = model(crop)["logits"]
            if logits.shape != (1, configuration["num_classes"], *patch) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Sliding-window model emitted invalid logits.")
            totals[(slice(None), *region)] += torch.softmax(logits.float(), dim=1)[0].cpu()
            coverage[region] += 1
        if not bool((coverage > 0).all()):
            raise RuntimeError("Sliding-window inference left uncovered voxels.")
        return (totals / coverage.unsqueeze(0)).argmax(0)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model, manifest_path, manifest, split, model_name, device, configuration, dependencies,
             output_dir=None):
    """Metrics see ground truth only after image-only full-grid inference."""
    volumes = CachedVolumes(manifest_path, manifest, split)
    if not len(volumes):
        raise ValueError("Cannot evaluate an empty split.")
    previous = model.training
    model.eval()
    cases = []
    try:
        for index, record in enumerate(volumes.records):
            image, target, spacing = volumes.load(index)
            tensor = torch.from_numpy(image).unsqueeze(0)
            if index == 0:
                predict_image(model, tensor, model_name, device, configuration)
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            prediction = predict_image(model, tensor, model_name, device, configuration)
            _sync(device)
            milliseconds = 1000 * (time.perf_counter() - start)
            memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            case = {"patient": record["patient"], "visit": record["visit"],
                **dependencies.utils.compute_case_metrics(prediction, target, 6),
                "inference_ms": milliseconds, "peak_allocated_cuda_bytes": memory,
                "shape_dhw": list(prediction.shape), "spacing_dhw": spacing.tolist()}
            if output_dir:
                destination = Path(output_dir) / "predictions" / (Path(record["cache_path"]).stem + ".npz")
                destination.parent.mkdir(exist_ok=True)
                np.savez_compressed(destination, prediction=prediction.numpy().astype(np.uint8),
                                    spacing_dhw=spacing, label_ids=np.asarray(manifest["label_ids"]),
                                    affine_xyz=np.asarray(record["resampled_affine_xyz"]))
                case["prediction_file"] = destination.relative_to(output_dir).as_posix()
            cases.append(case)
    finally:
        model.train(previous)
    result = {"cases": cases, "aggregate": dependencies.utils.aggregate_by_patient(cases),
              "metrics_grid": "v2_downsampled_resampled_grid_96x192x192",
              "native_resolution_accuracy_claimed": False,
              "test_status": "previously_inspected_evaluation_set" if split == "test" else "validation_for_model_selection",
              "mean_inference_ms": float(np.mean([r["inference_ms"] for r in cases])),
              "peak_allocated_cuda_bytes": max((r["peak_allocated_cuda_bytes"] or 0 for r in cases), default=0) or None,
              "latency_protocol": "One full-case warmup; synchronized transfer+forward+softmax-window-average/argmax+output-transfer; excludes disk/preprocessing/metrics. CPU window accumulation included."}
    result["patient_balanced_prostate_dice"] = result["aggregate"]["per_class"][5]["dice"]
    return result


def training_class_weights(manifest_path, manifest):
    """Actual new TRAIN mask counts; no VAL/TEST masks in loss weighting."""
    volumes = CachedVolumes(manifest_path, manifest, "train")
    counts = np.zeros(6, dtype=np.int64)
    for index in range(len(volumes)):
        _, mask, _ = volumes.load(index)
        counts += np.bincount(mask.reshape(-1), minlength=6)
    if np.any(counts == 0):
        raise ValueError("Every declared class must occur in TRAIN before formal training.")
    weights = 1 / np.sqrt(counts.astype(np.float64) / counts.sum())
    weights /= weights.mean()
    weights = np.clip(weights, .05, 10)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts.tolist()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]}}


def _restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    state = state["numpy"]
    np.random.set_state((state["algorithm"], np.asarray(state["keys"], dtype=np.uint32), state["position"], state["has_gauss"], state["cached_gaussian"]))


def _save_checkpoint(path, payload):
    temporary = Path(path).with_suffix(".pending.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def create_model(model_name, configuration, dependencies):
    if model_name not in MODELS:
        raise ValueError("Unknown model.")
    factory = dependencies.modules.UNet2D if model_name == "unet2d" else dependencies.modules.ImprovedUNet3D
    return factory(configuration["num_classes"], configuration["base_channels"])


def _optimizer_step(model, optimizer, scaler, clip_grad, amp):
    previous_scale = scaler.get_scale()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, error_if_nonfinite=not amp)
    finite = bool(torch.isfinite(norm))
    if amp and not finite and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise RuntimeError("Gradient reduction overflowed despite finite individual gradients.")
    scaler.step(optimizer)
    scaler.update()
    return float(norm) if finite else None, bool(amp and scaler.get_scale() < previous_scale)


def _identity(manifest_path, protocol, model_name, dependencies):
    return {"experiment_id": protocol["experiment_id"], "protocol_sha256": protocol["_sha256"],
            "manifest_sha256": sha256(manifest_path), "model": model_name,
            "configuration": protocol["configuration"], "source_hashes": dependencies.source_hashes,
            "v2_pipeline_sha256": sha256(__file__)}


def _plot_history(history, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["epoch"] for r in history], [r["train_loss"] for r in history])
    selected = [r for r in history if r["validation_prostate_dice"] is not None]
    axes[1].plot([r["epoch"] for r in selected], [r["validation_prostate_dice"] for r in selected], "o-")
    axes[0].set(xlabel="Epoch", ylabel="CE + multiclass soft Dice")
    axes[1].set(xlabel="Epoch", ylabel="VAL patient-balanced prostate Dice", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def validate_qa_receipt(qa_path, manifest_path, protocol):
    """Verify the existing real-data gates and final fixed fit step, not a flag alone."""
    if not qa_path:
        raise ValueError("Formal training requires the bound real-data QA receipt.")
    qa_path = Path(qa_path).resolve(strict=True)
    receipt = read_json(qa_path)
    identity = {"manifest_sha256": sha256(manifest_path), "protocol_sha256": protocol["_sha256"]}
    if receipt.get("status") != "passed" or any(receipt.get(k) != value for k, value in identity.items()):
        raise ValueError("Real-data QA has not passed for this manifest/protocol.")
    flags = ("full_image_only_inference_passed", "native_geometry_serialisation_passed", "independent_metric_check_passed", "passed")
    for name in MODELS:
        if any(receipt.get("models", {}).get(name, {}).get(flag) is not True for flag in flags):
            raise ValueError("Both models require complete image-only, native-geometry, metric and fit QA.")
    fit_path = qa_path.parent / "fit/fit_receipt.json"
    if sha256(fit_path) != receipt.get("fit_receipt_sha256"):
        raise ValueError("Real fit receipt digest differs from the combined QA receipt.")
    fit, gates = read_json(fit_path), protocol["qa"]
    if (fit.get("status") != "passed" or fit.get("seed") != gates["fit_seed"]
            or fit.get("steps") != gates["fit_steps"] or fit.get("case") != gates["fit_case"]
            or fit.get("amp") != gates["fit_amp"] or any(fit.get(k) != value for k, value in identity.items())):
        raise ValueError("Fit QA identity, case, precision or fixed budget differs from the protocol.")
    for name in MODELS:
        result = fit.get("models", {}).get(name, {})
        history = result.get("history", [])
        if not history or history[0].get("step") != 0 or history[-1].get("step") != gates["fit_steps"]:
            raise ValueError("Fit QA does not end at the single fixed final step.")
        first, last = history[0], history[-1]
        values = [first.get("loss"), last.get("loss"), last.get("prostate_dice"), last.get("foreground_dice")]
        if any(type(value) not in (int, float) or not np.isfinite(value) for value in values) or first["loss"] <= 0:
            raise ValueError("Fit QA contains undefined/nonfinite final metrics.")
        if (result.get("passed") is not True
                or last["prostate_dice"] < gates["minimum_fit_prostate_dice"]
                or last["foreground_dice"] < gates["minimum_fit_foreground_dice"]
                or last["loss"] / first["loss"] > gates["maximum_final_initial_loss_ratio"]
                or receipt["models"][name].get("final_fit_prostate_dice") != last["prostate_dice"]
                or receipt["models"][name].get("final_fit_foreground_dice") != last["foreground_dice"]):
            raise ValueError("Combined QA does not substantiate the predeclared final-step fit gates.")
    return receipt


def train(manifest_path, model_name, output_dir, protocol, dependencies, qa_receipt,
          device, resume=None):
    manifest = load_new_manifest(manifest_path, protocol, dependencies)
    identity = _identity(manifest_path, protocol, model_name, dependencies)
    validate_qa_receipt(qa_receipt, manifest_path, protocol)
    config = protocol["configuration"]
    if config["amp"] and device.type != "cuda":
        raise ValueError("Frozen formal training requires CUDA AMP. Use the Python API for CPU fixtures/QA.")
    seed_everything(config["seed"])
    dataset = SampledTrainingDataset(manifest_path, model_name, config, dependencies, manifest)
    batch_size = config["train_batch_size_2d"] if model_name == "unet2d" else config["train_batch_size_3d"]
    expected_steps = (len(dataset) + batch_size - 1) // batch_size
    model = create_model(model_name, config, dependencies).to(device)
    weights, counts = training_class_weights(manifest_path, manifest)
    weights = weights.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    history, best_score, best_epoch, start_epoch = [], None, None, 1
    if resume:
        output = Path(output_dir).resolve(strict=True)
        if Path(resume).resolve().parent != output:
            raise ValueError("Resume only in the original checkpoint directory.")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["identity"] != identity:
            raise ValueError("Resume identity differs: model/config/data/protocol/source.")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        history, best_score, best_epoch = checkpoint["history"], checkpoint["best_score"], checkpoint["best_epoch"]
        start_epoch = checkpoint["epoch"] + 1
        if start_epoch > config["epochs"]:
            raise ValueError("The fixed 30-epoch budget is already exhausted.")
        _restore_rng(checkpoint["rng_state"])
    else:
        output = _fresh_directory(output_dir)
    run = {"status": "running", "started_at": utc_now(), "identity": identity,
           "qa_receipt_sha256": sha256(qa_receipt), "class_names": CLASS_NAMES,
           "class_weights": weights.detach().cpu().tolist(), "training_voxel_counts": counts,
           "class_weight_formula": "TRAIN full-new-grid counts only; inverse sqrt frequency; normalize class mean; clamp [.05,10]; normalize class mean again; CE only",
           "requested_epochs": config["epochs"], "expected_samples_per_epoch": len(dataset),
           "expected_batches_per_epoch": expected_steps, "validation_epochs": config["validation_epochs"],
           "selection": config["selection"], "parameters": sum(p.numel() for p in model.parameters()),
           "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
           "python_version": platform.python_version(), "torch_version": str(torch.__version__),
           "numpy_version": np.__version__, "cuda_version": torch.version.cuda,
           "metrics_grid": "v2_downsampled_resampled_grid_96x192x192", "native_resolution_accuracy_claimed": False,
           "test_pixels_loaded_during_training": False,
           "completed_epochs": start_epoch - 1}
    write_json(output / "run.json", run)
    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            if sha256(manifest_path) != identity["manifest_sha256"]:
                raise RuntimeError("Immutable manifest changed during training.")
            dataset.set_epoch(epoch)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                num_workers=config["workers"], pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(config["seed"] + epoch), drop_last=False)
            model.train()
            _sync(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            examples, batches, skipped, loss_sum = 0, 0, 0, 0.0
            sampling = {"foreground_requested": 0, "foreground_missing_fallback": 0,
                        "selected_class_counts": [0] * 6, "hw_flip_count": 0}
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["mask"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=config["amp"]):
                    outputs = model(images)
                    loss = dependencies.modules.segmentation_loss(outputs, targets, class_weights=weights)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite training loss; checkpoint not advanced.")
                scaler.scale(loss).backward()
                gradient_norm, overflow = _optimizer_step(model, optimizer, scaler, config["clip_grad"], config["amp"])
                skipped += int(overflow)
                count = int(images.shape[0])
                examples += count
                batches += 1
                loss_sum += float(loss.detach()) * count
                sampling["foreground_requested"] += int(batch["foreground_requested"].sum())
                sampling["foreground_missing_fallback"] += int(batch["foreground_missing_fallback"].sum())
                sampling["hw_flip_count"] += int(batch["hw_flip_count"].sum())
                for chosen in batch["selected_class"].tolist():
                    if chosen > 0:
                        sampling["selected_class_counts"][chosen] += 1
            _sync(device)
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device)
            if batches != expected_steps or examples != len(dataset):
                raise RuntimeError("Actual sampling schedule did not complete its fixed per-epoch budget.")
            optimizer.zero_grad(set_to_none=True)
            del outputs, loss, images, targets
            evaluation = evaluate(model, manifest_path, manifest, "validation", model_name, device,
                config, dependencies) if epoch in config["validation_epochs"] else None
            score = evaluation["patient_balanced_prostate_dice"] if evaluation else None
            if evaluation and score is None:
                raise RuntimeError("Validation prostate score is undefined; no model selection is possible.")
            improved = score is not None and (best_score is None or score > best_score)
            if improved:
                best_score, best_epoch = score, epoch
            history.append({"epoch": epoch, "train_loss": loss_sum / examples,
                "training_examples": examples, "training_batches": batches,
                "optimizer_steps": batches - skipped, "amp_overflow_skipped_optimizer_steps": skipped,
                "last_gradient_norm_before_clipping": gradient_norm,
                "train_seconds": seconds, "peak_allocated_cuda_bytes_training": peak,
                "sampling": sampling, "validation_prostate_dice": score, "validation": evaluation})
            payload = {"schema_version": 2, "identity": identity, "config": {**config, "model": model_name,
                "label_ids": manifest["label_ids"]}, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                "best_score": best_score, "best_epoch": best_epoch, "history": history, "rng_state": _rng_state()}
            _save_checkpoint(output / "latest.pt", payload)
            if improved:
                _save_checkpoint(output / "best.pt", payload)
            if epoch == config["epochs"]:
                _save_checkpoint(output / "final.pt", payload)
            write_json(output / "history.json", history)
            _plot_history(history, output / "curves.png")
            _restore_rng(payload["rng_state"])
            run.update(completed_epochs=epoch, best_epoch=best_epoch, best_validation_prostate_dice=best_score,
                       actual_optimizer_steps=sum(r["optimizer_steps"] for r in history),
                       actual_training_examples=sum(r["training_examples"] for r in history))
            write_json(output / "run.json", run)
            print(f"{model_name} epoch {epoch}/30 loss={history[-1]['train_loss']:.5f} "
                  f"VAL prostate={score if score is not None else 'not scheduled'} steps={batches-skipped}/{batches}", flush=True)
        run.update(status="completed", completed_at=utc_now(),
                   best_checkpoint_sha256=sha256(output / "best.pt"),
                   final_checkpoint_sha256=sha256(output / "final.pt"))
        write_json(output / "run.json", run)
    except BaseException as error:
        run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", run)
        raise
    return output


def evaluate_checkpoint(manifest_path, checkpoint_path, model_name, split, output_dir,
                        protocol, dependencies, device, selection_freeze=None):
    manifest = load_new_manifest(manifest_path, protocol, dependencies)
    identity = _identity(manifest_path, protocol, model_name, dependencies)
    checkpoint_digest = sha256(checkpoint_path)
    freeze_digest = require_selection_freeze(selection_freeze, manifest_path, protocol, model_name,
        checkpoint_digest) if split == "test" else None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("identity") != identity:
        raise ValueError("Checkpoint does not match new manifest/protocol/model/source identity.")
    model = create_model(model_name, protocol["configuration"], dependencies).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    output = _fresh_directory(output_dir)
    result = evaluate(model, manifest_path, manifest, split, model_name, device,
                      protocol["configuration"], dependencies, output)
    if sha256(checkpoint_path) != checkpoint_digest or sha256(manifest_path) != identity["manifest_sha256"]:
        raise RuntimeError("Checkpoint/manifest changed during evaluation.")
    result.update(identity=identity, checkpoint_sha256=checkpoint_digest,
                  checkpoint_epoch=checkpoint["epoch"], split=split,
                  chosen_by="predeclared validation class-5 score; no test-based model selection",
                  class_names=CLASS_NAMES, completed_at=utc_now())
    if freeze_digest:
        result["selection_freeze_sha256"] = freeze_digest
    write_json(output / "metrics.json", result)
    write_json(output / "evaluation.json", result)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "prepare-test", "train", "evaluate"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--qa-receipt", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--selection-freeze", type=Path,
                        help="Required before TEST cache preparation/evaluation; freezes both completed models.")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    protocol["_sha256"] = sha256(args.protocol)
    verify_review_sources(protocol, args.protocol.resolve().parent)
    dependencies = load_dependencies(args.project_root, protocol)
    if args.mode in ("prepare", "prepare-test"):
        if not args.raw_root or (args.mode == "prepare" and not args.output_dir):
            parser.error("Cache preparation requires --raw-root and stage 1 requires a fresh --output-dir.")
        result = prepare(args.manifest, args.output_dir, args.raw_root, protocol, args.protocol, dependencies) if args.mode == "prepare" else prepare_test(args.manifest, args.raw_root, protocol, dependencies, args.selection_freeze)
    else:
        if not args.model or not args.output_dir:
            parser.error("Train/evaluate requires --model and --output-dir.")
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                              else "cpu" if args.device == "auto" else args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable.")
        if args.mode == "train":
            result = train(args.manifest, args.model, args.output_dir, protocol, dependencies,
                           args.qa_receipt, device, args.resume)
        else:
            if not args.checkpoint:
                parser.error("Evaluation requires --checkpoint.")
            result = evaluate_checkpoint(args.manifest, args.checkpoint, args.model, args.split,
                                         args.output_dir, protocol, dependencies, device, args.selection_freeze)
    print(f"Completed {args.mode}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
