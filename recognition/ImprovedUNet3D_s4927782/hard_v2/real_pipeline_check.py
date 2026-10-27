"""Fixed two-TRAIN-patient v1 image-only deployment and metric cross-check.

Reads only B006/Week0 and B038/Week0 source images/cache pixels. Ground truth is
read by this verifier after image-only deployment, never by infer_native.py.
No training, hyperparameter/model selection, TEST pixels or diagnostic failure
patients are involved. A passed check is software QA, not generalisation proof.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import infer_native as adapter

MANIFEST_SHA = "c47fe880d353629348786b852e3f699291af1a4d5628b2980bf42a74635d0424"
IDENTITY_SHA = "eacb84bf20626a6ec9d7f6d36489a81e2d4e4af9fbe4ade1ce8f508a2022e08a"
UTILS_SHA = "63c3111f457f531085c4658cdd03e4824172c82ac721c69983215b3115275a84"
CASE_KEYS = (("B006", "Week0"), ("B038", "Week0"))


def array_digest(value):
    import hashlib
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def confusion_metrics(prediction, target, classes):
    if prediction.shape != target.shape or min(int(prediction.min()), int(target.min())) < 0:
        raise ValueError("Metric input shape/range mismatch.")
    if max(int(prediction.max()), int(target.max())) >= classes:
        raise ValueError("Metric input contains undeclared class IDs.")
    confusion = np.bincount(classes * target.astype(np.int64).ravel() + prediction.astype(np.int64).ravel(),
                            minlength=classes ** 2).reshape(classes, classes)
    rows = []
    for c in range(classes):
        tp = int(confusion[c, c]); actual = int(confusion[c].sum()); predicted = int(confusion[:, c].sum())
        denominator = actual + predicted; union = denominator - tp
        rows.append({"class_index": c, "intersection": tp, "target_voxels": actual,
                     "predicted_voxels": predicted, "union": union,
                     "dice": 2 * tp / denominator if denominator else None,
                     "iou": tp / union if union else None})
    return rows, confusion


def select_cases(manifest):
    training = sorted((r for r in manifest["records"] if r["split"] == "train"),
                      key=lambda r: (r["patient"], r["visit"]))
    selected, seen = [], set()
    for row in training:
        if row["patient"] not in seen:
            selected.append(row); seen.add(row["patient"])
        if len(selected) == 2:
            break
    if tuple((r["patient"], r["visit"]) for r in selected) != CASE_KEYS:
        raise ValueError("Unexpected first two lexicographic distinct TRAIN patients/visits.")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    import torch
    import nibabel as nib

    project, data_root = args.project_root.resolve(), args.data_root.resolve()
    output = args.output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new QA output directory; existing evidence is preserved.")
    manifest_path = project / "artifacts/cache64/manifest.json"
    identity_path = manifest_path.with_name("cache_identity.json")
    utils_path = project / "utils.py"
    for path, expected in [(manifest_path, MANIFEST_SHA), (identity_path, IDENTITY_SHA), (utils_path, UTILS_SHA)]:
        if adapter.sha256(path) != expected:
            raise ValueError(f"Frozen source identity differs: {path.name}")
    manifest, identity = json.loads(manifest_path.read_text()), json.loads(identity_path.read_text())
    selected = select_cases(manifest)
    if identity["manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("Cache identity does not bind the frozen manifest.")
    spec = importlib.util.spec_from_file_location("frozen_v1_pipeline_utils", utils_path)
    utils = importlib.util.module_from_spec(spec); spec.loader.exec_module(utils)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "purpose": "TRAIN-only v1 software QA; no generalisation or new candidate-selection evidence",
        "selected_case_keys": [list(key) for key in CASE_KEYS],
        "selection_rule": "First visit of lexicographic first two different frozen TRAIN patients; fixed before execution.",
        "manifest_sha256": MANIFEST_SHA, "cache_identity_sha256": IDENTITY_SHA,
        "original_utils_sha256": UTILS_SHA, "original_modules_sha256": adapter.MODULES_SHA256,
        "adapter_sha256": adapter.sha256(HERE / "infer_native.py"),
        "verifier_sha256": adapter.sha256(__file__),
        "device": str(device), "amp": False, "cases": [],
        "scope": {"raw_images": 2, "cached_train_masks": 2, "models_per_case": 2,
                  "test_arrays_read": 0, "raw_masks_read": 0},
    }
    def save():
        (output / "pipeline_receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        for row in selected:
            key = (row["patient"], row["visit"])
            image_relative, cache_relative = Path(row["source_image"]), Path(row["cache_path"])
            if (image_relative.is_absolute() or ".." in image_relative.parts
                    or cache_relative.is_absolute() or ".." in cache_relative.parts):
                raise ValueError("Unexpected non-relative source/cache path.")
            image_path = (data_root / image_relative).resolve(strict=True)
            if not image_path.is_relative_to(data_root):
                raise ValueError("Image path escapes the requested dataset root.")
            case_output = output / f"{key[0]}_{key[1]}"
            image_directory = case_output / "image_only_input"
            image_directory.mkdir(parents=True)
            alias = image_directory / "image.nii.gz"
            alias.symlink_to(image_path)
            if list(image_directory.iterdir()) != [alias]:
                raise ValueError("Image-only directory contains unexpected files.")
            original_image = nib.load(alias)
            preprocessed, _, geometry = adapter.preprocess_image(alias, tuple(manifest["target_shape_dhw"]))
            # Opening this fixed TRAIN cache belongs to the verifier only.
            cache_path = manifest_path.parent / cache_relative
            with np.load(cache_path, allow_pickle=False) as archive:
                cache = {name: archive[name].copy() for name in ("image", "mask", "spacing")}
            pinned = identity["records"][cache_relative.as_posix()]
            if (pinned["patient"], pinned["visit"], pinned["split"]) != (*key, "train"):
                raise ValueError("Cache identity scope differs.")
            for name, value in cache.items():
                if array_digest(value) != pinned["arrays"][name]["sha256"]:
                    raise ValueError(f"Frozen {name} array digest differs for {key}.")
            if not np.array_equal(preprocessed, cache["image"]):
                raise ValueError(f"Image-only preprocessing differs from frozen v1 cache for {key}.")
            if not np.allclose(geometry["cache_affine_xyz"], row["resampled_affine_xyz"], rtol=0, atol=0):
                raise ValueError("Cache affine differs from frozen metadata.")
            case_report = {"patient": key[0], "visit": key[1], "split": "train",
                           "raw_image_sha256": adapter.sha256(image_path),
                           "cache_file_sha256": adapter.sha256(cache_path),
                           "array_sha256": {name: array_digest(value) for name, value in cache.items()},
                           "image_only_directory_entries": [p.name for p in image_directory.iterdir()],
                           "preprocessing_elementwise_equal": True, "models": {}}
            for name in ("unet2d", "improved3d"):
                checkpoint = project / "artifacts/full_seed42" / name / "best.pt"
                deployment = case_output / name
                # Production entry point gets only the image symlink and frozen checkpoint.
                adapter.main(["--image", str(alias), "--checkpoint", str(checkpoint),
                              "--output-dir", str(deployment), "--project-root", str(project),
                              "--device", str(device), "--no-amp", "--batch-size", "32"])
                with np.load(deployment / "prediction_cache.npz", allow_pickle=False) as archive:
                    deployed = archive["prediction"].copy()
                model, config, epoch, checkpoint_sha, _ = adapter.load_model(project, checkpoint, device)
                frozen_prediction, _, _ = utils.predict_volume(model, torch.from_numpy(cache["image"]).unsqueeze(0),
                                                               name, device, slice_batch_size=32, amp=False)
                if not np.array_equal(deployed, frozen_prediction.numpy()):
                    raise ValueError(f"Image-only and frozen-cache forward labels differ for {key}/{name}.")
                measured = utils.compute_case_metrics(deployed, cache["mask"], 6)
                own, confusion = confusion_metrics(deployed, cache["mask"], 6)
                if measured["per_class"] != own:
                    raise ValueError("Independent confusion-matrix metrics differ from original evaluator.")
                native = nib.load(deployment / "prediction_native.nii.gz")
                native_values = np.asanyarray(native.dataobj)
                if (native.shape != original_image.shape
                        or not np.allclose(native.affine, original_image.affine, rtol=0, atol=1e-5)
                        or not set(np.unique(native_values)).issubset(config["label_ids"])):
                    raise ValueError("Restored native output shape/affine/labels differ.")
                case_report["models"][name] = {
                    "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": epoch,
                    "cache_prediction_elementwise_equal": True,
                    "prediction_array_sha256": array_digest(deployed),
                    "independent_confusion_matrix": confusion.tolist(), "metrics": measured,
                    "native_shape_xyz": list(native.shape), "native_affine_matches_image": True,
                    "native_raw_label_ids": [int(v) for v in np.unique(native_values)],
                    "image_only_receipt_sha256": adapter.sha256(deployment / "receipt.json"),
                }
                print(f"PASS {key[0]}/{key[1]} {name}: image-only/cache pixel labels and metrics agree", flush=True)
                del model
            report["cases"].append(case_report)
            save()
        report["status"] = "passed"
        report["generalisation_evidence"] = False
        save()
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
