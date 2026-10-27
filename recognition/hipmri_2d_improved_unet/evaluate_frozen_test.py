"""Evaluate the preselected frozen pair once; never select or tune on test data."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch
from torch.utils.data import DataLoader

from audit_test_manifest import verify_test_content
from dataset import HipMRIPairDataset, collate_padded, collect_pairs, patient_ids
from modules import build_model
from resource_monitor import StageMeasurement
from train import manifest_hash, run_epoch, seed_everything


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
# A different output directory or plan cannot create a second attempt.
ONCE_GUARD = "outputs/frozen_test_once_20261026.guard.json"
PERSIST_REPLACE_ATTEMPTS = 6
PERSIST_RETRY_INITIAL_SECONDS = 0.025
MODEL_EPOCHS = (("baseline_unet", 5), ("improved_unet", 17))
REQUIRED_SOURCES = {
    "modules.py", "dataset.py", "losses.py", "metrics.py", "train.py", "predict.py",
    "resource_monitor.py", "profile_inference.py", "label_schema.json",
    "checkpoint_selection.py", "audit_test_manifest.py", "evaluate_frozen_test.py",
}
SPLIT_FOLDERS = {
    "training": ("keras_slices_train", "keras_slices_seg_train"),
    "validation": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def persist(path, value):
    """Atomically update this attempt, retrying only transient replacement I/O.

    Each call owns a unique temporary file. Windows scanners/sync clients may
    briefly hold the destination open; retry that PermissionError without
    invoking model code again. A permanent error preserves the prior target.
    """
    path = Path(path)
    serialized = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary, failure = None, None
    try:
        descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(PERSIST_REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, path)
                temporary = None
                return
            except PermissionError:
                if attempt + 1 == PERSIST_REPLACE_ATTEMPTS:
                    raise
                time.sleep(PERSIST_RETRY_INITIAL_SECONDS * 2 ** attempt)
    except BaseException as error:
        failure = error
        raise
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"Could not remove this call's temporary file {temporary}: {cleanup_error}")


def finalize_report(report, output_dir, guard, primary_error=None):
    """Try both state destinations without masking an existing model error."""
    first_persistence_error = None
    destinations = ([output_dir / "evaluation.json"] if output_dir.exists() else []) + [guard]
    for path in destinations:
        try:
            persist(path, report)
        except BaseException as error:
            report.setdefault("persistence_errors", []).append(
                {"path": str(path), "type": type(error).__name__, "message": str(error)})
            if primary_error is not None:
                primary_error.add_note(f"Final state persistence also failed for {path}: {type(error).__name__}: {error}")
            elif first_persistence_error is None:
                first_persistence_error = error
                report["status"] = "incomplete"
                report["error"] = {"type": type(error).__name__, "message": str(error)}
            else:
                first_persistence_error.add_note(
                    f"Final state persistence also failed for {path}: {type(error).__name__}: {error}")
    if first_persistence_error is not None:
        raise first_persistence_error


def inside(root, name):
    relative = Path(name)
    require(not relative.is_absolute(), "Frozen file paths must be relative")
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Frozen path leaves its declared root")
    return path


def verify_hashes(hashes, root, label):
    require(isinstance(hashes, dict) and bool(hashes), f"Missing {label} hashes")
    for name, expected in hashes.items():
        require(isinstance(expected, str) and len(expected) == 64, f"Invalid {label} SHA256")
        require(sha(inside(root, name)) == expected, f"Frozen {label} changed: {name}")


def state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(cpu.dtype), list(cpu.shape)]).encode("utf-8"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def verify_checkpoint(spec, root):
    path = inside(root, spec["checkpoint"])
    require(sha(path) == spec["checkpoint_sha256"], f"{spec['model']}: checkpoint bytes changed")
    original = inside(root, spec["original_checkpoint"])
    require(sha(original) == spec["checkpoint_sha256"], f"{spec['model']}: original differs from frozen copy")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "model_config", "epoch", "validation_cross_entropy", "validation_id5_dice"):
        require(saved[key] == spec[key], f"{spec['model']}: checkpoint {key} differs from frozen choice")
    require(saved["seed"] == 3710 and saved["limited_training_smoke"] is False,
            "Expected the approved full-training checkpoint")
    require(saved["selection_metric"] == "id5_dice", "Checkpoint was not selected by validation ID5")
    weights = saved["model_state"]
    require(bool(weights) and all(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                                 and bool(torch.isfinite(tensor).all()) for tensor in weights.values()),
            "Checkpoint model tensors must be finite CPU values")
    require(sum(tensor.numel() for tensor in weights.values()) == spec["parameters"],
            "Checkpoint parameter inventory differs from frozen model")
    require(all(not tensor.is_floating_point() or tensor.dtype == torch.float32 for tensor in weights.values()),
            "Frozen model must retain FP32 weights")
    del saved
    return path


def validate_plan(plan, plan_path, data_root, device, root, project, synthetic=False):
    require(plan["status"] == "approved_for_execution" and plan["user_confirmation_received"] is True,
            "Test execution requires the recorded explicit approval")
    require(plan["once_guard"] == ONCE_GUARD, "The fixed once guard cannot be changed")
    require(not (root / ONCE_GUARD).exists(), "Frozen test attempt already exists; do not rerun")
    require([(item["model"], item["epoch"]) for item in plan["models"]] == list(MODEL_EPOCHS),
            "Only frozen baseline5 and Improved17 may be evaluated")
    for item in plan["models"]:
        require(item["model_config"] == {"in_channels": 1, "num_classes": 6, "base_channels": 16},
                "Frozen architecture configuration changed")
    evaluation = plan["evaluation"]
    expected = {"seed": 3710, "batch_size": 4, "num_workers": 0, "precision": "FP32",
                "device": "cpu" if synthetic else "cuda", "shuffle": False, "drop_last": False,
                "aggregation": "pixel_pooled", "prediction": "argmax", "num_classes": 6,
                "test_time_augmentation": False, "threshold_tuning": False,
                "one_pass_per_model": True, "post_test_tuning": False}
    require(all(evaluation.get(key) == value for key, value in expected.items()),
            "Evaluation controls differ from the frozen protocol")
    require(str(device) == evaluation["device"], "Device differs from the frozen protocol")
    require(not synthetic or plan.get("synthetic_test_only") is True, "CPU execution is only for constructed checks")
    require(REQUIRED_SOURCES.issubset(plan["source_sha256"]), "Missing frozen evaluation dependencies")
    verify_hashes(plan["source_sha256"], project, "source")
    verify_hashes(plan["protected_original_files_sha256"], root, "original record")
    recipe_path, audit_path = inside(root, plan["recipe_file"]), inside(root, plan["audit_file"])
    require(sha(recipe_path) == plan["recipe_sha256"] and sha(audit_path) == plan["audit_sha256"],
            "Frozen recipe or metadata/content audit changed")
    recipe, audit = read(recipe_path), read(audit_path)
    require(recipe["frozen_before_test_manifest_or_pixel_read"] is True and recipe["user_confirmation_received"] is True,
            "Model recipe must precede test data inspection")
    require(recipe["models"] == plan["models"] and recipe["evaluation"] == evaluation,
            "Recipe's model choices or evaluation rules changed")
    require(all(plan["source_sha256"].get(name) == digest for name, digest in recipe["source_sha256"].items()),
            "Original shared pipeline differs from the pre-audit recipe")
    require(bool(recipe["protected_original_files_sha256"]) and all(
        plan["protected_original_files_sha256"].get(name) == digest
        for name, digest in recipe["protected_original_files_sha256"].items()),
        "Protected original records differ from the pre-audit recipe")
    require(audit["status"] == "passed_full", "A successful complete metadata/content audit is required")
    require(audit["frozen_plan_sha256"] == plan["recipe_sha256"],
            "Audit is bound to a different frozen recipe")
    require(audit["prior_plan_sha256"] == recipe["prior_id5_plan_sha256"],
            "Audit is bound to a different prior ID5 plan")
    require(bool(audit["source_sha256"]) and all(
        plan["source_sha256"].get(name) == digest for name, digest in audit["source_sha256"].items()),
        "Audit source hashes differ from the frozen execution plan")
    require(audit["test_content_manifest_sha256"] == plan["data"]["test"]["test_content_manifest_sha256"],
            "Test content manifest differs from the frozen audit")
    split_pairs, split_ids = {}, {}
    for split, folders in SPLIT_FOLDERS.items():
        declared = plan["data"][split]
        actual_folders = (declared.get("image_folder", folders[0]), declared.get("mask_folder", folders[1]))
        require(actual_folders == folders, "Split folders cannot be remapped")
        pairs = collect_pairs(data_root / folders[0], data_root / folders[1])
        ids = sorted(patient_ids(pairs))
        expected_ids = declared.get("case_ids", declared.get("case_proxy_ids"))
        require(len(pairs) == declared["pairs"] and manifest_hash(pairs) == declared["manifest_sha256"]
                and ids == expected_ids, f"{split} metadata differs from plan")
        audited = audit["splits"][split]
        require(audited["pairs"] == len(pairs) and audited["manifest_sha256"] == manifest_hash(pairs)
                and audited["case_proxy_ids"] == ids, f"{split} differs from the frozen audit")
        split_pairs[split], split_ids[split] = pairs, set(ids)
    require(all(not split_ids[left] & split_ids[right] for left, right in
                (("training", "validation"), ("training", "test"), ("validation", "test"))),
            "Case groups overlap across train/validation/test")
    content = verify_test_content(split_pairs["test"], audit)
    for spec in plan["models"]:
        verify_checkpoint(spec, root)
    require(sha(plan_path) == plan["_plan_sha256"], "Execution plan changed during setup")
    return split_pairs["test"], audit, content


def claim_once(root, binding):
    """Atomic exclusive claim: incomplete attempts remain consumed."""
    path = root / ONCE_GUARD
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(binding, indent=2, allow_nan=False) + "\n")
    return path


def evaluate_model(spec, pairs, device, output_dir, report, guard, root):
    name = spec["model"]
    entry = {"status": "running", "checkpoint_epoch": spec["epoch"], "checkpoint_sha256": spec["checkpoint_sha256"],
             "forward_calls": 0, "forward_samples": 0, "optimizer_steps": 0}
    report["models"][name] = entry
    persist(guard, report)
    persist(output_dir / "evaluation.json", report)
    path = verify_checkpoint(spec, root)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(name, **spec["model_config"]).to(device).eval()
    model.load_state_dict(saved["model_state"], strict=True)
    del saved
    require(sum(parameter.numel() for parameter in model.parameters()) == spec["parameters"],
            "Constructed model parameter count differs from frozen choice")
    before = state_sha256(model)
    entry["state_sha256_before"] = before

    def count_forward(module, inputs, output):
        require(not module.training and not torch.is_grad_enabled(), "Test forward must disable training and gradients")
        require(not torch.is_autocast_enabled(device.type), "Test forward must not enable AMP")
        require(inputs[0].dtype == output.dtype == torch.float32, "Test forward must preserve FP32")
        require(output.ndim == 4 and output.shape[1] == 6 and output.shape[0] == inputs[0].shape[0],
                "Unexpected frozen model output layout")
        entry["forward_calls"] += 1
        entry["forward_samples"] += int(output.shape[0])
        # An interruption after any batch still leaves a persistent consumed attempt.
        persist(guard, report)

    hook = model.register_forward_hook(count_forward)
    loader = DataLoader(HipMRIPairDataset(pairs), batch_size=4, shuffle=False,
                        num_workers=0, drop_last=False, collate_fn=collate_padded)
    measurement = StageMeasurement(device, f"frozen_test_{name}",
        "One restored FP32 eval model, test inputs/logits; no optimizer or training gradients. "
        "Prior model is released before the next model; CUDA allocator cache is retained. "
        "Includes input I/O, shared metrics and persistent batch-count recording; not isolated forward latency.")
    try:
        with measurement, torch.inference_mode():
            actual = run_epoch(model, loader, device, num_classes=6, optimizer=None,
                               progress_every=50, progress_label=f"frozen test {name}")
    finally:
        hook.remove()
        entry["resource_measurement"] = measurement.result
    expected_batches = math.ceil(len(pairs) / 4)
    require((actual["samples"], actual["batches"], actual["optimizer_steps"]) == (len(pairs), expected_batches, 0),
            "Test pass samples/batches/update budget differs")
    require(entry["forward_calls"] == expected_batches and entry["forward_samples"] == len(pairs),
            "Expected exactly one forward per test batch and one pass per sample")
    after = state_sha256(model)
    require(before == after and all(parameter.grad is None for parameter in model.parameters()),
            "Frozen test must not create gradients or change model state")
    require(sha(path) == spec["checkpoint_sha256"], "Test changed frozen checkpoint bytes")
    entry.update(status="passed", state_sha256_after=after, parameters_have_no_gradients=True,
                 metrics=actual, full_test_passes=1)
    persist(output_dir / f"{name}_result.json", entry)
    persist(output_dir / "evaluation.json", report)
    persist(guard, report)
    del model
    return actual


def execute(args, root=None, project=None, allow_cpu_for_synthetic=False):
    root, project = Path(root or REPO).resolve(), Path(project or PROJECT).resolve()
    output_dir = Path(args.output_dir).resolve()
    require(output_dir.is_relative_to(root), "Evaluation output must stay inside this repository")
    require(not output_dir.exists(), "Use a new output directory; previous evidence cannot be overwritten")
    require(not (root / ONCE_GUARD).exists(), "Frozen test attempt already exists; do not rerun")
    plan_path = Path(args.plan).resolve()
    plan = read(plan_path)
    plan["_plan_sha256"] = sha(plan_path)
    device = torch.device(args.device)
    require(device.type in ("cpu", "cuda"), "Unsupported device")
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    pairs, audit, before_content = validate_plan(plan, plan_path, Path(args.data_root), device,
                                               root, project, allow_cpu_for_synthetic)
    torch.set_num_threads(1)
    seed_everything(3710)
    math_settings = {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                     "float32_matmul_precision": torch.get_float32_matmul_precision()}
    if not allow_cpu_for_synthetic:
        require(math_settings == {"matmul_allow_tf32": False, "cudnn_allow_tf32": True,
                                  "float32_matmul_precision": "highest"}, "Backend math controls changed")
    report = {"created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(), "status": "running",
              "performed_by": "Codex automation; not user-performed verification", "split": "test",
              "source_revision": plan["source_revision"], "plan_sha256": plan["_plan_sha256"],
              "recipe_sha256": plan["recipe_sha256"], "audit_sha256": plan["audit_sha256"],
              "test_manifest_sha256": plan["data"]["test"]["manifest_sha256"],
              "test_content_manifest_sha256": audit["test_content_manifest_sha256"],
              "frozen_checkpoint_sha256": {item["model"]: item["checkpoint_sha256"] for item in plan["models"]},
              "output_directory": str(output_dir.relative_to(root)), "once_guard": ONCE_GUARD,
              "evaluation": plan["evaluation"], "models": {}, "source_sha256": plan["source_sha256"],
              "content_verification_before": before_content,
              "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                          "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                          "math_settings": math_settings},
              "test_split_read": True, "new_training": False, "post_test_tuning": False,
              "claim_boundary": "One final test pass per frozen model. No test-based model, epoch, threshold or hyperparameter selection. Numeric case is a grouping proxy and ID5 interpretation retains the accepted derivative-ID assumption."}
    guard = claim_once(root, report)
    primary_error = None
    try:
        output_dir.mkdir(parents=True)
        persist(output_dir / "evaluation.json", report)
        for spec in plan["models"]:
            actual = evaluate_model(spec, pairs, device, output_dir, report, guard, root)
            expected_pixels = plan["data"]["test"].get("valid_pixels")
            if expected_pixels is not None:
                require(actual["valid_pixels"] == expected_pixels, "Evaluated pixels differ from audited original shapes")
        report["content_verification_after"] = verify_test_content(pairs, audit)
        verify_hashes(plan["source_sha256"], project, "source")
        verify_hashes(plan["protected_original_files_sha256"], root, "original record")
        for spec in plan["models"]:
            require(sha(inside(root, spec["checkpoint"])) == spec["checkpoint_sha256"], "Frozen weight file changed")
        require(sha(plan_path) == plan["_plan_sha256"], "Execution plan changed")
        require(sha(inside(root, plan["recipe_file"])) == plan["recipe_sha256"], "Frozen recipe changed")
        require(sha(inside(root, plan["audit_file"])) == plan["audit_sha256"], "Frozen audit changed")
        report.update(status="passed", full_test_passes_per_model=1, protected_original_files_unchanged=True,
                      frozen_sources_and_weights_unchanged=True)
    except BaseException as error:
        primary_error = error
        report["status"] = "incomplete"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        for entry in report["models"].values():
            if entry["status"] == "running":
                entry["status"] = "incomplete"
        raise
    finally:
        finalize_report(report, output_dir, guard, primary_error)
    print(f"Completed one frozen test pass per model: {output_dir}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT.parents[3] / "Final Report/HipMRI_2D_Improved_UNet_学习/data/keras_slices_data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
