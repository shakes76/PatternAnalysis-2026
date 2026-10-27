"""Run the predeclared five validation examples with frozen epoch5/17 models.

This diagnostic reads only the selected validation pairs and two frozen weights.
It never scans or reads test images, invokes a test evaluator, updates parameters,
reselects a model, or rewrites a historical result. Local arrays are retained in
ignored outputs so subsequent figure work need not run another model forward.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
PLAN = PROJECT / "evidence/final_validation_display_plan_20261026.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def inside_repo(name):
    require(isinstance(name, str) and not Path(name).is_absolute(), "Expected repository-relative path")
    path = (REPO / name).resolve()
    require(path.is_relative_to(REPO.resolve()), "Path escapes repository")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT.parents[3] / "Final Report/HipMRI_2D_Improved_UNet_学习/data/keras_slices_data")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/final_validation_display_20261026/run")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(output.is_relative_to((REPO / "outputs").resolve()), "Diagnostic outputs must stay under ignored repository outputs")
    require(not output.exists(), "Use a fresh output directory; preserve prior attempts")
    plan_hash = file_sha(PLAN)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    require(plan["prepared_before_selected_validation_pixel_reads_and_new_model_forward"] is True, "Missing predeclared plan")
    require(plan["inference"]["split"] == "validation" and len(plan["selection"]["cases"]) == 5, "Only five preselected validation cases are allowed")
    for name, expected in plan["unchanged_shared_sources"].items():
        require(file_sha(PROJECT / name) == expected, "Shared experiment source changed: " + name)
    require(file_sha(PROJECT / "evidence/selection_diagnosis_20261026.json") == plan["selection"]["source_sha256"], "Historical selection record changed")
    recipe = PROJECT / "evidence/frozen_test_recipe_20261026.json"
    require(file_sha(recipe) == plan["frozen_recipe_sha256"], "Frozen recipe changed")
    frozen_recipe = json.loads(recipe.read_text(encoding="utf-8"))
    require(len(plan["models"]) == 2 and {x["model"] for x in plan["models"]} == {"baseline_unet", "improved_unet"}, "Exactly one of each final model is required")
    require(plan["models"] == frozen_recipe["models"], "Diagnostic model specs must match the hash-bound frozen recipe exactly")
    old_cases = json.loads((PROJECT / "evidence/selection_diagnosis_20261026.json").read_text(encoding="utf-8"))["failure_cases"]
    require([case["filename"] for case in old_cases] == [case["filename"] for case in plan["selection"]["cases"]], "Preselected filenames changed")
    require([{key: case[key] for key in ("case_label", "filename", "case", "week", "slice")} for case in old_cases] == plan["selection"]["cases"], "Selected case metadata changed")
    selected = []
    for case in plan["selection"]["cases"]:
        filename = case["filename"]
        require(Path(filename).name == filename and filename.startswith("case_") and filename.endswith(".nii.gz"), "Invalid selected basename")
        image = (args.data_root / "keras_slices_validate" / filename).resolve()
        mask = (args.data_root / "keras_slices_seg_validate" / filename.replace("case_", "seg_", 1)).resolve()
        require(image.parent.name == "keras_slices_validate" and mask.parent.name == "keras_slices_seg_validate", "Only validation directories are allowed")
        require(image.is_file() and mask.is_file(), "Selected validation pair missing")
        selected.append((case, image, mask))
    checkpoint_paths = {}
    for spec in plan["models"]:
        require((spec["model"], spec["epoch"]) in {( "baseline_unet", 5), ("improved_unet", 17)}, "Only frozen final models allowed")
        path = inside_repo(spec["checkpoint"])
        require(file_sha(path) == spec["checkpoint_sha256"], "Frozen checkpoint digest mismatch")
        checkpoint_paths[spec["model"]] = path
    output.mkdir(parents=True)
    source_files = ["render_final_validation_cases.py", "predict.py", "prediction_panels.py", "dataset.py", "modules.py", "losses.py", "metrics.py"]
    started = {"created_at": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_hash,
               "source_sha256": {name: file_sha(PROJECT / name) for name in source_files},
               "split": "validation", "filenames": [x[0]["filename"] for x in selected],
               "new_training": False, "real_test_pixels_read": False}
    write_new_json(output / "execution_start.json", started)

    import numpy as np
    import torch
    from unittest.mock import patch
    import dataset
    from modules import build_model
    from predict import infer_preprocessed_sample
    from prediction_panels import save_prediction_panels

    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(3710)

    def state_digest(model):
        digest = sha256()
        for name, tensor in sorted(model.state_dict().items()):
            values = tensor.detach().cpu().contiguous()
            digest.update(json.dumps([name, str(values.dtype), list(values.shape)]).encode("utf-8"))
            digest.update(values.numpy().tobytes())
        return digest.hexdigest()

    allowed_inputs = {path for _, image, mask in selected for path in (image, mask)}
    file_hashes = {str(path): file_sha(path) for path in allowed_inputs}
    loaded = []
    load_nifti = dataset.nib.load

    def checked_nifti_load(path, *positional, **keywords):
        path = Path(path).resolve()
        require(path in allowed_inputs, "Attempted unselected or non-validation NIfTI access")
        nifti = load_nifti(path, *positional, **keywords)
        loaded.append({"directory": path.parent.name, "filename": path.name, "shape": list(nifti.shape),
                       "units": list(nifti.header.get_xyzt_units()), "affine": np.asarray(nifti.affine).tolist()})
        return nifti

    report = {"status": "running", "created_at": started["created_at"], "performed_by": "Codex authorized validation-only inference",
              "plan_sha256": plan_hash, "source_sha256": started["source_sha256"], "batch_size": 1,
              "selection_rule": plan["selection"]["rule"], "models": {}, "cases": [],
              "new_training": False, "optimizer_steps": 0, "real_test_nifti_reads": 0,
              "test_evaluator_or_recovery_called": False,
              "scope": "Five predeclared validation displays, not full-split scores or a new test evaluation",
              "limitations": plan["limits"]}
    hooks = []
    try:
        models = {}
        for spec in plan["models"]:
            name = spec["model"]
            checkpoint = torch.load(checkpoint_paths[name], map_location="cpu", weights_only=True)
            require(checkpoint["model"] == name and checkpoint["epoch"] == spec["epoch"] and checkpoint["model_config"] == spec["model_config"], "Checkpoint metadata differs from frozen choice")
            model = build_model(name, **checkpoint["model_config"]).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            require(sum(x.numel() for x in model.parameters()) == spec["parameters"], "Parameter inventory mismatch")
            report["models"][name] = {"epoch": spec["epoch"], "checkpoint_sha256": spec["checkpoint_sha256"],
                                      "state_before": state_digest(model), "forward_calls": 0, "samples": 0,
                                      "mode_checks": [], "parameters": spec["parameters"]}
            def track(module, inputs, result, tag=name):
                row = report["models"][tag]
                row["forward_calls"] += 1
                row["samples"] += int(inputs[0].shape[0])
                row["mode_checks"].append({"training": module.training, "grad_enabled": torch.is_grad_enabled(), "inference_mode": torch.is_inference_mode_enabled()})
                require(not module.training and not torch.is_grad_enabled() and torch.is_inference_mode_enabled(), "Invalid diagnostic inference mode")
            hooks.append(model.register_forward_hook(track))
            models[name] = model
            del checkpoint
        with patch.object(dataset.nib, "load", side_effect=checked_nifti_load):
            for case_index, (case, image_path, mask_path) in enumerate(selected):
                sample = dataset.load_pair(image_path, mask_path)
                values = {name: infer_preprocessed_sample(model, sample, device, 6) for name, model in models.items()}
                image = values["baseline_unet"]["image"]
                target = values["baseline_unet"]["target"]
                require(np.array_equal(target, values["improved_unet"]["target"]), "Models did not share the same reference")
                require(np.array_equal(image, values["improved_unet"]["image"]), "Models did not share the same normalized MRI")
                require(int((target == 5).sum()) == old_cases[case_index]["models"]["baseline_best"]["reference_pixels"], "Reference differs from historical case")
                label = case["case_label"].split("_", 1)[0]
                figure_name = f"{label}_final_validation_epoch5_17.png"
                panel_result = save_prediction_panels(output / figure_name, image, target,
                    {"Baseline epoch 5": values["baseline_unet"]["prediction"], "Improved epoch 17": values["improved_unet"]["prediction"]},
                    {"split": "validation", "filename": case["filename"], "case_label": label + " | predeclared historical case",
                     "claim_boundary": "Final frozen models; selected validation only, not a test or full-validation score. ID5 preservation assumed; anatomy/physical units unconfirmed."},
                    num_classes=6, target_id=5, zoom=True)
                array_name = f"{label}_original_size_arrays.npz"
                with (output / array_name).open("xb") as stream:
                    np.savez_compressed(stream, image=image, reference=target,
                        baseline_prediction=values["baseline_unet"]["prediction"], improved_prediction=values["improved_unet"]["prediction"])
                row = {**case, "split": "validation", "anatomical_base_apex": "unconfirmed", "original_size": list(target.shape),
                       "source_files": {"image": image_path.name, "mask": mask_path.name,
                                        "image_sha256": file_hashes[str(image_path)], "mask_sha256": file_hashes[str(mask_path)]},
                       "models": {name: {"epoch": report["models"][name]["epoch"], "cross_entropy": result["cross_entropy"], "metrics": result["metrics"], "logit_shape": result["logit_shape"]} for name, result in values.items()},
                       "panels": panel_result, "figure": figure_name, "figure_sha256": file_sha(output / figure_name),
                       "local_arrays": array_name, "local_arrays_sha256": file_sha(output / array_name),
                       "array_policy": "Saved only in ignored outputs; never add the arrays to Git."}
                report["cases"].append(row)
                print(f"Validation display {len(report['cases'])}/5: {case['filename']}", flush=True)
        require(len(loaded) == 10 and {item["filename"] for item in loaded} == {path.name for path in allowed_inputs}, "NIfTI load scope mismatch")
        for spec in plan["models"]:
            name = spec["model"]
            model = models[name]
            row = report["models"][name]
            row["state_after"] = state_digest(model)
            row["state_unchanged"] = row["state_after"] == row["state_before"]
            row["parameters_have_no_gradients"] = all(x.grad is None for x in model.parameters())
            require(row["state_unchanged"] and row["parameters_have_no_gradients"] and row["forward_calls"] == 5 and row["samples"] == 5, "Model state or forward scope mismatch")
            require(file_sha(checkpoint_paths[name]) == spec["checkpoint_sha256"], "Checkpoint bytes changed")
        require(all(file_sha(Path(path)) == expected for path, expected in file_hashes.items()), "Selected input bytes changed")
        require(file_sha(PLAN) == plan_hash, "Plan changed during display run")
        require(sum(x["forward_calls"] for x in report["models"].values()) == 10, "Expected exactly ten final-model validation forwards")
        require(all(file_sha(PROJECT / name) == expected for name, expected in started["source_sha256"].items()), "Source changed during display run")
        report.update(status="passed", completed_at=datetime.now(timezone.utc).isoformat(), selected_input_files_unchanged=True,
                      checkpoint_files_unchanged=True, nifti_loads=loaded, original_pixel_counts_only=True,
                      total_model_forward_calls=sum(x["forward_calls"] for x in report["models"].values()),
                      runtime={"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                               "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device": str(device),
                               "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                               "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                               "float32_matmul_precision": torch.get_float32_matmul_precision(), "cpu_threads": torch.get_num_threads(),
                               "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark})
        write_new_json(output / "validation_display_summary.json", report)
    except Exception as error:
        report.update(status="incomplete", error={"type": type(error).__name__, "message": str(error)}, nifti_loads=loaded)
        write_new_json(output / "validation_display_incomplete.json", report)
        raise
    finally:
        for hook in hooks:
            hook.remove()


if __name__ == "__main__":
    main()
