"""Summarize a fixed seed-42/43 TRAIN/VAL robustness study after v1 TEST was seen.

Reads JSON evidence only; never loads predictions, models or test evaluations.
No model selection, training, significance test or replacement final score occurs.
Protocol fields: training_seeds, training_epochs_per_model, manifest_sha256,
v1_test_already_observed, allowed_splits, training_sources_sha256, seed42_models.
The original training_sources.json must be beside the supplied protocol.
Cloud roots additionally require --cloud-amendment and their execution identity;
cloud comparisons jointly change seed and platform/software, not seed alone.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


MODELS = ("unet2d", "improved3d")
SEEDS = (42, 43)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, (dict, list)):
        raise ValueError(f"Expected JSON object/list: {path}.")
    return data


def mean_defined(values):
    values = [v for v in values if v is not None]
    return math.fsum(values) / len(values) if values else None


def difference(before, after):
    return after - before if before is not None and after is not None else None


def paired_patient_summary(rows):
    """Mean patient differences on common defined support; zero is eligible."""
    result = {"support": {}}
    for metric in ("dice", "iou"):
        eligible = [row for row in rows if row[metric] is not None]
        result[metric] = mean_defined(row[metric] for row in eligible)
        result["support"][metric] = {
            "eligible_n": len(eligible),
            "eligible_patient_ids": [row["patient"] for row in eligible],
            "excluded_ids": [row["patient"] for row in rows if row[metric] is None],
        }
    return result


def same(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Different metric fields: {label}.")
        for key in expected:
            same(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Different metric list: {label}.")
        for index, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, f"{label}/{index}")
    elif expected is None or actual is None:
        if actual is not None or expected is not None:
            raise ValueError(f"Undefined metric differs: {label}.")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Metric differs: {label}.")
    elif actual != expected:
        raise ValueError(f"Value differs: {label}.")


def class_metrics(rows, num_classes):
    if [row["class_index"] for row in rows] != list(range(num_classes)):
        raise ValueError("Every class must appear exactly once in class-index order.")
    result = []
    for row in rows:
        item = {"class_index": row["class_index"]}
        for metric in ("dice", "iou"):
            value = row[metric]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Dice/IoU must be null or finite values in [0,1].")
            item[metric] = value
        result.append(item)
    return result


def checked_metrics(evaluation, manifest, standalone=False):
    """Recompute case -> patient -> population means, excluding only nulls."""
    if evaluation.get("split", "validation") != "validation":
        raise ValueError("TEST/non-validation metric objects are forbidden, including within histories.")
    if standalone and (evaluation.get("split") != "validation" or evaluation.get("final_test_explicitly_requested")):
        raise ValueError("Only standalone validation evaluations are permitted; TEST is forbidden.")
    if evaluation.get("development_only"):
        raise ValueError("Smoke/development evidence is forbidden.")
    expected_keys = {(r["patient"], r["visit"]) for r in manifest["records"] if r["split"] == "validation"}
    cases = {}
    grouped = defaultdict(list)
    for row in evaluation["cases"]:
        key = (row["patient"], row["visit"])
        if key not in expected_keys or key in cases:
            raise ValueError("Validation case keys are duplicate or outside the frozen validation split.")
        classes = class_metrics(row["per_class"], manifest["num_classes"])
        case = {"patient": key[0], "visit": key[1], "per_class": classes,
                **{f"foreground_mean_{m}": mean_defined(c[m] for c in classes[1:]) for m in ("dice", "iou")}}
        count_fields = ("target_voxels", "predicted_voxels", "intersection", "union")
        case["per_class_counts"] = []
        for original, metrics in zip(row["per_class"], classes):
            counts = {name: original[name] for name in count_fields}
            if any(type(value) is not int or value < 0 for value in counts.values()):
                raise ValueError("Per-class voxel counts must be nonnegative integers.")
            target, predicted, intersection, union = (counts[name] for name in count_fields)
            if intersection > min(target, predicted) or union != target + predicted - intersection:
                raise ValueError("Per-class voxel counts are inconsistent.")
            same(metrics["dice"], 2 * intersection / (target + predicted) if target + predicted else None, "case Dice algebra")
            same(metrics["iou"], intersection / union if union else None, "case IoU algebra")
            case["per_class_counts"].append({"class_index": metrics["class_index"], **counts})
        for metric in ("dice", "iou"):
            same(row[f"foreground_mean_{metric}"], case[f"foreground_mean_{metric}"], "case foreground")
        cases[key] = case
        grouped[key[0]].append(case)
    if set(cases) != expected_keys or len(grouped) != 6:
        raise ValueError("Every validation visit of the same six patients must be present exactly once.")
    patients = []
    for patient, visits in sorted(grouped.items()):
        patients.append({"patient": patient, "num_cases": len(visits),
                         "per_class": [{"class_index": c, **{m: mean_defined(v["per_class"][c][m] for v in visits)
                                                           for m in ("dice", "iou")}} for c in range(manifest["num_classes"])],
                         **{f"foreground_mean_{m}": mean_defined(v[f"foreground_mean_{m}"] for v in visits) for m in ("dice", "iou")}})
    aggregate = {"num_cases": len(cases), "num_patients": len(patients), "patients": patients,
                 "per_class": [{"class_index": c, **{m: mean_defined(p["per_class"][c][m] for p in patients)
                                                   for m in ("dice", "iou")}} for c in range(manifest["num_classes"])],
                 **{f"foreground_mean_{m}": mean_defined(p[f"foreground_mean_{m}"] for p in patients) for m in ("dice", "iou")}}
    reported = evaluation["aggregate"]
    reported = {**reported, "patients": sorted(reported["patients"], key=lambda p: p["patient"])}
    same(reported, aggregate, "independent patient-balanced aggregation")
    return {"aggregate": aggregate, "cases": [cases[k] for k in sorted(cases)]}


def nonnegative(value, label, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or integer and type(value) is not int:
        raise ValueError(f"Invalid nonnegative resource value: {label}.")
    return value


def resources(history, run, evaluation):
    nonnegative(run["parameters"], "parameters", True)
    if evaluation.get("parameters", run["parameters"]) != run["parameters"]:
        raise ValueError("Evaluation and training parameter counts differ.")
    for key in ("mean_inference_ms", "peak_allocated_cuda_bytes"):
        if evaluation[key] is not None:
            nonnegative(evaluation[key], key, key != "mean_inference_ms")
    for row in history:
        for key in ("train_seconds", "optimizer_steps", "amp_overflow_skipped_optimizer_steps", "training_batches"):
            nonnegative(row[key], key, key != "train_seconds")
        if row["optimizer_steps"] + row["amp_overflow_skipped_optimizer_steps"] != row["training_batches"]:
            raise ValueError("Successful/skipped optimizer steps do not sum to training batches.")
    peaks = [r["peak_allocated_cuda_bytes_training"] for r in history if r["peak_allocated_cuda_bytes_training"] is not None]
    for peak in peaks:
        nonnegative(peak, "training VRAM", True)
    return {"parameters": run["parameters"], "train_loop_seconds": math.fsum(r["train_seconds"] for r in history),
            "successful_optimizer_steps": sum(r["optimizer_steps"] for r in history),
            "amp_overflow_skipped_steps": sum(r["amp_overflow_skipped_optimizer_steps"] for r in history),
            "training_batches": sum(r["training_batches"] for r in history),
            "peak_training_allocated_cuda_bytes": max(peaks) if peaks else None,
            "validation_mean_inference_ms": evaluation["mean_inference_ms"],
            "validation_peak_allocated_cuda_bytes": evaluation["peak_allocated_cuda_bytes"],
            "training_gpu": run.get("gpu_name"), "evaluation_gpu": evaluation.get("gpu_name"),
            "torch_version": run.get("torch_version"), "cuda_version": run.get("cuda_version"),
            "latency_protocol": evaluation.get("latency_protocol")}


def load_run(seed, model, run_root, val_root, manifest, manifest_sha, protocol, sourcebook, source_sha):
    paths = {name: run_root / model / filename for name, filename in
             (("run", "run.json"), ("history", "history.json"), ("provenance", "source_provenance.json"))}
    paths["evaluation"] = val_root / model / "evaluation.json"
    run, history, provenance, evaluation = (read_json(paths[name]) for name in ("run", "history", "provenance", "evaluation"))
    config = run["config"]
    if config["seed"] != seed or config["model"] != model or evaluation["model_config"] != config:
        raise ValueError("Run/evaluation model or seed/config differs from the explicitly requested study.")
    if {k: v for k, v in config.items() if k != "seed"} != protocol["training_conditions"][model]:
        raise ValueError("Run config differs from protocol training_conditions beyond seed.")
    if config.get("development_only") or config.get("max_batches") is not None or run["status"] != "completed":
        raise ValueError("Only completed full-data runs are permitted.")
    if any(x != 30 for x in (run["completed_epochs"], run["requested_total_epochs"], len(history))):
        raise ValueError("Both models at both seeds must have exactly 30 completed epochs.")
    if [r["epoch"] for r in history] != list(range(1, 31)) or any(r.get("development_only") for r in history):
        raise ValueError("History must contain ordered full-data epochs 1..30 exactly once.")
    if any(x["manifest_sha256"] != manifest_sha for x in (config, provenance, evaluation)):
        raise ValueError("Manifest SHA differs across study evidence.")
    for key in ("num_classes", "label_ids", "target_shape_dhw", "patient_splits"):
        if config[key] != manifest[key]:
            raise ValueError(f"Frozen config differs from manifest: {key}.")
    if (provenance["expected_sha256"] != source_sha or provenance["source_file_sha256"] != sourcebook["source_file_sha256"]
            or provenance["training_code_commit"] != sourcebook["training_code_commit"]):
        raise ValueError("Training sourcebook/provenance identity differs from the frozen study.")
    checkpoint_sha = evaluation["checkpoint_sha256"]
    if checkpoint_sha != provenance["checkpoint_sha256"]:
        raise ValueError("Evaluation checkpoint differs from the recorded trained checkpoint.")
    best_epoch = run["best_epoch"]
    if evaluation["checkpoint_epoch"] != best_epoch or evaluation["checkpoint_best_epoch"] != best_epoch:
        raise ValueError("Standalone validation is not from the selected checkpoint.")
    if seed == 42:
        pinned = protocol["seed42_models"][model]
        if pinned["checkpoint_sha256"] != checkpoint_sha or pinned["selected_epoch"] != best_epoch:
            raise ValueError("Seed-42 reference differs from its fixed protocol checkpoint/epoch.")
    metrics = checked_metrics(evaluation, manifest, standalone=True)
    historical = [checked_metrics(row["validation"], manifest) for row in history]
    scores = [row["aggregate"]["foreground_mean_dice"] for row in historical]
    defined = [(score, -(i + 1)) for i, score in enumerate(scores) if score is not None]
    if not defined or -max(defined)[1] != best_epoch:
        raise ValueError("Checkpoint selection is not the first maximum foreground validation Dice.")
    same(run["best_validation_foreground_dice"], scores[best_epoch - 1], "run best validation score")
    same(metrics, historical[best_epoch - 1], "selected history vs actual standalone validation")
    aggregate = metrics["aggregate"]
    return {"seed": seed, "model": model, "selected_epoch": best_epoch, "epochs": 30,
            "manifest_sha256": manifest_sha, "checkpoint_sha256": checkpoint_sha,
            "checkpoint_digest_origin": "evaluation and training provenance metadata; checkpoint bytes are not loaded",
            "training_sources_sha256": source_sha, "training_code_commit": provenance["training_code_commit"],
            "input_evidence": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()},
            "config": config, "foreground": {m: aggregate[f"foreground_mean_{m}"] for m in ("dice", "iou")},
            "prostate": {m: aggregate["per_class"][5][m] for m in ("dice", "iou")},
            "per_class": aggregate["per_class"], "patients": aggregate["patients"],
            "validation_cases": aggregate["num_cases"], "validation_patients": aggregate["num_patients"],
            "resources": resources(history, run, evaluation), "selected_history_matches_standalone": True}


def check_cloud_evidence(amendment_path, protocol_path, protocol, seed43_root, seed43_val, runs):
    amendment_path = Path(amendment_path).expanduser().resolve()
    amendment = read_json(amendment_path)
    base_sha = sha256(protocol_path)
    if (amendment.get("amendment_id") != "paired_seed43_cloud_validation" or amendment.get("authorized") is not True
            or amendment.get("base_protocol_sha256") != base_sha or amendment.get("training_seed") != 43
            or amendment.get("training_epochs_per_model") != 30
            or set(amendment.get("allowed_splits", [])) != {"train", "validation"}):
        raise ValueError("Cloud amendment authorization/base protocol/TRAIN-VAL scope differs.")
    platform = amendment["platform"]
    if (platform.get("provider") != "Virtual GPU Lab" or platform.get("region") != "ap-northeast-1"
            or platform.get("instance_type") != "g6e.xlarge" or platform.get("cuda_device_count") != 1
            or type(platform.get("cuda_device_count")) is not int
            or not isinstance(platform.get("gpu_name"), str) or not platform["gpu_name"]):
        raise ValueError("Cloud amendment platform/GPU identity is incomplete or unauthorized.")
    for field, supplied, prefix in (("training", seed43_root, "followup_seed43_cloud"),
                                    ("validation", seed43_val, "validation_seed43_cloud")):
        relative = Path(amendment["output_roots"][field])
        expected = (protocol_path.parent / relative).resolve()
        if (len(relative.parts) != 2 or relative.parts[0] != "artifacts"
                or not (relative.parts[1] == prefix or relative.parts[1].startswith(prefix + "_"))
                or not expected.is_relative_to(protocol_path.parent) or expected != Path(supplied).resolve()):
            raise ValueError("Cloud CLI roots must match the independent amended cloud output roots.")
        for protected in ("full_seed42", "validation_seed42", "followup_seed43", "validation_seed43", "test_final"):
            path = (protocol_path.parent / "artifacts" / protected).resolve()
            if expected == path or expected.is_relative_to(path) or path.is_relative_to(expected):
                raise ValueError("Cloud outputs overlap protected Rangpur/v1 evidence.")
        if any((expected / model).resolve().parent != expected for model in MODELS):
            raise ValueError("Cloud model directory symlink escapes its amended output root.")
    required = amendment.get("required_file_sha256", {})
    mandatory = dict(protocol["required_file_sha256"])
    mandatory.update({"followup_protocol.json": base_sha,
                      "tools/analyze_seed_followup.py": sha256(__file__)})
    if "jobs/train_pair_seed43_cloud.sh" not in required or any(required.get(k) != v for k, v in mandatory.items()):
        raise ValueError("Cloud amendment does not freeze the base inputs, runner and this analysis source.")
    for name, expected in required.items():
        relative = Path(name)
        path = (protocol_path.parent / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(protocol_path.parent)
                or str(relative).startswith("artifacts/test") or path.suffix not in (".json", ".py", ".sh", ".slurm", ".txt")
                or sha256(path) != expected):
            raise ValueError(f"Cloud required frozen file identity differs: {name}.")
    execution_path = Path(seed43_root).resolve() / "cloud_execution.json"
    execution = read_json(execution_path)
    if (execution["amendment_sha256"] != sha256(amendment_path) or execution["base_protocol_sha256"] != base_sha
            or execution["runner_sha256"] != required["jobs/train_pair_seed43_cloud.sh"]
            or execution["manifest_sha256"] != protocol["manifest_sha256"]
            or execution["training_sources_sha256"] != protocol["training_sources_sha256"]
            or execution["platform_operator_declaration"] != platform
            or execution["output_roots"] != amendment["output_roots"]
            or execution["required_file_sha256_verified"] != required
            or type(execution["visible_cuda_device_count"]) is not int or execution["visible_cuda_device_count"] != 1
            or execution["measured_gpu_name"] != platform["gpu_name"]):
        raise ValueError("Cloud execution identity differs from amendment/protocol/GPU.")
    environment = amendment["environment"]
    measured = execution["measured_environment"]
    fields = ("python_version", "torch_version", "cuda_version", "numpy_version")
    if set(measured) != set(fields) or set(environment) - set(fields) - {"extra_versions"}:
        raise ValueError("Unexpected/incomplete amended cloud environment fields.")
    if any(not isinstance(environment.get(k), str) or environment[k] != measured[k] for k in fields):
        raise ValueError("Cloud execution runtime differs from its authorized environment.")
    extras = environment.get("extra_versions", {})
    packages = execution["measured_extra_packages"]
    if (not isinstance(extras, dict) or not isinstance(packages, dict) or not isinstance(packages.get("matplotlib"), str)
            or any(not isinstance(v, str) or not v or packages.get(k) != v for k, v in extras.items())):
        raise ValueError("Cloud execution package versions differ from amendment.")
    for model in MODELS:
        for kind in ("run", "evaluation"):
            data = read_json(runs["43"][model]["input_evidence"][kind]["path"])
            if (data.get("gpu_name") != platform["gpu_name"]
                    or any(data.get(k) != measured[k] for k in fields if k != "numpy_version")
                    or not str(data.get("device", "")).startswith("cuda")):
                raise ValueError(f"Cloud {model} {kind} runtime/GPU differs from its execution identity.")
            if kind == "evaluation" and data.get("amp") is not True:
                raise ValueError("Cloud validation must use the authorized AMP inference setting.")
    reference_gpus = {runs["42"][model]["resources"][kind] for model in MODELS
                      for kind in ("training_gpu", "evaluation_gpu")}
    if len(reference_gpus) != 1 or not isinstance(next(iter(reference_gpus)), str) or "A100" not in next(iter(reference_gpus)):
        raise ValueError("Cloud comparison requires the recorded seed-42 A100 reference.")
    return {"amendment_path": str(amendment_path), "amendment_sha256": sha256(amendment_path),
            "cloud_execution_path": str(execution_path), "cloud_execution_sha256": sha256(execution_path),
            "reference_gpu": next(iter(reference_gpus)), "followup_gpu": platform["gpu_name"],
            "platform_operator_declaration": platform, "measured_environment": measured,
            "measured_extra_packages": execution["measured_extra_packages"], "required_file_sha256_verified": required,
            "runtime_validation_scope": "Run/evaluation metadata independently match GPU/Python/Torch/CUDA; NumPy and extra package versions are recorded in cloud_execution, not in the original run/evaluation schemas.",
            "platform_attestation_limit": "Provider/region/instance are operator declarations, not independent cloud attestation."}


def build_summary(seed42_root, seed43_root, seed42_val, seed43_val, manifest_path, protocol_path, cloud_amendment=None):
    manifest_path, protocol_path = Path(manifest_path).resolve(), Path(protocol_path).resolve()
    if cloud_amendment is None and ((Path(seed43_root) / "cloud_execution.json").exists()
                                    or Path(seed43_root).name.startswith("followup_seed43_cloud")):
        raise ValueError("Cloud output requires --cloud-amendment; it cannot be labelled a pure seed study.")
    manifest, protocol = read_json(manifest_path), read_json(protocol_path)
    manifest_sha = sha256(manifest_path)
    if (protocol["manifest_sha256"] != manifest_sha or protocol["training_seeds"] != [42, 43]
            or protocol["training_epochs_per_model"] != 30 or protocol["v1_test_already_observed"] is not True
            or set(protocol["allowed_splits"]) != {"train", "validation"}
            or protocol.get("experiment_id") != "paired_seed43_validation" or protocol.get("training_seed") != 43
            or protocol.get("diagnostic_gate", {}).get("approved_to_proceed") is not True):
        raise ValueError("Protocol must explicitly fix the post-v1-TEST seed-42/43 TRAIN/VAL-only 30-epoch study.")
    if manifest.get("partial") or manifest.get("development_only") or manifest["label_ids"] != list(range(6)) or manifest["num_classes"] != 6:
        raise ValueError("Expected the complete frozen six-class HipMRI manifest.")
    memberships = manifest["patient_splits"]
    sets = [set(memberships[name]) for name in ("train", "validation", "test")]
    if any(len(sets[i]) != len(memberships[name]) for i, name in enumerate(("train", "validation", "test"))) or any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Manifest patient partitions overlap or contain duplicates.")
    if any(r["patient"] not in memberships[r["split"]] for r in manifest["records"]):
        raise ValueError("Manifest case partition disagrees with patient assignment.")
    source_path = protocol_path.parent / "training_sources.json"
    sourcebook, source_sha = read_json(source_path), sha256(source_path)
    if (source_sha != protocol["training_sources_sha256"] or sourcebook["manifest_sha256"] != manifest_sha
            or sourcebook["training_code_commit"] != protocol["training_code_commit"]):
        raise ValueError("Training sourcebook identity differs from protocol/manifest.")
    required = protocol.get("required_file_sha256")
    if not isinstance(required, dict) or not required:
        raise ValueError("Protocol must record the required frozen input-file identities.")
    for relative, expected in required.items():
        path = Path(relative)
        actual = (protocol_path.parent / path).resolve()
        if path.is_absolute() or ".." in path.parts or not actual.is_relative_to(protocol_path.parent) or sha256(actual) != expected:
            raise ValueError(f"Required frozen input-file identity differs: {relative}.")
    runs = {}
    for seed, run_root, val_root in ((42, seed42_root, seed42_val), (43, seed43_root, seed43_val)):
        runs[str(seed)] = {m: load_run(seed, m, Path(run_root), Path(val_root), manifest, manifest_sha, protocol, sourcebook, source_sha) for m in MODELS}
    for model in MODELS:
        configs = [{k: v for k, v in runs[str(seed)][model]["config"].items() if k != "seed"} for seed in SEEDS]
        if configs[0] != configs[1]:
            raise ValueError(f"Seed-42/43 configurations differ beyond seed: {model}.")
    deltas, patient_deltas, secondary_aggregate_differences = {}, {}, {}
    for seed in SEEDS:
        pair = runs[str(seed)]
        patient_deltas[str(seed)] = [{"patient": a["patient"], **{m: difference(a["per_class"][5][m], b["per_class"][5][m]) for m in ("dice", "iou")}}
                                    for a, b in zip(pair["unet2d"]["patients"], pair["improved3d"]["patients"])]
        deltas[str(seed)] = paired_patient_summary(patient_deltas[str(seed)])
        secondary_aggregate_differences[str(seed)] = {
            m: difference(pair["unet2d"]["prostate"][m], pair["improved3d"]["prostate"][m])
            for m in ("dice", "iou")}
    changes = {}
    for model in MODELS:
        a, b = runs["42"][model], runs["43"][model]
        changes[model] = {field: {m: difference(a[field][m], b[field][m]) for m in ("dice", "iou")} for field in ("foreground", "prostate")}
        changes[model]["patients"] = [{"patient": x["patient"], **{m: difference(x["per_class"][5][m], y["per_class"][5][m]) for m in ("dice", "iou")}}
                                       for x, y in zip(a["patients"], b["patients"])]
    summary = {"schema_version": 1, "study": "post-v1-TEST fixed TRAIN/VAL seed robustness follow-up", "split": "validation",
            "v1_test_already_observed": True, "validation_patients_unique": 6, "training_seeds": [42, 43],
            "manifest_sha256": manifest_sha, "protocol_sha256": sha256(protocol_path), "training_sources_sha256": source_sha,
            "analysis_source_sha256": sha256(__file__),
            "required_file_sha256_verified": required,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "runs": runs,
            "per_seed_prostate_3d_minus_2d": deltas, "per_seed_patient_prostate_3d_minus_2d": patient_deltas,
            "secondary_prostate_difference_of_separate_aggregate_means": secondary_aggregate_differences,
            "seed43_minus_seed42": changes,
            "limits": ["This fixed follow-up was planned after v1 TEST was observed; it uses TRAIN/VAL only and cannot replace v1 TEST or claim to rescue M015.",
                       "The same six validation patients recur across models/seeds; they are not twelve independent patients. No pooled significance test or two-seed confidence interval is computed.",
                       "Only undefined metrics are excluded from means; every zero is retained. Null differences remain null if either operand is undefined.",
                       "Primary per-seed model differences average defined paired patient differences; support is reported separately for Dice/IoU. Secondary differences of separate aggregate means can use different patient support.",
                       "Comparisons describe seed sensitivity of the complete models, not an isolated residual/deep-supervision/3D-context ablation.",
                       "Training loop time excludes validation, checkpoint writing and queue wait. CUDA figures are allocated memory, not total device usage.",
                       "Checkpoint hashes are checked between JSON provenance/evaluations and pinned seed-42 protocol metadata; this tool does not load or rehash checkpoint bytes."]}
    if cloud_amendment is not None:
        summary["cloud"] = check_cloud_evidence(cloud_amendment, protocol_path, protocol, seed43_root, seed43_val, runs)
        summary["study"] = "post-v1-TEST TRAIN/VAL joint seed + cloud platform/software follow-up"
        summary["comparison_factors"] = ["training_seed", "GPU/platform", "software_environment"]
        summary["isolated_seed_effect_claimed"] = False
        summary["seed43_minus_seed42_interpretation"] = "Between seed42+A100 reference and seed43+amended cloud GPU/software; jointly confounded, not an isolated seed effect."
        summary["limits"] = [line for line in summary["limits"] if not line.startswith("Comparisons describe seed sensitivity")]
        summary["limits"].extend([
            "Seed42 is the A100 reference; seed43 changes seed, cloud GPU/platform and software together. Effects cannot be attributed independently to seed, hardware or software.",
            "Within-run paired 3D-minus-2D retains the original prespecified common-patient estimator. Cross-platform resource changes are hardware/software observations, not isolated seed or architecture effects.",
            "Within-run comparisons remain comparisons of complete models, not isolated residual/deep-supervision/3D-context component ablations.",
            "Cloud authorization is checked against recorded execution and run/evaluation metadata; this analysis does not independently attest the physical cloud instance."])
    return summary


def plot_patients(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    patients = summary["runs"]["42"]["unet2d"]["patients"]
    names = [row["patient"] for row in patients]
    figure, axis = plt.subplots(figsize=(10, 5.8))
    undefined = []
    for seed in SEEDS:
        for model, color in (("unet2d", "#2865A8"), ("improved3d", "#CA5B26")):
            run = summary["runs"][str(seed)][model]
            values = [row["per_class"][5]["dice"] for row in run["patients"]]
            undefined.extend(f"{names[i]}/{model}/{seed}" for i, value in enumerate(values) if value is None)
            context = f"seed {seed}"
            if "cloud" in summary:
                context = "seed 42 / A100 reference" if seed == 42 else f"seed 43 / {summary['cloud']['followup_gpu']} / amended software"
            axis.plot(range(6), [v if v is not None else float("nan") for v in values],
                      color=color, linestyle="-" if seed == 42 else "--", marker="o" if seed == 42 else "s",
                      label=f"{model}, {context}, selected epoch {run['selected_epoch']}")
    axis.set_xticks(range(6), names)
    axis.set(ylabel="Validation prostate Dice (mean visits per patient)", xlabel="Same six validation patients", ylim=(-0.03, 1.03))
    axis.set_title("Post-v1-TEST TRAIN/VAL robustness follow-up: frozen models, two seeds" if "cloud" not in summary
                   else "Post-v1-TEST TRAIN/VAL: joint seed + platform/software change\nSeed 42 / A100 reference vs seed 43 / amended cloud GPU and software")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    caption = "No test scores, pooled significance or replacement v1 result. Zero values retained; undefined values shown as gaps."
    if "cloud" in summary:
        caption += "\nCloud comparison cannot isolate a seed effect; within-run paired estimator is unchanged."
    if undefined:
        caption += "\nUndefined: " + ", ".join(undefined)
    figure.text(0.5, 0.015, caption, ha="center", fontsize=8, wrap=True)
    figure.tight_layout(rect=(0, 0.12 if "cloud" in summary else 0.08 if undefined else 0.05, 1, 1))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seed42-root", "seed43-root", "seed42-val", "seed43-val", "manifest", "protocol", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cloud-amendment", type=Path, help="Required for cloud roots; validates joint seed/platform/software follow-up identity.")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError("Use a new output directory; prior evidence must not be overwritten.")
    summary = build_summary(args.seed42_root, args.seed43_root, args.seed42_val, args.seed43_val, args.manifest, args.protocol, args.cloud_amendment)
    output.mkdir(parents=True, exist_ok=False)
    plot_patients(summary, output / "patient_seed_prostate_dice.png")
    summary["figure"] = "patient_seed_prostate_dice.png"
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Validated TRAIN/VAL follow-up only: {output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
