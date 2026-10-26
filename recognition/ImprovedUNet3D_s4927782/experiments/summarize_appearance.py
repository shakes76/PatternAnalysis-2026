"""Audit metadata and independently summarize the complete TRAIN/VAL study.

No model, tensor, cache array or TEST inference is loaded. This post-test
motivated exploratory study cannot replace frozen v1 results or diagnose M015.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MODELS = ("seed42_reference", "seed43_reference", "gamma_only", "gamma_polarity")
CONDITIONS = ("clean", "gamma_0.7", "gamma_1.5", "polarity", "spatial_gain")
EXPERIMENT = "appearance_augmentation_v2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + x)))


def safe(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and ".." not in p.parts, "Unsafe evidence path.")
    result = (root / p).resolve(strict=True)
    require(result.is_relative_to(root.resolve()), "Evidence path escapes its root.")
    return result


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def equal(got, expected, context):
    """Strict structural/count checks; tolerate only floating roundoff."""
    if isinstance(expected, dict):
        require(isinstance(got, dict) and set(expected) <= set(got), context + ": missing fields.")
        for key, value in expected.items():
            equal(got[key], value, context + "/" + key)
    elif isinstance(expected, list):
        require(isinstance(got, list) and len(got) == len(expected), context + ": list length differs.")
        for i, value in enumerate(expected):
            equal(got[i], value, context + f"/{i}")
    elif isinstance(expected, float):
        require(type(got) in (int, float) and math.isfinite(got)
                and math.isclose(got, expected, rel_tol=1e-10, abs_tol=1e-10), context + ": number differs.")
    else:
        require(type(got) is type(expected) and got == expected, context + ": value differs.")


def recompute(cases, expected_keys, shape):
    """Counts -> case metrics -> visit means -> equally weighted patients."""
    keys = [(c["patient"], c["visit"]) for c in cases]
    require(len(keys) == len(expected_keys) and len(set(keys)) == len(keys)
            and set(keys) == set(expected_keys), "Missing, duplicate or out-of-scope evaluation case.")
    grouped, compact = defaultdict(list), []
    for case in cases:
        require(case["shape_dhw"] == shape, "Evaluation grid differs.")
        require([v["class_index"] for v in case["per_class"]] == list(range(6)), "Expected six ordered classes.")
        per_class = []
        for item in case["per_class"]:
            tp, pred, target = (item[k] for k in ("intersection", "predicted_voxels", "target_voxels"))
            require(all(type(v) is int and v >= 0 for v in (tp, pred, target))
                    and tp <= min(pred, target), "Invalid integer voxel counts.")
            union, denominator = pred + target - tp, pred + target
            row = {"class_index": item["class_index"], "intersection": tp,
                   "predicted_voxels": pred, "target_voxels": target, "union": union,
                   "dice": 2 * tp / denominator if denominator else None,
                   "iou": tp / union if union else None}
            equal(item, row, "case metric")
            per_class.append(row)
        require(sum(v["predicted_voxels"] for v in per_class) == math.prod(shape)
                and sum(v["target_voxels"] for v in per_class) == math.prod(shape), "Voxel totals differ from grid.")
        fg = {f"foreground_mean_{metric}": mean(v[metric] for v in per_class[1:]) for metric in ("dice", "iou")}
        equal(case, fg, "case foreground")
        grouped[case["patient"]].append({"per_class": per_class, **fg})
        p = per_class[5]
        compact.append({"patient": case["patient"], "visit": case["visit"], "tp": p["intersection"],
                        "fp": p["predicted_voxels"] - p["intersection"],
                        "fn": p["target_voxels"] - p["intersection"], "dice": p["dice"], "iou": p["iou"]})
    patients = []
    for patient, visits in sorted(grouped.items()):
        patients.append({"patient": patient, "num_cases": len(visits), "per_class": [
            {"class_index": i, **{metric: mean(v["per_class"][i][metric] for v in visits)
                                    for metric in ("dice", "iou")}} for i in range(6)],
            **{f"foreground_mean_{metric}": mean(v[f"foreground_mean_{metric}"] for v in visits) for metric in ("dice", "iou")}})
    aggregate = {"num_cases": len(cases), "num_patients": len(patients), "patients": patients,
                 "per_class": [{"class_index": i, **{m: mean(p["per_class"][i][m] for p in patients)
                                                    for m in ("dice", "iou")}} for i in range(6)],
                 **{f"foreground_mean_{m}": mean(p[f"foreground_mean_{m}"] for p in patients) for m in ("dice", "iou")}}
    return aggregate, sorted(compact, key=lambda x: (x["patient"], int(x["visit"].removeprefix("Week"))))


def engineering_gate(results, paired):
    a, b = results["gamma_only"], results["gamma_polarity"]
    def delta(condition, metric):
        return b[condition]["aggregate"][metric] - a[condition]["aggregate"][metric]
    values = {"clean_prostate_dice_delta": delta("clean", "prostate_dice"),
              "clean_foreground_dice_delta": delta("clean", "foreground_mean_dice"),
              "polarity_prostate_dice_delta": delta("polarity", "prostate_dice")}
    checks = {"clean_prostate_drop_le_0.02": values["clean_prostate_dice_delta"] >= -0.02,
              "clean_foreground_drop_le_0.02": values["clean_foreground_dice_delta"] >= -0.02,
              "polarity_prostate_gain_ge_0.05": values["polarity_prostate_dice_delta"] >= 0.05}
    return {"deltas_B_minus_A": values, "numeric_checks": checks,
            "numeric_gate_passes": all(checks.values()), "paired_execution_verified": paired,
            "development_indication_supported": paired and all(checks.values()),
            "interpretation": "Prespecified development indication only; no statistical superiority, new TEST or v1 replacement claim."}


def summarize(protocol_path, evaluation_path, project=PROJECT):
    project = Path(project).resolve()
    protocol, index = read(protocol_path), read(evaluation_path)
    protocol_sha, files = sha(protocol_path), {}
    require(protocol["experiment_id"] == EXPERIMENT and protocol["allowed_splits"] == ["train", "validation"]
            and protocol["training_seed"] == 42 and protocol["training_epochs_per_model"] == 30, "Protocol identity/budget differs.")
    require(tuple(protocol["conditions"]) == CONDITIONS and set(protocol["models"]) == set(MODELS), "Incomplete planned design.")
    equal(protocol["engineering_gate"], {"B_clean_prostate_Dice_minus_A_min": -0.02,
          "B_clean_foreground_Dice_minus_A_min": -0.02, "B_polarity_prostate_Dice_minus_A_min": 0.05}, "planned guard")
    for relative, digest in protocol["required_file_sha256"].items():
        require(sha(safe(project, relative)) == digest, "Planned source/input SHA differs: " + relative)
    manifest_path = safe(project, protocol["manifest"])
    require(sha(manifest_path) == protocol["manifest_sha256"], "Manifest identity differs.")
    require(sha(safe(project, "training_sources.json")) == protocol["training_sources_sha256"], "Original sourcebook differs.")
    manifest = read(manifest_path)
    require(not manifest.get("partial") and not manifest.get("development_only") and manifest["num_classes"] == 6
            and manifest["label_ids"] == list(range(6)), "Expected complete six-class manifest.")
    partitions = manifest["patient_splits"]
    require(set(partitions) == {"train", "validation", "test"}
            and len(set(sum(partitions.values(), []))) == sum(map(len, partitions.values())), "Patient partitions overlap.")
    records = manifest["records"]
    require(Counter(r["split"] for r in records) == {"train": 143, "validation": 38, "test": 30}, "Manifest scope differs.")
    require(all(r["patient"] in partitions[r["split"]] for r in records), "Record patient partition differs.")
    keys = {(r["patient"], r["visit"]) for r in records if r["split"] == "validation"}
    require(len(keys) == 38 and len(partitions["validation"]) == 6
            and {p for p, _ in keys} == set(partitions["validation"]), "Expected 38 VAL scans/six patients.")
    identity_sha = sha(safe(project, protocol["cache_identity"]))
    equal(index, {"status": "completed", "experiment_id": EXPERIMENT, "split": "validation",
          "protocol_sha256": protocol_sha, "manifest_sha256": protocol["manifest_sha256"],
          "cache_identity_sha256": identity_sha, "conditions": list(CONDITIONS), "n_scans_per_condition": 38,
          "n_unique_patients": 6, "saved_arrays": False}, "evaluation index")
    require(set(index["results"]) == set(index["models"]) == set(MODELS), "Index model scope differs.")
    require(index["runtime"]["device"] == "cuda" and index["runtime"]["amp"] == protocol["inference"]["amp"]
            and protocol["inference"]["split"] == "validation", "Inference device/precision/split differs.")
    results, training = {}, {}
    for name in MODELS:
        spec, receipt = protocol["models"][name], index["models"][name]
        checkpoint_path = safe(project, spec["checkpoint"])
        require(sha(checkpoint_path) == receipt["checkpoint_sha256"] and receipt["state_unchanged"] is True,
                "Checkpoint identity or unchanged-state receipt differs.")
        require(receipt["seed"] == spec["seed"] and receipt["parameters"] > 0, "Model metadata differs.")
        if name.endswith("reference"):
            require(receipt["checkpoint_sha256"] == spec["checkpoint_sha256"]
                    and receipt["checkpoint_epoch"] == spec["selected_epoch"], "Frozen reference differs.")
        else:
            root = checkpoint_path.parent
            run, history, provenance = (read(root / f) for f in ("run.json", "history.json", "provenance.json"))
            experiment = {"experiment_id": EXPERIMENT, "arm_tag": name, "protocol_sha256": protocol_sha}
            equal(run, {"experiment": experiment, "config": protocol["training_conditions"], "status": "completed",
                       "completed_epochs": 30, "requested_total_epochs": 30, "development_only": False,
                       "source_file_sha256": protocol["required_file_sha256"]}, "arm run")
            require(run["config"] == protocol["training_conditions"], "Arm configuration differs beyond allowed augmentation.")
            equal(provenance, {"experiment": experiment, "protocol_sha256": protocol_sha,
                  "manifest_sha256": protocol["manifest_sha256"], "source_file_sha256": protocol["required_file_sha256"],
                  "checkpoint_sha256": receipt["checkpoint_sha256"]}, "arm provenance")
            for filename, field in (("run.json", "run_sha256"), ("history.json", "history_sha256"), ("provenance.json", "provenance_sha256")):
                digest = sha(root / filename)
                require(digest == receipt[field] and (filename == "provenance.json" or digest == provenance[field]), "Arm receipt SHA differs.")
                files[str((root / filename).relative_to(project))] = digest
            verification_path = safe(project, provenance["cache_verification_path"])
            digest, verification = sha(verification_path), read(verification_path)
            require(digest == provenance["cache_verification_sha256"] == receipt["cache_verification_sha256"]
                    == run["cache_verification_sha256"], "Cache verification SHA differs.")
            expected = {(r["patient"], r["visit"], r["split"]) for r in records if r["split"] in ("train", "validation")}
            rows = verification["records"]
            equal(verification, {"status": "verified", "manifest_sha256": protocol["manifest_sha256"],
                  "counts": {"train": 143, "validation": 38}, "test_arrays_opened": 0}, "cache receipt")
            require(len(rows) == 181 and {(r["patient"], r["visit"], r["split"]) for r in rows} == expected
                    and all(r["decoded_identity_verified"] is True for r in rows), "Incomplete or out-of-scope cache verification.")
            require(len(history) == 30 and [r["epoch"] for r in history] == list(range(1, 31)), "Incomplete epoch history.")
            curve = []
            for row in history:
                require(row["training_examples"] == row["training_batches"] == 143 and row["development_only"] is False
                        and all(type(row[k]) is int and 0 <= row[k] <= 143 for k in ("optimizer_steps", "amp_overflow_skipped_optimizer_steps"))
                        and row["optimizer_steps"] + row["amp_overflow_skipped_optimizer_steps"] == 143, "Incomplete epoch/optimizer coverage.")
                aggregate, _ = recompute(row["validation"]["cases"], keys, manifest["target_shape_dhw"])
                equal(row["validation"]["aggregate"], aggregate, "epoch aggregate")
                equal(row["validation_foreground_dice"], aggregate["foreground_mean_dice"], "epoch selector")
                require(math.isfinite(row["train_loss"]) and math.isfinite(row["train_seconds"]) and row["train_seconds"] > 0
                        and type(row["peak_allocated_cuda_bytes_training"]) is int and row["peak_allocated_cuda_bytes_training"] > 0,
                        "Nonfinite loss or invalid training resources.")
                curve.append({k: row[k] for k in ("epoch", "train_loss", "validation_foreground_dice", "train_seconds",
                              "optimizer_steps", "amp_overflow_skipped_optimizer_steps", "peak_allocated_cuda_bytes_training")})
            best = max(history, key=lambda r: r["validation_foreground_dice"])
            require(best["epoch"] == run["best_epoch"] == receipt["checkpoint_epoch"], "Earlier-tie clean selector differs.")
            equal(run["best_validation_foreground_dice"], best["validation_foreground_dice"], "best run score")
            training[name] = {"selected_epoch": best["epoch"], "initial_parameter_state_sha256": run["initial_parameter_state_sha256"],
                              "runtime": run["runtime"], "epochs": curve, "total_train_seconds": sum(r["train_seconds"] for r in history),
                              "total_optimizer_steps": sum(r["optimizer_steps"] for r in history),
                              "total_amp_skipped_steps": sum(r["amp_overflow_skipped_optimizer_steps"] for r in history),
                              "peak_allocated_cuda_bytes_training": max(r["peak_allocated_cuda_bytes_training"] or 0 for r in history)}
            files[str(verification_path.relative_to(project))] = digest
        require(set(index["results"][name]) == set(CONDITIONS), "Missing or extra stress condition.")
        results[name] = {}
        for condition in CONDITIONS:
            entry = index["results"][name][condition]
            path = safe(Path(evaluation_path).resolve().parent, entry["evaluation_file"])
            require(sha(path) == entry["evaluation_sha256"], "Evaluation file SHA differs.")
            evaluation = read(path)
            equal(evaluation, {"model_id": name, "condition": condition, "split": "validation", "saved_arrays": False,
                  "protocol_sha256": protocol_sha, "manifest_sha256": protocol["manifest_sha256"],
                  "checkpoint_sha256": receipt["checkpoint_sha256"], "amp": index["runtime"]["amp"]}, "evaluation metadata")
            aggregate, compact = recompute(evaluation["cases"], keys, manifest["target_shape_dhw"])
            equal(evaluation["aggregate"], aggregate, "evaluation aggregate")
            equal(entry["aggregate"], aggregate, "index aggregate")
            for field in ("mean_inference_ms", "peak_allocated_cuda_bytes"):
                equal(entry[field], evaluation[field], "index inference resources")
            require(all(type(c["inference_ms"]) in (int, float) and math.isfinite(c["inference_ms"])
                        and c["inference_ms"] >= 0 and type(c["peak_allocated_cuda_bytes"]) is int
                        and c["peak_allocated_cuda_bytes"] > 0 for c in evaluation["cases"]), "Invalid CUDA inference resources.")
            equal(evaluation["mean_inference_ms"], mean(c["inference_ms"] for c in evaluation["cases"]), "mean latency")
            equal(evaluation["peak_allocated_cuda_bytes"], max(c["peak_allocated_cuda_bytes"] for c in evaluation["cases"]), "peak memory")
            patient_rows = [{"patient": p["patient"], "n_visits": p["num_cases"], "prostate_dice": p["per_class"][5]["dice"],
                             "prostate_iou": p["per_class"][5]["iou"]} for p in aggregate["patients"]]
            totals = {k: v for k, v in aggregate.items() if k != "patients"}
            totals.update(prostate_dice=aggregate["per_class"][5]["dice"], prostate_iou=aggregate["per_class"][5]["iou"])
            worst = min((p["prostate_dice"] for p in patient_rows if p["prostate_dice"] is not None), default=None)
            results[name][condition] = {"aggregate": totals, "patients": patient_rows, "cases_prostate": compact,
                "worst_patient_prostate_dice": worst, "worst_patients_prostate": [p["patient"] for p in patient_rows if p["prostate_dice"] == worst],
                "mean_inference_ms": evaluation["mean_inference_ms"], "peak_allocated_cuda_bytes": evaluation["peak_allocated_cuda_bytes"]}
            files[str(path.relative_to(project)) if path.is_relative_to(project) else "evaluation/" + entry["evaluation_file"]] = entry["evaluation_sha256"]
    a, b = training["gamma_only"], training["gamma_polarity"]
    paired = {"initial_parameter_state_sha256_equal": a["initial_parameter_state_sha256"] == b["initial_parameter_state_sha256"],
              "training_runtime_exactly_equal": a["runtime"] == b["runtime"], "protocol_sources_and_receipts_verified": True}
    require(all(isinstance(training[n]["initial_parameter_state_sha256"], str)
                and len(training[n]["initial_parameter_state_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in training[n]["initial_parameter_state_sha256"])
                for n in MODELS[2:]), "Invalid initial state SHA.")
    patient_deltas = []
    for patient in sorted(partitions["validation"]):
        row = {"patient": patient}
        for condition in ("clean", "polarity"):
            values = [{p["patient"]: p["prostate_dice"] for p in results[n][condition]["patients"]} for n in MODELS[2:]]
            row[condition] = values[1][patient] - values[0][patient]
        patient_deltas.append(row)
    return {"schema_version": 1, "status": "completed", "experiment_id": EXPERIMENT, "split": "validation",
            "protocol_sha256": protocol_sha, "evaluation_index_sha256": sha(evaluation_path),
            "manifest_sha256": protocol["manifest_sha256"], "input_file_sha256": files,
            "scope": {"split": "validation", "n_scans": 38, "n_patients": 6, "n_models": 4, "n_conditions": 5,
                      "test_arrays_read_by_summarizer": 0, "arrays_loaded": False, "grid": "shared downsampled cache; not native clinical accuracy"},
            "paired_execution_checks": paired, "confounds": [] if all(paired.values()) else ["Initial states or runtimes differ; isolated augmentation comparison is not supported."],
            "engineering_gate": engineering_gate(results, all(paired.values())), "training": training, "results": results,
            "resource_scope": "Training seconds sum only timed training loops, excluding validation/checkpoint/plot overhead; inference uses recorded synchronized warmup-excluded cache-volume timings. CUDA bytes are allocated-memory peaks, not reserved VRAM.",
            "plot_data": {"prostate_dice_heatmap": {"model_ids": list(MODELS), "conditions": list(CONDITIONS),
                         "values": [[results[n][c]["aggregate"]["prostate_dice"] for c in CONDITIONS] for n in MODELS]},
                         "patient_arm_deltas": patient_deltas},
            "limits": protocol["limits"] + ["Independent reaggregation checks arithmetic and recorded identities, not clinical labels or hidden historical training behavior."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "evaluation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require(not os.path.lexists(args.output), "Summary output must be new; refusing overwrite.")
    result = summarize(args.protocol, args.evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
