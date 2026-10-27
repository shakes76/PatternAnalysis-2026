"""Count-audit the frozen B/C/D M015 recheck without loading arrays or weights.

The existing VAL-only recommendation is retained even if another arm has better
M015 measurements. Retention checks are descriptive, never model selection.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from tools import summarize_m015_followup as counts
from tools import evaluate_recovery_m015 as evaluator

COUNT_HELPER_SHA = "347273495c1376384b94966b8ff2ba44aeb4c2babca13b8bda177b5ee1f50e87"
EVALUATOR_SHA = "8287f71c60843dfab93bb4e1f5e619bb12ec29d924521627aab328467290fb4e"
EXPERIMENT, MODELS, KEYS = evaluator.EXPERIMENT, evaluator.MODEL_IDS, evaluator.CASE_KEYS
require, sha, read, safe, equal = counts.require, counts.sha, counts.read, counts.safe, counts.equal


def summarize(protocol_path, evaluation_path, project=PROJECT):
    project = Path(project).resolve()
    protocol_path, evaluation_path = Path(protocol_path).resolve(strict=True), Path(evaluation_path).resolve(strict=True)
    require(protocol_path.is_relative_to(project) and evaluation_path.is_relative_to(project), "Evidence must reside inside the project.")
    require(sha(Path(counts.__file__)) == COUNT_HELPER_SHA and sha(Path(evaluator.__file__)) == EVALUATOR_SHA,
            "An independently pinned count/evaluation helper changed.")
    plan, index = read(protocol_path), read(evaluation_path)
    r1_path, val_path = (safe(project, plan[k]) for k in ("recovery_protocol_path", "val_summary_path"))
    require(sha(r1_path) == plan["recovery_protocol_sha256"] and sha(val_path) == plan["val_summary_sha256"], "Frozen R1/VAL summary SHA differs.")
    r1, validation = read(r1_path), read(val_path)
    evaluator.validate_protocol(plan, r1)
    required = plan["required_file_sha256"]
    require(all(required.get(k) == v for k, v in r1["required_file_sha256"].items()), "M2 changed the frozen R1 source plan.")
    files = {}
    def record(path):
        require(path.suffix not in (".pt", ".npz", ".npy", ".nii", ".gz"), "Metadata summary cannot read weights or array files.")
        files[str(path.relative_to(project))] = sha(path)
    for path in (protocol_path, evaluation_path, r1_path, val_path): record(path)
    for book in (required, validation["input_file_sha256"]):
        for relative, digest in book.items():
            path = safe(project, relative); record(path)
            require(sha(path) == digest, "Protected source/evidence SHA differs: " + relative)
    manifest_path = safe(project, r1["manifest"])
    require(sha(manifest_path) == plan["manifest_sha256"], "Manifest SHA differs.")
    manifest = read(manifest_path); record(manifest_path)
    identity_path = safe(project, r1["cache_identity"])
    require(sha(identity_path) == plan["cache_identity_sha256"], "Cache identity SHA differs.")
    record(identity_path)
    recommendation = evaluator.validate_val_summary(validation, plan, r1, manifest)
    equal(r1["known_patient_retention_check"], {"mean_prostate_Dice_min": .6818369277754766,
          "minimum_visit_prostate_Dice_min": .6})
    common = {"schema_version": 1, "experiment_id": EXPERIMENT, "split": "test_M015_only", "condition": "clean",
              "protocol_sha256": sha(protocol_path), "manifest_sha256": plan["manifest_sha256"], "device": "cpu", "amp": False,
              "saved_arrays": False, "exploratory_after_test": True, "replaces_v1_test": False}
    equal(index, {**common, "status": "completed", "protected_files_unchanged": True, "training_performed": False,
          "n_cases": 8, "n_patients": 1, "case_keys": list(KEYS), "source_file_sha256": required,
          "recovery_protocol_sha256": plan["recovery_protocol_sha256"], "val_summary_sha256": plan["val_summary_sha256"],
          "cache_identity_sha256": plan["cache_identity_sha256"], "frozen_val_recommendation": recommendation})
    equal(index["runtime"], {"device": "cpu", "gpu_name": None, "amp": False, "torch_num_threads": 4, "omp_num_threads": 4})
    require(all(isinstance(index["runtime"][k], str) and re.fullmatch(r"[0-9][A-Za-z0-9.+_-]*", index["runtime"][k]) for k in ("python", "torch"))
            and (index["runtime"]["cuda"] is None or isinstance(index["runtime"]["cuda"], str)
                 and re.fullmatch(r"[0-9][A-Za-z0-9.+_-]*", index["runtime"]["cuda"]))
            and type(index["runtime"]["mkldnn_enabled"]) is bool, "Invalid public runtime version fields.")
    require(set(index["models"]) == set(index["results"]) == set(MODELS), "M2 must retain exactly B/C/D.")
    target = {f"{r['patient']}/{r['visit']}": r for r in manifest["records"] if r["patient"] == "M015"}
    require(set(target) == set(KEYS) and all(r["split"] == "test" for r in target.values()), "Manifest M015 scope differs.")
    receipts = index["cache_verification"]
    require(len(receipts) == 8 and {r["case_key"] for r in receipts} == set(KEYS)
            and all(r["decoded_identity_verified"] is True and r["cache_path"] == target[r["case_key"]]["cache_path"]
                    and re.fullmatch(r"[0-9a-f]{64}", r["cache_file_sha256"]) for r in receipts), "M2 decoded-cache receipt scope differs.")
    results = {}
    for name in MODELS:
        receipt, spec, entry = index["models"][name], plan["models"][name], index["results"][name]
        equal(receipt, {"checkpoint_sha256": spec["checkpoint_sha256"], "checkpoint_epoch": spec["selected_epoch"], "state_unchanged": True})
        require(re.fullmatch(r"[0-9a-f]{64}", receipt["state_before_sha256"])
                and receipt["state_before_sha256"] == receipt["state_after_sha256"], "M2 model state changed.")
        checkpoint_root = Path(spec["checkpoint"]).parent
        for filename, field in (("run.json", "run_sha256"), ("history.json", "history_sha256"), ("provenance.json", "provenance_sha256")):
            path = safe(project, str(checkpoint_root / filename)); record(path)
            require(sha(path) == receipt[field], "M2 run/history/provenance SHA differs.")
        provenance = read(safe(project, str(checkpoint_root / "provenance.json")))
        for filename, field in (("run.json", "run_sha256"), ("history.json", "history_sha256")):
            require(provenance[field] == receipt[field], "Provenance file binding differs.")
        require(provenance["checkpoint_sha256"] == spec["checkpoint_sha256"], "Provenance checkpoint identity differs.")
        verification_path = safe(project, provenance["cache_verification_path"]); record(verification_path)
        require(sha(verification_path) == receipt["cache_verification_sha256"] == provenance["cache_verification_sha256"], "Cache-verification SHA differs.")
        equal(entry["evaluation_file"], f"{name}/evaluation.json")
        path = safe(evaluation_path.parent, entry["evaluation_file"]); record(path)
        require(sha(path) == entry["evaluation_sha256"], "M2 evaluation SHA differs.")
        raw = read(path)
        equal(raw, {**common, "model_id": name, "checkpoint_sha256": spec["checkpoint_sha256"]})
        cases, aggregate, totals = counts.recompute(raw["cases"])
        equal(raw["aggregate"], aggregate); equal(entry["aggregate"], aggregate)
        require(raw["peak_allocated_cuda_bytes"] is None and all(c["peak_allocated_cuda_bytes"] is None
                and type(c["inference_ms"]) in (int, float) and math.isfinite(c["inference_ms"]) and c["inference_ms"] >= 0
                for c in raw["cases"]), "Invalid CPU inference resource fields.")
        equal(raw["mean_inference_ms"], counts.mean(c["inference_ms"] for c in raw["cases"]))
        for case in cases:
            require([r["target_voxels"] for r in case["per_class"]] == target[f"M015/{case['visit']}"]["class_voxel_counts"],
                    "GT counts differ from frozen manifest.")
        values = [c["per_class"][5]["dice"] for c in cases]
        require(all(type(v) is float for v in values), "Every M015 prostate Dice must be defined, including zeros.")
        results[name] = {"checkpoint_sha256": spec["checkpoint_sha256"], "checkpoint_epoch": spec["selected_epoch"],
            "cases": cases, "aggregate": aggregate, "pooled_counts": totals, "mean_prostate_dice": aggregate["per_class"][5]["dice"],
            "minimum_visit_prostate_dice": min(values), "mean_inference_ms": raw["mean_inference_ms"]}
    m1_path = safe(project, plan["m1_reference"]["evaluation"]); record(m1_path)
    require(sha(m1_path) == plan["m1_reference"]["evaluation_sha256"] == evaluator.M1_SHA, "Original M1 B reference SHA differs.")
    old = read(m1_path)
    equal(old, {"model_id": "gamma_polarity", "checkpoint_sha256": plan["models"]["gamma_polarity"]["checkpoint_sha256"],
               "manifest_sha256": plan["manifest_sha256"], "device": "cpu", "amp": False})
    old_cases, old_aggregate, _ = counts.recompute(old["cases"])
    equal(old["aggregate"], old_aggregate)
    current = results["gamma_polarity"]
    comparison = {"M1_prostate_dice": old_aggregate["per_class"][5]["dice"], "M2_prostate_dice": current["mean_prostate_dice"],
                  "M2_minus_M1_prostate_dice": current["mean_prostate_dice"] - old_aggregate["per_class"][5]["dice"], "cases": []}
    for before, after in zip(old_cases, current["cases"]):
        per_class = []
        for a, b in zip(before["per_class"], after["per_class"]):
            require(a["target_voxels"] == b["target_voxels"], "M1/M2 GT marginal counts differ.")
            per_class.append({"class_index": a["class_index"], "M1_dice": a["dice"], "M2_dice": b["dice"],
                "dice_delta": b["dice"]-a["dice"] if a["dice"] is not None and b["dice"] is not None else None,
                **{k+"_delta": b[k]-a[k] for k in ("tp", "fp", "fn")}})
        comparison["cases"].append({"patient": "M015", "visit": after["visit"], "per_class": per_class})
    equal(index["M1_B_reference_comparison"], comparison)
    mean_min = r1["known_patient_retention_check"]["mean_prostate_Dice_min"]
    visit_min = r1["known_patient_retention_check"]["minimum_visit_prostate_Dice_min"]
    retention = {name: {"mean_prostate_dice": result["mean_prostate_dice"], "minimum_visit_prostate_dice": result["minimum_visit_prostate_dice"],
        "mean_check_passes": result["mean_prostate_dice"] >= mean_min, "minimum_visit_check_passes": result["minimum_visit_prostate_dice"] >= visit_min,
        "both_checks_pass": result["mean_prostate_dice"] >= mean_min and result["minimum_visit_prostate_dice"] >= visit_min} for name, result in results.items()}
    comparisons = {name: [{"visit": a["visit"], "prostate_dice_delta": a["per_class"][5]["dice"] - b["per_class"][5]["dice"],
                          "tp_delta": a["per_class"][5]["tp"] - b["per_class"][5]["tp"],
                          "fp_delta": a["per_class"][5]["fp"] - b["per_class"][5]["fp"],
                          "fn_delta": a["per_class"][5]["fn"] - b["per_class"][5]["fn"]}
                         for a, b in zip(results[name]["cases"], results["gamma_polarity"]["cases"])] for name in MODELS[1:]}
    require(all(sha(safe(project, p)) == digest for p, digest in files.items()), "A metadata input changed during the audit.")
    return {"schema_version": 1, "status": "completed", "experiment_id": EXPERIMENT, "split": "test_M015_only",
        "protocol_sha256": sha(protocol_path), "recovery_protocol_sha256": plan["recovery_protocol_sha256"], "val_summary_sha256": plan["val_summary_sha256"],
        "evaluation_index_sha256": sha(evaluation_path), "manifest_sha256": plan["manifest_sha256"], "cache_identity_sha256": plan["cache_identity_sha256"],
        "input_file_sha256": files, "source_file_sha256": required, "summarizer_sha256": sha(__file__),
        "count_helper_sha256": COUNT_HELPER_SHA, "evaluator_helper_sha256": EVALUATOR_SHA,
        "scope": {"models": 3, "cases_per_model": 8, "case_rows": 24, "independent_patients": 1, "device": "cpu", "amp": False,
                  "arrays_loaded": False, "checkpoint_files_read": False, "model_or_epoch_selection_performed": False},
        "runtime": {k: index["runtime"][k] for k in ("device", "gpu_name", "python", "torch", "cuda", "amp", "torch_num_threads", "omp_num_threads", "mkldnn_enabled")},
        "frozen_val_recommendation": {k: recommendation[k] for k in ("selected_model_id", "selection_basis", "eligible_model_ids", "same_platform_verified", "tie_break_order")}, "results": results,
        "new_minus_same_cpu_B": comparisons, "M1_B_reference_comparison": comparison,
        "prespecified_retention_check": {"mean_prostate_dice_min": mean_min, "minimum_visit_prostate_dice_min": visit_min,
            "selection_criterion": False, "models": retention},
        "limits": ["Post-test-informed recheck of one known patient/eight repeated visits, not independent TEST or a new benchmark.",
                   "VAL-only recommendation stays frozen regardless of retention outcomes; no new arm/epoch/threshold selection occurs.",
                   "All B/C/D visits and six classes, including zero/undefined metrics and negative results, are retained.",
                   "Counts independently reconstruct Dice/IoU and FP/FN; GT marginal counts match the manifest. Pixel overlap and annotations are not remeasured here.",
                   "Checkpoint identity is bound to guarded evaluator/protocol/VAL receipts; no weight bytes, MRI, mask or prediction arrays are read by this builder.",
                   "Pooled counts are descriptive, not the visit-mean Dice estimator. CPU-host differences are quantified, not assumed bit-identical.",
                   "Original v1/A1/M1 results, failed gates and releases remain unchanged; this cannot establish a unique root cause or clinical reliability."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "evaluation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require(not os.path.lexists(args.output), "Output already exists; refusing overwrite.")
    report = summarize(args.protocol, args.evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            import json
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno()); os.link(temporary, args.output)
        finally: temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
