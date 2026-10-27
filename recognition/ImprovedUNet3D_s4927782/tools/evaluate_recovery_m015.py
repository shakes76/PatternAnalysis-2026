"""One post-selection CPU FP32 recheck of B/C/D on all eight clean M015 visits.

The complete VAL summary and its VAL-only recommendation must already be frozen
by this separate protocol. No training, candidate/epoch selection, other TEST
patient, augmentation, postprocessing, logits or prediction arrays are provided.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import platform
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from experiments import evaluate_recovery as recovery
from tools import evaluate_m015_followup as m1

EXPERIMENT = "recovery_m015_followup_v1"
MODEL_IDS = ("gamma_polarity", *recovery.ARMS)
CASE_KEYS = m1.CASE_KEYS
M1_SHA = "3d38d202b8e04bd9f09d363fceaf21be20d612a0e500aa7f7036757be75e8f30"
require, sha, read, child = recovery.require, recovery.sha, recovery.read, recovery.child


def validate_protocol(plan, r1):
    require(plan["experiment_id"] == EXPERIMENT and plan["test_informed_exploratory_acknowledged"] is True,
            "A separate acknowledged post-test M2 protocol is required.")
    require(tuple(plan["case_keys"]) == CASE_KEYS and set(plan["models"]) == set(MODEL_IDS), "Only three checkpoints and eight M015 visits are allowed.")
    require(plan["manifest_sha256"] == recovery.original.MANIFEST_SHA
            and plan["cache_identity_sha256"] == recovery.original.IDENTITY_SHA
            and plan["training_sources_sha256"] == recovery.original.SOURCES_SHA, "Frozen data/source identity differs.")
    require(plan["inference"] == {"split": "test", "device": "cpu", "amp": False, "saved_arrays": False}
            and type(plan["inference"]["amp"]) is bool and type(plan["inference"]["saved_arrays"]) is bool,
            "M2 requires clean CPU FP32 with no saved arrays.")
    for name in MODEL_IDS:
        spec = plan["models"][name]
        expected = r1["warm_start"] if name == "gamma_polarity" else r1["models"][name]
        require(spec["checkpoint"] == expected["checkpoint"], "M2 checkpoint path differs from frozen recovery.")
        digest = spec["checkpoint_sha256"]
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Invalid checkpoint SHA.")
        require(type(spec["selected_epoch"]) is int and 1 <= spec["selected_epoch"] <= (29 if name == "gamma_polarity" else 8), "Invalid selected epoch.")
    require(plan["models"]["gamma_polarity"]["checkpoint_sha256"] == r1["warm_start"]["checkpoint_sha256"]
            and plan["models"]["gamma_polarity"]["selected_epoch"] == 29, "Frozen B reference identity differs.")
    require(plan["m1_reference"]["evaluation_sha256"] == M1_SHA, "The original M1 B reference must remain frozen.")


def validate_val_summary(summary, plan, r1, manifest):
    """Bind all25 VAL groups and independently check the frozen recommendation."""
    require(summary["experiment_id"] == recovery.EXPERIMENT and summary["split"] == "validation"
            and summary["status"] == "completed" and summary["protocol_sha256"] == plan["recovery_protocol_sha256"]
            and summary["manifest_sha256"] == plan["manifest_sha256"], "Frozen recovery VAL summary identity differs.")
    scope = {"n_models": 5, "n_conditions": 5, "n_scans": 38, "n_patients": 6, "n_case_evaluations": 950,
             "test_arrays_read_by_summarizer": 0, "arrays_loaded": False}
    require(all(type(summary["scope"][k]) is type(v) and summary["scope"][k] == v for k, v in scope.items()), "Incomplete/non-VAL summary scope.")
    require(set(summary["models"]) == set(summary["results"]) == set(recovery.MODEL_IDS), "Summary must contain all five models.")
    evidence_paths = summary["input_file_sha256"]
    require(all(sum(p.endswith(f"/{n}/{c}/evaluation.json") for p in evidence_paths) == 1
                for n in recovery.MODEL_IDS for c in recovery.CONDITIONS), "Summary must identify all25 unique input evaluation files.")
    keys = {(r["patient"], r["visit"]) for r in manifest["records"] if r["split"] == "validation"}
    patients = {p for p, _ in keys}
    require(len(keys) == 38 and len(patients) == 6, "Expected complete frozen validation scope.")
    for name in recovery.MODEL_IDS:
        meta = summary["models"][name]
        expected = plan["models"][name] if name in MODEL_IDS else r1["references"][name]
        require(meta["checkpoint_sha256"] == expected["checkpoint_sha256"] and meta["checkpoint_epoch"] == expected["selected_epoch"]
                and meta["state_unchanged"] is True, "Summary/checkpoint identity differs.")
        require(set(summary["results"][name]) == set(recovery.CONDITIONS), "Summary lacks a fixed condition.")
        for condition in recovery.CONDITIONS:
            result = summary["results"][name][condition]
            cases, visits = result["cases_prostate"], result["patients"]
            require(len(cases) == 38 and {(c["patient"], c["visit"]) for c in cases} == keys
                    and len(visits) == 6 and {p["patient"] for p in visits} == patients, "Summary group lacks complete VAL cases/patients.")
    results = summary["results"]
    baseline = results["seed42_reference"]["clean"]["aggregate"]
    polarity = results["gamma_polarity"]["polarity"]["aggregate"]["prostate_dice"]
    eligible = []
    for name in recovery.ARMS:
        clean = results[name]["clean"]["aggregate"]
        values = (clean["prostate_dice"], clean["foreground_mean_dice"], baseline["prostate_dice"],
                  baseline["foreground_mean_dice"], results[name]["polarity"]["aggregate"]["prostate_dice"], polarity)
        require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values), "Undefined/nonfinite VAL gate metric.")
        if values[0] - values[2] >= 0 and values[1] - values[3] >= -.005 and values[4] - values[5] >= -.02:
            eligible.append(name)
    ranked = sorted(eligible, key=lambda n: (-results[n]["clean"]["aggregate"]["prostate_dice"],
                    -results[n]["clean"]["aggregate"]["foreground_mean_dice"], n))
    recommendation = summary["recommendation"]
    require(recommendation["selection_basis"] == "validation_only" and recommendation["same_platform_verified"] is True
            and recommendation["tie_break_order"] == ["clean_prostate_dice_desc", "clean_foreground_dice_desc", "model_id_asc"]
            and len(recommendation["eligible_model_ids"]) == len(eligible) and set(recommendation["eligible_model_ids"]) == set(eligible)
            and recommendation["selected_model_id"] == (ranked[0] if ranked else None), "Frozen VAL-only recommendation/gates differ.")
    return recommendation


def compare_m1(new, old):
    """Compare reported counts, not voxel-array equality across CPU hosts."""
    require(old["model_id"] == "gamma_polarity" and old["checkpoint_sha256"] == recovery.REFERENCES["gamma_polarity"][1]
            and old["manifest_sha256"] == recovery.original.MANIFEST_SHA and old["device"] == "cpu" and old["amp"] is False,
            "Historical M1 CPU reference differs.")
    old_rows = {(c["patient"], c["visit"]): c for c in old["cases"]}
    require(len(old_rows) == len(old["cases"]) == 8 and {f"{p}/{v}" for p, v in old_rows} == set(CASE_KEYS), "Incomplete historical M1 scope.")
    rows = []
    for case in new["cases"]:
        before = old_rows[(case["patient"], case["visit"])]
        require([r["class_index"] for r in before["per_class"]] == list(range(6)), "M1 class scope differs.")
        per_class = []
        for a, b in zip(before["per_class"], case["per_class"]):
            require(a["target_voxels"] == b["target_voxels"], "M1/M2 GT counts differ.")
            per_class.append({"class_index": b["class_index"], "M1_dice": a["dice"], "M2_dice": b["dice"],
                "dice_delta": b["dice"] - a["dice"] if a["dice"] is not None and b["dice"] is not None else None,
                "tp_delta": b["intersection"] - a["intersection"],
                "fp_delta": (b["predicted_voxels"] - b["intersection"]) - (a["predicted_voxels"] - a["intersection"]),
                "fn_delta": (b["target_voxels"] - b["intersection"]) - (a["target_voxels"] - a["intersection"])})
        rows.append({"patient": case["patient"], "visit": case["visit"], "per_class": per_class})
    a, b = old["aggregate"]["per_class"][5]["dice"], new["aggregate"]["per_class"][5]["dice"]
    return {"M1_prostate_dice": a, "M2_prostate_dice": b, "M2_minus_M1_prostate_dice": b - a,
            "cases": rows, "interpretation": "Count/metric differences across CPU executions; prediction-array equality is not tested or required."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "manifest", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().absolute()
    require(not os.path.lexists(output), "Output must be new; old evidence cannot be overwritten.")
    protocol_path, manifest_path = args.protocol.expanduser().resolve(), args.manifest.expanduser().resolve()
    plan, protocol_sha = read(protocol_path), sha(protocol_path)
    r1_path, summary_path = child(plan["recovery_protocol_path"]), child(plan["val_summary_path"])
    require(sha(r1_path) == plan["recovery_protocol_sha256"] and sha(summary_path) == plan["val_summary_sha256"], "Frozen R1 protocol/VAL summary SHA differs.")
    r1, summary = read(r1_path), read(summary_path)
    appearance_path = child(r1["appearance_protocol"])
    appearance, appearance_sha = read(appearance_path), sha(appearance_path)
    require(appearance_sha == r1["appearance_protocol_sha256"], "Frozen A1 protocol SHA differs.")
    recovery.original.validate_protocol(appearance, sha(manifest_path))
    recovery.validate_protocol(r1, appearance, sha(manifest_path))
    validate_protocol(plan, r1)
    required = plan["required_file_sha256"]
    require(all(required.get(p) == digest for p, digest in r1["required_file_sha256"].items())
            and {"tools/evaluate_recovery_m015.py", "tests/test_evaluate_recovery_m015.py", "tools/summarize_recovery.py",
                 "tools/evaluate_m015_followup.py"} <= set(required), "M2 guard must extend, never change, the frozen R1 source plan.")
    def verify_files():
        require(sha(protocol_path) == protocol_sha and sha(r1_path) == plan["recovery_protocol_sha256"]
                and sha(summary_path) == plan["val_summary_sha256"], "A frozen protocol/summary changed.")
        for book in (appearance["required_file_sha256"], required, summary["input_file_sha256"]):
            require(all(Path(p).suffix not in (".npz", ".npy", ".nii", ".gz") and sha(child(p)) == digest for p, digest in book.items()),
                    "A protected source/VAL evidence SHA differs or an array path was supplied.")
    verify_files()
    from dataset import load_manifest
    from tools.cache_identity import load_verified_book, safe_relative
    manifest = load_manifest(manifest_path)
    recommendation = validate_val_summary(summary, plan, r1, manifest)
    m1.select_records(manifest)
    identity_path = manifest_path.parent / "cache_identity.json"
    require(sha(identity_path) == plan["cache_identity_sha256"], "Frozen cache identity differs.")
    identity = load_verified_book(identity_path, manifest_path, manifest)
    m1_path = child(plan["m1_reference"]["evaluation"])
    require(sha(m1_path) == M1_SHA, "Frozen M1 B evaluation differs.")
    old = read(m1_path)
    import torch
    from modules import ImprovedUNet3D
    from utils import evaluate_volumes, write_json
    from tools.diagnose_model_mechanism import state_sha256
    threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    require(threads == 4, "M2 requires OMP_NUM_THREADS=4.")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cpu")
    models, evidence, paths, receipts = {}, {}, {}, {}
    for name in MODEL_IDS:
        spec = plan["models"][name]
        path = paths[name] = child(spec["checkpoint"])
        require(not output.resolve().is_relative_to(path.parent) and not output.resolve().is_relative_to(manifest_path.parent), "Output enters protected checkpoint/cache directory.")
        require(sha(path) == spec["checkpoint_sha256"], "Selected checkpoint SHA differs.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        require(checkpoint["epoch"] == spec["selected_epoch"], "Selected epoch differs from frozen VAL choice.")
        if name == "gamma_polarity":
            recovery.original.validate_payload(checkpoint, appearance["models"][name], name, manifest, appearance_sha)
            receipt = recovery.original.validate_arm_files(checkpoint, checkpoint["config"], name, path, manifest, appearance_sha, appearance["required_file_sha256"])
            require(read(path.parent / "provenance.json")["history_sha256"] == receipt["history_sha256"], "B history receipt differs.")
        else:
            receipt = recovery.validate_arm_files(checkpoint, name, path, manifest, r1, plan["recovery_protocol_sha256"])
            receipts[name] = receipt
        model = ImprovedUNet3D(6, 16).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.register_forward_hook(recovery.original.finite_main_logits)
        models[name] = model
        evidence[name] = {"checkpoint_sha256": spec["checkpoint_sha256"], "checkpoint_epoch": checkpoint["epoch"],
                          "state_before_sha256": recovery.original.state_sha(model), **receipt}
        del checkpoint
    recovery.validate_pair(receipts, state_sha256(models["gamma_polarity"]), receipts[recovery.ARMS[0]]["training_runtime"])
    dataset = m1.VerifiedM015(manifest_path, manifest, identity)
    output.mkdir(parents=True, exist_ok=False)
    index = {"schema_version": 1, "experiment_id": EXPERIMENT, "status": "running", "split": "test_M015_only", "condition": "clean",
             "protocol_sha256": protocol_sha, "recovery_protocol_sha256": plan["recovery_protocol_sha256"], "val_summary_sha256": plan["val_summary_sha256"],
             "frozen_val_recommendation": recommendation, "manifest_sha256": plan["manifest_sha256"], "cache_identity_sha256": plan["cache_identity_sha256"],
             "n_cases": 8, "n_patients": 1, "case_keys": list(CASE_KEYS), "device": "cpu", "amp": False,
             "saved_arrays": False, "training_performed": False, "replaces_v1_test": False, "exploratory_after_test": True,
             "source_file_sha256": required, "cache_verification": dataset.receipts, "models": evidence, "results": {},
             "runtime": {"device": "cpu", "gpu_name": None, "python": platform.python_version(), "torch": str(torch.__version__),
                         "cuda": torch.version.cuda, "amp": False, "torch_num_threads": torch.get_num_threads(), "omp_num_threads": threads,
                         "mkldnn_enabled": torch.backends.mkldnn.enabled},
             "limits": ["One previously examined patient/eight repeated visits; this is not independent TEST evidence or a new benchmark.",
                        "VAL-only choice was frozen before this recheck. M015 cannot select an arm, epoch or new training/search.",
                        "All three models use clean cache/main-head CPU FP32; no augmentation/TTA/postprocessing/arrays.",
                        "M1 comparison is descriptive across CPU executions; no exact array identity or unique causal explanation is claimed.",
                        "Original v1/A1 results, failed gates and all positive/negative outcomes remain unchanged."]}
    for name, model in models.items():
        result = evaluate_volumes(model, dataset, "improved3d", 6, device, amp=False)
        require(len(result["cases"]) == 8 and {f"{c['patient']}/{c['visit']}" for c in result["cases"]} == set(CASE_KEYS)
                and result["aggregate"]["num_patients"] == 1, "Incomplete M2 M015 result scope.")
        m1.add_error_counts(result)
        result.update(schema_version=1, experiment_id=EXPERIMENT, model_id=name, condition="clean", split="test_M015_only", device="cpu", amp=False,
                      saved_arrays=False, exploratory_after_test=True, replaces_v1_test=False, protocol_sha256=protocol_sha,
                      manifest_sha256=plan["manifest_sha256"], checkpoint_sha256=evidence[name]["checkpoint_sha256"])
        require(recovery.original.state_sha(model) == evidence[name]["state_before_sha256"], "Inference changed model state.")
        evidence[name].update(state_after_sha256=recovery.original.state_sha(model), state_unchanged=True)
        relative = f"{name}/evaluation.json"
        write_json(output / relative, result)
        index["results"][name] = {"evaluation_file": relative, "evaluation_sha256": sha(output / relative), "aggregate": result["aggregate"]}
        if name == "gamma_polarity": index["M1_B_reference_comparison"] = compare_m1(result, old)
    verify_files()
    require(sha(identity_path) == plan["cache_identity_sha256"] and sha(m1_path) == M1_SHA
            and all(sha(paths[n]) == plan["models"][n]["checkpoint_sha256"] for n in MODEL_IDS)
            and all(sha(safe_relative(manifest_path.parent, r["cache_path"])) == r["cache_file_sha256"] for r in dataset.receipts), "A protected model/data file changed.")
    index.update(status="completed", protected_files_unchanged=True)
    write_json(output / "recovery_m015.json", index)
    print("Completed one frozen B/C/D x8 clean M015 CPU FP32 recheck; no selection or arrays.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
