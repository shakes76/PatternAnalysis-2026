"""Five fixed 3D checkpoints x five conditions x all 38 frozen VAL volumes.

The two eight-epoch recovery arms start from the same frozen B checkpoint.
No TEST pixels, arrays, training, postprocessing or stress-based epoch selection.
Conditions reuse the same six patients; this is post-test-informed development.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from experiments import evaluate_appearance as original

EXPERIMENT = "appearance_recovery_v1"
ARMS = ("clean_adapt", "polarity_rehearsal")
REFERENCES = {
    "seed42_reference": (29, "9ea46e13a95617ec4ad80c21ba459087fb62941fb04406c7ffeade22ad014b62"),
    "gamma_only": (30, "293138ddcd31fa4c7c81167edaf6be04481415fdaf5d4d9eb307b09b8939fce6"),
    "gamma_polarity": (29, "b13ba58fe8837fb9522b6108412281c51210a3be486011d59f902347ade2a82b"),
}
MODEL_IDS = (*REFERENCES, *ARMS)
CONDITIONS = original.CONDITIONS
require, sha, read, child = original.require, original.sha256, original.read_json, original.child


def validate_protocol(plan, appearance, manifest_sha):
    require(plan["experiment_id"] == EXPERIMENT, "Wrong recovery experiment identity.")
    require(plan["allowed_splits"] == ["train", "validation"], "Recovery permits TRAIN/VAL only, never TEST.")
    require(plan["manifest_sha256"] == manifest_sha == original.MANIFEST_SHA
            and plan["cache_identity_sha256"] == original.IDENTITY_SHA
            and plan["training_sources_sha256"] == original.SOURCES_SHA, "Frozen data/source identity differs.")
    require(plan["training_seed"] == 42 and plan["training_epochs_per_model"] == 8, "Recovery budget must be seed42/eight epochs.")
    require(tuple(plan["conditions"]) == CONDITIONS and set(plan["models"]) == set(ARMS)
            and set(plan["references"]) == set(REFERENCES), "All five models and five fixed conditions are required.")
    require(plan["inference"] == {"split": "validation", "device": "cuda", "amp": True, "saved_arrays": False}
            and type(plan["inference"]["amp"]) is bool and type(plan["inference"]["saved_arrays"]) is bool,
            "Formal recovery evaluation must be VAL CUDA AMP with no arrays.")
    expected = dict(appearance["training_conditions"], lr=1e-4)
    require(plan["training_conditions"] == expected, "Recovery config may differ from A1 only by lr=1e-4.")
    for name, (epoch, digest) in REFERENCES.items():
        spec = plan["references"][name]
        require(spec["checkpoint"] == appearance["models"][name]["checkpoint"]
                and spec["checkpoint_sha256"] == digest and spec["selected_epoch"] == epoch,
                "An existing reference checkpoint identity changed.")
    require(plan["warm_start"] == {"checkpoint": appearance["models"]["gamma_polarity"]["checkpoint"],
                                   "checkpoint_sha256": REFERENCES["gamma_polarity"][1]}, "Warm start must be frozen B best.pt.")
    for name, probability in zip(ARMS, (0.0, 0.1)):
        spec = plan["models"][name]
        require(spec["kind"] == "recovery_arm" and spec["polarity_probability"] == probability,
                "Recovery arm/polarity policy differs.")


def validate_payload(checkpoint, name, plan, protocol_sha):
    expected = {"experiment_id": EXPERIMENT, "arm_tag": name, "protocol_sha256": protocol_sha}
    require(checkpoint["experiment"] == expected and checkpoint["config"] == plan["training_conditions"],
            "Recovery checkpoint experiment/config differs.")
    require(checkpoint["parent_checkpoint_sha256"] == plan["warm_start"]["checkpoint_sha256"], "Wrong parent checkpoint.")
    require(type(checkpoint["epoch"]) is int and 1 <= checkpoint["epoch"] <= 8
            and checkpoint["epoch"] == checkpoint["best_epoch"], "Recovery checkpoint must be a selected epoch1..8.")


def validate_arm_files(checkpoint, name, path, manifest, plan, protocol_sha):
    """Check complete budget, earliest clean-FG selector and source/data receipts."""
    validate_payload(checkpoint, name, plan, protocol_sha)
    run_path, history_path, provenance_path = (path.parent / f for f in ("run.json", "history.json", "provenance.json"))
    run, history, provenance = (read(p) for p in (run_path, history_path, provenance_path))
    require(run["experiment"] == provenance["experiment"] == checkpoint["experiment"]
            and run["config"] == checkpoint["config"], "Run/provenance experiment or config differs.")
    parent = plan["warm_start"]["checkpoint_sha256"]
    require(run["parent_checkpoint_sha256"] == provenance["parent_checkpoint_sha256"] == parent,
            "Run/provenance parent identity differs.")
    require(all(data["parent_epoch"] == 29 and all(data[k] is True for k in
                ("warm_start_weights_only", "optimizer_reset", "scaler_reset")) for data in (run, provenance)),
            "Recovery must load only B29 weights and reset optimizer/scaler.")
    require(run["status"] == "completed" and run["completed_epochs"] == 8 and len(history) == 8
            and [r["epoch"] for r in history] == list(range(1, 9)), "Expected eight complete recovery epochs.")
    require(all(not r.get("development_only") and r["training_examples"] == r["training_batches"] == 143 for r in history),
            "Recovery must cover all143 TRAIN volumes in every epoch.")
    require(all(1 <= r["optimizer_steps"] <= 143 and r["optimizer_steps"] + r["amp_overflow_skipped_optimizer_steps"] == 143
                and r["amp_overflow_skipped_optimizer_steps"] >= 0 for r in history), "Invalid successful/skipped update accounting.")
    require(all(set(r["policy_counts"]) == {"noninverted", "inverted", "gamma_applied"}
                and all(type(v) is int and 0 <= v <= 143 for v in r["policy_counts"].values())
                and r["policy_counts"]["noninverted"] + r["policy_counts"]["inverted"] == 143 for r in history),
            "Invalid TRAIN appearance exposure counts.")
    totals = {k: sum(r["policy_counts"][k] for r in history) for k in ("noninverted", "inverted", "gamma_applied")}
    require(run["training_policy_counts"] == totals and (name != "clean_adapt" or totals["inverted"] == 0),
            "Recovery total exposure counts/clean policy differ.")
    best = max(history, key=lambda r: r["validation_foreground_dice"])
    require(best["epoch"] == run["best_epoch"] == checkpoint["epoch"]
            and best["validation_foreground_dice"] == checkpoint["best_score"], "Earliest clean foreground selector/history differs.")
    protected = {str(p.relative_to(PROJECT)): sha(p) for p in (run_path, history_path, provenance_path)}
    require(provenance["checkpoint_sha256"] == sha(path) and provenance["run_sha256"] == sha(run_path)
            and provenance["history_sha256"] == sha(history_path) and provenance["protocol_sha256"] == protocol_sha
            and provenance["manifest_sha256"] == original.MANIFEST_SHA, "Recovery receipt file identities differ.")
    require(run["source_file_sha256"] == provenance["source_file_sha256"] == plan["required_file_sha256"],
            "Recovery source plan differs.")
    verification_path = child(provenance["cache_verification_path"])
    require(sha(verification_path) == run["cache_verification_sha256"] == provenance["cache_verification_sha256"],
            "Recovery cache-verification receipt differs.")
    verification = read(verification_path)
    rows = verification["records"]
    expected_keys = {(r["patient"], r["visit"], r["split"]) for r in manifest["records"] if r["split"] in ("train", "validation")}
    require(verification["status"] == "verified" and verification["manifest_sha256"] == original.MANIFEST_SHA
            and verification["counts"] == {"train": 143, "validation": 38} and verification["test_arrays_opened"] == 0
            and len(rows) == 181 and {(r["patient"], r["visit"], r["split"]) for r in rows} == expected_keys
            and all(r["decoded_identity_verified"] is True for r in rows), "Expected exact181 TRAIN/VAL-only cache verification.")
    protected[str(verification_path.relative_to(PROJECT))] = sha(verification_path)
    return {"run_sha256": sha(run_path), "history_sha256": sha(history_path), "provenance_sha256": sha(provenance_path),
            "cache_verification_sha256": sha(verification_path), "initial_parameter_state_sha256": run["initial_parameter_state_sha256"],
            "training_runtime": run["runtime"], "protected_file_sha256": protected}


def validate_pair(receipts, parent_state, runtime):
    require(all(receipts[n]["initial_parameter_state_sha256"] == parent_state for n in ARMS),
            "Both recovery arms must start at the exact frozen B parameter state.")
    require(receipts[ARMS[0]]["training_runtime"] == receipts[ARMS[1]]["training_runtime"], "Recovery arms used different training runtimes.")
    require(all(all(receipts[n]["training_runtime"][k] == v for k, v in runtime.items()) for n in ARMS),
            "Evaluation runtime must match both recovery training runtimes.")


def validate_selected_clean(result, history, epoch):
    """Ignore latency; require the selected checkpoint's exact clean measurements."""
    def signature(data):
        rows = {(r["patient"], r["visit"]): {k: r[k] for k in
                ("per_class", "foreground_mean_dice", "foreground_mean_iou")} for r in data["cases"]}
        require(len(rows) == len(data["cases"]), "Duplicate selected-validation case keys.")
        return rows, data["aggregate"]
    require(history[epoch - 1]["epoch"] == epoch and signature(result) == signature(history[epoch - 1]["validation"]),
            "Standalone clean validation differs from the selected history.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "manifest", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().absolute()
    require(not os.path.lexists(output), "Output must be new; no previous evidence may be overwritten.")
    protocol_path, manifest_path = args.protocol.expanduser().resolve(), args.manifest.expanduser().resolve()
    plan, protocol_sha = read(protocol_path), sha(protocol_path)
    appearance_path = child(plan["appearance_protocol"])
    appearance, appearance_sha = read(appearance_path), sha(appearance_path)
    require(appearance_sha == plan["appearance_protocol_sha256"], "Frozen A1 protocol differs.")
    original.validate_protocol(appearance, sha(manifest_path))
    validate_protocol(plan, appearance, sha(manifest_path))
    required = plan["required_file_sha256"]
    require({"experiments/evaluate_recovery.py", "experiments/evaluate_appearance.py", "experiments/recovery.py",
             "modules.py", "dataset.py", "utils.py", "tools/diagnose_model_mechanism.py"} <= set(required), "Recovery source guard omits dependencies.")
    def verify_sources():
        require(sha(protocol_path) == protocol_sha and sha(appearance_path) == appearance_sha, "A protocol changed during evaluation.")
        for book in (appearance["required_file_sha256"], required):
            require(all(sha(child(p)) == digest for p, digest in book.items()), "A protected source/input changed.")
    verify_sources()
    import torch
    from dataset import load_manifest
    from modules import ImprovedUNet3D
    from utils import evaluate_volumes, write_json
    from experiments.appearance import STRESS_CONDITIONS, StressValidationDataset
    from tools.cache_identity import load_verified_book
    from tools.diagnose_model_mechanism import state_sha256
    require(torch.cuda.is_available(), "CUDA is required for formal recovery evaluation.")
    require(tuple(STRESS_CONDITIONS) == CONDITIONS, "Fixed appearance conditions differ.")
    manifest = load_manifest(manifest_path)
    require(not manifest["partial"] and not manifest["development_only"], "Full frozen manifest required.")
    identity_path = manifest_path.parent / "cache_identity.json"
    require(sha(identity_path) == original.IDENTITY_SHA, "Frozen cache identity differs.")
    identity = load_verified_book(identity_path, manifest_path, manifest)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    torch.use_deterministic_algorithms(True, warn_only=True)
    runtime = {"gpu_name": torch.cuda.get_device_name(device), "python": platform.python_version(),
               "torch": str(torch.__version__), "cuda": torch.version.cuda}
    models, evidence, paths, receipts = {}, {}, {}, {}
    for name in MODEL_IDS:
        spec = plan["references"][name] if name in REFERENCES else plan["models"][name]
        path = paths[name] = child(spec["checkpoint"])
        require(not output.resolve().is_relative_to(path.parent) and not output.resolve().is_relative_to(manifest_path.parent),
                "Output enters a protected checkpoint/cache directory.")
        digest = sha(path)
        if name in REFERENCES:
            require(digest == REFERENCES[name][1], "Frozen reference checkpoint SHA differs.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if name in REFERENCES:
            config = original.validate_payload(checkpoint, appearance["models"][name], name, manifest, appearance_sha)
            require(config == appearance["training_conditions"] and checkpoint["epoch"] == REFERENCES[name][0], "Reference configuration/epoch differs.")
            receipt = original.validate_arm_files(checkpoint, config, name, path, manifest, appearance_sha,
                                                  appearance["required_file_sha256"]) if name != "seed42_reference" else {}
            if name != "seed42_reference":
                require(read(path.parent / "provenance.json")["history_sha256"] == receipt["history_sha256"], "A1 history receipt differs.")
        else:
            receipt = validate_arm_files(checkpoint, name, path, manifest, plan, protocol_sha)
            receipts[name] = receipt
        model = ImprovedUNet3D(6, 16).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.register_forward_hook(original.finite_main_logits)
        models[name] = model
        evidence[name] = {"checkpoint_sha256": digest, "checkpoint_epoch": checkpoint["epoch"],
                          "state_before_sha256": original.state_sha(model), "parameters": sum(p.numel() for p in model.parameters()), **receipt}
        del checkpoint
    validate_pair(receipts, state_sha256(models["gamma_polarity"]), runtime)
    dataset = original.VerifiedValidation(manifest_path, manifest, identity)
    expected_keys = {(r["patient"], r["visit"]) for r in dataset.records}
    output.mkdir(parents=True, exist_ok=False)
    index = {"schema_version": 1, "experiment_id": EXPERIMENT, "status": "running", "split": "validation",
             "protocol_sha256": protocol_sha, "appearance_protocol_sha256": appearance_sha,
             "manifest_sha256": original.MANIFEST_SHA, "cache_identity_sha256": original.IDENTITY_SHA,
             "source_file_sha256": required, "models": evidence, "conditions": list(CONDITIONS),
             "n_scans_per_condition": 38, "n_unique_patients": 6, "saved_arrays": False, "test_arrays_opened": 0,
             "runtime": {"device": "cuda", "amp": True, **runtime}, "results": {},
             "limits": ["Post-test-informed TRAIN/VAL recovery experiment; no new TEST or independent confirmation.",
                        "All five models and five fixed conditions reported; no best-stress selection or inference postprocessing.",
                        "Conditions reuse the same six validation patients, not thirty independent patients; VAL also selects checkpoints.",
                        "Paired initialization/runtime do not remove warn-only CUDA nondeterminism. Synthetic stresses are cache-space probes.",
                        "M015 is not opened here and cannot select an epoch or candidate; original benchmarks and failed A1 guard remain unchanged."]}
    for name, model in models.items():
        model.to(device)  # Other four models remain on CPU.
        index["results"][name] = {}
        for condition in CONDITIONS:
            result = evaluate_volumes(model, StressValidationDataset(dataset, condition), "improved3d", 6, device, amp=True)
            require(len(result["cases"]) == 38 and {(c["patient"], c["visit"]) for c in result["cases"]} == expected_keys
                    and result["aggregate"]["num_patients"] == 6, "Incomplete/out-of-scope validation results.")
            if name in ARMS and condition == "clean":
                validate_selected_clean(result, read(paths[name].parent / "history.json"), evidence[name]["checkpoint_epoch"])
            result.update(schema_version=1, experiment_id=EXPERIMENT, model_id=name, condition=condition, split="validation",
                          saved_arrays=False, amp=True, device="cuda", protocol_sha256=protocol_sha,
                          manifest_sha256=original.MANIFEST_SHA, checkpoint_sha256=evidence[name]["checkpoint_sha256"])
            relative = f"{name}/{condition}/evaluation.json"
            write_json(output / relative, result)
            index["results"][name][condition] = {"evaluation_file": relative, "evaluation_sha256": sha(output / relative),
                "aggregate": result["aggregate"], "mean_inference_ms": result["mean_inference_ms"],
                "peak_allocated_cuda_bytes": result["peak_allocated_cuda_bytes"]}
        require(original.state_sha(model) == evidence[name]["state_before_sha256"], "Inference changed model state.")
        evidence[name]["state_unchanged"] = True
        model.to("cpu")
    verify_sources()
    require(all(sha(paths[n]) == evidence[n]["checkpoint_sha256"] for n in MODEL_IDS), "A checkpoint changed.")
    require(all(sha(child(p)) == digest for receipt in receipts.values() for p, digest in receipt["protected_file_sha256"].items()),
            "A recovery receipt/history changed during evaluation.")
    index["status"] = "completed"
    write_json(output / "recovery_evaluation.json", index)
    print("Completed five models x five conditions x38 VAL volumes; no TEST access or saved arrays.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
