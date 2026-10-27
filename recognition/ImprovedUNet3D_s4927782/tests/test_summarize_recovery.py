"""Synthetic full 25-group receipts, selectors, gates and scope; no arrays."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recovery_summary", ROOT / "tools/summarize_recovery.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)
FIXTURE_SPEC = importlib.util.spec_from_file_location("appearance_fixture", ROOT / "tests/test_summarize_appearance.py")
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)
write = fixtures.write


def study(root):
    appearance_path, old_index_path = fixtures.study_fixture(root)
    appearance, old_index = summary.read(appearance_path), summary.read(old_index_path)
    manifest = summary.read(root / "manifest.json")
    references = {name: {"checkpoint": appearance["models"][name]["checkpoint"],
                         "checkpoint_sha256": old_index["models"][name]["checkpoint_sha256"],
                         "selected_epoch": old_index["models"][name]["checkpoint_epoch"]} for name in summary.REFERENCES}
    models = {n: {"kind": "recovery_arm", "checkpoint": f"recovery/{n}/best.pt", "polarity_probability": p}
              for n, p in zip(summary.ARMS, (0., .1))}
    plan = {"experiment_id": summary.EXPERIMENT, "allowed_splits": ["train", "validation"], "training_seed": 42,
            "training_epochs_per_model": 8, "conditions": list(summary.CONDITIONS), "models": models, "references": references,
            "manifest": "manifest.json", "manifest_sha256": summary.sha(root / "manifest.json"),
            "cache_identity": "identity.json", "cache_identity_sha256": summary.sha(root / "identity.json"),
            "training_sources_sha256": summary.sha(root / "training_sources.json"),
            "appearance_protocol": "protocol.json", "appearance_protocol_sha256": summary.sha(appearance_path),
            "required_file_sha256": appearance["required_file_sha256"],
            "warm_start": {"checkpoint_sha256": references["gamma_polarity"]["checkpoint_sha256"]},
            "training_conditions": dict(appearance["training_conditions"], lr=.0001),
            "inference": {"split": "validation", "device": "cuda", "amp": True, "saved_arrays": False},
            "engineering_gate": summary.GATE.copy()}
    protocol = root / "recovery_protocol.json"; write(protocol, plan)
    protocol_sha = summary.sha(protocol)
    runtime = {"gpu_name": "synthetic GPU", "python": "3.11", "torch": "2.13", "cuda": "13.0"}
    index = {"schema_version": 1, "status": "completed", "experiment_id": summary.EXPERIMENT, "split": "validation",
             "protocol_sha256": protocol_sha, "manifest_sha256": plan["manifest_sha256"], "saved_arrays": False,
             "appearance_protocol_sha256": plan["appearance_protocol_sha256"], "cache_identity_sha256": plan["cache_identity_sha256"],
             "source_file_sha256": plan["required_file_sha256"], "conditions": list(summary.CONDITIONS),
             "n_scans_per_condition": 38, "n_unique_patients": 6, "test_arrays_opened": 0,
             "runtime": {"device": "cuda", "amp": True, **runtime}, "models": {}, "results": {}}
    original = summary.read(old_index_path.parent / old_index["results"]["seed42_reference"]["clean"]["evaluation_file"])
    for name in summary.MODELS:
        if name in summary.REFERENCES:
            receipt = copy.deepcopy(old_index["models"][name])
        else:
            directory = root / "recovery" / name
            parent = {"parent_checkpoint_sha256": plan["warm_start"]["checkpoint_sha256"], "parent_epoch": 29,
                      "warm_start_weights_only": True, "optimizer_reset": True, "scaler_reset": True}
            experiment = {"experiment_id": summary.EXPERIMENT, "arm_tag": name, "protocol_sha256": protocol_sha}
            counts = {"noninverted": 143 if name == "clean_adapt" else 133, "inverted": 0 if name == "clean_adapt" else 10, "gamma_applied": 70}
            verification = {"status": "verified", "manifest_sha256": plan["manifest_sha256"], "counts": {"train": 143, "validation": 38},
                            "test_arrays_opened": 0, "records": [{**r, "decoded_identity_verified": True} for r in manifest["records"] if r["split"] != "test"]}
            verification_path = directory / "cache_verification.json"; write(verification_path, verification)
            run = {"experiment": experiment, "config": plan["training_conditions"], "status": "completed", "completed_epochs": 8,
                   "requested_total_epochs": 8, "development_only": False, "source_file_sha256": plan["required_file_sha256"],
                   "cache_verification_sha256": summary.sha(verification_path), "initial_parameter_state_sha256": "a" * 64,
                   "runtime": runtime, "best_epoch": 1, "best_validation_foreground_dice": 1.,
                   "training_policy_counts": {k: 8 * v for k, v in counts.items()}, **parent}
            history = [{"epoch": e, "train_loss": .1, "train_seconds": 1., "validation_foreground_dice": 1.,
                        "validation_foreground_iou": 1.,
                        "validation": original, "training_examples": 143, "training_batches": 143, "development_only": False,
                        "policy_counts": counts, "optimizer_steps": 143, "amp_overflow_skipped_optimizer_steps": 0,
                        "peak_allocated_cuda_bytes_training": 200} for e in range(1, 9)]
            write(directory / "run.json", run); write(directory / "history.json", history)
            receipt = {"checkpoint_sha256": "c" * 64, "checkpoint_epoch": 1, "state_before_sha256": "b" * 64,
                       "state_unchanged": True, "parameters": 100, "initial_parameter_state_sha256": "a" * 64,
                       "training_runtime": runtime, "cache_verification_sha256": summary.sha(verification_path)}
            provenance = {"experiment": experiment, "protocol_sha256": protocol_sha, "manifest_sha256": plan["manifest_sha256"],
                          "source_file_sha256": plan["required_file_sha256"], "checkpoint_sha256": receipt["checkpoint_sha256"],
                          "run_sha256": summary.sha(directory / "run.json"), "history_sha256": summary.sha(directory / "history.json"),
                          "cache_verification_sha256": summary.sha(verification_path), "cache_verification_path": str(verification_path.relative_to(root)), **parent}
            write(directory / "provenance.json", provenance)
            receipt.update({field: summary.sha(directory / filename) for field, filename in
                            (("run_sha256", "run.json"), ("history_sha256", "history.json"), ("provenance_sha256", "provenance.json"))})
            receipt["protected_file_sha256"] = {str((directory / f).relative_to(root)): summary.sha(directory / f)
                for f in ("run.json", "history.json", "provenance.json", "cache_verification.json")}
        index["models"][name] = receipt
        index["results"][name] = {}
        for condition in summary.CONDITIONS:
            raw = copy.deepcopy(original)
            raw.update(schema_version=1, experiment_id=summary.EXPERIMENT, model_id=name, condition=condition,
                       device="cuda", amp=True, protocol_sha256=protocol_sha, checkpoint_sha256=receipt["checkpoint_sha256"])
            relative = f"{name}/{condition}/evaluation.json"; destination = root / "recovery_eval" / relative
            write(destination, raw)
            index["results"][name][condition] = {"evaluation_file": relative, "evaluation_sha256": summary.sha(destination),
                "aggregate": raw["aggregate"], "mean_inference_ms": raw["mean_inference_ms"], "peak_allocated_cuda_bytes": raw["peak_allocated_cuda_bytes"]}
    path = root / "recovery_eval/recovery_evaluation.json"; write(path, index)
    return protocol, path


class RecoverySummaryTests(unittest.TestCase):
    def test_complete_950_case_receipts_and_earliest_tie_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol, index = study(root)
            report = summary.summarize(protocol, index, root)
            self.assertEqual(report["scope"]["n_case_evaluations"], 950)
            self.assertEqual(sum(len(group["cases_prostate"]) for model in report["results"].values() for group in model.values()), 950)
            self.assertEqual(report["recommendation"]["selected_model_id"], "clean_adapt")
            self.assertTrue(all(report["paired_execution_checks"].values()))
            self.assertEqual(report["training"]["polarity_rehearsal"]["policy_counts"]["inverted"], 80)
            self.assertEqual(report["training"]["clean_adapt"]["selected_epoch"], 1)

    def test_all_gate_terms_runtime_and_val_only_tie_break(self):
        values = {n: {c: {"aggregate": {"prostate_dice": .8, "foreground_mean_dice": .85}}
                      for c in summary.CONDITIONS} for n in summary.MODELS}
        gates, choice = summary.candidate_choice(values, True)
        self.assertTrue(all(g["eligible_candidate"] for g in gates.values()))
        self.assertEqual(choice["selected_model_id"], "clean_adapt")
        values["polarity_rehearsal"]["clean"]["aggregate"]["prostate_dice"] = .81
        self.assertEqual(summary.candidate_choice(values, True)[1]["selected_model_id"], "polarity_rehearsal")
        values["polarity_rehearsal"]["polarity"]["aggregate"]["prostate_dice"] = .5
        self.assertFalse(summary.candidate_choice(values, True)[0]["polarity_rehearsal"]["eligible_candidate"])
        self.assertIsNone(summary.candidate_choice(values, False)[1]["selected_model_id"])
        values["clean_adapt"]["clean"]["aggregate"]["foreground_mean_dice"] = .84
        self.assertIsNone(summary.candidate_choice(values, True)[1]["selected_model_id"])

    def test_sha_test_scope_and_standalone_clean_drift_are_rejected(self):
        for mistake in ("sha", "test", "selected"):
            with self.subTest(mistake=mistake), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); protocol, index_path = study(root)
                index = summary.read(index_path)
                source = root / "recovery_eval/clean_adapt/clean/evaluation.json"
                if mistake == "sha": source.write_text(source.read_text() + " ")
                elif mistake == "test": index["split"] = "test"
                else:
                    raw = summary.read(source); first = raw["cases"][0]
                    raw["cases"][0] = fixtures.case(first["patient"], first["visit"], True, first["shape_dhw"])
                    keys = {(c["patient"], c["visit"]) for c in raw["cases"]}
                    aggregate, _ = summary.recompute(raw["cases"], keys, first["shape_dhw"])
                    raw["aggregate"] = aggregate; write(source, raw)
                    index["results"]["clean_adapt"]["clean"].update(aggregate=aggregate, evaluation_sha256=summary.sha(source))
                write(index_path, index)
                with self.assertRaises(ValueError): summary.summarize(protocol, index_path, root)

    def test_patient_weighting_retains_zero_and_empty_class(self):
        cases = [fixtures.case("P1", "Week0", True), fixtures.case("P1", "Week1", True), fixtures.case("P2", "Week0")]
        aggregate, compact = summary.recompute(cases, {(c["patient"], c["visit"]) for c in cases}, [1, 1, 6])
        self.assertEqual(aggregate["per_class"][5]["dice"], .5)
        self.assertEqual([r["dice"] for r in compact], [0, 0, 1])
        empty = fixtures.case()
        empty["per_class"][0].update(intersection=2, predicted_voxels=2, target_voxels=2, union=2)
        empty["per_class"][5].update(intersection=0, predicted_voxels=0, target_voxels=0, union=0, dice=None, iou=None)
        self.assertIsNone(summary.recompute([empty], {("P1", "Week0")}, [1, 1, 6])[0]["per_class"][5]["dice"])

    def test_runtime_confound_keeps_all_results_but_disables_recommendation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol, index_path = study(root)
            arm = root / "recovery/polarity_rehearsal"
            run = summary.read(arm / "run.json"); run["runtime"]["gpu_name"] = "different GPU"
            write(arm / "run.json", run)
            provenance = summary.read(arm / "provenance.json"); provenance["run_sha256"] = summary.sha(arm / "run.json")
            write(arm / "provenance.json", provenance)
            index = summary.read(index_path); receipt = index["models"]["polarity_rehearsal"]
            receipt.update(run_sha256=provenance["run_sha256"], provenance_sha256=summary.sha(arm / "provenance.json"), training_runtime=run["runtime"])
            for filename in ("run.json", "provenance.json"):
                receipt["protected_file_sha256"][str((arm / filename).relative_to(root))] = summary.sha(arm / filename)
            write(index_path, index)
            report = summary.summarize(protocol, index_path, root)
            self.assertTrue(report["confounds"])
            self.assertIsNone(report["recommendation"]["selected_model_id"])
            self.assertEqual(sum(len(g["cases_prostate"]) for m in report["results"].values() for g in m.values()), 950)

    def test_existing_output_is_refused_before_reading_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"; output.write_text("keep")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                summary.main(["--protocol", "/missing", "--evaluation", "/missing", "--output", str(output)])
            self.assertEqual(output.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
