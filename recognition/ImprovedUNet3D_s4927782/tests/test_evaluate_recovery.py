"""Synthetic recovery protocol/receipt guards; no real MRI, weights or GPU job."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments import evaluate_recovery as tool

SHA = "a" * 64
STATE = "b" * 64
RUNTIME = {"gpu_name": "synthetic GPU", "python": "3.11", "torch": "fixture", "cuda": "fixture"}


def plans():
    config = {"model": "improved3d", "base_channels": 16, "num_classes": 6, "seed": 42,
              "lr": 0.001, "max_batches": None, "development_only": False}
    appearance = {"training_conditions": config,
                  "models": {n: {"checkpoint": f"artifacts/{n}/best.pt"} for n in tool.REFERENCES}}
    recovery = {"experiment_id": tool.EXPERIMENT, "allowed_splits": ["train", "validation"],
                "manifest_sha256": tool.original.MANIFEST_SHA, "cache_identity_sha256": tool.original.IDENTITY_SHA,
                "training_sources_sha256": tool.original.SOURCES_SHA, "training_seed": 42, "training_epochs_per_model": 8,
                "conditions": list(tool.CONDITIONS), "training_conditions": dict(config, lr=1e-4),
                "inference": {"split": "validation", "device": "cuda", "amp": True, "saved_arrays": False},
                "required_file_sha256": {},
                "references": {n: {"checkpoint": appearance["models"][n]["checkpoint"],
                                   "selected_epoch": e, "checkpoint_sha256": h} for n, (e, h) in tool.REFERENCES.items()},
                "models": {n: {"kind": "recovery_arm", "polarity_probability": p,
                               "checkpoint": f"artifacts/recovery/{n}/best.pt"} for n, p in zip(tool.ARMS, (0.0, 0.1))}}
    recovery["warm_start"] = {"checkpoint": appearance["models"]["gamma_polarity"]["checkpoint"],
                              "checkpoint_sha256": tool.REFERENCES["gamma_polarity"][1]}
    return recovery, appearance


def manifest():
    return {"records": [{"patient": "TRAIN", "visit": f"Week{i}", "split": "train"} for i in range(143)]
                       + [{"patient": "VAL", "visit": f"Week{i}", "split": "validation"} for i in range(38)]}


def checkpoint(plan, arm="clean_adapt"):
    return {"experiment": {"experiment_id": tool.EXPERIMENT, "arm_tag": arm, "protocol_sha256": SHA},
            "config": copy.deepcopy(plan["training_conditions"]), "epoch": 3, "best_epoch": 3, "best_score": .9,
            "parent_checkpoint_sha256": plan["warm_start"]["checkpoint_sha256"]}


class ProtocolGuards(unittest.TestCase):
    def test_exact_five_models_five_conditions_and_two_arm_plan(self):
        plan, appearance = plans()
        tool.validate_protocol(plan, appearance, tool.original.MANIFEST_SHA)
        self.assertEqual(len(tool.MODEL_IDS), 5)

    def test_test_precision_budget_parent_config_and_reference_changes_fail(self):
        for issue in ("test", "precision", "budget", "parent", "config", "reference", "arm", "condition", "model"):
            with self.subTest(issue=issue):
                plan, appearance = plans()
                if issue == "test": plan["allowed_splits"].append("test")
                if issue == "precision": plan["inference"]["amp"] = False
                if issue == "budget": plan["training_epochs_per_model"] = 9
                if issue == "parent": plan["warm_start"]["checkpoint_sha256"] = SHA
                if issue == "config": plan["training_conditions"]["weight_decay"] = .5
                if issue == "reference": plan["references"]["gamma_only"]["checkpoint_sha256"] = SHA
                if issue == "arm": plan["models"]["polarity_rehearsal"]["polarity_probability"] = .25
                if issue == "condition": plan["conditions"].pop()
                if issue == "model": plan["references"].pop("gamma_only")
                with self.assertRaises(ValueError): tool.validate_protocol(plan, appearance, tool.original.MANIFEST_SHA)

    def test_checkpoint_experiment_parent_epoch_and_config_are_bound(self):
        plan, _ = plans()
        tool.validate_payload(checkpoint(plan), "clean_adapt", plan, SHA)
        for issue in ("tag", "parent", "epoch", "config"):
            with self.subTest(issue=issue):
                cp = checkpoint(plan)
                if issue == "tag": cp["experiment"]["arm_tag"] = "polarity_rehearsal"
                if issue == "parent": cp["parent_checkpoint_sha256"] = SHA
                if issue == "epoch": cp["epoch"] = cp["best_epoch"] = 9
                if issue == "config": cp["config"]["lr"] = .001
                with self.assertRaises(ValueError): tool.validate_payload(cp, "clean_adapt", plan, SHA)

    def test_existing_output_refused_before_reading_input_or_importing_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must be new"):
                tool.main(["--protocol", "/missing.json", "--manifest", "/missing.json", "--output-dir", directory])


class ReceiptGuards(unittest.TestCase):
    def fixture(self, root):
        plan, _ = plans()
        best = root / "best.pt"; best.write_bytes(b"opaque synthetic weight identity")
        cp = checkpoint(plan)
        history = [{"epoch": i, "training_examples": 143, "training_batches": 143, "optimizer_steps": 143,
                    "amp_overflow_skipped_optimizer_steps": 0, "validation_foreground_dice": .9 if i in (3, 5) else .5,
                    "policy_counts": {"noninverted": 143, "inverted": 0, "gamma_applied": 70}}
                   for i in range(1, 9)]
        run = {"experiment": cp["experiment"], "config": cp["config"], "parent_checkpoint_sha256": cp["parent_checkpoint_sha256"],
               "parent_epoch": 29, "warm_start_weights_only": True, "optimizer_reset": True, "scaler_reset": True,
               "status": "completed", "completed_epochs": 8, "best_epoch": 3, "source_file_sha256": {},
               "initial_parameter_state_sha256": STATE, "runtime": RUNTIME,
               "training_policy_counts": {"noninverted": 1144, "inverted": 0, "gamma_applied": 560}}
        verification = {"status": "verified", "manifest_sha256": tool.original.MANIFEST_SHA,
                        "counts": {"train": 143, "validation": 38}, "test_arrays_opened": 0,
                        "records": [{**r, "decoded_identity_verified": True} for r in manifest()["records"]]}
        for name, data in (("history.json", history), ("verification.json", verification)):
            (root / name).write_text(json.dumps(data))
        run["cache_verification_sha256"] = tool.sha(root / "verification.json")
        (root / "run.json").write_text(json.dumps(run))
        provenance = {"experiment": cp["experiment"], "parent_checkpoint_sha256": cp["parent_checkpoint_sha256"],
                      "parent_epoch": 29, "warm_start_weights_only": True, "optimizer_reset": True, "scaler_reset": True,
                      "checkpoint_sha256": tool.sha(best), "run_sha256": tool.sha(root / "run.json"),
                      "history_sha256": tool.sha(root / "history.json"), "protocol_sha256": SHA,
                      "manifest_sha256": tool.original.MANIFEST_SHA, "source_file_sha256": {},
                      "cache_verification_path": "verification.json", "cache_verification_sha256": tool.sha(root / "verification.json")}
        (root / "provenance.json").write_text(json.dumps(provenance))
        return best, cp, plan

    def test_complete_eight_epoch_receipt_and_reported_initial_state(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(tool, "PROJECT", Path(directory).resolve()), patch.object(tool.original, "PROJECT", Path(directory).resolve()):
            best, cp, plan = self.fixture(Path(directory).resolve())
            result = tool.validate_arm_files(cp, "clean_adapt", best, manifest(), plan, SHA)
            self.assertEqual(result["initial_parameter_state_sha256"], STATE)
            self.assertEqual(result["training_runtime"], RUNTIME)
            self.assertEqual(len(result["protected_file_sha256"]), 4)

    def test_incomplete_wrong_parent_updates_cache_scope_and_tied_selector_fail(self):
        for issue in ("incomplete", "parent", "updates", "test", "tie", "reset", "policy"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as directory, patch.object(tool, "PROJECT", Path(directory).resolve()), patch.object(tool.original, "PROJECT", Path(directory).resolve()):
                root = Path(directory).resolve(); best, cp, plan = self.fixture(root)
                if issue in ("incomplete", "updates"):
                    p = root / "history.json"; rows = tool.read(p)
                    if issue == "incomplete": rows.pop()
                    else: rows[0]["optimizer_steps"] = 0
                    p.write_text(json.dumps(rows))
                if issue == "parent":
                    p = root / "run.json"; data = tool.read(p); data["parent_checkpoint_sha256"] = SHA; p.write_text(json.dumps(data))
                if issue == "reset":
                    p = root / "run.json"; data = tool.read(p); data["optimizer_reset"] = False; p.write_text(json.dumps(data))
                if issue == "policy":
                    p = root / "history.json"; rows = tool.read(p); rows[0]["policy_counts"]["inverted"] = 1; p.write_text(json.dumps(rows))
                if issue == "tie":
                    cp["epoch"] = cp["best_epoch"] = 5
                    p = root / "run.json"; data = tool.read(p); data["best_epoch"] = 5; p.write_text(json.dumps(data))
                if issue == "test":
                    p = root / "verification.json"; data = tool.read(p); data["records"][0]["split"] = "test"; p.write_text(json.dumps(data))
                    p = root / "run.json"; data = tool.read(p); data["cache_verification_sha256"] = tool.sha(root / "verification.json"); p.write_text(json.dumps(data))
                    p = root / "provenance.json"; data = tool.read(p); data["run_sha256"] = tool.sha(root / "run.json"); data["cache_verification_sha256"] = tool.sha(root / "verification.json"); p.write_text(json.dumps(data))
                with self.assertRaises(ValueError): tool.validate_arm_files(cp, "clean_adapt", best, manifest(), plan, SHA)

    def test_receipt_detects_tampered_run_and_history_bytes(self):
        for filename in ("run.json", "history.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory, patch.object(tool, "PROJECT", Path(directory).resolve()), patch.object(tool.original, "PROJECT", Path(directory).resolve()):
                root = Path(directory).resolve(); best, cp, plan = self.fixture(root)
                p = root / filename; p.write_text(p.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "identities"):
                    tool.validate_arm_files(cp, "clean_adapt", best, manifest(), plan, SHA)


class PairGuards(unittest.TestCase):
    def test_actual_parent_initialization_and_same_training_eval_runtime_required(self):
        rows = {n: {"initial_parameter_state_sha256": STATE, "training_runtime": copy.deepcopy(RUNTIME)} for n in tool.ARMS}
        tool.validate_pair(rows, STATE, RUNTIME)
        for issue in ("state", "pair_runtime", "eval_runtime"):
            with self.subTest(issue=issue):
                bad = copy.deepcopy(rows); current = copy.deepcopy(RUNTIME)
                if issue == "state": bad["clean_adapt"]["initial_parameter_state_sha256"] = SHA
                if issue == "pair_runtime": bad["clean_adapt"]["training_runtime"]["gpu_name"] = "other GPU"
                if issue == "eval_runtime": current["gpu_name"] = "other GPU"
                with self.assertRaises(ValueError): tool.validate_pair(bad, STATE, current)

    def test_selected_clean_checks_counts_while_ignoring_resource_measurements(self):
        result = {"cases": [{"patient": "VAL", "visit": "Week0", "per_class": [
            {"class_index": i, "target_voxels": 1, "predicted_voxels": 1, "intersection": 1,
             "union": 1, "dice": 1., "iou": 1.} for i in range(6)],
            "foreground_mean_dice": 1., "foreground_mean_iou": 1., "inference_ms": 10.}],
            "aggregate": {"foreground_mean_dice": 1., "foreground_mean_iou": 1.}}
        history = [{"epoch": 1, "validation": copy.deepcopy(result)}]
        result["cases"][0]["inference_ms"] = 999.
        tool.validate_selected_clean(result, history, 1)
        bad = copy.deepcopy(result); bad["cases"][0]["per_class"][5]["intersection"] = 0
        with self.assertRaisesRegex(ValueError, "selected history"):
            tool.validate_selected_clean(bad, history, 1)
        bad = copy.deepcopy(result); bad["cases"].append(copy.deepcopy(bad["cases"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            tool.validate_selected_clean(bad, history, 1)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch required for finite-output guard")
class FiniteGuard(unittest.TestCase):
    def test_nonfinite_main_logits_rejected(self):
        import torch
        with self.assertRaisesRegex(RuntimeError, "before argmax"):
            tool.original.finite_main_logits(None, (), {"logits": torch.tensor([float("nan")])})


if __name__ == "__main__":
    unittest.main()
