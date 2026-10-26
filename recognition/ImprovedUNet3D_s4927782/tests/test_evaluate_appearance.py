"""Protocol/receipt guards without real checkpoints, cache pixels or GPU jobs."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("appearance_evaluator", PROJECT / "experiments/evaluate_appearance.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)
SHA = "a" * 64
try:
    import torch
except ImportError:
    torch = None


def protocol():
    models = {name: {"kind": "frozen_v1", "seed": seed, "selected_epoch": epoch, "checkpoint_sha256": digest,
                     "checkpoint": f"artifacts/{name}/best.pt"} for name, (seed, epoch, digest) in tool.REFERENCES.items()}
    models.update({name: {"kind": "appearance_arm", "experiment_tag": name, "seed": 42,
                         "checkpoint": f"artifacts/{name}/best.pt"} for name in tool.MODEL_IDS[2:]})
    return {"experiment_id": tool.EXPERIMENT, "manifest_sha256": tool.MANIFEST_SHA,
            "training_sources_sha256": tool.SOURCES_SHA, "allowed_splits": ["train", "validation"],
            "training_epochs_per_model": 30, "training_seed": 42, "conditions": list(tool.CONDITIONS),
            "models": models, "inference": {"split": "validation", "amp": True}}


def manifest():
    return {"label_ids": list(range(6)), "target_shape_dhw": [64, 128, 128], "patient_splits": {"train": ["A"], "validation": ["B"], "test": ["C"]},
            "records": [{"patient": "A", "visit": f"Week{i}", "split": "train"} for i in range(143)]
                       + [{"patient": "B", "visit": f"Week{i}", "split": "validation"} for i in range(38)]}


def checkpoint(name="gamma_only"):
    data = manifest()
    config = {"model": "improved3d", "base_channels": 16, "num_classes": 6, "manifest_sha256": tool.MANIFEST_SHA,
              "label_ids": data["label_ids"], "target_shape_dhw": data["target_shape_dhw"], "patient_splits": data["patient_splits"],
              "seed": 42, "development_only": False, "max_batches": None}
    experiment = {"experiment_id": tool.EXPERIMENT, "arm_tag": name, "protocol_sha256": SHA}
    return {"config": config, "experiment": experiment, "epoch": 3, "best_epoch": 3, "best_score": 1.}


class ProtocolGuards(unittest.TestCase):
    def test_all_four_models_and_five_conditions_are_required(self):
        tool.validate_protocol(protocol(), tool.MANIFEST_SHA)
        bad = protocol(); bad["conditions"].pop()
        with self.assertRaisesRegex(ValueError, "all five"):
            tool.validate_protocol(bad, tool.MANIFEST_SHA)
        bad = protocol(); bad["models"].pop("gamma_only")
        with self.assertRaisesRegex(ValueError, "All four"):
            tool.validate_protocol(bad, tool.MANIFEST_SHA)

    def test_test_and_reference_identity_changes_are_rejected(self):
        for change in ("split", "weight"):
            bad = protocol()
            if change == "split":
                bad["allowed_splits"].append("test")
            else:
                bad["models"]["seed42_reference"]["checkpoint_sha256"] = SHA
            with self.subTest(change=change), self.assertRaises(ValueError):
                tool.validate_protocol(bad, tool.MANIFEST_SHA)
        bad = protocol(); bad["inference"]["split"] = "test"
        with self.assertRaises(ValueError):
            tool.validate_protocol(bad, tool.MANIFEST_SHA)

    def test_original_and_new_checkpoint_contracts(self):
        cp = checkpoint(); spec = protocol()["models"]["gamma_only"]
        tool.validate_payload(cp, spec, "gamma_only", manifest(), SHA)
        cp["experiment"]["arm_tag"] = "gamma_polarity"
        with self.assertRaisesRegex(ValueError, "tag"):
            tool.validate_payload(cp, spec, "gamma_only", manifest(), SHA)
        cp = checkpoint(); cp["epoch"] = 29; cp["best_epoch"] = 29
        tool.validate_payload(cp, protocol()["models"]["seed42_reference"], "seed42_reference", manifest(), SHA)
        cp["config"]["max_batches"] = 1
        with self.assertRaisesRegex(ValueError, "development"):
            tool.validate_payload(cp, protocol()["models"]["seed42_reference"], "seed42_reference", manifest(), SHA)

    def test_existing_output_refused_before_any_dependency_or_input_read(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must be new"):
                tool.main(["--protocol", "/missing.json", "--manifest", "/missing.json", "--output-dir", directory])


class ArmReceiptGuards(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        best = root / "best.pt"; best.write_bytes(b"opaque synthetic checkpoint receipt")
        cp = checkpoint(); config = cp["config"]
        history = [{"epoch": i, "training_examples": 143, "training_batches": 143,
                    "validation_foreground_dice": 1. if i == 3 else .5} for i in range(1, 31)]
        run = {"experiment": cp["experiment"], "config": config, "status": "completed", "completed_epochs": 30, "best_epoch": 3}
        verification = {"status": "verified", "manifest_sha256": tool.MANIFEST_SHA, "counts": {"train": 143, "validation": 38},
                        "records": [{**r, "decoded_identity_verified": True} for r in manifest()["records"]]}
        for name, data in (("run.json", run), ("history.json", history), ("verification.json", verification)):
            (root / name).write_text(json.dumps(data))
        receipt = {"experiment": cp["experiment"], "checkpoint_sha256": tool.sha256(best), "run_sha256": tool.sha256(root / "run.json"),
                   "protocol_sha256": SHA, "manifest_sha256": tool.MANIFEST_SHA, "source_file_sha256": {},
                   "cache_verification_path": "verification.json", "cache_verification_sha256": tool.sha256(root / "verification.json")}
        (root / "provenance.json").write_text(json.dumps(receipt))
        return root, best, cp

    def test_complete_receipt_and_tampered_run_hash(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(tool, "PROJECT", Path(directory).resolve()):
            root, best, cp = self.fixture(directory)
            result = tool.validate_arm_files(cp, cp["config"], "gamma_only", best, manifest(), SHA, {})
            self.assertEqual(result["run_sha256"], tool.sha256(root / "run.json"))
            run = json.loads((root / "run.json").read_text()); run["extra"] = "tampering"
            (root / "run.json").write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, "identities"):
                tool.validate_arm_files(cp, cp["config"], "gamma_only", best, manifest(), SHA, {})

    def test_incomplete_training_and_test_verification_are_rejected(self):
        for error in ("incomplete", "test"):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory, patch.object(tool, "PROJECT", Path(directory).resolve()):
                root, best, cp = self.fixture(directory)
                if error == "incomplete":
                    path = root / "history.json"; rows = json.loads(path.read_text()); rows.pop(); path.write_text(json.dumps(rows))
                else:
                    path = root / "verification.json"; data = json.loads(path.read_text()); data["records"][0]["split"] = "test"; path.write_text(json.dumps(data))
                    path = root / "provenance.json"; data = json.loads(path.read_text()); data["cache_verification_sha256"] = tool.sha256(root / "verification.json"); path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    tool.validate_arm_files(cp, cp["config"], "gamma_only", best, manifest(), SHA, {})


@unittest.skipIf(torch is None, "PyTorch required for finite-logit runtime guards")
class FiniteLogitsGuard(unittest.TestCase):
    def test_nan_is_rejected_before_argmax(self):
        class BadModel(torch.nn.Module):
            def forward(self, _image):
                return {"logits": torch.tensor([float("nan"), 1.])}
        model = BadModel()
        model.register_forward_hook(tool.finite_main_logits)
        with self.assertRaisesRegex(RuntimeError, "before argmax"):
            model(None)

    def test_finite_output_is_unchanged(self):
        output = {"logits": torch.tensor([0., 1., -2.])}
        before = output["logits"].clone()
        self.assertIsNone(tool.finite_main_logits(None, (), output))
        self.assertTrue(torch.equal(before, output["logits"]))


if __name__ == "__main__":
    unittest.main()
