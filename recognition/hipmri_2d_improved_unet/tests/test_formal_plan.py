"""Formal-protocol guard checks on constructed filenames; no MRI or training."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from train import main, validate_formal_plan


class FormalPlanChecks(unittest.TestCase):
    def setUp(self):
        # Keep generated plans under ignored outputs on this Windows host.
        temporary_root = PROJECT_DIR.parents[1] / "outputs" / "test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.assertTrue(Path(directory.name).resolve().is_relative_to(temporary_root.resolve()))
        self.addCleanup(directory.cleanup)
        self.plan_path = Path(directory.name) / "approved_plan.json"
        self.training_pairs = [self.pair("004", 0), self.pair("010", 1)]
        self.validation_pairs = [self.pair("036", 0)]
        self.args = argparse.Namespace(
            formal_plan=self.plan_path, epochs=20, batch_size=4,
            base_channels=16, num_classes=6, lr=0.001, seed=3710,
            device="cuda", train_limit=None, model="baseline_unet",
        )
        self.plan = {
            "status": "approved_for_execution",
            "user_confirmation_received": True,
            "epochs_per_model": 20,
            "shared_protocol": {
                "models": ["baseline_unet", "improved_unet"],
                "batch_size": 4,
                "model_config": {"base_channels": 16, "num_classes": 6},
                "optimizer": {"learning_rate": 0.001},
                "seed": 3710,
            },
            "data": {
                "training": {"manifest_sha256": self.manifest(self.training_pairs)},
                "validation": {"manifest_sha256": self.manifest(self.validation_pairs)},
            },
            "source_sha256": {},
        }

    @staticmethod
    def pair(case, slice_id):
        suffix = f"{case}_week_0_slice_{slice_id}.nii.gz"
        return Path("constructed") / f"case_{suffix}", Path("constructed") / f"seg_{suffix}"

    @staticmethod
    def manifest(pairs):
        names = "\n".join(f"{image.name}|{mask.name}" for image, mask in pairs)
        return hashlib.sha256(names.encode("utf-8")).hexdigest()

    def validate(self):
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        return validate_formal_plan(self.args, self.training_pairs, self.validation_pairs)

    def test_absent_formal_plan_keeps_smoke_path(self):
        self.args.formal_plan = None
        self.assertIsNone(validate_formal_plan(self.args, [], []))

    def test_approved_matching_protocol_accepts_each_listed_model(self):
        for name in ("baseline_unet", "improved_unet"):
            with self.subTest(model=name):
                self.args.model = name
                self.assertEqual(self.validate(), self.plan)

    def test_unconfirmed_or_unapproved_plan_is_rejected(self):
        for key, invalid in (("user_confirmation_received", False), ("status", "draft")):
            with self.subTest(field=key):
                original = self.plan[key]
                self.plan[key] = invalid
                with self.assertRaisesRegex(ValueError, "explicit user confirmation"):
                    self.validate()
                self.plan[key] = original

    def test_smoke_limits_or_epoch_drift_are_rejected(self):
        for key, invalid in (("train_limit", 64), ("epochs", 2)):
            with self.subTest(field=key):
                original = getattr(self.args, key)
                setattr(self.args, key, invalid)
                with self.assertRaisesRegex(ValueError, "CLI differs"):
                    self.validate()
                setattr(self.args, key, original)

    def test_training_and_validation_manifest_drift_are_rejected(self):
        for split in ("training", "validation"):
            with self.subTest(split=split):
                pairs = getattr(self, f"{split}_pairs")
                original = pairs[0]
                pairs[0] = self.pair("099", 9)
                with self.assertRaisesRegex(ValueError, f"{split} manifest differs"):
                    self.validate()
                pairs[0] = original

    def test_source_hash_drift_is_rejected(self):
        self.plan["source_sha256"] = {"train.py": "0" * 64}
        with self.assertRaisesRegex(ValueError, "Source changed.*train.py"):
            self.validate()

    def test_model_not_listed_in_plan_is_rejected(self):
        self.args.model = "unapproved_model"
        with self.assertRaisesRegex(ValueError, "Model not listed"):
            self.validate()

    def test_formal_setup_failure_is_saved_as_incomplete(self):
        self.args.output_dir = self.plan_path.parent / "fresh_run"
        with patch("train.parse_args", return_value=self.args), \
                patch("train.execute_training", side_effect=RuntimeError("constructed setup failure")):
            with self.assertRaisesRegex(RuntimeError, "constructed setup failure"):
                main()
        report = json.loads((self.args.output_dir / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["failure_phase"], "setup")
        self.assertFalse(report["formal_training_started"])
        self.assertEqual(report["optimizer_steps"], 0)

    def test_existing_output_is_never_modified_by_setup_error_recording(self):
        self.args.output_dir = self.plan_path.parent / "existing_run"
        self.args.output_dir.mkdir()
        sentinel = self.args.output_dir / "keep.txt"
        sentinel.write_text("prior evidence", encoding="utf-8")
        with patch("train.parse_args", return_value=self.args), \
                patch("train.execute_training", side_effect=FileExistsError("existing output")):
            with self.assertRaises(FileExistsError):
                main()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "prior evidence")
        self.assertEqual(sorted(path.name for path in self.args.output_dir.iterdir()), ["keep.txt"])


if __name__ == "__main__":
    unittest.main()
