"""Constructed CPU-only recovery checks; never read the real test attempt."""

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_frozen_test as frozen
import recover_frozen_test as recovery
from metrics import SegmentationMetrics
from test_frozen_test import fixture, TinyFrozenModel, TEMP_ROOT, write


def recovery_fixture(root):
    value = fixture(root)
    old = deepcopy(value["plan"])
    metric = SegmentationMetrics(6)
    metric.confusion_matrix[5, 5] = 784
    metric.confusion_matrix[0, 5] = 512
    metrics = metric.compute()
    metrics.update(samples=5, batches=2, optimizer_steps=0, elapsed_seconds=.01,
                   cross_entropy=math.log(5 + math.exp(3)) - 3 * 784 / 1296)
    state = frozen.state_sha256(TinyFrozenModel())
    baseline = {"status": "passed", "checkpoint_epoch": 5,
                "checkpoint_sha256": old["models"][0]["checkpoint_sha256"],
                "forward_calls": 2, "forward_samples": 5, "optimizer_steps": 0, "full_test_passes": 1,
                "state_sha256_before": state, "state_sha256_after": state,
                "parameters_have_no_gradients": True, "metrics": metrics}
    runtime = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu_name": None,
               "math_settings": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                 "float32_matmul_precision": torch.get_float32_matmul_precision()}}
    attempt = {"status": "incomplete", "plan_sha256": frozen.sha(value["plan_path"]),
               "source_revision": old["source_revision"], "source_sha256": deepcopy(old["source_sha256"]),
               "recipe_sha256": old["recipe_sha256"], "audit_sha256": old["audit_sha256"],
               "evaluation": deepcopy(old["evaluation"]), "runtime": runtime,
               "test_manifest_sha256": old["data"]["test"]["manifest_sha256"],
               "test_content_manifest_sha256": old["data"]["test"]["test_content_manifest_sha256"],
               "frozen_checkpoint_sha256": {item["model"]: item["checkpoint_sha256"] for item in old["models"]},
               "models": {"baseline_unet": deepcopy(baseline), "improved_unet": {
                   "status": "incomplete", "checkpoint_epoch": 17,
                   "checkpoint_sha256": old["models"][1]["checkpoint_sha256"],
                   "forward_calls": 1, "forward_samples": 4, "optimizer_steps": 0, "state_sha256_before": state}}}
    guard = root / frozen.ONCE_GUARD
    guard.parent.mkdir(parents=True)
    attempt_path, baseline_path = root / "original_attempt.json", root / "original_baseline.json"
    write(attempt_path, attempt)
    write(guard, attempt)
    write(baseline_path, baseline)
    approval = {"user_answer": "允许，仅补跑 Improved", "user_recovery_confirmation_received": True,
                "additional_full_evaluations_authorized": {"baseline_unet": 0, "improved_unet": 1},
                "recovery_model": "improved_unet", "epoch": 17,
                "checkpoint_sha256": old["models"][1]["checkpoint_sha256"],
                "same_original_model_preprocessing_batch_and_evaluation_rules_required": True,
                "repeated_prior_test_forward_samples_disclosed": 4,
                "original_failure_and_baseline_must_be_preserved": True,
                "post_test_model_or_hyperparameter_tuning_authorized": False,
                "original_plan_sha256": frozen.sha(value["plan_path"]),
                "original_attempt_sha256": frozen.sha(attempt_path),
                "original_consumed_guard_sha256": frozen.sha(guard),
                "original_baseline_sha256": frozen.sha(baseline_path), "recovery_execution_started": False}
    approval_path = root / "recovery_approval.json"
    write(approval_path, approval)
    (value["project"] / "evaluate_frozen_test.py").write_bytes(b"constructed I/O-only repair")
    (value["project"] / "recover_frozen_test.py").write_bytes(b"constructed recovery entry")
    sources = {name: frozen.sha(value["project"] / name)
               for name in frozen.REQUIRED_SOURCES | {"recover_frozen_test.py"}}
    plan = {"status": "approved_for_execution", "user_recovery_confirmation_received": True,
            "recovery_model": "improved_unet", "checkpoint_epoch": 17,
            "recovery_guard": recovery.RECOVERY_GUARD, "synthetic_test_only": True,
            "source_revision": "constructed-recovery-only", "models": deepcopy(old["models"]),
            "evaluation": deepcopy(old["evaluation"]), "data": deepcopy(old["data"]), "source_sha256": sources,
            "original_evaluation_source_sha256": old["source_sha256"]["evaluate_frozen_test.py"],
            "protected_original_files_sha256": deepcopy(old["protected_original_files_sha256"])}
    links = {"original_plan": value["plan_path"], "original_attempt": attempt_path,
             "original_guard": guard, "baseline_result": baseline_path,
             "original_recipe": root / old["recipe_file"], "audit": root / old["audit_file"],
             "recovery_authorization": approval_path}
    for key, path in links.items():
        relative = path.relative_to(root).as_posix()
        plan[key + "_file"], plan[key + "_sha256"] = relative, frozen.sha(path)
        plan["protected_original_files_sha256"][relative] = frozen.sha(path)
    plan_path = root / "recovery_plan.json"
    write(plan_path, plan)
    value.update(plan=plan, plan_path=plan_path, old=old, original_attempt=attempt, original_baseline=baseline,
                 approval=approval, links=links,
                 args=argparse.Namespace(plan=plan_path, data_root=value["args"].data_root,
                                         output_dir=root / "new_recovery", device="cpu"))
    return value


def resign(value):
    """Construct changed metadata with matching file hashes for negative checks."""
    root = value["plan_path"].parent
    plan = value["plan"]
    approval = frozen.read(value["links"]["recovery_authorization"])
    for field, link in (("original_plan_sha256", "original_plan"), ("original_attempt_sha256", "original_attempt"),
                        ("original_consumed_guard_sha256", "original_guard"), ("original_baseline_sha256", "baseline_result")):
        approval[field] = frozen.sha(value["links"][link])
    write(value["links"]["recovery_authorization"], approval)
    for key, path in value["links"].items():
        plan[key + "_sha256"] = frozen.sha(path)
        plan["protected_original_files_sha256"][path.relative_to(root).as_posix()] = frozen.sha(path)
    write(value["plan_path"], plan)


class FrozenRecoveryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)

    def run_recovery(self, root, value, build=None):
        with patch("evaluate_frozen_test.HipMRIPairDataset", return_value=value["samples"]), \
                patch("evaluate_frozen_test.build_model", side_effect=build or (lambda *a, **k: TinyFrozenModel())), \
                patch("recover_frozen_test.torch.cuda.is_available", return_value=False):
            return recovery.execute(value["args"], root=root, project=value["project"], allow_cpu_for_synthetic=True)

    def test_only_improved_runs_and_all_original_files_remain_unchanged(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder)
            value = recovery_fixture(root)
            hashes = {name: frozen.sha(root / name) for name in value["plan"]["protected_original_files_sha256"]}
            built = []
            def build(name, **kwargs):
                self.assertEqual(name, "improved_unet")
                built.append(name)
                return TinyFrozenModel()
            report = self.run_recovery(root, value, build)
            self.assertEqual(built, ["improved_unet"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(set(report["models"]), {"improved_unet"})
            self.assertEqual(report["preserved_baseline_result"], value["original_baseline"])
            self.assertEqual(report["baseline_forward_calls_this_recovery"], 0)
            entry = report["models"]["improved_unet"]
            self.assertEqual((entry["forward_calls"], entry["forward_samples"], entry["optimizer_steps"]), (2, 5, 0))
            self.assertEqual(entry["metrics"]["valid_pixels"], 1296)
            self.assertEqual(entry["state_sha256_before"], entry["state_sha256_after"])
            self.assertEqual(report["exposure"]["repeated_prior_samples"], 4)
            self.assertEqual(report["exposure"]["planned_repeated_prior_samples"], 4)
            self.assertEqual(report["exposure"]["cumulative_improved_forward_calls"], 3)
            self.assertEqual(report["exposure"]["cumulative_improved_forward_samples"], 9)
            self.assertFalse(report["exposure"]["partial_metrics_recovered"])
            self.assertEqual(hashes, {name: frozen.sha(root / name) for name in hashes})
            self.assertEqual(frozen.read(root / frozen.ONCE_GUARD)["status"], "incomplete")
            self.assertEqual(frozen.read(root / recovery.RECOVERY_GUARD)["status"], "passed")

    def test_recovery_guard_blocks_different_plan_or_output(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder)
            value = recovery_fixture(root)
            self.run_recovery(root, value)
            value["args"].output_dir = root / "different_output"
            value["plan"]["extra_note"] = "must not bypass guard"
            another = root / "different_plan.json"
            write(another, value["plan"])
            value["args"].plan = another
            with self.assertRaisesRegex(ValueError, "Recovery attempt already exists"):
                self.run_recovery(root, value)
            self.assertFalse(value["args"].output_dir.exists())

    def test_extra_human_authorization_must_match_model_budget_and_repeated_exposure(self):
        for field, changed in (("user_recovery_confirmation_received", False),
                               ("additional_full_evaluations_authorized", {"baseline_unet": 1, "improved_unet": 1}),
                               ("repeated_prior_test_forward_samples_disclosed", 0),
                               ("post_test_model_or_hyperparameter_tuning_authorized", True)):
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
                root = Path(folder)
                value = recovery_fixture(root)
                approval = frozen.read(value["links"]["recovery_authorization"])
                approval[field] = changed
                write(value["links"]["recovery_authorization"], approval)
                resign(value)
                with self.assertRaisesRegex(ValueError, "Human approval does not authorize"):
                    self.run_recovery(root, value)
                self.assertFalse((root / recovery.RECOVERY_GUARD).exists())

    def test_models_rules_and_non_persistence_sources_cannot_change(self):
        for change in ("model", "batch", "core_source"):
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
                root = Path(folder)
                value = recovery_fixture(root)
                if change == "model":
                    value["plan"]["models"][1]["epoch"] = 18
                    message = "changed the original models"
                elif change == "batch":
                    value["plan"]["evaluation"]["batch_size"] = 1
                    message = "changed the original models"
                else:
                    (value["project"] / "dataset.py").write_bytes(b"changed preprocessing")
                    value["plan"]["source_sha256"]["dataset.py"] = frozen.sha(value["project"] / "dataset.py")
                    message = "non-persistence original dependency changed"
                write(value["plan_path"], value["plan"])
                with self.assertRaisesRegex(ValueError, message):
                    self.run_recovery(root, value)
                self.assertFalse((root / recovery.RECOVERY_GUARD).exists())

    def test_original_guard_must_equal_the_preserved_incomplete_attempt(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder)
            value = recovery_fixture(root)
            guard = frozen.read(value["links"]["original_guard"])
            guard["status"] = "passed"
            write(value["links"]["original_guard"], guard)
            resign(value)
            with self.assertRaisesRegex(ValueError, "incomplete attempt/guard provenance differs"):
                self.run_recovery(root, value)
            self.assertFalse((root / recovery.RECOVERY_GUARD).exists())

    def test_partial_improved_metrics_cannot_be_claimed_as_recovered(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder)
            value = recovery_fixture(root)
            attempt = deepcopy(value["original_attempt"])
            attempt["models"]["improved_unet"]["metrics"] = {"dice_per_class": [None] * 6}
            write(value["links"]["original_attempt"], attempt)
            write(value["links"]["original_guard"], attempt)
            resign(value)
            with self.assertRaisesRegex(ValueError, "incomplete partial pass without recovered metrics"):
                self.run_recovery(root, value)
            self.assertFalse((root / recovery.RECOVERY_GUARD).exists())

    def test_failed_recovery_preserves_original_guard_and_blocks_automatic_retry(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder)
            value = recovery_fixture(root)
            original = frozen.sha(root / frozen.ONCE_GUARD)
            def build(*args, **kwargs):
                raise RuntimeError("constructed recovery failure")
            with self.assertRaisesRegex(RuntimeError, "constructed recovery failure"):
                self.run_recovery(root, value, build)
            report = frozen.read(root / recovery.RECOVERY_GUARD)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["models"]["improved_unet"]["forward_calls"], 0)
            self.assertEqual(report["baseline_forward_calls_this_recovery"], 0)
            self.assertEqual(report["exposure"]["planned_repeated_prior_samples"], 4)
            self.assertEqual(report["exposure"]["repeated_prior_samples"], 0)
            self.assertEqual(frozen.sha(root / frozen.ONCE_GUARD), original)
            value["args"].output_dir = root / "retry_output"
            with self.assertRaisesRegex(ValueError, "Recovery attempt already exists"):
                self.run_recovery(root, value)

    def test_both_frozen_checkpoint_copies_are_rechecked_without_baseline_forward(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder)
            value = recovery_fixture(root)
            built = []
            def build(name, **kwargs):
                self.assertEqual(name, "improved_unet")
                built.append(name)
                (root / value["plan"]["models"][0]["checkpoint"]).write_bytes(b"constructed baseline-copy mutation")
                return TinyFrozenModel()
            with self.assertRaisesRegex(ValueError, "frozen checkpoint copy changed"):
                self.run_recovery(root, value, build)
            self.assertEqual(built, ["improved_unet"])
            report = frozen.read(root / recovery.RECOVERY_GUARD)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["models"]["improved_unet"]["forward_calls"], 2)
            self.assertEqual(report["baseline_forward_calls_this_recovery"], 0)


if __name__ == "__main__":
    unittest.main()
