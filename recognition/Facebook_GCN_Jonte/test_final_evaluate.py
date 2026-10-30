"""Final-evaluation guard/pipeline tests use development labels, not test labels."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from confidence import failure_cases
from dataset import load_facebook
from evaluate import PROJECT
from final_evaluate import freeze_manifest, run_final, verify_manifest
from predict import DEFAULT_POLICY


class FinalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_facebook()
        cls.checkpoints = {name: PROJECT / f"outputs/{name.lower()}_seed42/best.pt" for name in ("MLP", "GCN")}

    def test_release_guards_fail_before_label_access(self):
        with patch.object(self.dataset, "targets", side_effect=AssertionError("Unexpected labels")):
            with self.assertRaises(PermissionError):
                run_final(self.dataset, self.checkpoints, DEFAULT_POLICY, "absent.json", "unused")
            with self.assertRaises(PermissionError):
                failure_cases(self.dataset, {}, {}, {}, partition="test")

    def test_freeze_and_manifest_mismatch_do_not_request_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.json"
            with patch.object(self.dataset, "targets", side_effect=AssertionError("Unexpected labels")):
                record = freeze_manifest(self.dataset, self.checkpoints, DEFAULT_POLICY, path)
                self.assertFalse(record["test_labels_released"])
                self.assertEqual(verify_manifest(self.dataset, self.checkpoints, DEFAULT_POLICY, path), record)
                with self.assertRaises(FileExistsError):
                    freeze_manifest(self.dataset, self.checkpoints, DEFAULT_POLICY, path)
                record["source_sha256"]["modules.py"] = "tampered"
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "source_sha256"):
                    run_final(self.dataset, self.checkpoints, DEFAULT_POLICY, path,
                              Path(directory) / "out", partition="selection")

    def test_full_pipeline_on_selection_only_matches_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "freeze.json"
            freeze_manifest(self.dataset, self.checkpoints, DEFAULT_POLICY, manifest)
            access = self.dataset.targets

            def permitted(role, **kwargs):
                if role not in ("selection", "train"):
                    raise AssertionError(f"Development smoke accessed {role} labels")
                return access(role, **kwargs)

            with patch.object(self.dataset, "targets", side_effect=permitted), \
                 patch("confidence.fit_temperature", side_effect=AssertionError("Refitting")), \
                 patch("confidence.choose_threshold", side_effect=AssertionError("Retuning")):
                with redirect_stdout(io.StringIO()):
                    result = run_final(self.dataset, self.checkpoints, DEFAULT_POLICY, manifest,
                                       Path(directory) / "out", device="cpu", partition="selection")
            old = json.loads((PROJECT / "outputs/evaluation_selection_seed42/selection_metrics.json").read_text())
            for row in old["models"]:
                actual = result["models"][row["model"]]["classification"]["raw"]
                self.assertAlmostEqual(actual["accuracy"], row["metrics"]["accuracy"], places=10)
                self.assertAlmostEqual(actual["macro_f1"], row["metrics"]["macro_f1"], places=10)
                self.assertEqual(actual["confusion_matrix"], row["metrics"]["confusion_matrix"])
            self.assertEqual(len(result["failure_cases"]), 5)
            self.assertTrue(result["no_fitting_or_threshold_search"])
            self.assertEqual(result["partition"], "selection")
            self.assertTrue((Path(directory) / "out/reliability_selection.png").is_file())


if __name__ == "__main__":
    unittest.main()
