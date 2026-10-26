"""Focused diagnostic guards/formulas; no real data, training or GPU needed."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


path = Path(__file__).resolve().parents[1] / "tools" / "diagnose_frozen_inference.py"
spec = importlib.util.spec_from_file_location("frozen_inference_diagnostic", path)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class IndependentFormulaTests(unittest.TestCase):
    def test_confusion_orientation_counts_exact_dice_and_iou(self):
        result = diagnostic.metrics_from_confusion([[3, 1, 0], [2, 4, 0], [0, 0, 0]])
        self.assertEqual(result["gt_counts"], [4, 6, 0])
        self.assertEqual(result["prediction_counts"], [5, 5, 0])
        background, foreground, absent = result["per_class"]
        self.assertEqual((background["tp"], background["fp"], background["fn"]), (3, 2, 1))
        self.assertAlmostEqual(background["dice"], 6 / 9)
        self.assertAlmostEqual(foreground["dice"], 8 / 11)
        self.assertAlmostEqual(foreground["iou"], 4 / 7)
        self.assertIsNone(absent["dice"])
        self.assertIsNone(absent["iou"])

    def test_false_positive_only_scores_zero_and_empty_scores_none(self):
        result = diagnostic.metrics_from_confusion([[2, 1, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(result["per_class"][1]["dice"], 0)
        self.assertEqual(result["per_class"][1]["iou"], 0)
        self.assertIsNone(result["per_class"][2]["dice"])

    def test_invalid_confusion_counts_fail(self):
        for matrix in ([], [[1]], [[1, 2], [3]], [[1, -1], [0, 0]], [[True, 0], [0, 1]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                diagnostic.metrics_from_confusion(matrix)

    def test_dice_delta_preserves_undefined_transition(self):
        before = diagnostic.metrics_from_confusion([[2, 0], [0, 0]])
        after = diagnostic.metrics_from_confusion([[1, 1], [0, 0]])
        delta = diagnostic.dice_deltas(before, after)
        self.assertAlmostEqual(delta[0]["after_minus_before"], -1 / 3)
        self.assertTrue(delta[1]["defined_status_changed"])
        self.assertIsNone(delta[1]["after_minus_before"])


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"partial": False, "development_only": False,
                         "records": [{"patient": p, "visit": v, "split": s}
                                     for p, v, s in (("B006", "Week0", "train"),
                                                     ("B040", "Week0", "validation"),
                                                     ("M015", "Week2", "test"),
                                                     ("W029", "Week0", "test"))]}

    def test_explicit_controls_preserve_order(self):
        rows = diagnostic.select_cases(self.manifest, ["B040/Week0", "B006/Week0"])
        self.assertEqual([r["split"] for r in rows], ["validation", "train"])

    def test_m015_test_requires_ack_and_other_test_is_forbidden(self):
        with self.assertRaises(ValueError):
            diagnostic.select_cases(self.manifest, ["M015/Week2"])
        rows = diagnostic.select_cases(self.manifest, ["M015/Week2"], True)
        self.assertEqual(rows[0]["patient"], "M015")
        with self.assertRaises(ValueError):
            diagnostic.select_cases(self.manifest, ["W029/Week0"], True)

    def test_partial_smoke_duplicate_unknown_or_implicit_selection_fail(self):
        for field in ("partial", "development_only"):
            modified = copy.deepcopy(self.manifest)
            modified[field] = True
            with self.assertRaises(ValueError):
                diagnostic.select_cases(modified, ["B006/Week0"])
        for keys in ([], ["B006"], ["B006/Week0/extra"], ["unknown/Week0"], ["B006/Week0"] * 2):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                diagnostic.select_cases(self.manifest, keys, True)

    def test_array_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(diagnostic.safe_child(root, "predictions/a.npz"), (root / "predictions/a.npz").resolve())
            for relative in ("../a.npz", str(root / "a.npz")):
                with self.assertRaises(ValueError):
                    diagnostic.safe_child(root, relative)

    def test_existing_output_rejected_before_dependency_or_input_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ["--manifest", "missing.json", "--baseline-checkpoint", "a.pt",
                    "--model-checkpoint", "b.pt", "--baseline-eval", "a.json",
                    "--model-eval", "b.json", "--case", "B006/Week0", "--output-dir", directory]
            with self.assertRaisesRegex(ValueError, "must be new"):
                diagnostic.main(args)

    def test_evaluation_digest_and_config_binding(self):
        config = {"model": "unet2d", "development_only": False}
        base = {"development_only": False, "manifest_sha256": "manifest",
                "checkpoint_sha256": "checkpoint", "model_config": config,
                "split": "validation", "amp": True,
                "cases": [{"patient": "B040", "visit": "Week0"}]}
        with tempfile.TemporaryDirectory() as directory:
            evaluation = Path(directory) / "evaluation.json"
            evaluation.write_text(json.dumps(base))
            refs, evidence = diagnostic.load_evaluations([evaluation], "unet2d", "checkpoint", "manifest", config, self.manifest)
            self.assertIn(("B040", "Week0"), refs)
            self.assertEqual(evidence[0]["sha256"], diagnostic.file_sha256(evaluation))
            for field, value in (("checkpoint_sha256", "wrong"), ("manifest_sha256", "wrong"),
                                 ("development_only", True), ("split", "train")):
                modified = copy.deepcopy(base)
                modified[field] = value
                evaluation.write_text(json.dumps(modified))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    diagnostic.load_evaluations([evaluation], "unet2d", "checkpoint", "manifest", config, self.manifest)


try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy not installed locally")
class ArrayFormulaTests(unittest.TestCase):
    def test_bincount_matches_hand_counted_confusion_and_differences(self):
        target = np.array([0, 0, 1, 1], dtype=np.uint8)
        saved = np.array([0, 1, 0, 1], dtype=np.uint8)
        direct = np.array([0, 0, 1, 1], dtype=np.uint8)
        old = diagnostic.independent_metrics(saved, target, 3)
        new = diagnostic.independent_metrics(direct, target, 3)
        self.assertEqual(old["confusion_rows_gt_columns_prediction"], [[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        comparison = diagnostic.compare_predictions(saved, direct, old, new)
        self.assertEqual(comparison["different_voxels"], 2)
        self.assertEqual(comparison["changed_from_counts"], [1, 1, 0])
        self.assertEqual(comparison["changed_into_counts"], [1, 1, 0])
        self.assertEqual(comparison["per_class_gt_dice_difference"][1]["after_minus_before"], 0.5)

    def test_invalid_arrays_rejected(self):
        target = np.zeros((2, 2), dtype=np.uint8)
        for bad in (np.zeros((3, 2), dtype=np.uint8), target.astype(float), target + 2):
            with self.assertRaises(ValueError):
                diagnostic.independent_metrics(bad, target, 2)


if __name__ == "__main__":
    unittest.main()
