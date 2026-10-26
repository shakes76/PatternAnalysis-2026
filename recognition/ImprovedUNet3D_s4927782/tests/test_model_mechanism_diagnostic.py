"""Synthetic-only contracts for the frozen mechanism diagnostic."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mechanism", PROJECT / "tools/diagnose_model_mechanism.py")
mechanism = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mechanism)

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None


def manifest():
    return {"records": [{"patient": key.split("/")[0], "visit": key.split("/")[1], "split": split}
                        for key, split in mechanism.ALLOWED_CASES.items()]}


class Guards(unittest.TestCase):
    def test_fixed_cases_and_test_ack(self):
        self.assertEqual(len(mechanism.select_cases(manifest(), list(mechanism.ALLOWED_CASES), [], True)), 14)
        with self.assertRaisesRegex(ValueError, "ack"):
            mechanism.select_cases(manifest(), ["M015/Week2"], [], False)
        with self.assertRaisesRegex(ValueError, "fixed fourteen"):
            mechanism.select_cases(manifest(), ["M020/Week0"], [], True)

    def test_exploratory_train_controls_are_forward_only(self):
        keys = ["R016/Week7", "T014/Week3"]
        selected = mechanism.select_cases(manifest(), keys, [], False)
        self.assertTrue(all(record["split"] == "train" for record in selected))
        self.assertEqual(set(keys), mechanism.EXPLORATORY_TRAIN_CASES)
        with self.assertRaisesRegex(ValueError, "three prespecified"):
            mechanism.select_cases(manifest(), keys, [keys[0]], False)

    def test_backward_is_explicit_selected_and_prespecified(self):
        with self.assertRaisesRegex(ValueError, "three prespecified"):
            mechanism.select_cases(manifest(), ["M015/Week1"], ["M015/Week1"], True)
        with self.assertRaisesRegex(ValueError, "also be"):
            mechanism.select_cases(manifest(), ["B006/Week0"], ["M015/Week2"], True)
        with self.assertRaisesRegex(ValueError, "unique"):
            mechanism.select_cases(manifest(), ["B006/Week0", "B006/Week0"], [], False)
        bad = manifest()
        bad["records"][0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "split differs"):
            mechanism.select_cases(bad, ["B006/Week0"], [], True)

    def test_documents_reject_wrong_sha_before_importing_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "protocol SHA"):
                mechanism.verify_documents({"protocol": path})

    def test_output_overwrite_refused_before_dependencies_or_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must be new"):
                mechanism.main(["--manifest", "/missing/manifest.json", "--output-dir", directory,
                                "--case", "B006/Week0"])


@unittest.skipIf(np is None, "NumPy required")
class LogitSummaries(unittest.TestCase):
    def test_rank_softmax_margin_and_errors_include_zeros(self):
        logits = np.array([[0, 0, 0, 0, 0, 2], [3, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1]], dtype=np.float32)
        result = mechanism.class5_summary(logits)
        self.assertEqual(result["class5_rank_counts"], {"1": 2, "2": 1, "3": 0, "4": 0, "5": 0, "6": 0})
        self.assertEqual(result["gt_class5_predicted_as_class_counts"], [2, 0, 0, 0, 0, 1])
        self.assertEqual(result["gt_class5_errors_predicted_as_class_counts"], [2, 0, 0, 0, 0, 0])
        self.assertEqual(result["class5_tied_with_max_competitor_voxels"], 1)
        self.assertEqual(result["class5_margin_over_max_competitor_quantiles"]["0"], -2)
        self.assertEqual(result["class5_margin_over_max_competitor_quantiles"]["0.5"], 0)
        self.assertAlmostEqual(sum(result["mean_softmax_probability_by_class"]), 1.)
        json.dumps(result, allow_nan=False)

    def test_nonfinite_rows_explicit_and_no_nan_json(self):
        result = mechanism.class5_summary(np.array([[np.nan, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]))
        self.assertEqual(result["nonfinite_gt_logit_rows"], 1)
        self.assertEqual(sum(result["gt_class5_predicted_as_class_counts"]), 1)
        json.dumps(result, allow_nan=False)

    def test_absent_gt_is_undefined_not_zero_probability(self):
        result = mechanism.class5_summary(np.empty((0, 6)))
        self.assertEqual(result["gt_class5_voxels"], 0)
        self.assertEqual(result["mean_softmax_probability_by_class"], [None] * 6)
        self.assertTrue(all(v is None for v in result["class5_margin_over_max_competitor_quantiles"].values()))


@unittest.skipIf(torch is None or np is None, "PyTorch and NumPy required")
class TorchMechanisms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        spec = importlib.util.spec_from_file_location("mechanism_core", PROJECT / "modules.py")
        cls.core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.core)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_independent_nearest_and_losses_match_core_for_odd_sizes(self):
        torch.manual_seed(12)
        target = torch.randint(6, (1, 9, 17, 19))
        logits = torch.randn(1, 6, 4, 8, 9)
        weights = torch.tensor([.1, .2, .4, 1., 2., 2.3])
        ce, dice, counts, resized = mechanism.independent_head_loss(logits, target, weights)
        expected_target = torch.nn.functional.interpolate(target[:, None].float(), size=logits.shape[2:], mode="nearest")[:, 0].long()
        self.assertTrue(torch.equal(resized, expected_target))
        self.assertEqual(sum(counts), 4 * 8 * 9)
        self.assertAlmostEqual(ce.item(), self.core._categorical_head_loss(logits, target, 1, 0, 1e-5, weights).item(), places=5)
        self.assertAlmostEqual(dice.item(), self.core._categorical_head_loss(logits, target, 0, 1, 1e-5, weights).item(), places=5)

    def test_real_graph_backward_keeps_state_and_captures_every_head(self):
        torch.manual_seed(2)
        model = self.core.ImprovedUNet3D(6, 2).eval()
        image = np.random.default_rng(2).normal(size=(8, 16, 16)).astype(np.float32)
        target = np.random.default_rng(3).integers(0, 6, size=image.shape, dtype=np.uint8)
        result = mechanism.diagnose_3d(model, image, target, torch.device("cpu"), torch.ones(6), True, self.core)
        self.assertTrue(result["parameters_and_buffers_unchanged"])
        self.assertTrue(result["loss"]["formula_matches_at_1e_minus5"])
        self.assertEqual(len(result["heads"]), 3)
        self.assertTrue(all(h["nearest_target_matches_torch"] for h in result["heads"]))
        self.assertTrue(result["backward"]["all_parameter_gradients_present"])
        self.assertTrue(result["backward"]["all_parameter_gradients_finite"])
        self.assertTrue(all(h["gradient_present"] and not h["all_zero"] for h in result["backward"]["head_logits"]))
        self.assertEqual(len(result["group_norm_layers"]), 14)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        json.dumps(result, allow_nan=False)

    def test_2d_chunks_summarize_whole_volume_without_updates(self):
        model = self.core.UNet2D(6, 2).eval()
        image = np.zeros((33, 8, 8), dtype=np.float32)
        target = np.full(image.shape, 5, dtype=np.uint8)
        result = mechanism.diagnose_2d(model, image, target, torch.device("cpu"))
        self.assertEqual(result["gt_class5"]["gt_class5_voxels"], target.size)
        self.assertTrue(result["parameters_and_buffers_unchanged"])
        self.assertEqual(result["logit_elements"], 6 * target.size)


if __name__ == "__main__":
    unittest.main()
