"""Light synthetic tests for one-factor support removal; no real data or weights."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
SPEC = importlib.util.spec_from_file_location("support_d2", PROJECT / "tools/diagnose_support_intervention.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)
try:
    import nibabel as nib
    from scipy.ndimage import zoom
except ImportError:
    nib = zoom = None
try:
    import torch
except ImportError:
    torch = None


class SingleFactorTests(unittest.TestCase):
    def test_zeroing_reconstructs_baseline_and_only_removed_support_changes(self):
        before = np.arange(8, dtype=np.float32).reshape(2, 2, 2) - 3
        support = np.ones(before.shape, dtype=bool)
        support[0, 0, 0] = support[1, 1, 1] = False
        baseline = before.copy(); baseline[~support] = 0
        target = np.zeros(before.shape, dtype=np.uint8); target[0, 0, 0] = 5
        changed, summary = tool.prepare_intervention(before, support, baseline, target)
        np.testing.assert_array_equal(changed, before)
        self.assertTrue(summary["baseline_reconstruction_elementwise_equal"])
        self.assertEqual(summary["different_voxels"], 2)
        self.assertEqual(summary["gt_class5_affected_voxels"], 1)
        self.assertEqual(summary["changed_voxels_inside_retained_support"], 0)
        self.assertEqual(summary["bbox_dhw_half_open"], [[0, 2], [0, 2], [0, 2]])
        self.assertEqual(baseline[0, 0, 0], 0)

    def test_baseline_mismatch_refused_before_any_model_forward(self):
        before = np.ones((2, 2, 2), dtype=np.float32)
        baseline = before.copy(); baseline[0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "elementwise equal"):
            tool.prepare_intervention(before, np.ones_like(before, dtype=bool), baseline, np.zeros_like(before, dtype=np.uint8))

    def test_all_fourteen_cases_required_and_other_test_cases_forbidden(self):
        manifest = {"records": [{"patient": k.split('/')[0], "visit": k.split('/')[1], "split": s}
                                for k, s in tool.ALLOWED_CASES.items()]}
        self.assertEqual(len(tool.complete_cases(manifest, None, True)), 14)
        with self.assertRaisesRegex(ValueError, "ack"):
            tool.complete_cases(manifest, None, False)
        with self.assertRaisesRegex(ValueError, "14 allowed"):
            tool.complete_cases(manifest, ["M015/Week2"], True)
        names = list(tool.ALLOWED_CASES); names[-1] = "M020/Week0"
        with self.assertRaisesRegex(ValueError, "14 allowed"):
            tool.complete_cases(manifest, names, True)

    def test_class5_counts_and_empty_change_are_explicit(self):
        target = np.array([5, 5, 0, 0], dtype=np.uint8).reshape(1, 2, 2)
        prediction = np.array([5, 0, 5, 0], dtype=np.uint8).reshape(1, 2, 2)
        self.assertEqual(tool.target_metrics(prediction, target),
                         {"tp": 1, "fp": 1, "fn": 1, "gt_voxels": 2, "predicted_voxels": 2, "dice": .5})
        self.assertIsNone(tool.change_summary(prediction, prediction, target)["bbox_dhw_half_open"])


@unittest.skipIf(nib is None or zoom is None, "NiBabel/SciPy unavailable; native reconstruction fixture skipped")
class NativeReconstructionTests(unittest.TestCase):
    def test_synthetic_nonras_source_reconstructs_frozen_support_zeroing(self):
        data = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        data[0, :, :] = 0
        affine = np.diag([-2., -3., 4., 1.])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.nii.gz"
            nib.save(nib.Nifti1Image(data, affine), path)
            volume = nib.load(path); canonical = nib.as_closest_canonical(volume)
            record = {"original_image": {"shape_xyz": list(volume.shape), "affine_xyz": volume.affine.tolist()},
                      "canonical_image": {"affine_xyz": canonical.affine.tolist()}}
            before, support, _ = tool.reconstruct_before_support(path, record, (2, 2, 2), nib, zoom)
            normalized, active, _ = tool.normalize_nonzero(canonical.get_fdata(dtype=np.float32))
            expected = zoom(normalized, .5, order=1, grid_mode=True, mode="nearest", prefilter=False).transpose(2, 1, 0)
            expected_support = zoom(active.astype(np.uint8), .5, order=0, grid_mode=True, mode="nearest", prefilter=False).transpose(2, 1, 0).astype(bool)
            expected[~expected_support] = 0
            _, summary = tool.prepare_intervention(before, support, expected, np.zeros((2, 2, 2), dtype=np.uint8))
            self.assertTrue(summary["baseline_reconstruction_elementwise_equal"])


@unittest.skipIf(torch is None, "PyTorch unavailable; synthetic forward fixture skipped")
class ForwardOnlyTests(unittest.TestCase):
    def test_small_2d_forward_does_not_update_state_or_create_gradients(self):
        import modules
        model = modules.UNet2D(6, 2).eval()
        before = tool.state_sha256(model)
        image = np.zeros((2, 8, 8), dtype=np.float32)
        target = np.full(image.shape, 5, dtype=np.uint8)
        prediction, summary = tool.forward_summary(model, image, target, "unet2d", torch.device("cpu"))
        self.assertEqual(prediction.shape, image.shape)
        self.assertEqual(summary["gt_class5_logits"]["gt_class5_voxels"], target.size)
        self.assertTrue(summary["all_main_logits_finite"])
        self.assertEqual(before, tool.state_sha256(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
