"""Check timing coverage and real preprocessing without requiring a GPU."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_cases import benchmark, read_input
from dataset import HipMRIDataset
from modules import build_model
import test_evaluate_cases as fixtures


class CaseBenchmarkTests(unittest.TestCase):
    def test_image_only_preprocessing_matches_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, manifest = fixtures.CaseEvaluationTests().fixture(root, split="test")
            image = np.arange(32 * 16, dtype=np.float32).reshape(32, 16)
            nib.save(nib.Nifti1Image(image, np.eye(4)), root / "image0.nii.gz")
            for mode in ("pad", "stretch"):
                data = HipMRIDataset(manifest, "test", 32, [0, 5], resize_mode=mode)
                actual = read_input(str(root / "image0.nii.gz"), {"image_size": 32, "resize_mode": mode})
                self.assertTrue(torch.equal(actual, data[0]["image"]))

    def test_grouping_partial_batches_repetitions_and_no_labels(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoint, manifest = fixtures.CaseEvaluationTests().fixture(root, split="test")
                payload = torch.load(checkpoint, weights_only=True)
                payload["config"]["model"] = "improved_unet"
                model = build_model("improved_unet", num_classes=2, base_channels=2)
                for parameter in model.parameters():
                    parameter.data.zero_()
                payload["model_state"] = model.state_dict()
                other = root / "improved.pt"
                torch.save(payload, other)
                for mask in root.glob("mask*.nii.gz"):
                    mask.unlink()  # Timed inference must not load targets.
                original = checkpoint.read_bytes()
                result = benchmark([checkpoint, other], manifest, root/"out", "cpu", 3, 2, 1)
                self.assertEqual((result["case_count"], result["slice_count"]), (2, 3))
                self.assertEqual([x["slices"] for x in result["cases"]], [2, 1])
                self.assertEqual([x['model'] for x in result['execution_order'][:4]],
                                 ['unet', 'improved_unet', 'improved_unet', 'unet'])
                for case in result["cases"]:
                    expected = hashlib.sha256(torch.zeros((case['slices'], 32, 32), dtype=torch.int64).numpy().tobytes()).hexdigest()
                    for entry in case['models'].values():
                        self.assertEqual(len(entry['trials']), 2)
                        self.assertTrue(entry['predictions_identical_across_repeats'])
                        for trial in entry['trials']:
                            self.assertEqual(trial['prediction_sha256'], expected)
                            self.assertGreaterEqual(trial['pipeline_seconds'], trial['forward_seconds'])
                            self.assertGreater(trial['forward_seconds'], 0)
                self.assertEqual(checkpoint.read_bytes(), original)
                self.assertNotIn('private-A', json.dumps(result))
                self.assertEqual(json.loads((root/'out/summary.json').read_text()), result)
        finally:
            torch.set_num_threads(previous)

    def test_mismatched_manifest_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, manifest = fixtures.CaseEvaluationTests().fixture(root)
            with self.assertRaises(FileExistsError):
                benchmark([checkpoint], manifest, root, device="cpu", split="val")
            manifest.write_bytes(manifest.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'manifest_sha256'):
                benchmark([checkpoint], manifest, root/'out', device="cpu", split="val")
            self.assertFalse((root/'out').exists())

    def test_invalid_arguments_fail_before_io(self):
        for overrides in ({'batch_size': 0}, {'repeats': 0}, {'warmup': -1}, {'split': 'train'}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                benchmark([Path('missing')], Path('missing'), Path('missing'), **overrides)


if __name__ == '__main__':
    unittest.main()
