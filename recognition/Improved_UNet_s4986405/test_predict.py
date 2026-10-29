"""Check checkpoint metadata, label errors and NIfTI/PNG prediction exports."""
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

from modules import UNet2D
from predict import infer_mask, load_predictor, main, overlap_map


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.masks = self.root / 'masks'
        self.images.mkdir()
        self.masks.mkdir()
        self.values = np.arange(256, dtype=np.float32).reshape(16, 16)
        self.labels = (np.arange(256).reshape(16, 16) % 6).astype(np.int16)
        self.affine = np.diag([2., 3., 1., 1.])
        nib.save(nib.Nifti1Image(self.values, self.affine), str(self.images / 'case_check.nii.gz'))
        nib.save(nib.Nifti1Image(self.labels, self.affine), str(self.masks / 'seg_check.nii.gz'))
        model = UNet2D(base_channels=4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.out.bias[2] = 3
        self.checkpoint = self.root / 'best.pt'
        self.saved = {
            'schema_version': 1, 'epoch': 2, 'architecture': 'UNet2D',
            'model_config': {'in_channels': 1, 'num_classes': 6, 'base_channels': 4},
            'model_state': model.state_dict(),
            'config': {'partial_run': True, 'preprocessing': {'normalize': True}},
        }
        torch.save(self.saved, self.checkpoint)

    def arguments(self, output):
        return ['--checkpoint', str(self.checkpoint), '--image-dir', str(self.images),
                '--mask-dir', str(self.masks), '--output-dir', str(output),
                '--label-id', '1', '--allow-partial']

    def test_partial_and_metadata_guards(self):
        with self.assertRaisesRegex(ValueError, 'Partial checkpoint'):
            load_predictor(self.checkpoint)
        self.saved['config']['partial_run'] = False
        self.saved['config']['preprocessing']['normalize'] = 'unknown'
        torch.save(self.saved, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'normalization'):
            load_predictor(self.checkpoint)

    def test_inference_uses_saved_width_and_has_no_gradients(self):
        model, saved = load_predictor(self.checkpoint, allow_partial=True)
        prediction = infer_mask(model, torch.tensor(self.values).unsqueeze(0))
        self.assertEqual(saved['model_config']['base_channels'], 4)
        self.assertEqual(tuple(prediction.shape), (16, 16))
        self.assertEqual(prediction.dtype, torch.long)
        self.assertTrue(torch.all(prediction == 2).item())
        self.assertFalse(model.training)
        self.assertFalse(prediction.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_tp_fp_fn_map_matches_hand_example(self):
        target = torch.tensor([[1, 0, 1], [0, 1, 0]])
        prediction = torch.tensor([[1, 1, 0], [0, 1, 1]])
        expected = torch.tensor([[1, 2, 3], [0, 1, 2]])
        self.assertTrue(torch.equal(overlap_map(prediction, target, 1), expected))
        with self.assertRaises(ValueError):
            overlap_map(prediction, target, 6)

    def test_export_preserves_shape_affine_and_numeric_scores(self):
        output = self.root / 'prediction'
        with contextlib.redirect_stdout(io.StringIO()):
            main(self.arguments(output))
        self.assertTrue((output / 'prediction.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
        nifti = nib.load(str(output / 'prediction.nii.gz'))
        self.assertEqual(nifti.shape, (16, 16))
        self.assertEqual(nifti.get_data_dtype(), np.dtype('uint8'))
        np.testing.assert_array_equal(nifti.affine, self.affine)
        np.testing.assert_array_equal(nifti.get_fdata(), np.full((16, 16), 2))
        report = json.loads((output / 'prediction.json').read_text(encoding='utf-8'))
        self.assertTrue(report['partial_checkpoint'])
        self.assertEqual(report['normalization'], 'per-slice z-score')
        self.assertEqual(sum(map(sum, report['confusion'])), 256)
        count = int(np.count_nonzero(self.labels == 2))
        self.assertAlmostEqual(report['dice'][2], 2 * count / (256 + count))
        self.assertEqual(report['selected_label_counts'], {'tp': 0, 'fp': 0, 'fn': int(np.count_nonzero(self.labels == 1))})

    def test_existing_prediction_directory_is_preserved(self):
        output = self.root / 'existing'
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_text('existing result', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            main(self.arguments(output))
        self.assertEqual(marker.read_text(encoding='utf-8'), 'existing result')
        self.assertEqual(len(list(output.iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
