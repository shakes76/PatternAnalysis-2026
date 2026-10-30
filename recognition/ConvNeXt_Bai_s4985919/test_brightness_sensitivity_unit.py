"""Synthetic tests only: python -B -m unittest -v test_brightness_sensitivity_unit"""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import torch
import test_brightness_sensitivity as exp


def tensor_transform(image):
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32).copy()
    return torch.from_numpy(rgb).permute(2, 0, 1) / 255


class FakeModel(torch.nn.Module):
    def forward(self, x):
        brightness = x.mean((1, 2, 3))
        return torch.stack([1 - brightness * 3, torch.zeros_like(brightness)], 1)


class BrightnessTests(unittest.TestCase):
    def test_mask_rounding_background_saturation_and_original(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pixels.png'
            pixels = np.array([[0, 5, 6, 100, 240, 255]], dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            tensors, audit = exp.inputs_for_image(path, tensor_transform)
            expected = tensor_transform(Image.open(path).convert('RGB'))
            self.assertTrue(torch.equal(tensors[0], expected))
            arrays = [np.rint(t[0].numpy() * 255).astype(int) for t in tensors]
            for arr in arrays:
                self.assertEqual(arr[0, :2].tolist(), [0, 5])
            self.assertEqual(arrays[1][0].tolist(), [0, 5, 5, 90, 216, 230])
            self.assertEqual(arrays[2][0].tolist(), [0, 5, 7, 110, 255, 255])
            self.assertAlmostEqual(audit[2]['clipped_fraction_all'], 2 / 6)
            self.assertEqual(audit[2]['clipped_fraction_foreground'], .5)
            self.assertEqual(audit[0]['clipped_fraction_all'], 0)
            rgb = np.zeros((4, 4, 3), dtype=np.uint8)
            rgb[:, :, 0] = 40
            Image.fromarray(rgb).save(path)
            with self.assertRaises(exp.DiagnosticError):
                exp.inputs_for_image(path, tensor_transform)

    def test_sampling_reproducibility_and_scan_position_coverage(self):
        rows = []
        for patient, scans in [('A', 1), ('B', 2), ('C', 5)]:
            for scan in range(scans):
                for position in range(20):
                    rows.append(dict(patient=patient, scan=patient + str(scan), position=position,
                        raw_slice_index=100 + position, label=0, path='not_used'))
        selected = exp.sample_patients(rows)
        self.assertEqual(selected, exp.sample_patients(list(reversed(rows))))
        self.assertEqual(len(selected), 12)
        for patient, n_scans in [('A', 1), ('B', 2), ('C', 4)]:
            rr = [r for r in selected if r['patient'] == patient]
            self.assertEqual(len({r['position'] // 5 for r in rr}), 4)
            self.assertEqual(len({r['scan'] for r in rr}), n_scans)

    def test_paired_metrics_and_patient_bootstrap_outputs(self):
        rows = []
        for i, label in enumerate([0, 0, 1, 1]):
            base = [.8, .65, .6, .2][i]
            rows.append(dict(patient='anon' + str(i), scan='anon_scan' + str(i), label=label,
                position=i, raw_slice_index=100+i,
                scores={'1.0': base, '0.9': base+.05, '1.1': base-.05},
                audit=[dict(factor=f, clipped_fraction_all=0., clipped_fraction_foreground=0.,
                    saturated_fraction_all=0.) for f in exp.FACTORS]))
        ss = exp.paired_summary(rows)
        self.assertAlmostEqual(ss['0.9']['mean_delta'], .05)
        self.assertEqual(ss['0.9']['flip_fraction'], .25)
        self.assertEqual(ss['1.0']['accuracy'], .75)
        self.assertEqual(ss['0.9']['accuracy'], 1.)
        self.assertIsNone(exp.paired_summary(rows[:2])['1.0']['auroc_ad'])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exp.report(rows, root, 10)
            summary = json.loads((root / 'summary.json').read_text())
            self.assertEqual(summary['all']['0.9']['patient_cluster_ci']['mean_delta']['valid'], 10)
            self.assertTrue((root / 'local_paired_predictions.csv').is_file())
            report = (root / 'report.md').read_text()
            self.assertNotIn('anon_scan', report)
            self.assertNotIn('anon0', report)

    def test_check_no_forward_and_analyze_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'PRIVATE_SCAN_42.jpeg'
            Image.fromarray(np.full((8, 8), 80, np.uint8)).save(path)
            rows = [dict(patient='anonymous' + str(i), scan='scan_token' + str(i), label=i,
                path=str(path), position=i, raw_slice_index=42+i) for i in (0, 1)]
            model = FakeModel()
            out = root / 'results'
            fake = (model, rows, tensor_transform, {'version': 1}, {'selected_slices': 2, 'patients': 2})
            argv = ['test_brightness_sensitivity.py', 'check', '--output-dir', str(out), '--bootstrap', '5']
            # Avoid changing the global interop setting repeatedly in a test process.
            with mock.patch('sys.argv', argv), mock.patch.object(exp, 'prepare', return_value=fake), mock.patch.object(torch, 'set_num_interop_threads'), mock.patch.object(model, 'forward', wraps=model.forward) as forward, contextlib.redirect_stdout(io.StringIO()):
                exp.main()
                forward.assert_not_called()
            argv[1] = 'analyze'
            with mock.patch('sys.argv', argv), mock.patch.object(exp, 'prepare', return_value=fake), mock.patch.object(torch, 'set_num_interop_threads'), contextlib.redirect_stdout(io.StringIO()):
                exp.main()
            self.assertTrue((out / 'complete.json').exists())
            for file in out.iterdir():
                if file.suffix in ('.json', '.csv', '.md'):
                    contents = file.read_text()
                    self.assertNotIn('PRIVATE_SCAN', contents)
                    self.assertNotIn(str(path), contents)


if __name__ == '__main__':
    unittest.main()
