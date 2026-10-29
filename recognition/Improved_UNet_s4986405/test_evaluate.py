"""Verify pooled/patient metrics, split guards, timing scopes and export isolation."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from torch import nn

from evaluate import evaluate_cases, main, parse_case_name
from modules import UNet2D


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(3710)
        self.add_slice('validate', 36, 0, 0, (32, 32), 0)
        self.add_slice('validate', 36, 0, 1, (32, 48), 2)
        self.add_slice('validate', 37, 1, 0, (32, 32), 1)
        self.add_slice('test', 40, 0, 0, (32, 32), 2)
        model = UNet2D(base_channels=2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.out.bias[2] = 3
        self.saved = {
            'schema_version': 1, 'epoch': 1, 'architecture': 'UNet2D',
            'model_config': {'in_channels': 1, 'num_classes': 6, 'base_channels': 2},
            'model_state': model.state_dict(),
            'config': {'partial_run': False, 'preprocessing': {'normalize': False},
                       'splits': {'train_patients': [4], 'validation_patients': [36, 37], 'validation_slices': 3}},
        }
        self.checkpoint = self.root / 'best.pt'
        self.save()

    def save(self):
        torch.save(self.saved, self.checkpoint)

    def add_slice(self, split, patient, week, slice_id, shape, label):
        images = self.data / f'keras_slices_{split}'
        masks = self.data / f'keras_slices_seg_{split}'
        images.mkdir(parents=True, exist_ok=True)
        masks.mkdir(parents=True, exist_ok=True)
        suffix = f'{patient:03d}_week_{week}_slice_{slice_id}.nii.gz'
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        labels = np.full(shape, label, dtype=np.int16)
        nib.save(nib.Nifti1Image(values, np.eye(4)), images / ('case_' + suffix))
        nib.save(nib.Nifti1Image(labels, np.eye(4)), masks / ('seg_' + suffix))

    def arguments(self, name='evaluation', *extra):
        return ['--checkpoint', str(self.checkpoint), '--data-root', str(self.data),
                '--output-dir', str(self.root / name), '--warmup-steps', '1', *extra]

    def run_evaluation(self, name='evaluation', *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(self.arguments(name, *extra))

    def test_full_validation_pools_pixels_by_patient_and_case(self):
        report = self.run_evaluation()
        self.assertEqual(report['processed_samples'], 3)
        self.assertEqual(report['pooled']['pixels'], 3584)
        self.assertFalse(report['smoke_only'])
        self.assertEqual(report['processed_patient_ids'], [36, 37])
        expected = np.zeros((6, 6), dtype=np.int64)
        expected[0, 2], expected[2, 2], expected[1, 2] = 1024, 1536, 1024
        np.testing.assert_array_equal(report['pooled']['confusion'], expected)
        self.assertAlmostEqual(report['pooled']['classes'][2]['dice'], 0.6)
        self.assertAlmostEqual(report['pooled']['classes'][2]['iou'], 3 / 7)
        self.assertAlmostEqual(report['patients'][0]['classes'][2]['dice'], 0.75)
        self.assertAlmostEqual(report['patients'][0]['classes'][2]['iou'], 0.6)
        self.assertEqual(report['patients'][1]['classes'][2]['dice'], 0)
        self.assertEqual(len(report['cases']), 2)
        self.assertEqual(report['cases'][1]['week'], 1)
        self.assertEqual(sum(entry['pixels'] for entry in report['patients']), 3584)
        self.assertGreaterEqual(report['median_model_forward_ms'], 0)
        self.assertTrue(all(entry['processing_seconds'] >= 0 for entry in report['cases']))
        self.assertEqual(report['warmup_shapes'], [[32, 32], [32, 48]])
        self.assertEqual(len(report['checkpoint_sha256']), 64)
        self.assertIn('evaluate.py', report['evaluation_source_sha256'])
        output = self.root / 'evaluation'
        written = json.loads((output / 'evaluation.json').read_text(encoding='utf-8'))
        self.assertIsNone(written['pooled']['classes'][5]['dice'])
        self.assertNotIn('NaN', (output / 'evaluation.json').read_text(encoding='utf-8'))
        with (output / 'pooled_metrics.csv').open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[5]['dice'], '')
        self.assertEqual(rows[5]['iou'], '')
        self.assertEqual(rows[2]['tp'], '1536')
        self.assertEqual(rows[2]['smoke_only'], 'False')
        self.assertEqual(rows[2]['architecture'], 'UNet2D')
        for name in ('patients_metrics.csv', 'cases_metrics.csv', 'slice_metrics.csv', 'case_times.csv'):
            self.assertTrue((output / name).is_file())

    def test_limited_validation_is_explicit_and_counts_only_selected_samples(self):
        report = self.run_evaluation('limited', '--max-samples', '1')
        self.assertTrue(report['smoke_only'])
        self.assertTrue(report['limited_evaluation'])
        self.assertFalse(report['partial_checkpoint'])
        self.assertEqual(report['processed_samples'], 1)
        self.assertEqual(report['split_samples'], 3)
        self.assertEqual(report['processed_patient_ids'], [36])
        self.assertEqual(report['split_patient_ids'], [36, 37])
        self.assertEqual(report['pooled']['pixels'], 1024)

    def test_partial_checkpoint_guard_and_full_test_scope(self):
        self.saved['config']['partial_run'] = True
        self.save()
        with self.assertRaisesRegex(ValueError, 'Partial checkpoint'):
            self.run_evaluation('refused')
        report = self.run_evaluation('allowed', '--allow-partial')
        self.assertTrue(report['partial_checkpoint'])
        self.assertTrue(report['smoke_only'])
        with self.assertRaisesRegex(ValueError, 'refuses partial'):
            self.run_evaluation('partial_test', '--allow-partial', '--split', 'test')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_evaluation('limited_test', '--split', 'test', '--max-samples', '1')

    def test_training_overlap_and_validation_mismatch_rejected(self):
        original = copy.deepcopy(self.saved)
        invalid_splits = [
            {'train_patients': [36], 'validation_patients': [37]},
            {'train_patients': [4], 'validation_patients': [36]},
            {'train_patients': [4], 'validation_patients': [4, 36, 37]},
            {'validation_patients': [36, 37]},
            {'train_patients': [True], 'validation_patients': [36, 37]},
            {'train_patients': [4], 'validation_patients': [36, 37], 'validation_slices': 2},
        ]
        for index, splits in enumerate(invalid_splits):
            self.saved = copy.deepcopy(original)
            self.saved['config']['splits'] = splits
            self.save()
            with self.subTest(splits=splits), self.assertRaises(ValueError):
                self.run_evaluation(f'bad_split_{index}')
            self.assertFalse((self.root / f'bad_split_{index}').exists())

    def test_disjoint_full_test_fixture_and_validation_overlap_guard(self):
        report = self.run_evaluation('test_fixture', '--split', 'test')
        self.assertEqual(report['split'], 'test')
        self.assertFalse(report['smoke_only'])
        self.assertEqual(report['processed_patient_ids'], [40])
        self.assertEqual(report['pooled']['classes'][2]['dice'], 1.0)
        self.saved['config']['splits']['validation_patients'].append(40)
        self.save()
        with self.assertRaisesRegex(ValueError, 'overlap checkpoint validation'):
            self.run_evaluation('overlap_test', '--split', 'test')

    def test_checkpoint_change_during_evaluation_is_rejected(self):
        def changed_checkpoint(*args, **kwargs):
            result = evaluate_cases(*args, **kwargs)
            with self.checkpoint.open('ab') as handle:
                handle.write(b'changed during evaluation')
            return result
        with patch('evaluate.evaluate_cases', side_effect=changed_checkpoint):
            with self.assertRaisesRegex(RuntimeError, 'Checkpoint changed'):
                self.run_evaluation('changed_checkpoint')
        self.assertFalse((self.root / 'changed_checkpoint').exists())

    def test_existing_output_directory_is_preserved(self):
        output = self.root / 'existing'
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_text('existing result', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            self.run_evaluation('existing')
        self.assertEqual(marker.read_text(encoding='utf-8'), 'existing result')
        self.assertEqual(len(list(output.iterdir())), 1)

    def test_bad_names_and_nonfinite_model_output_rejected(self):
        self.assertEqual(parse_case_name('case_036_week_2_slice_17.nii.gz'), (36, 2, 17))
        for name in ('case_check.nii.gz', 'seg_036_week_0_slice_0.nii.gz', 'case_036_week_0_slice_0.nii'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_case_name(name)
        class NonfiniteModel(nn.Module):
            def forward(self, inputs):
                return torch.full((1, 6, *inputs.shape[-2:]), float('nan'))
        with patch('evaluate.load_predictor', return_value=(NonfiniteModel(), self.saved)):
            with self.assertRaisesRegex(RuntimeError, 'Invalid evaluation logits'):
                self.run_evaluation('nonfinite')
        self.assertFalse((self.root / 'nonfinite').exists())
        images = self.data / 'keras_slices_validate'
        masks = self.data / 'keras_slices_seg_validate'
        (images / 'case_036_week_00_slice_0.nii.gz').write_bytes(
            (images / 'case_036_week_0_slice_0.nii.gz').read_bytes())
        (masks / 'seg_036_week_00_slice_0.nii.gz').write_bytes(
            (masks / 'seg_036_week_0_slice_0.nii.gz').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Duplicate patient/week/slice'):
            self.run_evaluation('duplicate_slice')


if __name__ == '__main__':
    unittest.main()
