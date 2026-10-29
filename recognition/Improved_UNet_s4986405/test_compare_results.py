"""Audit comparison summaries using synthetic completed-run records, not trained models."""
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from compare_results import collect_comparison, main
import run_comparison as runner


def synthetic_metrics(architecture):
    matrix = [[0] * 6 for _ in range(6)]
    if architecture == 'UNet2D':
        matrix[0][:2], matrix[1][:2] = [4, 1], [2, 3]
        classes = [
            {'label_id': 0, 'target_pixels': 5, 'predicted_pixels': 6, 'tp': 4, 'fp': 2, 'fn': 1, 'dice': 8/11, 'iou': 4/7},
            {'label_id': 1, 'target_pixels': 5, 'predicted_pixels': 4, 'tp': 3, 'fp': 1, 'fn': 2, 'dice': 2/3, 'iou': 1/2},
        ]
    else:
        matrix[0][:2], matrix[1][:2] = [5, 0], [1, 4]
        classes = [
            {'label_id': 0, 'target_pixels': 5, 'predicted_pixels': 6, 'tp': 5, 'fp': 1, 'fn': 0, 'dice': 10/11, 'iou': 5/6},
            {'label_id': 1, 'target_pixels': 5, 'predicted_pixels': 4, 'tp': 4, 'fp': 0, 'fn': 1, 'dice': 8/9, 'iou': 4/5},
        ]
    for label in range(2, 6):
        classes.append({'label_id': label, 'target_pixels': 0, 'predicted_pixels': 0,
                        'tp': 0, 'fp': 0, 'fn': 0, 'dice': None, 'iou': None})
    return {'pixels': 10, 'confusion': matrix, 'classes': classes}


def create_synthetic_runs(root):
    """Small numerical fixtures for auditing/plot tests; no model training occurs."""
    root = Path(root)
    protocol = runner.read_json(Path(__file__).with_name('comparison_protocol.json'))
    protocol['training']['epochs'] = 2
    protocol['data'] = {'root': '/synthetic/data', 'train_slices': 2, 'validation_slices': 2,
                        'train_patients': 1, 'validation_patients': 1}
    manifest = {phase: [{'image': f'case_{patient}_week_0_slice_{index}.nii.gz',
                         'mask': f'seg_{patient}_week_0_slice_{index}.nii.gz',
                         'image_bytes': 10, 'mask_bytes': 10} for index in range(2)]
                for phase, patient in (('train', '004'), ('validation', '036'))}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    data = {'root': '/synthetic/data', 'splits': {'train': {'patients': [4], 'samples': 2},
                                                'validation': {'patients': [36], 'samples': 2}},
            'manifest': manifest, 'filename_size_manifest_sha256': digest,
            'manifest_scope': 'synthetic filenames and sizes only'}
    hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in runner.RUNTIME_FILES}
    directories = []
    for position, architecture in enumerate(runner.ARCHITECTURES):
        directory = root / architecture
        training = directory / 'training'
        training.mkdir(parents=True)
        pooled = synthetic_metrics(architecture)
        patient = {'patient_id': 36, 'samples': 2, 'processing_seconds': 0.2, **copy.deepcopy(pooled)}
        case = {'patient_id': 36, 'week': 0, 'samples': 2, 'processing_seconds': 0.2, **copy.deepcopy(pooled)}
        config = {'architecture': architecture, 'partial_run': False,
                  'arguments': copy.deepcopy(protocol['training']),
                  'model_config': {'in_channels': 1, 'num_classes': 6, 'base_channels': 16},
                  'preprocessing': {'normalize': True, 'one_hot': True, 'train_flip_probability': 0.5},
                  'splits': {'train_patients': [4], 'validation_patients': [36]},
                  'source_sha256': {'train.py': hashes['train.py'], 'modules.py': hashes['modules.py']}}
        if position:
            config['model_config']['dropout'] = 0.3
        history = [{'epoch': index, 'train': {'samples': 2, 'loss': 1.4 - index / 10 - position / 10},
                    'validation': {'samples': 2, 'loss': 1.5 - index / 10 - position / 10}}
                   for index in (1, 2)]
        profile = {'architecture': architecture, 'partial_run': False, 'completed_epochs': 2,
                   'peak_allocated_bytes': 134217728 * (position + 1),
                   'peak_reserved_bytes': 167772160 * (position + 1),
                   'device_name': 'Synthetic A100 fixture', 'node': 'fixture-' + str(position),
                   'run_wall_seconds_including_io_and_plot': 100 + position}
        provenance = {'status': 'completed', 'architecture': architecture, 'protocol_id': protocol['protocol_id'],
                      'source_commit': 'f' * 40, 'source_sha256': copy.deepcopy(hashes),
                      'filename_size_manifest_sha256': digest}
        runner.write_json(directory / 'protocol.json', protocol)
        runner.write_json(directory / 'data_manifest.json', data)
        runner.write_json(directory / 'experiment_provenance.json', provenance)
        runner.write_json(training / 'config.json', {'config': config})
        runner.write_json(training / 'history.json', history)
        runner.write_json(training / 'gpu_profile.json', profile)
        for index in range(1, 4):
            folder = directory / f'validation_{index}'
            folder.mkdir()
            report = {'architecture': architecture, 'split': 'validate', 'smoke_only': False,
                      'partial_checkpoint': False, 'limited_evaluation': False,
                      'processed_samples': 2, 'split_patient_ids': [36],
                      'checkpoint_epoch': 2, 'checkpoint_sha256': ('a' if position == 0 else 'b') * 64,
                      'pooled': pooled, 'patients': [patient], 'cases': [case],
                      'median_model_forward_ms': 2 + index + 2 * position,
                      'latency_scope': 'Synthetic numerical fixture only',
                      'device': 'cuda', 'batch_size': 1, 'warmup_steps_per_shape': 3}
            runner.write_json(folder / 'evaluation.json', report)
        result = {'status': 'completed', 'protocol_id': protocol['protocol_id'], 'source_commit': 'f' * 40,
                  'architecture': architecture, 'completed_epochs': 2, 'training_gpu_profile': profile,
                  **runner.summarize_validation(directory, protocol, architecture, data)}
        runner.write_json(directory / 'comparison_result.json', result)
        directories.append(directory)
    return directories


class ComparisonReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.baseline, self.improved = create_synthetic_runs(self.root)

    def arguments(self, destination, *extra):
        return ['--baseline-run', str(self.baseline), '--improved-run', str(self.improved),
                '--output-dir', str(destination), *extra]

    def test_hand_calculated_deltas_and_undefined_classes(self):
        report = collect_comparison(self.baseline, self.improved)
        row = report['per_class'][0]
        self.assertAlmostEqual(row['baseline_dice'], 8/11)
        self.assertAlmostEqual(row['improved_dice'], 10/11)
        self.assertAlmostEqual(row['dice_delta_improved_minus_baseline'], 2/11)
        self.assertIsNone(report['per_class'][5]['baseline_dice'])
        self.assertIsNone(report['per_class'][5]['dice_delta_improved_minus_baseline'])
        self.assertEqual(report['model_summary'][0]['median_of_forward_trial_medians_ms'], 4)
        self.assertEqual(report['model_summary'][1]['median_of_forward_trial_medians_ms'], 6)
        self.assertEqual(report['model_summary'][1]['peak_allocated_gib'], 0.25)
        self.assertEqual(len(report['per_patient']), 6)
        self.assertEqual(len(report['case_times']), 6)

    def test_export_keeps_validation_scope_and_null_cells(self):
        destination = self.root / 'report'
        with contextlib.redirect_stdout(io.StringIO()):
            main(self.arguments(destination, '--skip-plots'))
        result = json.loads((destination / 'comparison.json').read_text())
        self.assertIn('Validation', result['scope'])
        self.assertNotIn('NaN', (destination / 'comparison.json').read_text())
        with (destination / 'per_class_comparison.csv').open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[5]['baseline_dice'], '')
        self.assertEqual(rows[5]['dice_delta_improved_minus_baseline'], '')
        self.assertEqual(rows[0]['scope'], 'validation_only_numeric_labels')
        self.assertIn('undefined', (destination / 'comparison.md').read_text())

    def test_unfinished_or_partial_runs_are_rejected(self):
        path = self.improved / 'experiment_provenance.json'
        provenance = runner.read_json(path)
        provenance['status'] = 'running_training'
        runner.write_json(path, provenance)
        with self.assertRaisesRegex(ValueError, 'has not completed'):
            collect_comparison(self.baseline, self.improved)
        provenance['status'] = 'completed'
        runner.write_json(path, provenance)
        path = self.improved / 'training/config.json'
        config = runner.read_json(path)
        config['config']['partial_run'] = True
        runner.write_json(path, config)
        with self.assertRaisesRegex(RuntimeError, 'partial-run'):
            collect_comparison(self.baseline, self.improved)

    def test_different_source_hashes_are_rejected(self):
        path = self.improved / 'experiment_provenance.json'
        value = runner.read_json(path)
        value['source_sha256']['evaluate.py'] = 'a' * 64
        runner.write_json(path, value)
        with self.assertRaisesRegex(ValueError, 'source_sha256 differs'):
            collect_comparison(self.baseline, self.improved)

    def test_corrupted_data_manifest_and_label_scores_are_rejected(self):
        path = self.improved / 'data_manifest.json'
        original = runner.read_json(path)
        changed = copy.deepcopy(original)
        changed['manifest']['train'][0]['image_bytes'] += 1
        runner.write_json(path, changed)
        with self.assertRaisesRegex(ValueError, 'manifest digest mismatch'):
            collect_comparison(self.baseline, self.improved)
        runner.write_json(path, original)
        for index in range(1, 4):
            path = self.improved / f'validation_{index}/evaluation.json'
            value = runner.read_json(path)
            value['pooled']['classes'][0]['dice'] = 0.123
            runner.write_json(path, value)
        path = self.improved / 'comparison_result.json'
        cached = runner.read_json(path)
        cached['pooled']['classes'][0]['dice'] = 0.123
        runner.write_json(path, cached)
        with self.assertRaisesRegex(ValueError, 'overlap metric differs'):
            collect_comparison(self.baseline, self.improved)

    def test_mismatched_timing_configuration_is_rejected(self):
        path = self.improved / 'validation_1/evaluation.json'
        value = runner.read_json(path)
        value['device'] = 'cpu'
        runner.write_json(path, value)
        with self.assertRaisesRegex(ValueError, 'timing configuration'):
            collect_comparison(self.baseline, self.improved)

    def test_existing_report_is_preserved(self):
        destination = self.root / 'existing'
        destination.mkdir()
        marker = destination / 'keep.txt'
        marker.write_text('existing result')
        with self.assertRaises(FileExistsError):
            main(self.arguments(destination))
        self.assertEqual(marker.read_text(), 'existing result')
        self.assertEqual(len(list(destination.iterdir())), 1)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib unavailable on this test host')
    def test_three_png_figures_render(self):
        destination = self.root / 'plots'
        with contextlib.redirect_stdout(io.StringIO()):
            main(self.arguments(destination))
        for name in ('training_loss_comparison.png', 'validation_metrics_comparison.png', 'resource_comparison.png'):
            data = (destination / name).read_bytes()
            self.assertTrue(data.startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertGreater(len(data), 1000)


if __name__ == '__main__':
    unittest.main()
