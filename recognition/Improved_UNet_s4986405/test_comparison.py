"""Check protocol constraints and orchestration without launching model training."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_comparison as runner


class ComparisonProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'snapshot'
        self.source.mkdir()
        self.output = self.root / 'experiment'
        self.protocol = runner.read_json(Path(__file__).with_name('comparison_protocol.json'))
        self.protocol['training']['epochs'] = 2
        self.protocol['data'] = {'root': str(self.root / 'data'), 'train_slices': 2,
                                 'validation_slices': 1, 'train_patients': 1, 'validation_patients': 1}
        for name in runner.RUNTIME_FILES:
            (self.source / name).write_text('# isolated source fixture\n', encoding='utf-8')
        self.save_protocol()
        for split, patient, count in (('train', 4, 2), ('validate', 36, 1)):
            for prefix, folder in (('case', f'keras_slices_{split}'), ('seg', f'keras_slices_seg_{split}')):
                directory = self.root / 'data' / folder
                directory.mkdir(parents=True)
                for index in range(count):
                    (directory / f'{prefix}_{patient:03d}_week_0_slice_{index}.nii.gz').write_bytes(b'filename/size fixture only')
        self.calls = []

    def save_protocol(self):
        (self.source / 'comparison_protocol.json').write_text(json.dumps(self.protocol), encoding='utf-8')

    def arguments(self, *extra):
        return ['--architecture', 'UNet2D', '--source-commit', 'a' * 40,
                '--output-dir', str(self.output), *extra]

    def invoke(self, *extra, child=None):
        printed = io.StringIO()
        with patch.object(runner, '__file__', str(self.source / 'run_comparison.py')):
            with patch.object(runner.subprocess, 'run', side_effect=child or self.fake_child) as launched:
                with contextlib.redirect_stdout(printed):
                    result = runner.main(self.arguments(*extra))
        return result, printed.getvalue(), launched.call_count

    @staticmethod
    def option(command, name):
        return command[command.index(name) + 1]

    def fake_child(self, command, **kwargs):
        self.assertTrue(kwargs['check'])
        self.assertEqual(kwargs['cwd'], self.source)
        self.calls.append(command)
        script = Path(command[2]).name
        if script == 'train.py':
            folder = Path(self.option(command, '--output-dir'))
            folder.mkdir()
            architecture = self.option(command, '--architecture')
            config = {'architecture': architecture, 'partial_run': False,
                      'arguments': copy.deepcopy(self.protocol['training']),
                      'model_config': {'in_channels': 1, 'num_classes': 6,
                                       'base_channels': self.protocol['training']['base_channels']},
                      'preprocessing': {'normalize': True, 'one_hot': True, 'train_flip_probability': 0.5},
                      'splits': {'train_patients': [4], 'validation_patients': [36]}}
            if architecture == 'ImprovedUNet2D':
                config['model_config']['dropout'] = 0.3
            history = [{'epoch': index, 'train': {'samples': 2}, 'validation': {'samples': 1}}
                       for index in range(1, self.protocol['training']['epochs'] + 1)]
            profile = {'architecture': architecture, 'partial_run': False,
                       'completed_epochs': self.protocol['training']['epochs'], 'device_name': 'stub GPU'}
            runner.write_json(folder / 'config.json', {'config': config})
            runner.write_json(folder / 'history.json', history)
            runner.write_json(folder / 'gpu_profile.json', profile)
            (folder / 'best.pt').write_bytes(b'stub checkpoint, no model training')
        elif script == 'evaluate.py':
            folder = Path(self.option(command, '--output-dir'))
            folder.mkdir()
            confusion = [[0] * 6 for _ in range(6)]
            confusion[0][0] = 1024
            runner.write_json(folder / 'evaluation.json', {
                'architecture': runner.read_json(self.output / 'training/config.json')['config']['architecture'],
                'split': 'validate', 'smoke_only': False,
                'processed_samples': 1, 'split_patient_ids': [36],
                'checkpoint_sha256': hashlib.sha256((self.output / 'training/best.pt').read_bytes()).hexdigest(),
                'checkpoint_epoch': 2, 'pooled': {'confusion': confusion},
                'patients': [{'patient_id': 36}], 'cases': [{'patient_id': 36, 'week': 0}],
                'median_model_forward_ms': 2 + int(folder.name.rsplit('_', 1)[1]),
                'latency_scope': 'synthetic orchestration fixture only',
            })
        elif script != 'check_checkpoint.py':
            raise AssertionError('Unexpected child script')
        return subprocess.CompletedProcess(command, 0)

    def test_invalid_protocols_are_rejected_before_launch(self):
        invalid = []
        for key, value in (('epochs', 0), ('batch_size', True), ('lr', float('nan')),
                           ('max_train_batches', 1), ('device', 'cpu'), ('seed', True)):
            protocol = copy.deepcopy(self.protocol)
            protocol['training'][key] = value
            invalid.append(protocol)
        for key, value in (('split', 'test'), ('max_samples', 2), ('warmup_steps', 0), ('repeats', 100)):
            protocol = copy.deepcopy(self.protocol)
            protocol['evaluation'][key] = value
            invalid.append(protocol)
        protocol = copy.deepcopy(self.protocol)
        protocol['resource_budget']['time_per_task'] = '24:00:00'
        invalid.append(protocol)
        for index, protocol in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                runner.validate_protocol(protocol)
        self.assertEqual(self.calls, [])

    def test_matched_full_split_plans_never_access_test(self):
        plans = [runner.build_plan(self.protocol, architecture, self.output / architecture,
                                   self.source, '/python') for architecture in runner.ARCHITECTURES]
        def options(command):
            result, position = {}, 3
            while position < len(command):
                name = command[position]
                if name == '--augment':
                    result[name] = True
                    position += 1
                else:
                    result[name] = command[position + 1]
                    position += 2
            return result
        left, right = options(plans[0][0][1]), options(plans[1][0][1])
        self.assertEqual(left.pop('--architecture'), 'UNet2D')
        self.assertEqual(right.pop('--architecture'), 'ImprovedUNet2D')
        left.pop('--output-dir')
        right.pop('--output-dir')
        self.assertEqual(left, right)
        self.assertEqual(left['--max-train-batches'], '0')
        self.assertEqual(left['--max-val-batches'], '0')
        for plan in plans:
            self.assertEqual(len(plan), 5)
            for stage, command in plan[2:]:
                self.assertEqual(self.option(command, '--split'), 'validate')
                self.assertEqual(self.option(command, '--max-samples'), '0')
                self.assertNotIn('--allow-partial', command)

    def test_dry_run_creates_no_outputs_or_child_processes(self):
        plan, printed, count = self.invoke('--dry-run')
        self.assertFalse(self.output.exists())
        self.assertEqual(count, 0)
        self.assertEqual(len(plan), 5)
        self.assertEqual(json.loads(printed)['architecture'], 'UNet2D')

    def test_complete_orchestration_records_provenance_and_three_trials(self):
        for architecture in runner.ARCHITECTURES:
            self.output = self.root / architecture
            result, printed, count = self.invoke('--architecture', architecture)
            self.assertEqual(count, 5)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['completed_epochs'], 2)
            self.assertEqual(result['validation_model_forward_median_ms_trials'], [3, 4, 5])
            self.assertEqual(result['median_of_trial_medians_ms'], 4)
            self.assertEqual(runner.read_json(self.output / 'experiment_provenance.json')['status'], 'completed')
            manifest = runner.read_json(self.output / 'data_manifest.json')
            self.assertEqual(manifest['splits']['train']['samples'], 2)
            self.assertEqual(manifest['splits']['validation']['patients'], [36])
            self.assertEqual(len(manifest['filename_size_manifest_sha256']), 64)
            self.assertNotIn('test', manifest['manifest'])
            self.assertIn('comparison_protocol.json', runner.read_json(self.output / 'experiment_provenance.json')['source_sha256'])
            self.assertIn('comparison task: PASS', printed)

    def test_source_mutation_stops_before_check_or_evaluation(self):
        def mutate(command, **kwargs):
            response = self.fake_child(command, **kwargs)
            (self.source / 'modules.py').write_text('# modified source\n', encoding='utf-8')
            return response
        with self.assertRaisesRegex(RuntimeError, 'snapshot changed'):
            self.invoke(child=mutate)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((self.output / 'comparison_result.json').exists())
        self.assertEqual(runner.read_json(self.output / 'experiment_provenance.json')['status'], 'failed')

    def test_incomplete_training_sample_counts_are_rejected(self):
        def incomplete(command, **kwargs):
            response = self.fake_child(command, **kwargs)
            if Path(command[2]).name == 'train.py':
                path = self.output / 'training/history.json'
                history = runner.read_json(path)
                history[-1]['validation']['samples'] = 0
                runner.write_json(path, history)
            return response
        with self.assertRaisesRegex(RuntimeError, 'did not process every sample'):
            self.invoke(child=incomplete)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((self.output / 'comparison_result.json').exists())

    def test_existing_experiment_directory_is_preserved(self):
        self.output.mkdir()
        marker = self.output / 'keep.txt'
        marker.write_text('existing result', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(marker.read_text(encoding='utf-8'), 'existing result')
        self.assertEqual(len(list(self.output.iterdir())), 1)
        self.assertEqual(self.calls, [])

    def test_preflight_rejects_missing_masks_and_patient_overlap(self):
        mask = self.root / 'data/keras_slices_seg_train/seg_004_week_0_slice_0.nii.gz'
        mask.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing paired mask'):
            runner.audit_data(self.protocol)
        mask.write_bytes(b'fixture restored')
        folder = self.root / 'data/keras_slices_validate'
        old = folder / 'case_036_week_0_slice_0.nii.gz'
        old.rename(folder / 'case_004_week_0_slice_0.nii.gz')
        mask_folder = self.root / 'data/keras_slices_seg_validate'
        (mask_folder / 'seg_036_week_0_slice_0.nii.gz').rename(mask_folder / 'seg_004_week_0_slice_0.nii.gz')
        with self.assertRaisesRegex(ValueError, 'patient overlap'):
            runner.audit_data(self.protocol)


if __name__ == '__main__':
    unittest.main()
