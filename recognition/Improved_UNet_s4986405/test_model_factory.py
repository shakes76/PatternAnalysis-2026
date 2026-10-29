"""Exercise architecture selection, old checkpoints and both complete CLI paths."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

from check_checkpoint import main as check_run
from model_factory import build_model, model_from_checkpoint
from modules import ImprovedUNet2D, UNet2D
from predict import infer_mask, load_predictor, main as predict_main
from train import main as train_main
from training_io import load_checkpoint


class ModelFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(3710)
        self.model_config = {'in_channels': 1, 'num_classes': 6, 'base_channels': 2}

    def test_builds_both_architectures_with_configured_width_and_dropout(self):
        for architecture, model_type in (('UNet2D', UNet2D), ('ImprovedUNet2D', ImprovedUNet2D)):
            with self.subTest(architecture=architecture):
                config = dict(self.model_config)
                if architecture == 'ImprovedUNet2D':
                    config['dropout'] = 0.17
                model = build_model(architecture, config).eval()
                self.assertIsInstance(model, model_type)
                with torch.no_grad():
                    output = model(torch.randn(1, 1, 33, 49))
                self.assertEqual(tuple(output.shape), (1, 6, 33, 49))
                self.assertTrue(torch.isfinite(output).all().item())
                if architecture == 'ImprovedUNet2D':
                    self.assertEqual(model.context0.body[1].p, 0.17)

    def test_invalid_architecture_and_configuration_are_rejected(self):
        for architecture in ('UnknownModel', None, []):
            with self.subTest(architecture=architecture), self.assertRaises(ValueError):
                build_model(architecture, self.model_config)
        invalid = [None, {}, {**self.model_config, 'base_channels': True},
                   {**self.model_config, 'in_channels': 0},
                   {**self.model_config, 'unexpected': 1},
                   {**self.model_config, 'dropout': 0.3}]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_model('UNet2D', config)
        for dropout in (float('nan'), float('inf'), -0.1, 1.0, True):
            with self.subTest(dropout=dropout), self.assertRaises(ValueError):
                build_model('ImprovedUNet2D', {**self.model_config, 'dropout': dropout})

    def test_inconsistent_metadata_and_incompatible_weights_are_rejected(self):
        model = build_model('UNet2D', self.model_config)
        saved = {'architecture': 'UNet2D', 'model_config': self.model_config,
                 'model_state': model.state_dict(),
                 'config': {'architecture': 'UNet2D', 'model_config': dict(self.model_config)}}
        mismatch = copy.deepcopy(saved)
        mismatch['config']['architecture'] = 'ImprovedUNet2D'
        with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
            model_from_checkpoint(mismatch)
        mismatch = copy.deepcopy(saved)
        mismatch['config']['model_config']['base_channels'] = 3
        with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
            model_from_checkpoint(mismatch)
        wrong_weights = dict(saved, architecture='ImprovedUNet2D', config={})
        with self.assertRaises(RuntimeError):
            model_from_checkpoint(wrong_weights)
        missing = dict(saved)
        missing.pop('architecture')
        with self.assertRaisesRegex(ValueError, 'missing architecture'):
            model_from_checkpoint(missing)

    def test_legacy_baseline_prediction_uses_explicit_top_level_metadata(self):
        model = build_model('UNet2D', self.model_config).eval()
        saved = {'schema_version': 1, 'epoch': 1, 'architecture': 'UNet2D',
                 'model_config': self.model_config, 'model_state': model.state_dict(),
                 'config': {'partial_run': True, 'preprocessing': {'normalize': True}}}
        path = self.root / 'legacy.pt'
        torch.save(saved, path)
        with self.assertRaisesRegex(ValueError, 'Partial checkpoint'):
            load_predictor(path)
        restored, observed = load_predictor(path, allow_partial=True)
        self.assertIsInstance(restored, UNet2D)
        self.assertEqual(observed['model_config']['base_channels'], 2)
        images = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(restored(images), model(images), rtol=0, atol=0)
        prediction = infer_mask(restored, images[0])
        self.assertFalse(prediction.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in restored.parameters()))

    def run_cli_round_trip(self, architecture):
        data = self.root / 'data'
        rng = np.random.default_rng(3710)
        for split, patient in (('train', 4), ('validate', 36)):
            images = data / f'keras_slices_{split}'
            masks = data / f'keras_slices_seg_{split}'
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            for index, shape in enumerate(((32, 32), (32, 48))):
                suffix = f'{patient:03d}_week_0_slice_{index}.nii.gz'
                values = rng.normal(size=shape).astype(np.float32)
                labels = (np.arange(np.prod(shape)).reshape(shape) % 6).astype(np.int16)
                affine = np.diag([2., 3., 1., 1.])
                nib.save(nib.Nifti1Image(values, affine), images / ('case_' + suffix))
                nib.save(nib.Nifti1Image(labels, affine), masks / ('seg_' + suffix))
        run = self.root / 'run'
        arguments = ['--data-root', str(data), '--output-dir', str(run), '--epochs', '1',
                     '--batch-size', '2', '--base-channels', '2', '--device', 'cpu',
                     '--augment', '--max-train-batches', '2', '--max-val-batches', '2']
        # Omitting the flag must retain the old baseline default.
        if architecture != 'UNet2D':
            arguments += ['--architecture', architecture]
        with contextlib.redirect_stdout(io.StringIO()):
            train_main(arguments)
        saved = load_checkpoint(run / 'best.pt')
        self.assertEqual(saved['architecture'], architecture)
        self.assertEqual(saved['config']['architecture'], architecture)
        self.assertEqual(saved['config']['model_config'], saved['model_config'])
        self.assertEqual(saved['model_config']['base_channels'], 2)
        if architecture == 'ImprovedUNet2D':
            self.assertEqual(saved['model_config']['dropout'], 0.3)
        self.assertIn('model_factory.py', saved['config']['source_sha256'])
        self.assertEqual(saved['config']['batching']['strategy'], 'same_shape_buckets')
        for phase in ('train', 'validation'):
            self.assertEqual(saved['history'][0][phase]['samples'], 2)
            self.assertEqual(saved['history'][0][phase]['pixels'], 2560)
        self.assertTrue((run / 'training_curves.png').is_file())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            check_run(['--run-dir', str(run)])
        self.assertIn('Architecture: ' + architecture, output.getvalue())
        self.assertIn('Checkpoint reload: PASS', output.getvalue())
        destination = self.root / 'prediction'
        with contextlib.redirect_stdout(io.StringIO()):
            report = predict_main([
                '--checkpoint', str(run / 'best.pt'),
                '--image-dir', str(data / 'keras_slices_validate'),
                '--mask-dir', str(data / 'keras_slices_seg_validate'),
                '--output-dir', str(destination), '--index', '0', '--label-id', '1',
                '--device', 'cpu', '--allow-partial',
            ])
        self.assertEqual(report['architecture'], architecture)
        self.assertEqual(report['model_config'], saved['model_config'])
        self.assertTrue(report['partial_checkpoint'])
        self.assertEqual(report['shape'], [32, 32])
        self.assertEqual(sum(map(sum, report['confusion'])), 1024)
        written = json.loads((destination / 'prediction.json').read_text(encoding='utf-8'))
        self.assertEqual(written['architecture'], architecture)
        self.assertEqual(nib.load(destination / 'prediction.nii.gz').shape, (32, 32))
        self.assertTrue((destination / 'prediction.png').is_file())

    def test_default_baseline_train_check_and_prediction_cli_round_trip(self):
        self.run_cli_round_trip('UNet2D')

    def test_improved_train_check_and_prediction_cli_round_trip(self):
        self.run_cli_round_trip('ImprovedUNet2D')


if __name__ == '__main__':
    unittest.main()
