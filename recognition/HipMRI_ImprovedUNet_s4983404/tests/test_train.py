"""Start-up guards: settings, checkpoint architecture and run-directory ordering."""
from pathlib import Path
import argparse
import json
import sys
import tempfile
import unittest
from unittest import mock
import nibabel as nib
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import build_model
from train import main, resolve_architecture, validate_args


def arguments(**overrides) -> argparse.Namespace:
    """Mirror parse_args() defaults so a single field can be made invalid."""
    values = dict(root=Path('missing'), run=Path('run'), model=None, width=None,
                  epochs=1, batch_size=2, workers=0, lr=0.001, seed=2026,
                  limit=None, val_limit=None, augment=False, cache=False,
                  device='cpu', time_budget=60.0, evaluate=None, split='validate')
    values.update(overrides)
    return argparse.Namespace(**values)


def write_pairs(root: Path) -> None:
    """One 32x16 slice per split; enough for a CPU evaluation pass."""
    for split, patient in (('train', '004'), ('validate', '036'), ('test', '040')):
        for prefix in ('keras_slices_', 'keras_slices_seg_'):
            (root/f'{prefix}{split}').mkdir(parents=True)
        image = np.ones((32, 16), dtype=np.float32)
        labels = np.zeros_like(image)
        labels[8:10, 4:6] = 5
        nib.save(nib.Nifti1Image(image, np.eye(4)),
                 root/f'keras_slices_{split}'/f'case_{patient}_week_0_slice_0.nii.gz')
        nib.save(nib.Nifti1Image(labels, np.eye(4)),
                 root/f'keras_slices_seg_{split}'/f'seg_{patient}_week_0_slice_0.nii.gz')


def run_main(argv: list[str]) -> None:
    with mock.patch.object(sys, 'argv', ['train.py', *argv]):
        main()


class TrainingGuardTests(unittest.TestCase):
    def test_invalid_settings_rejected(self) -> None:
        for overrides in ({'epochs': 0}, {'batch_size': 0}, {'width': 0},
                          {'limit': 0}, {'val_limit': -1}, {'workers': -1},
                          {'lr': 0.0}, {'time_budget': 0.0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_args(arguments(**overrides))
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaises(ValueError):
                validate_args(arguments(device='cuda'))
            validate_args(arguments())          # CPU defaults must stay valid

    def test_architecture_follows_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'best.pt'
            torch.save({'state_dict': build_model('improved', 4).state_dict(),
                        'config': {'model': 'improved', 'width': 4}}, path)
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(resolve_architecture(arguments(evaluate=path), checkpoint),
                             ('improved', 4))
            self.assertEqual(resolve_architecture(
                arguments(evaluate=path, model='improved', width=4), checkpoint),
                ('improved', 4))
            for overrides in ({'model': 'basic'}, {'width': 8}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    resolve_architecture(arguments(evaluate=path, **overrides), checkpoint)
            with self.assertRaises(ValueError):
                resolve_architecture(arguments(evaluate=path), {'state_dict': {}})
            self.assertEqual(resolve_architecture(arguments(), None), ('basic', 16))

    def test_rejected_runs_create_no_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/'run'
            with self.assertRaises(FileNotFoundError):
                run_main(['--root', str(Path(directory)/'absent'), '--run', str(run),
                          '--device', 'cpu', '--workers', '0'])
            self.assertFalse(run.exists())
            checkpoint = Path(directory)/'best.pt'
            torch.save({'state_dict': build_model('improved', 4).state_dict(),
                        'config': {'model': 'improved', 'width': 4}}, checkpoint)
            with self.assertRaises(ValueError):
                run_main(['--root', str(Path(directory)/'absent'), '--run', str(run),
                          '--device', 'cpu', '--workers', '0',
                          '--evaluate', str(checkpoint), '--width', '8'])
            self.assertFalse(run.exists())

    def test_evaluation_uses_checkpoint_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, run = Path(directory)/'data', Path(directory)/'run'
            checkpoint = Path(directory)/'best.pt'
            write_pairs(root)
            torch.save({'state_dict': build_model('improved', 4).state_dict(),
                        'config': {'model': 'improved', 'width': 4}}, checkpoint)
            run_main(['--root', str(root), '--run', str(run), '--device', 'cpu',
                      '--workers', '0', '--evaluate', str(checkpoint)])
            config = json.loads((run/'config.json').read_text(encoding='utf-8'))
            self.assertEqual((config['model'], config['width']), ('improved', 4))
            evaluation = json.loads((run/'evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(evaluation['samples'], 1)
            self.assertTrue((run/'resources.json').is_file())


if __name__ == '__main__':
    unittest.main()
