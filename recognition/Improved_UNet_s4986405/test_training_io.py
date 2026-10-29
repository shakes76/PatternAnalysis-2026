"""Check serialization, validation-based selection and run isolation."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from modules import UNet2D
from training_io import RunRecorder, load_checkpoint


def result(loss):
    return {
        'loss': loss, 'samples': 2, 'batches': 1, 'pixels': 10,
        'confusion': torch.diag(torch.tensor([2, 2, 2, 2, 2, 0], dtype=torch.long)),
        'dice': torch.tensor([1., 1., 1., 1., 1., float('nan')]),
        'iou': torch.tensor([1., 1., 1., 1., 1., float('nan')]),
    }


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'run'
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(3710)
        self.model = UNet2D(base_channels=4).cpu()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.generator = torch.Generator().manual_seed(3710)
        self.model_config = {'in_channels': 1, 'num_classes': 6, 'base_channels': 4}
        self.config = {'model_config': self.model_config, 'architecture': 'UNet2D', 'partial_run': True}
        self.images = torch.randn(1, 1, 31, 17)

    def step(self, model, optimizer):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        model(self.images).square().mean().backward()
        optimizer.step()

    def record(self, recorder, epoch, train_loss=2., val_loss=2.):
        return recorder.record(epoch, self.model, self.optimizer,
                               result(train_loss), result(val_loss), self.generator)

    def test_model_and_optimizer_round_trip(self):
        self.step(self.model, self.optimizer)
        self.model.eval()
        with torch.no_grad():
            expected = self.model(self.images)
        recorder = RunRecorder(self.directory, self.config)
        self.record(recorder, 1)
        saved = load_checkpoint(self.directory / 'last.pt')
        restored = UNet2D(**saved['model_config']).cpu()
        restored.load_state_dict(saved['model_state'], strict=True)
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(self.images), expected, rtol=0, atol=0)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
        restored_optimizer.load_state_dict(saved['optimizer_state'])
        # The next identical optimizer step must also match after restoring Adam.
        self.step(self.model, self.optimizer)
        self.step(restored, restored_optimizer)
        for original, recovered in zip(self.model.parameters(), restored.parameters()):
            torch.testing.assert_close(original, recovered, rtol=0, atol=0)

    def test_best_selected_only_by_validation_loss(self):
        recorder = RunRecorder(self.directory, self.config)
        self.assertTrue(self.record(recorder, 1, train_loss=5, val_loss=2))
        first = load_checkpoint(self.directory / 'best.pt')
        with torch.no_grad():
            next(self.model.parameters()).add_(1)
        self.assertFalse(self.record(recorder, 2, train_loss=0.1, val_loss=3))
        best = load_checkpoint(self.directory / 'best.pt')
        last = load_checkpoint(self.directory / 'last.pt')
        self.assertEqual(best['epoch'], 1)
        self.assertEqual(last['epoch'], 2)
        for name in first['model_state']:
            self.assertTrue(torch.equal(best['model_state'][name], first['model_state'][name]))
        self.assertTrue(self.record(recorder, 3, train_loss=9, val_loss=1))
        self.assertEqual(load_checkpoint(self.directory / 'best.pt')['epoch'], 3)

    def test_histories_keep_absent_metrics_as_null_or_empty(self):
        recorder = RunRecorder(self.directory, self.config)
        self.record(recorder, 1)
        self.record(recorder, 2, val_loss=1.5)
        text = (self.directory / 'history.json').read_text(encoding='utf-8')
        self.assertNotIn('NaN', text)
        history = json.loads(text)
        self.assertEqual(len(history), 2)
        self.assertIsNone(history[0]['validation']['dice'][5])
        with (self.directory / 'history.csv').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['validation_dice_5'], '')
        self.assertEqual(rows[1]['epoch'], '2')

    def test_existing_run_directory_is_preserved(self):
        RunRecorder(self.directory, self.config)
        original = (self.directory / 'config.json').read_bytes()
        with self.assertRaises(FileExistsError):
            RunRecorder(self.directory, self.config)
        self.assertEqual((self.directory / 'config.json').read_bytes(), original)

    def test_invalid_epoch_and_nonfinite_loss_rejected(self):
        recorder = RunRecorder(self.directory, self.config)
        for epoch, val_loss in ((0, 2), (2, 2), (1, float('nan')), (1, float('inf'))):
            with self.subTest(epoch=epoch, loss=val_loss):
                with self.assertRaises(ValueError):
                    self.record(recorder, epoch, val_loss=val_loss)
        self.assertEqual(recorder.history, [])
        self.assertFalse((self.directory / 'last.pt').exists())

    def test_rng_and_run_metadata_are_retained(self):
        recorder = RunRecorder(self.directory, self.config)
        self.record(recorder, 1)
        saved = load_checkpoint(self.directory / 'last.pt')
        self.assertTrue(torch.equal(saved['rng_state']['torch_cpu'], torch.get_rng_state()))
        self.assertTrue(torch.equal(saved['rng_state']['shuffle'], self.generator.get_state()))
        self.assertEqual(saved['config']['partial_run'], True)
        self.assertEqual(saved['model_config'], self.model_config)
        self.assertEqual(saved['best_epoch'], 1)
        self.assertEqual(len(saved['history']), 1)


if __name__ == '__main__':
    unittest.main()
