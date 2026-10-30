"""Metric, padding and checkpoint checks with synthetic data only."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train import SegmentationLoss, dice_from_matrix, fit, validate
from modules import UNet2D


class EncodedPrediction(nn.Module):
    def forward(self, images):
        return F.one_hot(images[:, 0].long(), 6).movedim(-1, 1).float() * 10


def scan_samples():
    samples = []
    for patient in range(3):
        for index in range(60):
            image = torch.zeros(1, 8, 8)
            target = torch.zeros(8, 8, dtype=torch.long)
            target[:, -1] = 255; image[:, :, -1] = 5
            if index == 0 and patient == 0:
                target[0, 0] = 5; image[0, 0, :2] = 5
            if index == 0 and patient == 2:
                image[0, 0, 0] = 5
            samples.append((image, target, dict(patient_id=str(patient),
                            week_id=0, slice_id=index)))
    return samples


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_loss_ignores_padding_and_has_finite_gradients(self):
        torch.manual_seed(3)
        logits = torch.randn(2, 6, 12, 10, requires_grad=True)
        target = torch.randint(0, 6, (2, 12, 10)); target[:, :, 8:] = 255
        criterion = SegmentationLoss()
        loss = criterion(logits, target)
        altered = logits.detach().clone(); altered[:, :, :, 8:] = 500
        torch.testing.assert_close(loss, criterion(altered, target))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(int(torch.count_nonzero(logits.grad[:, :, :, 8:])), 0)
        self.assertGreater(float(logits.grad[:, :, :, :8].abs().sum()), 0)

    def test_scan_dice_invariant_to_batch_size_and_empty_class_excluded(self):
        samples = scan_samples()
        results = [validate(EncodedPrediction(), DataLoader(samples, batch_size=b),
                            SegmentationLoss(), 'cpu') for b in (17, 60)]
        self.assertEqual(results[0]['case_dice'], results[1]['case_dice'])
        for result in results:
            self.assertEqual(result['scans'], 3)
            self.assertEqual(result['valid_cases'][5], 2)
            self.assertAlmostEqual(result['case_dice'][5], 1 / 3)
            self.assertAlmostEqual(result['pooled_dice'][5], 0.5)
        self.assertIsNone(dice_from_matrix(torch.zeros(6, 6, dtype=torch.long))[5])

    def test_incomplete_formal_scan_rejected(self):
        with self.assertRaisesRegex(ValueError, 'all 60'):
            validate(EncodedPrediction(), DataLoader(scan_samples()[:2], batch_size=2),
                     SegmentationLoss(), 'cpu')

    def test_end_to_end_checkpoint_and_actual_early_stop(self):
        torch.manual_seed(8)
        model = UNet2D(base_features=1)
        samples = [(torch.randn(1, 32, 32), torch.full((32, 32), 5, dtype=torch.long),
                    dict(patient_id='004', week_id=0, slice_id=i)) for i in range(2)]
        loader = DataLoader(samples, batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=directory, dice_weight=.5, lr=0.,
                                   device='cpu', epochs=4, min_epochs=2,
                                   patience=1, smoke_steps=1)
            config = dict(model_config=dict(in_channels=1, num_classes=6, base_features=1))
            # Constant validation score makes stopping behavior deterministic;
            # the real forward/backward/validation code still runs every epoch.
            actual_validate = validate
            def fixed_score(*a, **kw):
                result = actual_validate(*a, **kw)
                result['case_dice'][5] = .2
                return result
            with patch('train.validate', side_effect=fixed_score), patch('train.plot_history'):
                result = fit(model, loader, loader, args, config)
            self.assertEqual(result['completed_epochs'], 2)
            out = Path(directory)
            best = torch.load(out/'best_model.pt', weights_only=True)
            last = torch.load(out/'last_model.pt', weights_only=True)
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(last['epoch'], 2)
            self.assertTrue(best['smoke_only'])
            restored = UNet2D(**best['model_config'])
            restored.load_state_dict(best['model_state_dict'], strict=True)
            self.assertEqual(len((out/'history.csv').read_text().splitlines()), 3)
            self.assertFalse(json.loads((out/'training_summary.json').read_text())['test_evaluated'])


if __name__ == '__main__':
    unittest.main()
