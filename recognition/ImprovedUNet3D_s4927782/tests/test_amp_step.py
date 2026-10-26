"""Exercise real GradScaler overflow handling on CPU without needing a GPU."""

import json
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("PyTorch is required for AMP optimizer-step checks.")

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from train import optimizer_step


class AmpStepTests(unittest.TestCase):
    def fixture(self, amp):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=amp, init_scale=8, growth_interval=100)
        loss = model(torch.ones(1, 1)).sum()
        scaler.scale(loss).backward()
        return model, optimizer, scaler

    def test_amp_overflow_skips_update_reduces_scale_and_has_finite_json(self):
        model, optimizer, scaler = self.fixture(True)
        original = model.weight.detach().clone()
        model.weight.grad.fill_(float("inf"))
        norm, skipped = optimizer_step(model, optimizer, scaler, 5.0, amp=True)
        self.assertIsNone(norm)
        self.assertTrue(skipped)
        self.assertEqual(scaler.get_scale(), 4)
        self.assertTrue(torch.equal(original, model.weight))
        json.dumps({"last_gradient_norm_before_clipping": norm, "skipped": skipped}, allow_nan=False)

    def test_non_amp_nonfinite_gradient_remains_an_error(self):
        model, optimizer, scaler = self.fixture(False)
        original = model.weight.detach().clone()
        model.weight.grad.fill_(float("inf"))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            optimizer_step(model, optimizer, scaler, 5.0, amp=False)
        self.assertTrue(torch.equal(original, model.weight))

    def test_finite_amp_gradient_is_clipped_and_optimizer_advances(self):
        model, optimizer, scaler = self.fixture(True)
        # Set a known scaled gradient: after unscale it is 200, clipped to 5.
        model.weight.grad.fill_(200 * scaler.get_scale())
        norm, skipped = optimizer_step(model, optimizer, scaler, 5.0, amp=True)
        self.assertAlmostEqual(norm, 200.0)
        self.assertFalse(skipped)
        self.assertAlmostEqual(float(model.weight.item()), 0.5, places=5)
        self.assertLessEqual(float(model.weight.grad.norm()), 5.00001)
        self.assertEqual(scaler.get_scale(), 8)


if __name__ == "__main__":
    unittest.main()
