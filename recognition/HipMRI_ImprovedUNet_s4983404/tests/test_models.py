"""CPU checks run before scheduling GPU training; no pytest dependency."""
from pathlib import Path
import io
import sys
import tempfile
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import build_model, load_checkpoint
from metrics import segmentation_loss, confusion_counts, scores


class ModelTests(unittest.TestCase):
    def test_shapes_gradients_checkpoint(self) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        for name in ('basic', 'improved'):
            model = build_model(name, width=4)
            image = torch.randn(2, 1, 64, 32)
            target = torch.nn.functional.one_hot(
                torch.randint(0, 6, (2, 64, 32)), 6).permute(0, 3, 1, 2).float()
            logits = model(image)
            self.assertEqual(tuple(logits.shape), tuple(target.shape))
            loss = segmentation_loss(logits, target)
            self.assertTrue(0 < loss.item() < 10)
            loss.backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            model.eval()
            expected = model(image)
            torch.testing.assert_close(expected.softmax(1).sum(1), torch.ones_like(image[:, 0]))
            checkpoint = io.BytesIO()
            torch.save(model.state_dict(), checkpoint)
            checkpoint.seek(0)
            restored = build_model(name, width=4).eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            torch.testing.assert_close(expected, restored(image), rtol=0, atol=0)

    def test_metric_edge_cases(self) -> None:
        confusion = torch.zeros(6, 6, dtype=torch.int64)
        self.assertIsNone(scores(confusion)['dice'][5])
        confusion[0, 0] = 10
        confusion[5, 5] = 3
        self.assertEqual(scores(confusion)['dice'][5], 1)
        confusion[5, 0] = 2
        confusion[0, 5] = 1
        self.assertAlmostEqual(scores(confusion)['dice'][5], 6/9)
        self.assertAlmostEqual(scores(confusion)['iou'][5], 3/6)
        confusion[5, 5] = 0
        self.assertEqual(scores(confusion)['dice'][5], 0)
        labels = torch.tensor([[[0, 5], [1, 2]]])
        target = torch.nn.functional.one_hot(labels, 6).permute(0, 3, 1, 2).float()
        self.assertTrue(torch.equal(confusion_counts(target*20, target).diag(),
                                    torch.tensor([1, 1, 1, 0, 0, 1])))

    def test_invalid_model(self) -> None:
        with self.assertRaises(ValueError):
            build_model('unknown')

    def test_checkpoint_loader_requires_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'best.pt'
            torch.save({'state_dict': {}}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, 'cpu')
            torch.save({'state_dict': build_model('basic', 4).state_dict(),
                        'config': {'model': 'basic', 'width': 4}}, path)
            self.assertEqual(load_checkpoint(path, 'cpu')['config']['width'], 4)


if __name__ == '__main__':
    unittest.main()
