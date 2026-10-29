"""Check raw values, missing scores, malformed inputs and protected PNG output."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from plot_history import load_history, metric_series, plot_history


class HistoryPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config = {'architecture': 'UNet2D', 'model_config': {'base_channels': 4}, 'partial_run': True}
        self.history = [
            {'epoch': epoch,
             'train': {'loss': loss, 'dice': [0.1] * 6, 'iou': [0.05] * 6},
             'validation': {'loss': loss + 0.2, 'dice': [0.2] * 5 + [None], 'iou': [0.1] * 5 + [None]}}
            for epoch, loss in ((1, 3.0), (2, 2.9))
        ]
        self.write()

    def write(self):
        (self.directory / 'config.json').write_text(json.dumps({'config': self.config}), encoding='utf-8')
        (self.directory / 'history.json').write_text(json.dumps(self.history), encoding='utf-8')

    def test_raw_values_and_null_metrics_preserved(self):
        config, history = load_history(self.directory)
        self.assertTrue(config['partial_run'])
        self.assertEqual(history[1]['train']['loss'], 2.9)
        self.assertEqual(metric_series(history, 'validation', 'dice', 0), [0.2, 0.2])
        self.assertTrue(all(math.isnan(value) for value in metric_series(history, 'validation', 'dice', 5)))
        self.assertIsNone(history[0]['validation']['dice'][5])

    def test_invalid_histories_rejected(self):
        original = copy.deepcopy(self.history)
        cases = [[], [dict(original[0], epoch=2)], [original[0], original[0]]]
        invalid_loss = copy.deepcopy(original)
        invalid_loss[0]['train']['loss'] = float('nan')
        cases.append(invalid_loss)
        invalid_metric = copy.deepcopy(original)
        invalid_metric[0]['validation']['dice'][0] = 1.2
        cases.append(invalid_metric)
        invalid_count = copy.deepcopy(original)
        invalid_count[0]['train']['dice'] = [0.1]
        cases.append(invalid_count)
        for history in cases:
            with self.subTest(history=history):
                self.history = history
                self.write()
                with self.assertRaises(ValueError):
                    load_history(self.directory)

    def test_missing_run_scope_rejected(self):
        self.config.pop('partial_run')
        self.write()
        with self.assertRaisesRegex(ValueError, 'partial_run'):
            load_history(self.directory)

    def test_png_rendering_for_one_and_two_epochs(self):
        original = copy.deepcopy(self.history)
        for count in (1, 2):
            with self.subTest(epochs=count):
                self.history = copy.deepcopy(original[:count])
                self.write()
                output = plot_history(self.directory, self.directory / f'curves_{count}.png')
                self.assertTrue(output.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
                self.assertGreater(output.stat().st_size, 1000)

    def test_existing_output_is_preserved(self):
        output = self.directory / 'existing.png'
        output.write_bytes(b'keep this existing result')
        with self.assertRaises(FileExistsError):
            plot_history(self.directory, output)
        self.assertEqual(output.read_bytes(), b'keep this existing result')


if __name__ == '__main__':
    unittest.main()
