"""Synthetic-only tests; never open ADNI or existing checkpoint files.

Run: python -B -m unittest -v test_diagnose_development
"""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import torch
import diagnose_development as diag


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return torch.full((3, 8, 8), row['position'] / 20.0), row['label']


class FakeModel(torch.nn.Module):
    def forward(self, images):
        score = images.mean((1, 2, 3))
        return torch.stack([score * 3, -score], 1)


class DevelopmentTests(unittest.TestCase):
    def make_data(self, root):
        rows, mapping, samples = {}, {}, {}
        for split in ('train', 'validation'):
            samples[split] = []
            for label in (0, 1):
                scan = split + str(label)
                mapping[scan] = 'PRIVATE_PATIENT_' + scan
                for position in range(1, 21):
                    path = root / (scan + '_' + str(position) + '.jpeg')
                    Image.fromarray(np.full((12, 16), 20 + position + label * 30, dtype=np.uint8)).save(path)
                    samples[split].append((str(path), label))
            records, stats = diag.inspect_samples(samples[split], mapping, lambda p: Path(p).stem.split('_')[0])
            self.assertEqual(stats['relative_positions'], list(range(20)))
            self.assertEqual([r['raw_slice_index'] for r in records[:20]], list(range(1, 21)))
            self.assertEqual([r['position'] for r in records[:20]], list(range(20)))
            for r in records:
                r['patient'] = diag.token(b'x' * 32, 'patient', r['patient'])
                r['scan'] = diag.token(b'x' * 32, 'scan', r['scan'])
            rows[split] = records
        return rows, samples, mapping

    def test_indices_and_failures(self):
        self.assertEqual(diag.parse_name('scan_12.jpeg'), ('scan', 12))
        for name in ('scan_x.jpeg', 'scan_2.png', 'scan.jpeg'):
            with self.assertRaises(diag.DiagnosticError):
                diag.parse_name(name)
        with tempfile.TemporaryDirectory() as temp:
            _, samples, mapping = self.make_data(Path(temp))
            bad = samples['train'][:-1]
            with self.assertRaises(diag.DiagnosticError):
                diag.inspect_samples(bad, mapping, lambda p: Path(p).stem.split('_')[0])
            with self.assertRaises(diag.DiagnosticError):
                diag.inspect_samples(samples['train'], {}, lambda p: Path(p).stem.split('_')[0])
            with self.assertRaises(diag.DiagnosticError):
                diag.inspect_samples(samples['train'][:-1] + samples['train'][:1], mapping, lambda p: Path(p).stem.split('_')[0])

    def test_shifted_indices_and_aggregate_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            samples = [('A_{}.jpeg'.format(i), 0) for i in range(40, 60)]
            samples += [('B_{}.jpeg'.format(i), 1) for i in range(101, 121)]
            rows, stats = diag.inspect_samples(samples, {'A': 'privateA', 'B': 'privateB'}, lambda p: Path(p).stem.split('_')[0], decode=False, audit_prefix=root / 'audit')
            self.assertEqual(stats['distinct_index_sets'], 2)
            self.assertTrue(stats['valid_scans_with_different_starts'])
            self.assertEqual(stats['start_index_counts'], {'40': 1, '101': 1})
            for row in rows:
                self.assertEqual(row['position'], row['raw_slice_index'] - (40 if row['scan'] == 'A' else 101))
            self.assertNotIn('privateA', (root / 'audit.json').read_text())
            bad = [('A_{}.jpeg'.format(i), 0) for i in list(range(19)) + [18]]
            bad[10] = ('A_30.jpeg', 0)
            bad += [('B_{}.jpeg'.format(i), 1) for i in range(19)] + [('B_bad.jpeg', 1)]
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured), self.assertRaises(diag.DiagnosticError):
                diag.inspect_samples(bad, {'A': 'privateA', 'B': 'privateB'}, lambda p: Path(p).stem.split('_')[0], decode=False, audit_prefix=root / 'bad')
            audit = json.loads((root / 'bad.json').read_text())
            for field in ('unparseable_filenames', 'scans_with_duplicate_indices', 'scans_with_internal_gaps', 'scans_with_wrong_slice_count'):
                self.assertEqual(audit['problems'][field], 1)
            self.assertIn('unparseable_filenames', captured.getvalue())

    def test_metrics_ties_and_weighting(self):
        rows = [dict(label=y, score=s, patient=str(i), scan=str(i)) for i, (y, s) in enumerate([(0, .9), (0, .4), (1, .4), (1, .1)])]
        m = diag.metrics(rows)
        self.assertEqual(m['accuracy'], .75)
        self.assertEqual(m['recall_ad'], .5)
        self.assertEqual(m['auroc_ad'], .875)
        self.assertEqual(diag.metrics(rows, [2, 1, 1, 1])['accuracy'], .8)
        self.assertIsNone(diag.metrics(rows[:2])['auroc_ad'])
        self.assertEqual(diag.correlation([1, 2, 2, 4], [4, 3, 3, 1], True), -1.0)

    def test_resume_reports_privacy_and_corrupt_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, _, _ = self.make_data(root)
            datasets = {split: FakeDataset(rr) for split, rr in rows.items()}
            output = root / 'output'
            output.mkdir()
            args = argparse.Namespace(output_dir=output, device='cpu', batch_size=7,
                                      max_seconds=0, bootstrap=2, seed=42)
            diag.STOP = False
            # Interrupt after one completed batch; resume must not duplicate it.
            real_intensity = diag.intensity
            calls = [0]
            def interrupt(path):
                calls[0] += 1
                if calls[0] == 7:
                    diag.STOP = True
                return real_intensity(path)
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(diag, 'intensity', side_effect=interrupt):
                self.assertFalse(diag.analyze(FakeModel(), datasets, rows, args))
            cache = output / 'local_slice_cache.jsonl'
            self.assertEqual(len(cache.read_text().splitlines()), 7)
            with cache.open('ab') as f:
                f.write(b'{"interrupted":')
            diag.STOP = False
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(diag.analyze(FakeModel(), datasets, rows, args))
            data = [json.loads(line) for line in cache.read_text().splitlines()]
            self.assertEqual(len(data), 80)
            self.assertEqual(len({(r['split'], r['scan'], r['position']) for r in data}), 80)
            for f in output.iterdir():
                if f.suffix in ('.json', '.jsonl', '.csv', '.md'):
                    content = f.read_text()
                    self.assertNotIn('PRIVATE_PATIENT_', content)
                    self.assertNotIn(str(root), content)
                    self.assertNotIn('.jpeg', content)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['validation']['slice']['n'], 40)
            self.assertTrue((output / 'intensity_correlations.png').is_file())
            # Deliberately change membership: old cache must be rejected.
            rows['train'][0]['position'] = 999
            with self.assertRaises(diag.DiagnosticError):
                diag.analyze(FakeModel(), datasets, rows, args)

    def test_real_convnext_cpu_forward(self):
        from modules import ConvNeXt
        model = ConvNeXt().eval()
        with torch.inference_mode():
            logits = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertTrue(torch.isfinite(logits).all())

    def test_check_cli_never_analyzes_and_resume_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'new'
            argv = ['diagnose_development.py', 'check', '--config', 'unused.json', '--output-dir', str(out)]
            fake = (FakeModel(), {}, {}, {'train': {'slices': 40}}, {'version': 1})
            with mock.patch('sys.argv', argv), mock.patch.object(diag, 'load_development', return_value=fake), mock.patch.object(diag, 'analyze') as analyze, contextlib.redirect_stdout(io.StringIO()):
                diag.main()
                analyze.assert_not_called()
            self.assertTrue(json.loads((out / 'manifest.json').read_text())['check_passed'])
            with mock.patch('sys.argv', argv + ['--resume']), mock.patch.object(diag, 'load_development', return_value=fake), contextlib.redirect_stdout(io.StringIO()):
                diag.main()
            mismatch = (FakeModel(), {}, {}, {}, {'version': 2})
            with mock.patch('sys.argv', argv + ['--resume']), mock.patch.object(diag, 'load_development', return_value=mismatch):
                with self.assertRaises(diag.DiagnosticError):
                    diag.main()


if __name__ == '__main__':
    unittest.main()
