import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler
from check_dataloader import AuditedDataset, make_loader, validate_batch, run_check

class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rows = []
        for s, split in enumerate(('train', 'validation', 'test')):
            for i in range(5):
                path = self.root / f'{split}_{i}.png'
                if split != 'test':
                    Image.new('L', (256,240), 20+s*60+i*10).save(path)
                rows.append(dict(file_path=str(path), original_split='test',
                    image_id=str(100+s), patient_id=f'001_S_{s:04d}',
                    scan_date='', label='AD' if s == 0 else 'NC', split=split))
        self.manifest = self.root / 'split_samples.csv'
        with self.manifest.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        self.summary = self.root / 'split_summary.json'
        self.summary.write_text(json.dumps({'status':'passed','output_sha256':{
            'split_samples':hashlib.sha256(self.manifest.read_bytes()).hexdigest()}}))
    def tearDown(self): self.tmp.cleanup()
    def test_batch_alignment_repeat_and_no_test_reads(self):
        r = run_check(self.manifest, self.summary, 4, 42)
        self.assertEqual(r['status'], 'passed')
        self.assertEqual(r['read_counts'], {'train':4,'validation':8,'test':0,'total':12,'unique_images':8})
        self.assertTrue(r['validation_repeat_identical'])
        self.assertEqual(r['batches']['train']['image_shape'],[4,1,240,256])
        self.assertEqual(r['batches']['train']['label_counts'], {'AD':4,'NC':0})
    def test_samplers_and_incomplete_tail(self):
        for split, kind in [('train',RandomSampler),('validation',SequentialSampler)]:
            ds = AuditedDataset(self.manifest,split,self.summary)
            loader = make_loader(ds,4,42)
            self.assertIsInstance(loader.sampler,kind)
            self.assertEqual([len(b['label']) for b in loader],[4,1])
    def test_fixed_training_seed(self):
        ds = AuditedDataset(self.manifest,'train',self.summary)
        a = next(iter(make_loader(ds,4,42)))
        b = next(iter(make_loader(ds,4,42)))
        self.assertEqual(a['file_path'],b['file_path'])
    def test_wrong_metadata_or_pixels_detected(self):
        for key in ('label','patient_id','image_id','image'):
            ds = AuditedDataset(self.manifest,'validation',self.summary)
            batch = next(iter(make_loader(ds,4,42)))
            if key == 'image': batch[key][0].zero_()
            elif key == 'label': batch[key][0] = 1
            else: batch[key][0] = 'wrong'
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_batch(batch,ds)
    def test_bad_image_fails(self):
        (self.root/'validation_0.png').write_bytes(b'broken')
        with self.assertRaisesRegex(RuntimeError,'validation_0.png'):
            run_check(self.manifest,self.summary,4,42)
    def test_invalid_batch_size_fails(self):
        with self.assertRaises(ValueError): run_check(self.manifest,self.summary,0,42)

    def cli(self, manifest=None):
        out = self.root / 'output'
        result = subprocess.run([sys.executable,
            str(Path(__file__).resolve().parents[1] / 'check_dataloader.py'),
            '--manifest', str(manifest or self.manifest),
            '--summary', str(self.summary), '--output-dir', str(out)],
            capture_output=True, text=True)
        return result, out

    def test_cli_rejects_wrong_cohort_before_reading_images(self):
        # Valid tiny manifest must not pass as the frozen real ADNI cohort.
        result, out = self.cli()
        self.assertNotEqual(result.returncode, 0)
        evidence = json.loads((out/'loader_check_summary.json').read_text())
        self.assertEqual(evidence['status'], 'failed')
        self.assertIn('Dataset lengths', evidence['error'])

    def test_cli_missing_input_still_saves_failure_evidence(self):
        result, out = self.cli(self.root/'missing.csv')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((out/'loader_check_summary.json').exists())
        evidence = json.loads((out/'loader_check_summary.json').read_text())
        self.assertEqual(evidence['status'], 'failed')
        self.assertIn('FileNotFoundError', evidence['error'])
