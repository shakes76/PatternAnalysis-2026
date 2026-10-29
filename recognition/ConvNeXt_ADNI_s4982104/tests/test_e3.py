import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import test_check_dataloader
from run_e3 import run_baseline, compare_checkpoints

class E3Tests(unittest.TestCase):
    def setUp(self):
        self.f=test_check_dataloader.LoaderTests();self.f.setUp();self.addCleanup(self.f.tearDown)
    def run_small(self):
        return run_baseline(self.f.manifest,self.f.summary,self.f.root/'e3',device='cpu',epochs=4,
            expected_lengths=None,expected_slices=5,expected_patients=1)
    def test_end_to_end_two_predictions_and_read_counts(self):
        r=self.run_small()
        self.assertEqual(r['status'],'passed')
        self.assertEqual(r['total_read_counts'],dict(train=20,validation=40,test=0))
        self.assertEqual(set(r['comparison']),{'first3','full'})
        self.assertTrue((self.f.root/'e3/comparison.csv').exists())
        self.assertTrue((self.f.root/'e3/training/first3/best.pt').exists())
        with self.assertRaises(FileExistsError):self.run_small()
    def test_comparison_rejects_tampered_best_epoch(self):
        self.run_small();p=self.f.root/'e3/training/training_summary.json'
        s=json.loads(p.read_text());s['best_epoch']=99;p.write_text(json.dumps(s))
        with self.assertRaises(ValueError):compare_checkpoints(self.f.root/'e3',expected_epochs=4)
    def test_failure_summary_saved_without_hiding_error(self):
        with patch('run_e3.run_training',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):self.run_small()
        s=json.loads((self.f.root/'e3/e3_summary.json').read_text())
        self.assertEqual(s['status'],'failed');self.assertTrue((self.f.root/'e3/e3_error.log').exists())
    def test_same_best_checkpoint_is_allowed(self):
        from train import run_epoch as real
        losses=iter([0.1,0.2,0.3,0.4])
        def controlled(model,loader,device,optimizer=None,**kw):
            r=real(model,loader,device,optimizer,**kw)
            if optimizer is None:r['loss']=r['unweighted_loss']=next(losses)
            return r
        with patch('train.run_epoch',side_effect=controlled):r=self.run_small()
        self.assertTrue(r['same_checkpoint'])
        self.assertEqual(r['comparison']['first3']['best_epoch'],1)
        self.assertEqual(r['comparison']['full']['best_epoch'],1)
