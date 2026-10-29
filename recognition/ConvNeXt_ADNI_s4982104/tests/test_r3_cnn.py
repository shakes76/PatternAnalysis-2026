import importlib.util
import json
import unittest
from unittest.mock import patch
import torch
from torch import nn
import test_n3
import train

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Conv2d(1,2,1);self.classifier=nn.Linear(2,2)
    def forward(self,x):return self.classifier(self.features(x).mean((2,3)))

class R3CNNTests(unittest.TestCase):
    def setUp(self):
        self.f=test_n3.N3Tests();self.f.setUp();self.p=self.f.p
    def tearDown(self):self.f.tearDown()
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_r3_cnn'))
        import run_r3_cnn
        return run_r3_cnn
    def run_arm(self,arm,name,seed=42):
        api=self.api()
        with patch('run_r3_cnn.SimpleCNN',TinyCNN):
            return api.run_cnn(self.f.manifest,self.f.summary,self.p/name,arm=arm,epochs=3,seed=seed,device='cpu',expected_lengths=None,expected_slices=3)
    def test_both_arms_full_epochs_dual_best_and_reloads(self):
        for arm in ('A','B'):
            s=self.run_arm(arm,arm);self.assertEqual(s['status'],'passed')
            self.assertEqual(s['training_read_counts'],dict(train=18,validation=18,test=0))
            self.assertEqual(s['reload_read_counts'],dict(train=0,validation=24,test=0))
            self.assertEqual(s['config']['batch_size'],16);self.assertEqual(s['optimizer_actual']['lr'],.001)
            for h in s['history']:
                self.assertTrue(h['diagnostic_rng_unchanged']);self.assertEqual(h['diagnostic_additional_reads'],0)
            for rule in ('patient_nll','slice_ce'):
                self.assertTrue(s['reload'][rule]['repeat_equal']);self.assertTrue(s['reload'][rule]['selected_epoch_equal'])
                best=min(s['selection']['history'],key=lambda h:(h[rule],h['epoch']))
                self.assertEqual(s['reload'][rule]['epoch'],best['epoch'])
    def test_original_v1_trajectory_and_rng_unchanged(self):
        for arm,weight in [('A','none'),('B','balanced')]:
            with patch('train.SimpleCNN',TinyCNN):
                old=train.run_training(self.f.manifest,self.f.summary,self.p/('old'+arm),device='cpu',epochs=3,class_weighting=weight)
            rng=torch.get_rng_state().clone()
            new=self.run_arm(arm,'new'+arm)
            self.assertEqual(old['initial_state_sha256'],new['initial_state_sha256'])
            for a,b in zip(old['history'],new['history']):
                self.assertEqual(a['train'],b['train']);self.assertEqual(a['validation'],b['validation'])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_seed_reproducibility_and_effect(self):
        a=self.run_arm('A','a');b=self.run_arm('A','b');c=self.run_arm('A','c',43)
        self.assertEqual(a['initial_state_sha256'],b['initial_state_sha256'])
        self.assertNotEqual(a['initial_state_sha256'],c['initial_state_sha256'])
        self.assertEqual([h['train'] for h in a['history']],[h['train'] for h in b['history']])
    def test_no_overwrite_and_failure_keeps_completed_epoch(self):
        api=self.api();out=self.p/'exists';out.mkdir()
        with self.assertRaises(FileExistsError):api.run_cnn(None,None,out)
        original=api.validation_with_rows;calls=[]
        def fail_second(*args):
            calls.append(1)
            if len(calls)==2:raise ValueError('injected failure')
            return original(*args)
        with patch('run_r3_cnn.validation_with_rows',side_effect=fail_second):
            with self.assertRaisesRegex(ValueError,'injected'):self.run_arm('A','fail')
        s=json.loads((self.p/'fail/training_summary.json').read_text());self.assertEqual(s['status'],'failed');self.assertEqual(len(s['history']),1)
        self.assertTrue((self.p/'fail/selection/best_patient_nll.pt').exists())
    def test_bad_protocol_rejected(self):
        api=self.api()
        for args in [dict(arm='C'),dict(seed=1),dict(epochs=2)]:
            with self.assertRaises(ValueError):api.run_cnn(self.f.manifest,self.f.summary,self.p/str(args),device='cpu',**args)
    def expand_training(self):
        import csv,hashlib,shutil
        with self.f.manifest.open() as f:rows=list(csv.DictReader(f))
        sources=[r for r in rows if r['split']=='train' and r['label']=='NC']
        for extra in range(4):
            for i,r in enumerate(sources):
                path=self.p/f'extra_{extra}_{i}.png';shutil.copyfile(r['file_path'],path)
                rows.append(dict(r,file_path=str(path),patient_id=f'002_S_{extra:04}',image_id=str(500+extra)))
        with self.f.manifest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        self.f.summary.write_text(json.dumps({'status':'passed','output_sha256':{'split_samples':hashlib.sha256(self.f.manifest.read_bytes()).hexdigest()}}))
    def test_unbalanced_weights_and_full_plus_tail_batches(self):
        self.expand_training();s=self.run_arm('B','tail')
        self.assertEqual(s['config']['class_weight_details']['weights'],[.6,3.])
        self.assertEqual(s['training_read_counts']['train'],54)
        self.assertTrue(all(h['train']['samples']==18 for h in s['history']))
        self.assertTrue(all(h['validation']['loss']==h['validation']['unweighted_loss'] for h in s['history']))
    def test_real_simplecnn_matches_legacy_on_synthetic_images(self):
        api=self.api();self.expand_training()
        for arm,mode in [('A','none'),('B','balanced')]:
            old=train.run_training(self.f.manifest,self.f.summary,self.p/('real_old'+arm),device='cpu',epochs=3,class_weighting=mode)
            new=api.run_cnn(self.f.manifest,self.f.summary,self.p/('real_new'+arm),arm=arm,device='cpu',expected_lengths=None,expected_slices=3)
            self.assertEqual(old['initial_state_sha256'],new['initial_state_sha256'])
            for a,b in zip(old['history'],new['history']):
                self.assertEqual(a['train'],b['train']);self.assertEqual(a['validation'],b['validation'])
    def test_new_checkpoint_rejects_wrong_config(self):
        s=self.run_arm('A','config')
        from dual_checkpoint import load_selected
        with self.assertRaisesRegex(ValueError,'configuration'):
            load_selected(self.p/'config/selection','patient_nll',TinyCNN,dict(s['config'],seed=44),s['provenance'])
