import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from PIL import Image
from run_n3 import run_pilot, load_checkpoint, protocol_config

class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Conv2d(3,2,1);self.classifier=nn.Linear(2,2)
    def forward(self,x):return self.classifier(self.features(x).mean((2,3)))

class N3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory();self.p=Path(self.temp.name);rows=[]
        for s,split in enumerate(['train','validation','test']):
            for patient in range(2):
                for i in range(3):
                    path=self.p/f'{split}_{patient}_{i}.png'
                    if split!='test':Image.new('L',(256,240),40+patient*160+i).save(path)
                    rows.append(dict(file_path=str(path),original_split='test',image_id=str(100+s*2+patient),patient_id=f'001_S_{s*2+patient:04}',scan_date='',label=['NC','AD'][patient],split=split))
        self.manifest=self.p/'split_samples.csv'
        with self.manifest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        self.summary=self.p/'split_summary.json';self.summary.write_text(json.dumps({'status':'passed','output_sha256':{'split_samples':hashlib.sha256(self.manifest.read_bytes()).hexdigest()}}))
    def tearDown(self):self.temp.cleanup()
    def test_three_full_epochs_tail_repeat_and_no_test_reads(self):
        with patch('run_n3.initialize_model',return_value=(Tiny(),'synthetic')),patch('run_n3.build_model',side_effect=lambda *a,**k:Tiny()):
            result=run_pilot(self.manifest,self.summary,self.p/'weights',self.p/'weight_summary',self.p/'out',device='cpu',expected_lengths=None,expected_slices=3,expected_patients=2)
        self.assertEqual(result['total_read_counts'],dict(train=18,validation=30,test=0))
        t=json.loads((self.p/'out/training/training_summary.json').read_text())
        self.assertEqual([h['train']['samples'] for h in t['history']],[6,6,6])
        self.assertTrue(all(h['validation']['loss']==h['validation']['unweighted_loss'] for h in t['history']))
        self.assertEqual(t['best_epoch'],min(t['history'],key=lambda h:h['validation']['loss'])['epoch'])
        self.assertTrue(result['prediction_repeat_equal'])
    def test_checkpoint_rejects_changed_config_and_preprocessing(self):
        c=protocol_config();model=Tiny();path=self.p/'best.pt'
        torch.save(dict(config=c,model_state_dict=model.state_dict()),path)
        for key,value in [('model','simple_cnn'),('preprocessing',{}),('batch_size',8)]:
            bad=dict(c);bad[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'config|protocol'):
                load_checkpoint(path,bad,'cpu')
    def test_missing_weights_fails_and_retains_summary(self):
        with self.assertRaises(FileNotFoundError):
            run_pilot(self.manifest,self.summary,self.p/'absent',self.p/'absent.json',self.p/'failed',device='cpu',expected_lengths=None,expected_slices=3,expected_patients=2)
        self.assertEqual(json.loads((self.p/'failed/n3_summary.json').read_text())['status'],'failed')
    def test_existing_output_not_overwritten(self):
        out=self.p/'exists';out.mkdir();(out/'keep').write_text('evidence')
        with self.assertRaises(FileExistsError):run_pilot(self.manifest,self.summary,None,None,out,device='cpu')
        self.assertEqual((out/'keep').read_text(),'evidence')

class ProvenanceTests(unittest.TestCase):
    def test_changed_prediction_inputs_fail(self):
        from run_n3 import verify_input_hashes
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'samples';b=Path(tmp)/'summary';a.write_text('old');b.write_text('summary')
            hashes=dict(split_samples=hashlib.sha256(a.read_bytes()).hexdigest(),split_summary=hashlib.sha256(b.read_bytes()).hexdigest())
            a.write_text('changed')
            with self.assertRaisesRegex(ValueError,'input'):verify_input_hashes(a,b,hashes)
    def test_frozen_patient_label_counts_fail(self):
        from run_n3 import validate_frozen_labels
        records=[dict(patient_id=f'001_S_{i:04}',label='NC') for i in range(408)]
        with self.assertRaisesRegex(ValueError,'class'):validate_frozen_labels(records,'train')
