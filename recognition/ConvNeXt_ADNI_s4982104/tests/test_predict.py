import json
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import test_check_dataloader
from train import run_training
from predict import aggregate_patients, run_prediction

class PredictionTests(unittest.TestCase):
    def rows(self):
        return [dict(patient_id='001_S_0001',image_id='1',label=1,p_ad=p,file_path=str(i))
                for i,p in enumerate([0.2,0.8])]

    def test_average_probability_and_threshold_tie(self):
        r=aggregate_patients(self.rows(),2)
        self.assertEqual(r[0]['p_ad'],0.5)
        self.assertEqual(r[0]['prediction'],1)
        self.assertEqual(r[0]['slice_count'],2)

    def test_reject_bad_groups(self):
        for field,value in [('label',0),('image_id','2'),('file_path','0'),('p_ad',float('nan'))]:
            rows=self.rows(); rows[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): aggregate_patients(rows,2)
        with self.assertRaises(ValueError): aggregate_patients(self.rows(),20)
        with self.assertRaises(ValueError): aggregate_patients([],20)

    def setUp(self):
        self.f=test_check_dataloader.LoaderTests();self.f.setUp();self.addCleanup(self.f.tearDown)

    def prepare(self):
        run_training(self.f.manifest,self.f.summary,self.f.root/'train',device='cpu',epochs=1,batch_size=4)

    def predict(self, name='pred'):
        return run_prediction(self.f.manifest,self.f.summary,self.f.root/'train/best.pt',
            self.f.root/'train/training_summary.json',self.f.root/name,device='cpu',batch_size=4,
            expected_slices=5,expected_patients=1,expected_lengths=None)

    def test_integration_two_passes_private_exports_and_no_test_reads(self):
        self.prepare(); r=self.predict()
        self.assertEqual(r['read_counts'],dict(train=0,validation=10,test=0))
        self.assertTrue(r['repeat_equal']);self.assertTrue(r['parameters_unchanged'])
        self.assertEqual(r['patient_count'],1);self.assertEqual(r['slice_count'],5)
        self.assertEqual(r['patient_metrics']['per_class']['NC']['support'],1)
        self.assertTrue((self.f.root/'pred/patient_predictions.csv').exists())
        with self.assertRaises(FileExistsError): self.predict()

    def test_checkpoint_tamper_saved_failure(self):
        self.prepare()
        with (self.f.root/'train/best.pt').open('ab') as f:f.write(b'tamper')
        with self.assertRaisesRegex(ValueError,'hash'):self.predict()
        self.assertEqual(json.loads((self.f.root/'pred/prediction_summary.json').read_text())['status'],'failed')

    def test_wrong_patient_slice_count_rejected(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError,'slice count'):
            run_prediction(self.f.manifest,self.f.summary,self.f.root/'train/best.pt',
                self.f.root/'train/training_summary.json',self.f.root/'pred',device='cpu',expected_lengths=None)

    def test_training_provenance_mismatch_rejected(self):
        self.prepare();p=self.f.root/'train/training_summary.json';s=json.loads(p.read_text())
        s['input_sha256']['split_samples']='bad';p.write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError,'input'):self.predict()
