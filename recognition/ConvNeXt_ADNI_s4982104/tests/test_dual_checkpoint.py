import importlib.util
import json
import tempfile
from pathlib import Path
import unittest
import torch
from torch import nn
from test_patient_evaluation import fixture

class DualCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        self.config=dict(model='synthetic_linear',seed=42);self.provenance=dict(manifest_sha256='a'*64,weights_sha256='b'*64)
    def tearDown(self):self.tmp.cleanup()
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('dual_checkpoint'),'R1 dual saver missing')
        import dual_checkpoint
        return dual_checkpoint
    def model(self,value):
        m=nn.Linear(2,2).eval()
        with torch.no_grad():
            for p in m.parameters():p.fill_(value)
        return m
    def test_two_selectors_diverge_earliest_ties_and_strict_reload(self):
        api=self.api();rows,expected=fixture();selector=api.DualCheckpointSelector(self.p/'out',self.config,self.provenance)
        selector.consider(1,self.model(1),rows,expected,.2)
        better=[dict(r,p_ad=.9 if r['label'] else .1) for r in rows]
        selector.consider(2,self.model(2),better,expected,.4)
        selector.consider(3,self.model(3),better,expected,.4)
        summary=json.loads((self.p/'out/selection_summary.json').read_text())
        self.assertEqual(summary['best']['patient_nll']['epoch'],2);self.assertEqual(summary['best']['slice_ce']['epoch'],1)
        for rule,epoch in [('patient_nll',2),('slice_ce',1)]:
            m,cp=api.load_selected(self.p/'out',rule,lambda:nn.Linear(2,2),self.config,self.provenance)
            self.assertEqual(cp['epoch'],epoch)
            self.assertTrue(all(torch.equal(p,torch.full_like(p,epoch)) for p in m.parameters()))
        with self.assertRaises(ValueError):api.load_selected(self.p/'out','patient_nll',lambda:nn.Linear(2,2),dict(seed=44),self.provenance)
    def test_no_rng_consumption_from_saving_or_loading(self):
        api=self.api();rows,expected=fixture();m=self.model(1)
        state=torch.random.get_rng_state().clone()
        s=api.DualCheckpointSelector(self.p/'out',self.config,self.provenance);s.consider(1,m,rows,expected,.2)
        api.load_selected(self.p/'out','patient_nll',lambda:nn.Linear(2,2),self.config,self.provenance)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
    def test_no_overwrite_bad_records_retain_previous(self):
        api=self.api();rows,expected=fixture();s=api.DualCheckpointSelector(self.p/'out',self.config,self.provenance)
        s.consider(1,self.model(1),rows,expected,.2)
        previous=(self.p/'out/best_patient_nll.pt').read_bytes()
        with self.assertRaises(ValueError):s.consider(2,self.model(2),rows[:-1],expected,.1)
        self.assertEqual((self.p/'out/best_patient_nll.pt').read_bytes(),previous)
        self.assertEqual(json.loads((self.p/'out/selection_summary.json').read_text())['status'],'failed')
        with self.assertRaises(FileExistsError):api.DualCheckpointSelector(self.p/'out',self.config,self.provenance)
    def test_hash_and_source_tampering_rejected(self):
        api=self.api();rows,expected=fixture();s=api.DualCheckpointSelector(self.p/'out',self.config,self.provenance);s.consider(1,self.model(1),rows,expected,.2)
        with self.assertRaises(ValueError):api.load_selected(self.p/'out','slice_ce',lambda:nn.Linear(2,2),self.config,dict(manifest_sha256='bad'))
        with (self.p/'out/best_slice_ce.pt').open('ab') as f:f.write(b'corrupt')
        with self.assertRaises(ValueError):api.load_selected(self.p/'out','slice_ce',lambda:nn.Linear(2,2),self.config,self.provenance)
    def test_sealed_test_split_and_bad_epoch_or_metric_rejected(self):
        api=self.api();rows,expected=fixture()
        for name,epoch,ce,records in [('epoch',2,.2,expected),('negative',1,-1,expected),('nan',1,float('nan'),expected),('test',1,.2,[dict(r,split='test') for r in expected])]:
            with self.subTest(name=name):
                s=api.DualCheckpointSelector(self.p/name,self.config,self.provenance)
                with self.assertRaises(ValueError):s.consider(epoch,self.model(1),rows,records,ce)
    def test_one_validation_pass_then_selection_adds_no_image_reads(self):
        api=self.api()
        import test_n3
        from dataset import ConvNeXtSliceDataset
        from check_dataloader import make_loader
        from run_n4 import validation_with_rows
        case=test_n3.N3Tests();case.setUp()
        try:
            ds=ConvNeXtSliceDataset(case.manifest,'validation',case.summary)
            m=test_n3.Tiny().eval();before=torch.random.get_rng_state().clone()
            metric,rows=validation_with_rows(m,make_loader(ds,4,42),'cpu')
            s=api.DualCheckpointSelector(self.p/'out',self.config,self.provenance,expected_slices=3)
            s.consider(1,m,rows,ds.records,metric['loss'])
            self.assertEqual(ds.read_calls,6)
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        finally:case.tearDown()
    def test_validation_cohort_cannot_change_between_epochs(self):
        api=self.api();rows,expected=fixture();s=api.DualCheckpointSelector(self.p/'out',self.config,self.provenance)
        s.consider(1,self.model(1),rows,expected,.2)
        with self.assertRaises(ValueError):
            s.consider(2,self.model(2),rows[:20],expected[:20],.1)
    def test_failed_selection_cannot_load_as_accepted(self):
        api=self.api();rows,expected=fixture();s=api.DualCheckpointSelector(self.p/'out',self.config,self.provenance)
        s.consider(1,self.model(1),rows,expected,.2)
        with self.assertRaises(ValueError):s.consider(2,self.model(2),rows[:-1],expected,.1)
        with self.assertRaises(ValueError):
            api.load_selected(self.p/'out','patient_nll',lambda:nn.Linear(2,2),self.config,self.provenance)
