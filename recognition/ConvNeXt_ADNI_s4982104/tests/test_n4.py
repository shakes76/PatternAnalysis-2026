"""N4 regressions: real optimizer/loop/I/O with a cheap synthetic network."""
import csv
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch import nn
import test_n3
Tiny = test_n3.Tiny
import run_n3

class N4Tests(unittest.TestCase):
    setUp = test_n3.N3Tests.setUp
    tearDown = test_n3.N3Tests.tearDown

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_n4'), 'N4 experiment entry point missing')
        import run_n4
        return run_n4

    def initialize(self, *args):
        torch.manual_seed(42)
        return Tiny(), 'synthetic'

    def reference(self):
        with patch('run_n3.initialize_model',side_effect=self.initialize), patch('run_n3.build_model',side_effect=lambda *a,**k:Tiny()):
            run_n3.run_pilot(self.manifest,self.summary,None,None,self.p/'reference','cpu',None,3,2)
        return self.p/'reference'

    def test_complete_control_actual_optimizer_diagnostics_and_sealed_test(self):
        n4=self.api();reference=self.reference();observed=[]
        real=torch.optim.AdamW
        def optimizer(*a,**k):
            obj=real(*a,**k);observed.append(obj.param_groups[0]['lr']);return obj
        with patch('run_n4.initialize_model',side_effect=self.initialize),patch('run_n4.build_model',side_effect=lambda *a,**k:Tiny()),patch('torch.optim.AdamW',side_effect=optimizer):
            r=n4.run_control(self.manifest,self.summary,None,None,self.p/'control',reference,'cpu',None,3,2)
        self.assertEqual(observed,[3e-5])
        self.assertEqual(r['total_read_counts'],dict(train=18,validation=30,test=0))
        self.assertTrue(r['paired_checks']['initial_state_equal'])
        self.assertTrue(r['paired_checks']['all_epoch_orders_equal'])
        t=json.loads((self.p/'control/training/training_summary.json').read_text())
        self.assertEqual(t['config']['learning_rate'],3e-5)
        self.assertEqual(t['best_epoch'],min(t['history'],key=lambda x:x['validation']['loss'])['epoch'])
        for epoch in (1,2,3):
            folder=self.p/f'control/diagnostics/epoch_{epoch}'
            rows=list(csv.DictReader((folder/'slice_predictions.csv').read_text().splitlines()))
            patients=list(csv.DictReader((folder/'patient_predictions.csv').read_text().splitlines()))
            self.assertEqual(len(rows),6);self.assertEqual(len(patients),2)
            self.assertTrue(all(int(x['slice_count'])==3 for x in patients))
            # Logits permit CE reconstruction even when float32 probabilities saturate.
            logits=torch.tensor([[float(x['logit_nc']),float(x['logit_ad'])] for x in rows])
            labels=torch.tensor([int(x['label']) for x in rows])
            self.assertAlmostEqual(nn.functional.cross_entropy(logits,labels).item(),t['history'][epoch-1]['validation']['loss'],places=6)
        with patch('run_n4.build_model',side_effect=lambda *a,**k:Tiny()):
            n4.load_checkpoint(self.p/'control/training/best.pt',t['config'],'cpu')
        self.assertEqual(r['status'],'passed')

    def test_reference_corruption_rejected_before_images(self):
        n4=self.api();reference=self.reference()
        p=reference/'training/config.json';p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'reference|Reference|N3'):
            n4.run_control(self.manifest,self.summary,None,None,self.p/'failed',reference,'cpu',None,3,2)
        self.assertEqual(json.loads((self.p/'failed/n4_summary.json').read_text())['status'],'failed')

    def test_checkpoint_rejects_original_lr_even_if_self_consistent(self):
        n4=self.api();c=run_n3.protocol_config();p=self.p/'wrong.pt'
        torch.save(dict(config=c,model_state_dict=Tiny().state_dict()),p)
        with self.assertRaisesRegex(ValueError,'protocol|learning_rate'):
            n4.load_checkpoint(p,c,'cpu')

    def test_validation_observation_preserves_metrics_rng_and_reads(self):
        n4=self.api()
        from dataset import ConvNeXtSliceDataset
        from check_dataloader import make_loader
        from train import run_epoch
        ds=ConvNeXtSliceDataset(self.manifest,'validation',self.summary)
        model=Tiny();state=torch.random.get_rng_state().clone()
        plain=run_epoch(model,make_loader(ds,4,42),'cpu')
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        measured,rows=n4.validation_with_rows(model,make_loader(ds,4,42),'cpu')
        self.assertEqual(plain,measured);self.assertEqual(ds.read_calls,12)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertEqual([x['file_path'] for x in rows],[x['file_path'] for x in ds.records])

    def test_joint_signal_requires_all_three_conditions(self):
        n4=self.api()
        base=[dict(loss=x,per_class={'AD':{'recall':a},'NC':{'recall':n}}) for x,a,n in [(1,.6,.8),(1,.4,.9),(3,.9,.4)]]
        good=[dict(loss=x,per_class={'AD':{'recall':a},'NC':{'recall':n}}) for x,a,n in [(1,.6,.8),(1,.5,.7),(2,.7,.6)]]
        self.assertTrue(n4.stability_signal(base,good)['joint_signal'])
        good[-1]['per_class']['NC']['recall']=.1
        self.assertFalse(n4.stability_signal(base,good)['joint_signal'])

    def test_prediction_failure_preserves_partial_read_counts(self):
        n4=self.api();reference=self.reference();real=n4.predict_pass
        def fail_after_read(*args):
            real(*args)
            raise RuntimeError('injected prediction failure')
        with patch('run_n4.initialize_model',side_effect=self.initialize),patch('run_n4.build_model',side_effect=lambda *a,**k:Tiny()),patch('run_n4.predict_pass',side_effect=fail_after_read):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                n4.run_control(self.manifest,self.summary,None,None,self.p/'failed_prediction',reference,'cpu',None,3,2)
        pred=json.loads((self.p/'failed_prediction/prediction/prediction_summary.json').read_text())
        self.assertEqual(pred['status'],'failed')
        self.assertEqual(pred['read_counts'],dict(train=0,validation=6,test=0))

    def test_pair_start_rejects_changed_source_initialization_and_config(self):
        n4=self.api();reference=self.reference()
        t=json.loads((reference/'training/training_summary.json').read_text())
        from copy import deepcopy
        actual=deepcopy(t);actual['config']['learning_rate']=3e-5
        n4.check_pair_start(t,actual)
        for key in ('initial_state_sha256','input_sha256','versions'):
            bad=deepcopy(actual);bad[key]='changed'
            with self.subTest(key=key),self.assertRaises(ValueError):n4.check_pair_start(t,bad)
        bad=deepcopy(actual);bad['config']['batch_size']=8
        with self.assertRaises(ValueError):n4.check_pair_start(t,bad)
        # Consistent rewritten summary chain must still reject a false source hash.
        t['code_sha256']['train.py']='0'*64
        tp=reference/'training/training_summary.json';tp.write_text(json.dumps(t))
        from check_dataloader import sha256
        summary=json.loads((reference/'n3_summary.json').read_text())
        summary['training_summary_sha256']=sha256(tp)
        prp=reference/'prediction/prediction_summary.json';pr=json.loads(prp.read_text());pr['training_summary_sha256']=sha256(tp);prp.write_text(json.dumps(pr))
        summary['prediction_summary_sha256']=sha256(prp);(reference/'n3_summary.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError,'source'):
            n4.read_reference(reference,self.manifest,self.summary)
