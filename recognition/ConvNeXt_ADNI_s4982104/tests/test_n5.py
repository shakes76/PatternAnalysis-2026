"""N5 CPU synthetic regressions; not evidence of real-MRI/GPU acceptance."""
import importlib.util
import json
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch import nn
import test_n3
import run_n3
import run_n4

class HeadTiny(nn.Module):
    """Small backbone with the production 768-dimensional classifier contract."""
    def __init__(self,num_classes=1000):
        super().__init__()
        self.features=nn.Conv2d(3,2,1)
        self.classifier=nn.Sequential(nn.Linear(2,768),nn.Linear(768,num_classes))
    def forward(self,x):return self.classifier(self.features(x).mean((2,3)))

class N5Tests(unittest.TestCase):
    setUp=test_n3.N3Tests.setUp
    tearDown=test_n3.N3Tests.tearDown

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_n5'),'Formal N5 entry missing')
        import run_n5
        return run_n5

    def patches(self,n5):
        """Only replace expensive backbone/weight acquisition; seed and loop remain real."""
        stack=ExitStack()
        path=self.p/'weights.pt'
        if not path.exists():
            torch.manual_seed(900);torch.save(HeadTiny().state_dict(),path)
        for module in (run_n3,n5):
            stack.enter_context(patch.object(module,'build_model',side_effect=lambda name,num_classes=2:HeadTiny(num_classes)))
            stack.enter_context(patch.object(module,'verify_weights',return_value='synthetic-verified-weight'))
        stack.enter_context(patch.object(run_n4,'build_model',side_effect=lambda name,num_classes=2:HeadTiny(num_classes)))
        return stack

    def reference(self,n5):
        with self.patches(n5):
            run_n3.run_pilot(self.manifest,self.summary,self.p/'weights.pt',None,self.p/'n3','cpu',None,3,2)
            run_n4.run_control(self.manifest,self.summary,self.p/'weights.pt',None,self.p/'n4',self.p/'n3','cpu',None,3,2)
        return self.p/'n4'

    def run_formal(self,n5,out,ref,seed=42):
        return n5.run_formal(self.manifest,self.summary,self.p/'weights.pt',None,self.p/out,ref,device='cpu',seed=seed,expected_lengths=None,expected_slices=3,expected_patients=2)

    def test_thirty_epochs_real_seed_lr_tail_diagnostics_and_prefix(self):
        n5=self.api();ref=self.reference(n5);actual_lr=[];real=torch.optim.AdamW
        def optimizer(*args,**kw):
            obj=real(*args,**kw);actual_lr.append(obj.param_groups[0]['lr']);return obj
        with self.patches(n5),patch('torch.optim.AdamW',side_effect=optimizer):
            result=self.run_formal(n5,'formal',ref)
        t=json.loads((self.p/'formal/training/training_summary.json').read_text())
        self.assertEqual(actual_lr,[3e-5]);self.assertEqual(t['config']['epochs'],30)
        self.assertEqual([h['epoch'] for h in t['history']],list(range(1,31)))
        self.assertTrue(all(h['train']['samples']==6 and h['validation']['samples']==6 for h in t['history']))
        self.assertEqual(result['total_read_counts'],dict(train=180,validation=192,test=0))
        self.assertEqual(len(list((self.p/'formal/diagnostics').glob('epoch_*'))),30)
        self.assertEqual(result['n4_prefix_check']['status'],'passed')
        self.assertEqual(t['best_epoch'],min(t['history'],key=lambda h:h['validation']['loss'])['epoch'])
        self.assertTrue(result['prediction_repeat_equal'])
        self.assertEqual(t['config']['split_seed'],42)

    def test_seed_changes_real_head_and_same_seed_matches_n4(self):
        n5=self.api()
        from run_n3 import state_digest
        with self.patches(n5):
            states=[]
            for seed in [42,43,44,43]:
                model,_=n5.initialize_model(self.p/'weights.pt',None,seed)
                states.append(state_digest(model))
            baseline,_=run_n3.initialize_model(self.p/'weights.pt',None)
        self.assertEqual(states[0],state_digest(baseline));self.assertEqual(states[1],states[3])
        self.assertEqual(len(set(states[:3])),3)

    def test_seed43_repeats_full_training_and_changes_batch_order(self):
        n5=self.api();ref=self.reference(n5)
        with self.patches(n5):
            self.run_formal(n5,'a43',ref,43);self.run_formal(n5,'b43',ref,43)
        a=json.loads((self.p/'a43/training/training_summary.json').read_text())
        b=json.loads((self.p/'b43/training/training_summary.json').read_text())
        old=json.loads((ref/'training/training_summary.json').read_text())
        self.assertEqual(a['initial_state_sha256'],b['initial_state_sha256'])
        self.assertNotEqual(a['initial_state_sha256'],old['initial_state_sha256'])
        self.assertEqual([h['train'] for h in a['history']],[h['train'] for h in b['history']])
        self.assertEqual([h['validation'] for h in a['history']],[h['validation'] for h in b['history']])
        self.assertNotEqual(a['history'][0]['train']['order_sha256'],old['history'][0]['train']['order_sha256'])
        self.assertEqual(a['input_sha256'],old['input_sha256'])

    def test_prefix_mismatch_retains_completed_epoch_and_stops(self):
        n5=self.api();ref=self.reference(n5);real=n5.validation_with_rows
        def drift(*a,**kw):
            metrics,rows=real(*a,**kw);metrics['loss']+=.1;return metrics,rows
        with self.patches(n5),patch.object(n5,'validation_with_rows',side_effect=drift):
            with self.assertRaisesRegex(ValueError,'prefix'):
                self.run_formal(n5,'drift',ref)
        t=json.loads((self.p/'drift/training/training_summary.json').read_text())
        self.assertEqual(len(t['history']),1);self.assertEqual(t['completed_epochs'],1)
        self.assertEqual(t['status'],'failed')
        self.assertTrue((self.p/'drift/n5_error.log').exists())

    def test_checkpoint_protocol_rejects_lr_seed_preprocessing_and_epochs(self):
        n5=self.api();c=n5.protocol_config(43)
        for k,v in [('learning_rate',1e-4),('seed',45),('preprocessing',{}),('epochs',3)]:
            bad=dict(c);bad[k]=v
            path=self.p/'bad.pt';torch.save(dict(config=bad,model_state_dict={}),path)
            with self.subTest(k=k),self.assertRaises(ValueError):n5.load_checkpoint(path,bad,'cpu')

    def test_best_checkpoint_tie_keeps_earlier_epoch(self):
        n5=self.api()
        self.assertTrue(n5.is_improvement(.5,.6))
        self.assertFalse(n5.is_improvement(.5,.5))
        self.assertFalse(n5.is_improvement(.6,.5))

    def test_diagnostic_observer_preserves_rng_and_full_validation(self):
        n5=self.api()
        from dataset import ConvNeXtSliceDataset
        from check_dataloader import make_loader
        ds=ConvNeXtSliceDataset(self.manifest,'validation',self.summary);model=HeadTiny(2)
        state=torch.random.get_rng_state().clone()
        plain=n5.run_epoch(model,make_loader(ds,4,42),'cpu')
        measured,rows=n5.validation_with_rows(model,make_loader(ds,4,42),'cpu')
        self.assertEqual(plain,measured);self.assertEqual(ds.read_calls,12)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()));self.assertEqual(len(rows),6)

    def test_existing_output_is_never_overwritten(self):
        n5=self.api();out=self.p/'keep';out.mkdir();(out/'marker').write_text('keep')
        with self.assertRaises(FileExistsError):self.run_formal(n5,'keep',self.p/'absent')
        self.assertEqual((out/'marker').read_text(),'keep')

    def test_failure_after_training_preserves_actual_prediction_reads(self):
        n5=self.api();ref=self.reference(n5);real=n5.predict_pass
        def fail(*a):real(*a);raise RuntimeError('prediction failure injected')
        with self.patches(n5),patch.object(n5,'predict_pass',side_effect=fail):
            with self.assertRaises(RuntimeError):self.run_formal(n5,'failed',ref,44)
        pred=json.loads((self.p/'failed/prediction/prediction_summary.json').read_text())
        self.assertEqual(pred['status'],'failed');self.assertEqual(pred['read_counts'],dict(train=0,validation=6,test=0))
        t=json.loads((self.p/'failed/training/training_summary.json').read_text())
        self.assertEqual(t['completed_epochs'],30)

    def test_equal_validation_losses_keep_actual_first_checkpoint(self):
        n5=self.api();ref=self.reference(n5);real=n5.validation_with_rows
        def tied(*a,**kw):
            metrics,rows=real(*a,**kw)
            metrics['loss']=.5;metrics['unweighted_loss']=.5
            return metrics,rows
        with self.patches(n5),patch.object(n5,'validation_with_rows',side_effect=tied):
            self.run_formal(n5,'ties',ref,44)
        t=json.loads((self.p/'ties/training/training_summary.json').read_text())
        cp=torch.load(self.p/'ties/training/best.pt',map_location='cpu',weights_only=True)
        self.assertEqual(t['best_epoch'],1);self.assertEqual(cp['epoch'],1)
        self.assertEqual([h['epoch'] for h in t['history'] if h['saved_best']],[1])

    def test_invalid_seed_fails_before_reference_or_mri_access(self):
        n5=self.api()
        with self.assertRaisesRegex(ValueError,'seed'):
            self.run_formal(n5,'invalid',self.p/'nonexistent',45)
        result=json.loads((self.p/'invalid/n5_summary.json').read_text())
        self.assertEqual(result['status'],'failed')
        training=json.loads((self.p/'invalid/training/training_summary.json').read_text())
        self.assertEqual(training['completed_epochs'],0)
