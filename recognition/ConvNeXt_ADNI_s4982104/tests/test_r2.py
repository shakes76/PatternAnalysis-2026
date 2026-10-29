import importlib.util
import json
import unittest
from unittest.mock import patch
import torch
from torch import nn
import test_n3
from test_n3 import Tiny

class R2Tests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_n3.N3Tests();self.fixture.setUp();self.p=self.fixture.p
    def tearDown(self):self.fixture.tearDown()
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_r2'))
        import run_r2
        return run_r2
    def execute(self,out='out'):
        api=self.api()
        with patch('run_r2.initialize_model',side_effect=lambda *a:(Tiny(),'synthetic')),patch('run_r2.build_model',side_effect=lambda *a,**k:Tiny()):
            return api.run_pair(self.fixture.manifest,self.fixture.summary,None,None,self.p/out,device='cpu',expected_lengths=None,expected_slices=3)
    def test_complete_pair_tail_counts_freezing_and_dual_reload(self):
        r=self.execute();self.assertEqual(r['status'],'passed')
        self.assertTrue(r['paired_initial_state_equal']);self.assertTrue(r['paired_order_equal'])
        for arm in ('F','H'):
            s=r['arms'][arm];self.assertEqual(len(s['history']),5)
            self.assertEqual(s['training_read_counts'],dict(train=30,validation=30,test=0))
            self.assertEqual(s['reload_read_counts'],dict(train=0,validation=24,test=0))
            self.assertTrue(all(x['train']['samples']==6 for x in s['history']))
            for rule in ('patient_nll','slice_ce'):self.assertTrue(s['reload'][rule]['repeat_equal'])
            self.assertEqual(len(list((self.p/'out'/arm/'epochs').glob('*.pt'))),5)
        self.assertTrue(all(x['backbone_unchanged'] and x['backbone_no_grad'] and x['head_updated'] for x in r['arms']['H']['history']))
        self.assertEqual(r['arms']['F']['config']['learning_rate'],3e-5)
    def test_only_final_linear_is_head_and_train_mode_preserved(self):
        api=self.api();m=Tiny();m.classifier=nn.Sequential(nn.LayerNorm(2),nn.Linear(2,2))
        names=api.configure_trainable(m,'H');m.train()
        self.assertEqual(names,['classifier.1.weight','classifier.1.bias'])
        self.assertTrue(m.features.training);self.assertTrue(m.classifier[0].training)
    def test_existing_output_and_failed_initialization_retained(self):
        api=self.api();out=self.p/'exists';out.mkdir()
        with self.assertRaises(FileExistsError):api.run_pair(None,None,None,None,out)
        with patch('run_r2.initialize_model',side_effect=ValueError('missing weights')):
            with self.assertRaises(ValueError):api.run_pair(self.fixture.manifest,self.fixture.summary,None,None,self.p/'failed',device='cpu',expected_lengths=None,expected_slices=3)
        self.assertEqual(json.loads((self.p/'failed/r2_summary.json').read_text())['status'],'failed')
    def test_n4_mismatch_stops_pair(self):
        api=self.api()
        with self.assertRaises(ValueError):api.check_n4_prefix({'initial_state_sha256':'a','history':[]},{'initial_state_sha256':'b','history':[]})

    def test_same_seed_repeats_initialization_and_order(self):
        a=self.execute('a');b=self.execute('b')
        for arm in ('F','H'):
            self.assertEqual(a['arms'][arm]['initial_state_sha256'],b['arms'][arm]['initial_state_sha256'])
            self.assertEqual([x['train']['order_sha256'] for x in a['arms'][arm]['history']],[x['train']['order_sha256'] for x in b['arms'][arm]['history']])
    def test_pair_start_rejects_different_initial_state(self):
        api=self.api()
        with self.assertRaises(ValueError):api.check_pair_start({'initial_state_sha256':'a'},{'initial_state_sha256':'b'})
    def test_frozen_parameter_accidental_update_stops(self):
        api=self.api();original=api.run_epoch
        def broken(model,*args,**kwargs):
            result=original(model,*args,**kwargs)
            frozen=[p for p in model.parameters() if not p.requires_grad]
            if frozen:
                with torch.no_grad():frozen[0].add_(1.)
            return result
        with patch('run_r2.run_epoch',side_effect=broken):
            with self.assertRaisesRegex(ValueError,'Frozen'):self.execute('bad_freeze')
        self.assertEqual(json.loads((self.p/'bad_freeze/H/training_summary.json').read_text())['status'],'failed')
    def test_n4_reference_preflight_rejects_failed_or_changed_input(self):
        api=self.api();ref=self.p/'reference.json'
        ref.write_text(json.dumps(dict(status='failed')))
        with self.assertRaises(ValueError):api.read_n4_reference(ref,self.fixture.manifest,self.fixture.summary)
        ref.write_text(json.dumps(dict(status='passed',config=api.n4_config(),input_sha256={'split_samples':'wrong','split_summary':'wrong'})))
        with self.assertRaises(ValueError):api.read_n4_reference(ref,self.fixture.manifest,self.fixture.summary)
    def test_frozen_backbone_forward_still_in_training_mode(self):
        api=self.api();seen=[]
        class Observed(Tiny):
            def forward(self,x):
                if not self.features.weight.requires_grad and torch.is_grad_enabled():seen.append(self.features.training)
                return super().forward(x)
        with patch('run_r2.initialize_model',side_effect=lambda *a:(Observed(),'synthetic')),patch('run_r2.build_model',side_effect=lambda *a,**k:Observed()):
            api.run_pair(self.fixture.manifest,self.fixture.summary,None,None,self.p/'mode',device='cpu',expected_lengths=None,expected_slices=3)
        self.assertEqual(seen,[True]*10)

    def prefix_fixture(self):
        from copy import deepcopy
        ref={'initial_state_sha256':'same', 'history':[
            {'epoch':e,'train':{'order_sha256':f't{e}','loss':.6,'unweighted_loss':.5},
             'validation':{'order_sha256':'v','loss':.4,'unweighted_loss':.4}} for e in (1,2,3)]}
        return deepcopy(ref),ref

    def test_n4_prefix_matching_and_small_float_difference_pass(self):
        actual,reference=self.prefix_fixture()
        actual['history'][1]['validation']['loss']+=1e-7
        actual['history'].append({'epoch':4})
        self.assertTrue(self.api().check_n4_prefix(actual,reference)['passed'])

    def test_n4_prefix_changed_loss_or_order_rejected(self):
        for split in ('train','validation'):
            for key,value in [('loss',.9),('unweighted_loss',.9),('order_sha256','wrong'),('loss',float('nan'))]:
                with self.subTest(split=split,key=key,value=value):
                    actual,reference=self.prefix_fixture();actual['history'][1][split][key]=value
                    with self.assertRaises(ValueError):self.api().check_n4_prefix(actual,reference)

    def test_n4_prefix_missing_or_misnumbered_epochs_rejected(self):
        for target in ('actual','reference'):
            for mutation in ('missing','duplicate','offset'):
                with self.subTest(target=target,mutation=mutation):
                    actual,reference=self.prefix_fixture();history=(actual if target=='actual' else reference)['history']
                    if mutation=='missing':history.pop()
                    else:history[1]['epoch']=1 if mutation=='duplicate' else 4
                    with self.assertRaises(ValueError):self.api().check_n4_prefix(actual,reference)
