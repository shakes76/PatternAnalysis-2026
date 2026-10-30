"""Synthetic revised-evaluation tests. No private data, network or GPU access.

End-to-end flow uses an artificial tiny network. Actual full A1 checkpoint
loading is covered by tests_selected_a1; no clinical performance is asserted.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import revised_evaluation as r
import selected_a1 as sa
from dataset import SliceRecord, SplitBundle, preprocessing_spec
from train_convnext_v2 import environment
from training_utils import seed_everything

torch.set_num_threads(1)


def row(subject, scan, label=0, index=80, split='test'):
    c = ['NC','AD'][label]
    return SliceRecord(f'AD_NC/train/{c}/{scan}_{index}.jpeg', subject, str(scan), index,
                       'train', c, 2*label, label, split)


def rule(tau=.75):
    return {'name': 'Synthetic', 'selected_epoch': 13, 'temperature': 1.8,
            'reference_confidence_threshold': tau, 'input_hashes': {'checkpoint_sha256': 'a'*64}}


class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.scale = nn.Parameter(torch.tensor(2.0))
    def forward(self, x):
        v = x.mean((1,2,3)) * self.scale
        return torch.stack([-v, v], 1)


class CountingTiny(Tiny):
    def __init__(self, fail_at=None):
        super().__init__(); self.calls = 0; self.fail_at = fail_at
    def forward(self, x):
        self.calls += 1
        if self.calls == self.fail_at: raise RuntimeError('simulated interruption')
        return super().forward(x)


def fixture(root):
    private = root/'private files'; private.mkdir()
    project = root/'repo with spaces'/'recognition'/'assignment'; project.mkdir(parents=True)
    (project/'README.md').write_bytes(b'# Historical report\r\nPreserve me.\r\n')
    for name in r.NEW_FILES:
        path = project/name; path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic source '+name)
    data = private/'data';data.mkdir()
    records=[];image_hashes={}
    for i in range(12):
        label = int(i >= 6)
        pixels = (35 if label == 0 else 220)
        if i in (0,2,7,8,9):pixels = 255-pixels
        for j in range(2):
            rec = row(f'PRIVATE_PATIENT_{i:02d}',f'SCAN_{i:02d}',label,index=80+j)
            path=data/rec.image_path;path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.full((240,256),pixels+j,dtype=np.uint8)).save(path)
            records.append(rec);image_hashes[rec.image_path]=r.sha256(path)
    hashes={'train.csv':'a'*64,'validation.csv':'b'*64,'test.csv':'c'*64}
    bundle=SplitBundle(data,{'train':(), 'validation':(), 'test':tuple(records)}, {}, hashes)
    model=Tiny()
    sources={name:r.sha256(project/name) for name in r.NEW_FILES}
    candidates={}
    for alias in ('cnn','a1'):
        f=private/alias;f.mkdir()
        cfg={'seed':42,'recipe':'scratch_augmented'} if alias=='a1' else {'model':'baseline','seed':42}
        ep=13 if alias=='a1' else 14
        report={'configuration':cfg,'selected_epoch':ep,'parameters':1,'source_code_sha256':sources}
        checkpoint={'format_version':2 if alias=='a1' else 1, 'epoch':ep, 'configuration':cfg,
                    'source_code_sha256':sources,'manifest_sha256':hashes,'model_label_map':{'NC':0,'AD':1},
                    'architecture':'ConvNeXtTiny' if alias=='a1' else 'BaselineCNN-v1',
                    'pretrained':{'used':False},'model_name':'baseline','preprocessing':preprocessing_spec(0),
                    'model_state':model.state_dict()}
        cp=f/'PRIVATE_best_model.pt';torch.save(checkpoint,cp)
        pred=f/'PRIVATE_validation_predictions.json';pred.write_text('{}')
        candidates[alias]=sa.Candidate(alias,alias,f,report,'report.json',cp,pred,[],[],[],[],
            {'checkpoint_sha256':r.sha256(cp),'run_summary_sha256':'d'*64,'validation_predictions_sha256':'e'*64})
    env=environment(torch.device('cpu'))
    conf={'models':{a:{'checkpoint_replay':{'passed':True,'binding':{'environment':env},
         'model_only_inference':{'median_ms_per_slice':1.,'p95_ms_per_slice':1.2,'scope':'synthetic only'}}}
           for a in candidates}}
    rules={'models':{a:{'temperature':1.8,'confidence_threshold':.75} for a in candidates}}
    return {'private':private,'project':project,'bundle':bundle,'candidates':candidates,'confidence':conf,
            'rules':rules,'bound_evidence':{'old':'a'*64},'audit_inventory_sha256':'b'*64,
            'test_hashes':image_hashes,'revised_sources':sources}


class MathTests(unittest.TestCase):
    def test_same_three_policy_ids(self):
        rep,_=r.score([row('A',1),row('B',2,1)],[.02,.9],rule())
        self.assertEqual(set(rep['policies']), {'reference','strict_cn','no_auto_cn'})
    def test_no_auto_cn_not_perfect(self):
        rep,_=r.score([row('A',1),row('B',2,1)],[.01,.01],rule())
        p=rep['policies']['no_auto_cn']
        self.assertEqual(p['n_auto_CN'],0)
        self.assertIsNone(p['negative_predictive_value'])
        self.assertIsNone(p['automated_research_label_accuracy'])
        self.assertEqual(p['n_clinical_reviews_if_all_AD_flags_also_reviewed'],2)
    def test_strict_does_not_change_classification(self):
        rep,p=r.score([row('A',1),row('B',2,1)],[.10,.10],rule())
        self.assertEqual([x['prediction'] for x in p['patients']],[0,0])
        self.assertEqual(rep['metrics']['patient']['accuracy'],.5)
    def test_missing_rule_propagates_refer_all(self):
        rep,_=r.score([row('A',1),row('B',2,1)],[.01,.99],rule(None))
        for p in rep['policies'].values():self.assertEqual(p['n_unresolved_referrals'],2)
    def test_below_target_is_result(self):
        rep,_=r.score([row('A',1),row('B',2,1)],[.9,.1],rule())
        self.assertFalse(rep['observed_accuracy_at_least_0_80_by_unit']['patient'])
    def test_all_units_reported(self):
        rep,_=r.score([row('A',1),row('B',2,1)],[.1,.9],rule())
        self.assertEqual(set(rep['metrics']), {'patient','scan','slice'})
    def test_group_average_not_slice_weighted(self):
        records=[row('A',1,index=80),row('A',1,index=81),row('A',2,index=80),row('B',3,1)]
        _,out=r.score(records,[0.,0.,1.,.8],rule())
        self.assertEqual(out['patients'][0]['p_ad'],.5)
    def test_non_test_rejected(self):
        with self.assertRaises(ValueError):r.score([row('A',1,split='validation'),row('B',2,1)],[.1,.9],rule())
    def test_invalid_probabilities(self):
        for p in (float('nan'),float('inf'),-1,2):
            with self.assertRaises(ValueError):r.score([row('A',1),row('B',2,1)],[p,.9],rule())
    def test_confidence_not_fit(self):
        with patch('confidence_analysis.fit_temperature',side_effect=AssertionError('no fitting')):
            rep,_=r.score([row('A',1),row('B',2,1)],[.1,.9],rule())
        self.assertTrue(rep['no_test_fitting_or_policy_selection'])


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.ctx=fixture(Path(self.temp.name))
        self.load_patch=patch.object(r,'load_development',return_value=self.ctx)
        self.load_patch.start();self.addCleanup(self.load_patch.stop)
    def freeze(self):return r.freeze(self.ctx['private'],'cpu')
    def evaluate(self):return r.evaluate(self.ctx['private'],enforce_git=False,factories={'cnn':Tiny,'a1':Tiny})
    def data(self):return r.VerifiedTestDataset(self.ctx['bundle'],self.ctx['test_hashes'])
    def test_freeze_does_not_decode(self):
        with patch('PIL.Image.open',side_effect=AssertionError('no MRI opening')):f=self.freeze()
        self.assertTrue((f/r.PLAN_FILE).exists())
        self.assertFalse((f/'REVISED_TEST_ACCESS_STARTED.json').exists())
    def test_plan_discloses_prior_exposure(self):
        f=self.freeze();p=r.read_json(f/r.PLAN_FILE)
        self.assertTrue(p['prior_test_exposure_disclosed']);self.assertFalse(p['fresh_independent_test'])
        self.assertEqual(p['configuration']['strict_CN_maximum_calibrated_p_AD'],.05)
    def test_repeated_freeze_same_bytes(self):
        f=self.freeze();before=(f/r.PLAN_FILE).read_bytes();self.freeze()
        self.assertEqual(before,(f/r.PLAN_FILE).read_bytes())
    def test_changed_rule_blocks_refreeze(self):
        self.freeze();self.ctx['rules']['models']['a1']['temperature']=2.2
        with self.assertRaises(ValueError):self.freeze()
    def test_environment_change_rejected(self):
        self.ctx['confidence']['models']['a1']['checkpoint_replay']['binding']['environment']['torch']='changed'
        with self.assertRaises(ValueError):self.freeze()
    def test_changed_plan_blocks_evaluation(self):
        f=self.freeze();p=r.read_json(f/r.PLAN_FILE);p['configuration']['diagnostic_threshold']=.4
        r.atomic_json(f/r.PLAN_FILE,p)
        with self.assertRaises(ValueError):self.evaluate()
    def test_wrong_checkpoint_bytes_rejected(self):
        self.freeze();self.ctx['candidates']['a1'].checkpoint.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.evaluate()
    def test_verified_image_change_rejected(self):
        data=self.data();path=data.source_path(0);path.write_bytes(b'changed')
        with self.assertRaises(ValueError):data[0]
    def test_audit_missing_hash_rejected(self):
        with self.assertRaises(ValueError):r.audited_hashes(self.data().records,{'images':[]})
    def test_audit_duplicate_path_rejected(self):
        rec=self.data().records[0];a={'relative_path':rec.image_path,'file_sha256':'a'*64,'status':'decoded'}
        with self.assertRaises(ValueError):r.audited_hashes(self.data().records,{'images':[a,a]})
    def test_full_flow_private_excluded_and_failure_templates(self):
        f=self.freeze();self.evaluate();report=r.read_json(f/r.SUMMARY)
        self.assertTrue(report['quantitative_evaluation_completed'])
        self.assertEqual(report['test_counts']['patients'],12)
        self.assertFalse(report['manual_failure_review_completed'])
        self.assertEqual(report['failure_review']['cases_selected'],5)
        with ZipFile(f/r.BUNDLE) as z:
            self.assertEqual(set(z.namelist()),set(r.PUBLIC))
            for name in z.namelist():
                if name.endswith(('.md','.json')):
                    self.assertNotIn('PRIVATE_PATIENT_',z.read(name).decode())
                    self.assertNotIn(str(self.ctx['private']),z.read(name).decode())
        self.assertIn('[COMPLETE', (f/'PRIVATE_failure_review'/'PRIVATE_failure_review.md').read_text())
    def test_completed_repeat_no_more_inference(self):
        f=self.freeze();self.evaluate();before=(f/r.SUMMARY).read_bytes()
        with patch.object(sa,'load_selected_checkpoint',side_effect=AssertionError('no reload')):
            self.evaluate()
        self.assertEqual(before,(f/r.SUMMARY).read_bytes())
    def test_tampered_completed_report_rejected(self):
        f=self.freeze();self.evaluate();(f/'README_revised_test_results.md').write_text('tampered')
        with self.assertRaises(ValueError):self.evaluate()
    def test_crash_in_report_generation_reuses_completed_predictions(self):
        self.freeze()
        with patch.object(r,'plots',side_effect=RuntimeError('interrupted reporting')):
            with self.assertRaises(RuntimeError):self.evaluate()
        with patch.object(Tiny,'forward',side_effect=AssertionError('should use cache')):
            self.evaluate()
    def test_written_manual_review_preserved_on_recovery(self):
        f=self.freeze()
        with patch.object(r,'plots',side_effect=RuntimeError('crash')):
            with self.assertRaises(RuntimeError):self.evaluate()
        w=f/'PRIVATE_failure_review'/'PRIVATE_failure_review.md';w.write_text('My manual observations')
        self.evaluate();self.assertEqual(w.read_text(),'My manual observations')
    def test_source_change_fails_completion(self):
        self.freeze();real=r.plots
        def mutate(*a):
            real(*a);self.ctx['bound_evidence']['old']='different'
        with patch.object(r,'plots',side_effect=mutate):
            with self.assertRaises(ValueError):self.evaluate()
    def test_publish_preserves_old_text_and_is_idempotent(self):
        f=self.freeze();self.evaluate();p=self.ctx['project']/'README.md';old=p.read_bytes()
        with patch.object(r,'require_committed',return_value='synthetic'):
            r.publish(self.ctx['private']);after=p.read_bytes();r.publish(self.ctx['private'])
        self.assertTrue(after.startswith(old));self.assertEqual(after,p.read_bytes())
        self.assertEqual(after.count(r.MARKER.encode()),1)
    def make_git(self):
        project=self.ctx['project']
        for args in [('init','-b','topic-recognition'),('config','user.name','Synthetic'),
                     ('config','user.email','synthetic@example.invalid'),('add','.'),('commit','-m','Synthetic frozen plan')]:
            subprocess.run(['git','-C',str(project.parents[1]),*args],check=True,capture_output=True)
    def test_uncommitted_plan_rejected(self):
        self.freeze()
        with self.assertRaises(ValueError):r.require_committed(self.ctx['project'],r.read_json(self.ctx['project']/r.PLAN_FILE))
    def test_committed_plan_accepted(self):
        self.freeze();self.make_git()
        commit=r.require_committed(self.ctx['project'],r.read_json(self.ctx['project']/r.PLAN_FILE))
        self.assertEqual(len(commit),40)
    def test_source_edit_after_commit_rejected(self):
        self.freeze();self.make_git();(self.ctx['project']/'predict.py').write_text('changed source')
        with self.assertRaises(ValueError):r.require_committed(self.ctx['project'],r.read_json(self.ctx['project']/r.PLAN_FILE))
    def test_no_old_pointer_changed(self):
        private=self.ctx['private'];old=private/'latest_final_evaluation.txt';old.write_text('old-run')
        self.freeze();self.evaluate();self.assertEqual(old.read_text(),'old-run')
    def test_policy_all_retained_in_report(self):
        f=self.freeze();self.evaluate();s=(f/'README_revised_test_results.md').read_text()
        for name in ('reference','strict_cn','no_auto_cn','prior','AD recall','Scan accuracy','Slice accuracy'):
            self.assertIn(name,s)


# A separate TestCase with only fixture setup helpers.
class CacheTests(unittest.TestCase):
    setUp=FixtureTests.setUp
    data=FixtureTests.data
    def call(self,model,path,identity=None):
        return r.infer_cached(model,self.data(),torch.device('cpu'),path,
                              identity or {'alias':'a1','plan':'x'},8)
    def test_interruption_resumes_without_repeating_saved_batch(self):
        p=self.ctx['private']/'cache.json'
        with self.assertRaises(RuntimeError):self.call(CountingTiny(fail_at=2),p)
        self.assertEqual(len(r.read_json(p)['probabilities']),8)
        m=CountingTiny();result,_=self.call(m,p)
        full,_=self.call(CountingTiny(),self.ctx['private']/'full.json')
        self.assertEqual(result,full);self.assertEqual(m.calls,2)
    def test_final_partial_batch_kept(self):
        # 24 images -> 8/8/8 here; independently slice to 22.
        bundle=self.ctx['bundle'];self.ctx['bundle']=replace(bundle,records={**bundle.records,'test':bundle.records['test'][:22]})
        self.ctx['test_hashes']={k:v for k,v in self.ctx['test_hashes'].items() if k in {x.image_path for x in self.ctx['bundle'].records['test']}}
        p=self.ctx['private']/'cache.json';values,_=self.call(CountingTiny(),p)
        self.assertEqual(len(values),22);values2,_=self.call(CountingTiny(),p);self.assertEqual(values,values2)
    def test_cache_plan_mismatch(self):
        p=self.ctx['private']/'cache.json';self.call(CountingTiny(),p)
        with self.assertRaises(ValueError):self.call(CountingTiny(),p,{'alias':'a1','plan':'different'})
    def test_cache_probability_digest(self):
        p=self.ctx['private']/'cache.json';self.call(CountingTiny(),p)
        c=r.read_json(p);c['probabilities'][0]=.5;r.atomic_json(p,c)
        with self.assertRaises(ValueError):self.call(CountingTiny(),p)
    def test_cache_invalid_boundary(self):
        p=self.ctx['private']/'cache.json';self.call(CountingTiny(),p)
        c=r.read_json(p);c['probabilities']=c['probabilities'][:9];c['completed_prediction_sha256']=r.object_hash(c['probabilities']);r.atomic_json(p,c)
        with self.assertRaises(ValueError):self.call(CountingTiny(),p)
    def test_cached_images_reverified(self):
        p=self.ctx['private']/'cache.json';self.call(CountingTiny(),p)
        self.data().source_path(0).write_bytes(b'changed')
        with self.assertRaises(ValueError):self.call(CountingTiny(),p)
    def test_nonfinite_logits_rejected_before_cache(self):
        class Bad(Tiny):
            def forward(self,x):return torch.full((len(x),2),float('nan'))
        p=self.ctx['private']/'cache.json'
        with self.assertRaises(ValueError):self.call(Bad(),p)
        self.assertFalse(p.exists())
    def test_inference_disables_gradients(self):
        class Checked(Tiny):
            def forward(self,x):
                if torch.is_grad_enabled() or self.training:raise RuntimeError('wrong mode')
                return super().forward(x)
        self.call(Checked(),self.ctx['private']/'cache.json')


class CommandTests(unittest.TestCase):
    def test_confirmation_required(self):
        with patch.object(r,'evaluate') as call:
            self.assertEqual(r.main(['revised-evaluate','--private-root','/not-here']),1)
            call.assert_not_called()
    def test_confirmed_command_routes(self):
        with patch.object(r,'evaluate') as call:
            self.assertEqual(r.main(['revised-evaluate','--private-root','/not-here','--confirm-reused-test']),0)
            call.assert_called_once()
    def test_predict_dispatches_revised_only(self):
        spec=importlib.util.spec_from_file_location('predict_for_test', PROJECT/'predict.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with patch.object(sys,'argv',['predict.py','revised-freeze','--private-root','example']),patch.object(r,'main',return_value=17) as call:
            self.assertEqual(m.main(),17);call.assert_called_once()
    def test_predict_legacy_freeze_not_redirected(self):
        spec=importlib.util.spec_from_file_location('predict_for_test',PROJECT/'predict.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with patch.object(sys,'argv',['predict.py','freeze','--private-root','example']),patch.object(m,'freeze') as call,patch.object(r,'main') as new:
            self.assertEqual(m.main(),0);call.assert_called_once();new.assert_not_called()
    def test_no_training_or_fit_call_in_new_source(self):
        import ast
        tree=ast.parse((PROJECT/'revised_evaluation.py').read_text())
        calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)]
        for name in ('fit_temperature','train_run','backward','optimizer_for','select_rule'):
            self.assertNotIn(name,calls)


if __name__=='__main__':unittest.main()
