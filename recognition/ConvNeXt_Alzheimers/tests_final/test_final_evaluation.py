"""Synthetic tests. No real ADNI images, patient records or GPU required.

The end-to-end evaluation test uses a deterministic tiny stand-in model, not a
claim about ConvNeXt accuracy; real checkpoint loading is checked separately.
"""
from __future__ import annotations
from dataclasses import asdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_evaluation as fe
import analyse_confidence as ac
from confidence_analysis import analyse_patients
from dataset import COLUMNS, FINAL_STATUS, SliceRecord, SplitBundle, counts_for_records, preprocessing_spec
from training_utils import aggregate_predictions
from train_experiment import ExperimentConfig, save_model, load_model
from modules import BaselineCNN


def row(subject, scan, label=0, split='test', index=80):
    name=['NC','AD'][label]
    return SliceRecord(f'AD_NC/train/{name}/{scan}_{index}.jpeg', subject, str(scan), index,'train',name,2*label,label,split)


def rule(name='Synthetic', threshold=.85):
    return dict(name=name,selected_epoch=1,temperature=1.6,diagnostic_threshold=.5,
        confidence_threshold=threshold,refer_all=threshold is None)


class MathTests(unittest.TestCase):
    def test_test_only_aggregate(self):
        with self.assertRaises(ValueError):fe.aggregate_test([row('a',1,split='train')],[.1])
    def test_no_empty_aggregate(self):
        with self.assertRaises(ValueError):fe.aggregate_test([],[])
    def test_no_missing_predictions(self):
        with self.assertRaises(ValueError):fe.aggregate_test([row('a',1)],[])
    def test_repeated_image_rejected(self):
        r=row('a',1)
        with self.assertRaises(ValueError):fe.aggregate_test([r,r],[.1,.2])
    def test_invalid_probability_rejected(self):
        for p in [-.1,1.1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):fe.aggregate_test([row('a',1)],[p])
    def test_scan_cannot_mix_patients(self):
        with self.assertRaises(ValueError):fe.aggregate_test([row('a',1,index=80),row('b',1,index=81)],[.1,.2])
    def test_patient_cannot_mix_diagnoses(self):
        with self.assertRaises(ValueError):fe.aggregate_test([row('a',1,0),row('a',2,1)],[.1,.2])
    def test_mean_of_scan_means_not_pooled_slices(self):
        records=[row('a',1,index=80),row('a',1,index=81),row('a',2,index=80),row('b',3,label=1)]
        _,p=fe.aggregate_test(records,[0,0,1,.8])
        self.assertEqual(p['patients'][0]['p_ad'],.5)
    def test_matches_inherited_validation_aggregation(self):
        rs=[row('a',1,0),row('b',2,1)]; p=[.2,.8]
        converted=[SliceRecord(**{**asdict(r),'split':'validation'}) for r in rs]
        expected,_=aggregate_predictions(converted,p)
        actual,_=fe.aggregate_test(rs,p)
        self.assertEqual(actual,expected)
    def test_calibration_keeps_classes(self):
        m,p=fe.score_model([row('a',1,0),row('b',2,1)],[.2,.8],rule())
        self.assertEqual([r['prediction'] for r in p['patients']],[0,1])
        self.assertEqual(m['metrics']['patient']['accuracy'],1)
    def test_refer_all_not_perfect_accuracy(self):
        m,_=fe.score_model([row('a',1,0),row('b',2,1)],[.9,.1],rule(threshold=None))
        self.assertEqual(m['triage']['coverage'],0)
        self.assertIsNone(m['triage']['accepted_error_rate'])
        self.assertFalse(m['descriptive_referral_constraints_met'])
    def test_target_not_met_is_result_not_failure(self):
        m,_=fe.score_model([row('a',1,0),row('b',2,1)],[.9,.1],rule())
        self.assertFalse(m['observed_all_patient_accuracy_at_least_0_80'])
    def test_pair_cohorts_must_match(self):
        with self.assertRaises(ValueError):fe.paired_comparison([{'subject_id':'a','label':0,'p_ad':.1}],[{'subject_id':'b','label':0,'p_ad':.1}])
    def test_paired_accuracy_difference_and_identical_interval(self):
        a=[dict(subject_id=str(i),label=i%2,p_ad=.1 if i%2==0 else .9) for i in range(10)]
        m=fe.paired_comparison(a,a,repeats=100)
        self.assertEqual(m['convnext_minus_cnn_accuracy'],0)
        self.assertEqual(m['paired_patient_bootstrap_95_percentile'],[0,0])
    def test_paired_direction(self):
        a=[dict(subject_id='a',label=0,p_ad=.9),dict(subject_id='b',label=1,p_ad=.1)]
        b=[dict(subject_id='a',label=0,p_ad=.1),dict(subject_id='b',label=1,p_ad=.9)]
        self.assertEqual(fe.paired_comparison(a,b,repeats=50)['convnext_minus_cnn_accuracy'],1)
    def test_no_test_refitting_functions_called(self):
        with (patch('confidence_analysis.fit_temperature',side_effect=AssertionError('No fit')),
              patch('confidence_analysis.select_referral',side_effect=AssertionError('No selection'))):
            fe.score_model([row('a',1,0),row('b',2,1)],[.4,.7],rule())


class CacheTests(unittest.TestCase):
    def test_accept_complete_nonmultiple_batch(self):
        self.assertEqual(len(fe.validate_progress({'identity':{},'probabilities':[.2]*10},{},10,8)),10)
    def test_reject_incomplete_batch(self):
        with self.assertRaises(ValueError):fe.validate_progress({'identity':{},'probabilities':[.2]*5},{},10,8)
    def test_reject_foreign_plan(self):
        with self.assertRaises(ValueError):fe.validate_progress({'identity':{'plan':1},'probabilities':[]},{'plan':2},10,8)
    def test_reject_oversized_cache(self):
        with self.assertRaises(ValueError):fe.validate_progress({'identity':{},'probabilities':[.2]*16},{},10,8)
    def test_reject_corrupt_cache_probability(self):
        for bad in [float('nan'),True,'0.5',2]:
            with self.assertRaises(ValueError):fe.validate_progress({'identity':{},'probabilities':[bad]},{},1,8)
    def test_image_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); r=row('a',1)
            file=root/r.image_path;file.parent.mkdir(parents=True);Image.new('L',(256,240),123).save(file)
            b=SplitBundle(root,{'test':(r,)},{},{})
            data=fe.VerifiedTestDataset(b,{r.image_path:ac.sha256(file)})
            self.assertEqual(data[0]['image'].shape,(1,224,224))
            file.write_bytes(b'changed')
            with self.assertRaises(ValueError):data[0]
    def test_missing_image_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):fe.VerifiedTestDataset(SplitBundle(Path(tmp),{'test':(row('a',1),)},{},{}),{})
    def test_completed_batches_reused_after_interruption(self):
        class Data:
            def __len__(self):return 12
            def __getitem__(self,i):return {'image':torch.ones(1,4,4)*i}
            def verified_bytes(self,i):return b'unchanged'
        class Model(torch.nn.Module):
            def __init__(self,fail=False):super().__init__();self.calls=0;self.fail=fail
            def forward(self,x):
                self.calls+=1
                if self.fail and self.calls==2:raise RuntimeError('synthetic interruption')
                t=x.mean((1,2,3));return torch.stack([-t,t],1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'progress.json'; identity={'alias':'test'}
            with self.assertRaises(RuntimeError):fe.cached_inference(Model(True),Data(),torch.device('cpu'),path,identity)
            self.assertEqual(len(ac.read_json(path)['probabilities']),8)
            model=Model();p,_=fe.cached_inference(model,Data(),torch.device('cpu'),path,identity)
            self.assertEqual(model.calls,1);self.assertEqual(len(p),12)
            model=Model();fe.cached_inference(model,Data(),torch.device('cpu'),path,identity)
            self.assertEqual(model.calls,0)


class FailureTests(unittest.TestCase):
    def patients(self):
        a=[];b=[]
        for i in range(8):
            label=i%2;p=.9 if label==0 else .1
            common={'subject_id':f'PRIVATE_SUBJECT_{i}','label':label}
            b.append({**common,'prediction':1-label,'confidence':.9-i*.01,'p_ad':p})
            a.append({**common,'prediction':label,'confidence':.8,'p_ad':.1 if label==0 else .9})
        return {'cnn':{'patients':a},'convnext':{'patients':b}}
    def test_five_distinct_error_cases(self):
        cases=fe.select_failure_cases(self.patients())
        self.assertEqual(len(cases),5);self.assertEqual(len({r['subject_id'] for r in cases}),5)
    def test_includes_false_negative_and_positive(self):
        cases=fe.select_failure_cases(self.patients())
        self.assertIn('negative',cases[0]['selection_reason']);self.assertIn('positive',cases[1]['selection_reason'])
    def test_no_errors_does_not_invent_cases(self):
        p=self.patients()
        for alias in p:
            for r in p[alias]['patients']:r['prediction']=r['label']
        self.assertEqual(fe.select_failure_cases(p),[])
    def test_fewer_errors_reports_actual_count(self):
        p=self.patients()
        for r in p['convnext']['patients'][2:]:r['prediction']=r['label']
        self.assertEqual(len(fe.select_failure_cases(p)),2)
    def test_failure_selection_deterministic(self):
        p=self.patients();first=fe.select_failure_cases(p)
        for v in p.values():v['patients'].reverse()
        self.assertEqual(first,fe.select_failure_cases(p))
    def test_marker_allows_only_same_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);out=p/'out';out.mkdir()
            fe.access_marker(p,out,'one');fe.access_marker(p,out,'one')
            with self.assertRaises(ValueError):fe.access_marker(p,out,'two')
    def test_recover_lock_requires_acknowledgement(self):
        with self.assertRaises(ValueError):fe.recover_lock(Path('.'),False)


class SyntheticFiles:
    """Complete small local experiment history; fake checkpoint bytes for orchestration."""
    def __init__(self, private):
        self.private=private;private.mkdir();self.project=Path(fe.__file__).resolve().parent
        (private/'project_path.txt').write_text(str(self.project))
        self.root=private/'data'/'ADNI';self.root.mkdir(parents=True)
        self.splits=private/'splits';self.splits.mkdir()
        self.records={};hashes={};counts={};inventory=[];number=1000
        for split in ('train','validation','test'):
            rs=[]
            for label,name in enumerate(('NC','AD')):
                for person in range(10 if split=='validation' else (3 if split=='test' else 1)):
                    number+=1; subject=f'PRIVATE_SYNTH_{split}_{label}_{person}'
                    for z in range(80,100):
                        r=row(subject,number,label,split,z);rs.append(r)
                        if split=='test':
                            path=self.root/r.image_path;path.parent.mkdir(parents=True,exist_ok=True)
                            value=210 if label else 45
                            if person==0:value=255-value
                            Image.new('L',(256,240),value).save(path)
                            inventory.append({'relative_path':r.image_path,'status':'decoded','file_sha256':ac.sha256(path)})
            self.records[split]=tuple(rs);counts[split]=counts_for_records(tuple(rs))
            stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=COLUMNS,lineterminator='\n');w.writeheader();w.writerows(asdict(r) for r in rs)
            file=self.splits/(split+'.csv');file.write_text(stream.getvalue(),encoding='utf-8');hashes[file.name]=ac.sha256(file)
        total=sum(v['candidate_subjects'] for v in counts.values())
        split_summary={'format_version':1,'status':FINAL_STATUS,'mapping_structural_checks_passed':True,
            'patient_identifier_semantics_confirmed_by_user':True,'tutor_split_change_approved_as_reported_by_user':True,
            'zero_group_overlap_verified':True,'splits_finalized':True,'new_pairwise_candidate_subject_overlap':0,
            'model_label_map':{'NC':0,'AD':1},'metadata_to_model_labels':{'0':0,'2':1},'manifest_sha256':hashes,
            'proposed_or_final_counts':counts,'evidence':{'images':total*20,'scan_prefixes':total,'candidate_subjects':total}}
        ac.write_json(self.splits/'split_summary.json',split_summary)
        (private/'latest_patient_splits.txt').write_text(str(self.splits))
        self.audit=private/'audit';self.audit.mkdir();ac.write_json(self.audit/'private_inventory.json',{'images':inventory})
        (private/'latest_adni_audit.txt').write_text(str(self.audit))
        sources={n:ac.sha256(self.project/n) for n in ('dataset.py','modules.py','training_utils.py','train.py','train_experiment.py','check_experiment.py')}
        self.runs={};reports={}
        for key,pointer in [('initial','latest_baseline_run.txt'),('baseline','latest_step8_baseline.txt'),('convnext','latest_step8_convnext.txt')]:
            folder=private/key;folder.mkdir();self.runs[key]=folder
            ps=[]
            for r in self.records['validation']:
                person=int(r.subject_id.rsplit('_',1)[-1]);p=.9 if r.label else .1
                if person%5==0:p=1-p
                if key=='initial':p=.5
                ps.append(p)
            metrics,pred=aggregate_predictions(self.records['validation'],ps)
            ac.write_json(folder/'PRIVATE_validation_predictions.json',pred)
            cp=folder/('PRIVATE_best_baseline.pt' if key=='initial' else 'PRIVATE_best_model.pt');cp.write_bytes(b'NOT_REAL_WEIGHTS_SYNTHETIC')
            report={'baseline_training_completed':True,'experiment_completed':True,'model_name':key,
                'final_test_model_evaluation_performed':False,'final_test_images_decoded':0,'selected_epoch':1,
                'manifest_sha256':hashes,'preprocessing':preprocessing_spec(0.0),'split_counts':{s:counts[s] for s in ('train','validation')},
                'configuration':{'model':key},'source_code_sha256':sources,'selected_validation_metrics':metrics,
                'selected_checkpoint_reloaded_and_checked':True,'selected_checkpoint_sha256':ac.sha256(cp)}
            ac.write_json(folder/('baseline_summary.json' if key=='initial' else 'experiment_summary.json'),report)
            (private/pointer).write_text(str(folder));reports[key]=report
        cmp=private/'comparison';cmp.mkdir()
        summary={'matched_step8_configuration_verified':True,'manifest_sha256':hashes,'preferred_cnn_by_prespecified_validation_rule':'Regularized CNN',
            'models':{name:{'selected_epoch':1,'patient_metrics':reports[k]['selected_validation_metrics']['patient']}
                      for k,name in [('initial','Initial CNN'),('baseline','Regularized CNN'),('convnext','ConvNeXt-Tiny')]}}
        ac.write_json(cmp/'comparison_summary.json',summary);(private/'latest_step8_comparison.txt').write_text(str(cmp))
        inputs=ac.check_inputs(private); s9=private/'confidence';s9.mkdir()
        summary={'step9_completed':True,'final_test_model_evaluation_performed':False,'comparison_sha256':inputs['comparison_sha256'],
                 'manifest_sha256':hashes,'source_code_sha256':{n:ac.sha256(self.project/n) for n in ac.NEW_SOURCE},'models':{}}
        rules={'models':{},'manifest_sha256':hashes,'input_probability_clip':[1e-7,1-1e-7]}
        for alias,m in inputs['models'].items():
            analyzed,_=analyse_patients(m['subjects'],m['labels'],m['p'])
            analyzed.update(name=m['name'],input_hashes=m['input_hashes'],original_validation_patient_metrics=m['report']['selected_validation_metrics']['patient'])
            summary['models'][alias]=analyzed
            rules['models'][alias]={'name':m['name'],'selected_epoch':1,'checkpoint_sha256':m['input_hashes']['selected_checkpoint_sha256'],
                'temperature':analyzed['full_validation_temperature_fit']['temperature'],'diagnostic_threshold':.5,
                'confidence_threshold':analyzed['operating_rule']['confidence_threshold'],
                'refer_all':analyzed['operating_rule']['confidence_threshold'] is None,'fit_partition':'validation','fit_patients':len(m['subjects'])}
        ac.write_json(s9/'confidence_summary.json',summary);ac.write_json(s9/'operating_rules.json',rules)
        (private/'latest_step9_confidence.txt').write_text(str(s9));self.step9=s9


class FrozenFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);actual=Path(fe.__file__).resolve().parent
        project=root/'repo'/'recognition'/'Synthetic';project.mkdir(parents=True)
        names=set(fe.NEW_FILES)|set(ac.NEW_SOURCE)|{'dataset.py','modules.py','training_utils.py',
            'train.py','train_experiment.py','check_experiment.py','analyse_confidence.py',
            'compare_step8.py','check_baseline.py'}
        for name in names:
            dst=project/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(actual/name,dst)
        # Every write stays in a temporary project, never the student's real repository.
        for module,name in [(fe,'final_evaluation.py'),(ac,'analyse_confidence.py')]:
            p=patch.object(module,'__file__',str(project/name));p.start();self.addCleanup(p.stop)
        self.f=SyntheticFiles(root/'private')
        self.plan_file=project/fe.PLAN_NAME
    def freeze(self):
        with patch.object(fe,'project_check'):
            return fe.freeze(self.f.private,'cpu')
    def test_freeze_opens_no_images(self):
        with patch.object(Image,'open',side_effect=AssertionError('No images during freeze')):
            out=self.freeze()
        self.assertTrue((out/fe.PLAN_NAME).exists());self.assertFalse((self.f.private/'FINAL_TEST_ACCESS_STARTED.json').exists())
    def test_second_freeze_rejected(self):
        self.freeze()
        with self.assertRaises(ValueError):self.freeze()
    def test_changed_rules_rejected_before_freeze(self):
        path=self.f.step9/'operating_rules.json';d=ac.read_json(path);d['models']['cnn']['temperature']+=1;ac.write_json(path,d)
        with self.assertRaises(ValueError):self.freeze()
    def test_missing_audit_image_hash_rejected(self):
        path=self.f.audit/'private_inventory.json';d=ac.read_json(path);d['images'].pop();ac.write_json(path,d)
        with self.assertRaises(ValueError):self.freeze()
    def test_changed_checkpoint_after_freeze_rejected(self):
        self.freeze();(self.f.runs['convnext']/'PRIVATE_best_model.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):fe.checked_plan(self.f.private,committed=False)
    def test_changed_plan_rejected(self):
        self.freeze();d=ac.read_json(self.plan_file);d['configuration']['batch_size']=16;ac.write_json(self.plan_file,d)
        with self.assertRaises(ValueError):fe.checked_plan(self.f.private,committed=False)
    def test_changed_step9_rules_after_freeze_rejected(self):
        self.freeze();path=self.f.step9/'operating_rules.json';d=ac.read_json(path);d['models']['cnn']['temperature']+=.1;ac.write_json(path,d)
        with self.assertRaises(ValueError):fe.checked_plan(self.f.private,committed=False)
    def test_changed_software_blocks_resume(self):
        self.freeze()
        with patch.object(fe,'software_versions',return_value={}):
            with self.assertRaises(ValueError):fe.checked_plan(self.f.private,committed=False)
    def test_lock_does_not_launch_concurrent_evaluation(self):
        out=self.freeze();(out/'EVALUATION_RUNNING.lock').write_text('{}')
        with patch.object(fe,'require_committed_plan',return_value='synthetic'):
            with self.assertRaises(ValueError):fe.evaluate(self.f.private)
    def test_complete_synthetic_freeze_evaluate_and_bundle(self):
        torch.set_num_threads(1);out=self.freeze()
        class Simple(torch.nn.Module):
            def forward(self,x):
                t=x.mean((1,2,3))*2;return torch.stack([-t,t],1)
        def model(*args,**kwargs):return Simple(),{'epoch':1}
        repo=self.f.project.parents[1]
        def g(*args):
            subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True)
        g('init','-b','topic-recognition');g('config','user.name','Synthetic Test');g('config','user.email','synthetic@example.invalid')
        g('add','recognition');g('commit','-m','Freeze synthetic plan')
        self.assertTrue(fe.require_committed_plan(self.f.project,ac.read_json(out/fe.PLAN_NAME)))
        with patch.object(fe,'load_model',side_effect=model):
            fe.evaluate(self.f.private)
        summary=ac.read_json(out/'final_test_summary.json')
        self.assertTrue(summary['quantitative_evaluation_completed']);self.assertFalse(summary['test_used_for_fitting_or_selection'])
        self.assertFalse(summary['manual_failure_review_completed'])
        self.assertEqual(summary['models']['cnn']['metrics']['patient']['n'],6)
        self.assertTrue((out/'PRIVATE_failure_review'/'PRIVATE_failure_review.md').exists())
        with ZipFile(out/'COMP3710_Step10_Results.zip') as z:
            self.assertEqual(set(z.namelist()),set(fe.PUBLIC_FILES))
            for name in z.namelist():
                if not name.endswith('.png'):
                    text=z.read(name).decode();self.assertNotIn('PRIVATE_SYNTH_',text);self.assertNotIn(str(self.f.private),text)
        with patch.object(fe,'load_model',side_effect=AssertionError('No repeat inference')):
            fe.evaluate(self.f.private)
    def test_uncommitted_plan_cannot_open_test(self):
        self.freeze()
        with self.assertRaises(ValueError):fe.require_committed_plan(self.f.project,ac.read_json(self.plan_file))
    def test_real_baseline_checkpoint_loading(self):
        torch.set_num_threads(1)
        model=BaselineCNN();cfg=ExperimentConfig(model='baseline')
        path=Path(self.tmp.name)/'real_baseline.pt';hashes={'train.csv':'a','validation.csv':'b','test.csv':'c'}
        save_model(path,model,cfg,hashes,2)
        actual,checkpoint=load_model(path,torch.device('cpu'),hashes,'baseline')
        self.assertEqual(checkpoint['epoch'],2)
        with torch.inference_mode():self.assertEqual(actual(torch.zeros(1,1,224,224)).shape,(1,2))
        with self.assertRaises(ValueError):load_model(path,torch.device('cpu'),hashes,'convnext')


if __name__=='__main__':unittest.main()
