"""Synthetic integration tests. No private ADNI, network or GPU required."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import selected_a1 as s
from dataset import SliceRecord, SplitBundle, ADNISliceDataset, preprocessing_spec
from modules import ConvNeXtTiny, BaselineCNN
from training_utils import aggregate_predictions
from confidence_analysis import analyse_patients

torch.set_num_threads(1)


class TinyModel(nn.Module):
    """Synthetic stand-in for integration flow only, not the actual architecture."""
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(1.25))
    def forward(self,x):
        v=x.mean((1,2,3))*self.scale
        return torch.stack([-v,v],1)


def make_fixture(root:Path):
    private=root/'private files';private.mkdir()
    project=root/'code with spaces'/'recognition'/'assignment';project.mkdir(parents=True)
    (project/'README.md').write_bytes(b'# Existing report\r\nKeep this evidence.\r\n')
    for name in s.NEW_FILES:
        p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic source '+name)
    (project/'historical.py').write_text('# unchanged source\n')
    sources={name:s.sha256(project/name) for name in (*s.NEW_FILES,'historical.py')}
    historical={'historical.py':sources['historical.py']}
    data=private/'data';data.mkdir()
    records=[]
    # >=5 per class for the unchanged five-fold cross-fitting implementation.
    for i in range(12):
        label=i//6
        for j in range(2):
            rel=f'validation/{label}/{i}_{j}.jpeg';p=data/rel;p.parent.mkdir(parents=True,exist_ok=True)
            pixels=np.full((240,256),40+label*150+i+j,dtype=np.uint8)
            Image.fromarray(pixels).save(p)
            records.append(SliceRecord(rel,f'PRIVATE_PERSON_{i}',f'SCAN_{i}',j,'train',
                                       'NC' if label==0 else 'AD',0 if label==0 else 2,label,'validation'))
    hashes={'train.csv':'a'*64,'validation.csv':'b'*64,'test.csv':'c'*64}
    bundle=SplitBundle(data,{'train':(), 'validation':tuple(records),'test':()}, {},hashes)
    model=TinyModel().eval();ds=ADNISliceDataset(bundle,'validation')
    probs,_=s.inference(model,ds,torch.device('cpu'))
    metrics,grouped=aggregate_predictions(records,probs)
    subjects=[r['subject_id'] for r in grouped['patients']]
    labels=[r['label'] for r in grouped['patients']]
    pp=[r['p_ad'] for r in grouped['patients']]
    candidates={}
    for alias in ('cnn','a1'):
        folder=private/alias;folder.mkdir()
        cfg={'recipe':'scratch_augmented','seed':42,'batch_size':8} if alias=='a1' else {'model':'baseline','batch_size':8}
        ep=13 if alias=='a1' else 14
        report={'status':'COMPLETED','experiment_completed':True,'selected_epoch':ep,
                'configuration':cfg,'parameters':1,'source_code_sha256':historical,
                'manifest_sha256':hashes,'selected_validation_metrics':metrics,'pretrained':{'used':False},
                'model_name':'baseline','recipe':'scratch_augmented','operating_threshold':.5,
                'preprocessing':preprocessing_spec(0)}
        reportname='v2_summary.json' if alias=='a1' else 'experiment_summary.json'
        checkpoint={'format_version':2 if alias=='a1' else 1,
            'architecture':'ConvNeXtTiny' if alias=='a1' else 'BaselineCNN-v1',
            'model_name':'baseline','epoch':ep,'configuration':cfg,'manifest_sha256':hashes,
            'preprocessing':preprocessing_spec(0),'model_label_map':{'NC':0,'AD':1},
            'source_code_sha256':historical,'pretrained':{'used':False},'model_state':model.state_dict()}
        ck=folder/'PRIVATE_best_model.pt';torch.save(checkpoint,ck)
        pred=folder/'PRIVATE_validation_predictions.json';s.write_json(pred,grouped)
        s.write_json(folder/reportname,report)
        candidates[alias]=s.Candidate(alias,alias,folder,report,reportname,ck,pred,subjects,labels,pp,probs,
            {'checkpoint_sha256':s.sha256(ck),'validation_predictions_sha256':s.sha256(pred),
             'run_summary_sha256':s.sha256(folder/reportname)})
    old,_=analyse_patients(subjects,labels,pp)
    ctx={'private':private,'project':project,'bundle':bundle,'candidates':candidates,
         'sources':sources,'pipeline_sha256':'d'*64,'old_confidence':{'models':{'cnn':old}}}
    return ctx


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.ctx=make_fixture(Path(self.tmp.name))
        self.records=self.ctx['bundle'].records['validation']
    def save_checkpoint(self,c,value):
        torch.save(value,c.checkpoint);c.hashes['checkpoint_sha256']=s.sha256(c.checkpoint)
    def read_checkpoint(self,c):
        return torch.load(c.checkpoint,map_location='cpu',weights_only=True)


class CachedEvidenceTests(FixtureCase):
    def test_original_cached_predictions_reaggregate(self):
        c=self.ctx['candidates']['a1']
        ids,y,p=s.verify_cached_predictions(c.prediction_file,self.records,c.report)
        self.assertEqual(ids,c.subjects);self.assertEqual(y,c.labels);self.assertEqual(p,c.p)
    def test_foreign_slice_path_rejected(self):
        c=self.ctx['candidates']['a1'];data=s.read_json(c.prediction_file)
        data['slices'][0]['image_path']='test/foreign.jpeg';s.write_json(c.prediction_file,data)
        with self.assertRaises(ValueError):s.verify_cached_predictions(c.prediction_file,self.records,c.report)
    def test_duplicate_slice_rejected(self):
        c=self.ctx['candidates']['a1'];data=s.read_json(c.prediction_file)
        data['slices'][1]=data['slices'][0];s.write_json(c.prediction_file,data)
        with self.assertRaises(ValueError):s.verify_cached_predictions(c.prediction_file,self.records,c.report)
    def test_changed_subject_rejected(self):
        c=self.ctx['candidates']['a1'];data=s.read_json(c.prediction_file)
        data['slices'][0]['subject_id']='WRONG';s.write_json(c.prediction_file,data)
        with self.assertRaises(ValueError):s.verify_cached_predictions(c.prediction_file,self.records,c.report)
    def test_changed_summary_metrics_rejected(self):
        c=self.ctx['candidates']['a1'];r=deepcopy(c.report);r['selected_validation_metrics']['patient']['accuracy']=.1
        with self.assertRaises(ValueError):s.verify_cached_predictions(c.prediction_file,self.records,r)
    def test_test_partition_cannot_reach_inference(self):
        ds=ADNISliceDataset(self.ctx['bundle'],'validation');ds.split='test'
        with self.assertRaises(ValueError):s.inference(TinyModel(),ds,torch.device('cpu'))


class LoaderTests(FixtureCase):
    def load(self,c):
        return s.load_selected_checkpoint(c,self.ctx['bundle'].manifest_hashes,torch.device('cpu'),
                                         {'a1':TinyModel,'cnn':TinyModel})
    def test_both_checkpoint_formats_load(self):
        for c in self.ctx['candidates'].values():
            self.assertFalse(self.load(c).training)
    def test_wrong_epoch_rejected(self):
        c=self.ctx['candidates']['a1'];v=self.read_checkpoint(c);v['epoch']=21;self.save_checkpoint(c,v)
        with self.assertRaises(ValueError):self.load(c)
    def test_pretrained_flag_rejected(self):
        c=self.ctx['candidates']['a1'];v=self.read_checkpoint(c);v['pretrained']={'used':True};self.save_checkpoint(c,v)
        with self.assertRaises(ValueError):self.load(c)
    def test_changed_checkpoint_bytes_rejected(self):
        c=self.ctx['candidates']['a1'];c.checkpoint.write_bytes(b'not the selected checkpoint')
        with self.assertRaises(ValueError):self.load(c)
    def test_wrong_manifest_rejected(self):
        c=self.ctx['candidates']['a1'];v=self.read_checkpoint(c);v['manifest_sha256']={};self.save_checkpoint(c,v)
        with self.assertRaises(ValueError):self.load(c)
    def test_wrong_labels_rejected(self):
        c=self.ctx['candidates']['cnn'];v=self.read_checkpoint(c);v['model_label_map']={'AD':0,'NC':1};self.save_checkpoint(c,v)
        with self.assertRaises(ValueError):self.load(c)
    def test_wrong_architecture_rejected(self):
        c=self.ctx['candidates']['a1'];v=self.read_checkpoint(c);v['architecture']='RGBTransferConvNeXt';self.save_checkpoint(c,v)
        with self.assertRaises(ValueError):self.load(c)
    def test_state_shape_mismatch_rejected(self):
        c=self.ctx['candidates']['a1'];v=self.read_checkpoint(c);v['model_state']['scale']=torch.ones(2);self.save_checkpoint(c,v)
        with self.assertRaises(RuntimeError):self.load(c)
    def test_singleton_vector_cannot_replace_scalar(self):
        c=self.ctx['candidates']['a1'];v=self.read_checkpoint(c)
        v['model_state']['scale']=torch.ones(1);self.save_checkpoint(c,v)
        with self.assertRaisesRegex(RuntimeError, 'shape mismatch for scale'):
            self.load(c)
    def test_state_non_tensor_entry_rejected(self):
        c=self.ctx['candidates']['a1'];v=self.read_checkpoint(c)
        v['model_state']['scale']=1.25;self.save_checkpoint(c,v)
        with self.assertRaisesRegex(RuntimeError, 'must be a torch.Tensor'):
            self.load(c)
    def test_state_missing_key_rejected(self):
        c=self.ctx['candidates']['a1'];v=self.read_checkpoint(c)
        del v['model_state']['scale'];self.save_checkpoint(c,v)
        with self.assertRaisesRegex(RuntimeError, 'keys do not match'):
            self.load(c)
    def test_state_unexpected_key_rejected(self):
        c=self.ctx['candidates']['a1'];v=self.read_checkpoint(c)
        v['model_state']['unexpected']=torch.tensor(1.0);self.save_checkpoint(c,v)
        with self.assertRaisesRegex(RuntimeError, 'keys do not match'):
            self.load(c)
    def test_state_non_dictionary_rejected(self):
        c=self.ctx['candidates']['a1'];v=self.read_checkpoint(c)
        v['model_state']=[];self.save_checkpoint(c,v)
        with self.assertRaisesRegex(RuntimeError, 'must be a tensor dictionary'):
            self.load(c)
    def test_bad_shape_rejected_before_pytorch_loading(self):
        c=self.ctx['candidates']['a1'];v=self.read_checkpoint(c)
        v['model_state']['scale']=torch.ones(2);self.save_checkpoint(c,v)
        with patch.object(TinyModel, 'load_state_dict') as load:
            with self.assertRaisesRegex(RuntimeError, 'shape mismatch for scale'):
                self.load(c)
            load.assert_not_called()
    def test_same_numel_different_shape_rejected(self):
        c=self.ctx['candidates']['a1'];v=self.read_checkpoint(c)
        factory=lambda: nn.Linear(3,2,bias=False)
        c.report['parameters']=6
        v['model_state']={'weight':torch.ones(3,2)};self.save_checkpoint(c,v)
        with self.assertRaisesRegex(RuntimeError, 'shape mismatch for weight'):
            s.load_selected_checkpoint(c,self.ctx['bundle'].manifest_hashes,
                                       torch.device('cpu'),{'a1':factory})
    def test_changed_source_metadata_rejected(self):
        c=self.ctx['candidates']['cnn'];v=self.read_checkpoint(c);v['source_code_sha256']={};self.save_checkpoint(c,v)
        with self.assertRaises(ValueError):self.load(c)
    def test_actual_full_convnext_state_load(self):
        c=self.ctx['candidates']['a1'];model=ConvNeXtTiny()
        c.report['parameters']=s.parameter_count(model)
        v=self.read_checkpoint(c);v['model_state']=model.state_dict();self.save_checkpoint(c,v)
        loaded=s.load_selected_checkpoint(c,self.ctx['bundle'].manifest_hashes,torch.device('cpu'))
        self.assertEqual(s.parameter_count(loaded),27818594)
        with torch.inference_mode():
            out=loaded(torch.zeros(1,1,224,224))
        self.assertEqual(tuple(out.shape),(1,2));self.assertTrue(torch.isfinite(out).all())
    def test_actual_baseline_state_load(self):
        c=self.ctx['candidates']['cnn'];model=BaselineCNN();c.report['parameters']=s.parameter_count(model)
        v=self.read_checkpoint(c);v['model_state']=model.state_dict();self.save_checkpoint(c,v)
        loaded=s.load_selected_checkpoint(c,self.ctx['bundle'].manifest_hashes,torch.device('cpu'))
        self.assertEqual(s.parameter_count(loaded),359922)


class ReplayTests(FixtureCase):
    def test_same_predictions_pass(self):
        p=self.ctx['candidates']['a1'].slice_probabilities
        self.assertTrue(s.replay_checks(self.records,p,p,2e-5)['passed'])
    def test_small_roundoff_passes(self):
        p=self.ctx['candidates']['a1'].slice_probabilities
        q=[v+1e-7 for v in p]
        self.assertTrue(s.replay_checks(self.records,p,q,2e-5)['passed'])
    def test_large_difference_stops(self):
        p=self.ctx['candidates']['a1'].slice_probabilities;q=list(p);q[0]+=.001
        with self.assertRaises(ValueError):s.replay_checks(self.records,p,q,2e-5)
    def test_small_difference_crossing_cutoff_stops(self):
        p=[.499999]*len(self.records);q=[.500001]*len(self.records)
        with self.assertRaises(ValueError):s.replay_checks(self.records,p,q,2e-5)
    def test_nan_probability_stops(self):
        p=self.ctx['candidates']['a1'].slice_probabilities;q=list(p);q[0]=float('nan')
        with self.assertRaises(ValueError):s.replay_checks(self.records,p,q,2e-5)
    def test_partial_replay_stops(self):
        p=self.ctx['candidates']['a1'].slice_probabilities
        with self.assertRaises(ValueError):s.replay_checks(self.records,p,p[:-1],2e-5)
    def test_receipt_reuses_completed_model(self):
        c=self.ctx['candidates']['cnn'];folder=self.ctx['private']/'replay';folder.mkdir()
        args=(c,self.ctx['bundle'],torch.device('cpu'),folder,{'plan':'fixed'})
        r=s.verify_candidate(*args,factories={'cnn':TinyModel},benchmark_repeats=1)
        with patch.object(s,'inference',side_effect=AssertionError('must not repeat')):
            r2=s.verify_candidate(*args,factories={'cnn':TinyModel},benchmark_repeats=1)
        self.assertEqual(r,r2)
    def test_changed_receipt_binding_stops(self):
        c=self.ctx['candidates']['cnn'];folder=self.ctx['private']/'replay';folder.mkdir()
        s.verify_candidate(c,self.ctx['bundle'],torch.device('cpu'),folder,{'plan':'a'},
                           {'cnn':TinyModel},1)
        with self.assertRaises(ValueError):
            s.verify_candidate(c,self.ctx['bundle'],torch.device('cpu'),folder,{'plan':'b'},
                               {'cnn':TinyModel},1)


class ReportTests(FixtureCase):
    def test_selection_is_id_free_and_non_test(self):
        text=json.dumps(s.selection(self.ctx))
        self.assertNotIn('PRIVATE_PERSON',text);self.assertNotIn(str(self.ctx['private']),text)
        self.assertFalse(s.selection(self.ctx)['test_evaluation_authorised_by_this_plan'])
    def test_prepare_identical_repeat_allowed(self):
        a=s.prepare(self.ctx);b=s.prepare(self.ctx);self.assertEqual(a,b)
    def test_prepare_changed_selection_rejected(self):
        s.prepare(self.ctx);self.ctx['candidates']['a1'].report['selected_epoch']=99
        with self.assertRaises(ValueError):s.prepare(self.ctx)
    def test_bad_seed_stops(self):
        report=deepcopy(self.ctx['candidates']['a1'].report);report['configuration']['seed']=43
        with self.assertRaises(ValueError):s.validate_report(report,'a1',self.ctx['bundle'].manifest_hashes)
    def test_bad_completion_stops(self):
        report=deepcopy(self.ctx['candidates']['cnn'].report);report['experiment_completed']=False
        with self.assertRaises(ValueError):s.validate_report(report,'cnn',self.ctx['bundle'].manifest_hashes)
    def test_end_to_end_private_outputs_public_bundle_and_idempotence(self):
        s.prepare(self.ctx)
        before={str(p):p.read_bytes() for c in self.ctx['candidates'].values()
                for p in (c.checkpoint,c.prediction_file,c.folder/c.report_name)}
        out=s.run(self.ctx,'cpu',factories={'cnn':TinyModel,'a1':TinyModel},benchmark_repeats=1,enforce_git=False)
        report=s.read_json(out/s.SUMMARY)
        self.assertTrue(report['a1_confidence_completed']);self.assertEqual(report['final_test_images_decoded'],0)
        self.assertTrue(report['models']['cnn']['previous_cnn_rules_reproduced'])
        with ZipFile(out/s.BUNDLE) as z:
            self.assertEqual(set(z.namelist()),set(s.PUBLIC))
            for name in z.namelist():
                if not name.endswith('.png'):
                    text=z.read(name).decode();self.assertNotIn('PRIVATE_PERSON',text)
                    self.assertNotIn(str(self.ctx['private']),text)
        old=report['created_utc']
        with patch.object(s,'inference',side_effect=AssertionError('do not repeat completed run')):
            s.run(self.ctx,'cpu',enforce_git=False)
        self.assertEqual(s.read_json(out/s.SUMMARY)['created_utc'],old)
        s.publish(self.ctx);text=(self.ctx['project']/'README.md').read_text()
        self.assertIn('Existing report',text);self.assertEqual(text.count(s.MARKER),1)
        s.publish(self.ctx);self.assertEqual((self.ctx['project']/'README.md').read_text(),text)
        for path,content in before.items():self.assertEqual(Path(path).read_bytes(),content)
    def test_interruption_after_first_model_reuses_it(self):
        s.prepare(self.ctx);original=s.verify_candidate
        def interrupt(c,*args,**kw):
            if c.alias=='a1':raise KeyboardInterrupt()
            return original(c,*args,**kw)
        with patch.object(s,'verify_candidate',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                s.run(self.ctx,'cpu',factories={'cnn':TinyModel,'a1':TinyModel},benchmark_repeats=1,enforce_git=False)
        self.assertFalse((self.ctx['private']/s.POINTER).exists())
        folder=self.ctx['private']/s.FOLDER
        self.assertTrue((folder/'cnn_verification.json').exists())
        s.run(self.ctx,'cpu',factories={'cnn':TinyModel,'a1':TinyModel},benchmark_repeats=1,enforce_git=False)
        self.assertTrue((self.ctx['private']/s.POINTER).is_file())
    def test_cnn_rule_change_rejected(self):
        self.ctx['old_confidence']['models']['cnn']['full_validation_temperature_fit']['temperature']=9
        with self.assertRaises(ValueError):s.analyse(self.ctx,{'cnn':{},'a1':{}})
    def test_source_change_rejected(self):
        (self.ctx['project']/'historical.py').write_text('changed')
        with self.assertRaises(ValueError):s.check_sources(self.ctx['project'],self.ctx['sources'])
    def test_public_plan_cannot_be_used_without_prepare(self):
        with self.assertRaises(ValueError):s.run(self.ctx,'cpu',enforce_git=False)


class GitPlanTests(FixtureCase):
    def setup_git(self):
        project=self.ctx['project'];repo=project.parent.parent
        def g(*args):
            return subprocess.run(['git','-C',str(repo),*args],capture_output=True,check=True)
        g('init');g('config','user.email','synthetic@example.invalid');g('config','user.name','Synthetic test')
        g('checkout','-b','topic-recognition')
        return project,g
    def test_project_location_and_commit_guard(self):
        project,g=self.setup_git();s.project_check(project);s.prepare(self.ctx)
        with self.assertRaises(ValueError):s.require_committed(project)
        g('add','.');g('commit','-m','Synthetic plan')
        s.require_committed(project)
        (project/'selected_a1.py').write_text('changed implementation')
        with self.assertRaises(ValueError):s.require_committed(project)
    def test_wrong_branch_rejected(self):
        project,g=self.setup_git();g('checkout','-b','wrong-branch')
        with self.assertRaises(ValueError):s.project_check(project)
    def test_root_directory_rejected(self):
        project,g=self.setup_git()
        with self.assertRaises(ValueError):s.project_check(project.parent.parent)


class ContextTests(unittest.TestCase):
    def setUp(self):
        # Reuse the earlier suite's known manifest fixture. It creates NO images.
        from tests_confidence.test_confidence import SyntheticRun
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=SyntheticRun(Path(self.tmp.name)/'private')
        private=self.f.private;project=self.f.project
        for key,alias,epoch,param in [('baseline','cnn',14,359922),('convnext','a1',13,27818594)]:
            folder=self.f.paths[key];r=self.f.reports[key]
            r.update(status='COMPLETED',parameters=param,selected_epoch=epoch,
                     model_name='baseline' if alias=='cnn' else 'convnext',
                     recipe='scratch_augmented',pretrained={'used':False},operating_threshold=.5)
            r['configuration'].update(recipe='scratch_augmented',seed=42)
            s.write_json(folder/('v2_summary.json' if alias=='a1' else 'experiment_summary.json'),r)
        (private/'latest_v2_scratch_augmented.txt').write_text(str(self.f.paths['convnext']))
        comparison=private/'comparison'/'comparison_summary.json';v=s.read_json(comparison)
        v['models']['Regularized CNN']['selected_epoch']=14;s.write_json(comparison,v)
        pipeline=private/'pipeline';pipeline.mkdir()
        s.write_json(pipeline/'data_pipeline_summary.json',{
            'data_pipeline_check_passed':True,'baseline_training_can_begin_after_visual_review':True,
            'manifest_sha256':self.f.reports['baseline']['manifest_sha256'],
            'source_code_sha256':{'dataset.py':s.sha256(project/'dataset.py')},
            'preprocessing':preprocessing_spec(0)})
        (private/'latest_data_pipeline.txt').write_text(str(pipeline))
        conf=private/'old_confidence';conf.mkdir()
        s.write_json(conf/'confidence_summary.json',{'step9_completed':True,
            'source_code_sha256':{'confidence_analysis.py':s.sha256(project/'confidence_analysis.py')}})
        (private/'latest_step9_confidence.txt').write_text(str(conf))
    def test_context_accepts_real_manifest_schema_without_images(self):
        with patch.object(s,'project_check'):
            ctx=s.context(self.f.private)
        self.assertEqual(ctx['candidates']['a1'].report['selected_epoch'],13)
        self.assertEqual(len(ctx['candidates']['a1'].subjects),40)
        self.assertFalse(any((self.f.private/'data').rglob('*.jpeg')))
    def test_wrong_a1_pointer_rejected(self):
        (self.f.private/'latest_v2_scratch_augmented.txt').write_text(str(self.f.paths['baseline']))
        with patch.object(s,'project_check'):
            with self.assertRaises(ValueError):s.context(self.f.private)
    def test_changed_confidence_source_hash_rejected(self):
        path=self.f.private/'old_confidence'/'confidence_summary.json';r=s.read_json(path)
        r['source_code_sha256']['confidence_analysis.py']='0'*64;s.write_json(path,r)
        with patch.object(s,'project_check'):
            with self.assertRaises(ValueError):s.context(self.f.private)


if __name__=='__main__':unittest.main()
