"""Test fixed-model full-test scope, provenance guards, isolated exports and interleaved plans."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

import run_heldout as subject
from run_comparison import RUNTIME_FILES


def template():
    return json.loads(Path(__file__).with_name('heldout_protocol.json').read_text(encoding='utf-8'))


def metrics(matrix):
    rows=[]
    for label in range(6):
        tp=matrix[label][label];actual=sum(matrix[label]);predicted=sum(row[label] for row in matrix)
        denominator=actual+predicted;union=actual+predicted-tp
        rows.append({'label_id':label,'tp':tp,'fp':predicted-tp,'fn':actual-tp,
                     'target_pixels':actual,'predicted_pixels':predicted,
                     'dice':2*tp/denominator if denominator else None,'iou':tp/union if union else None})
    return {'pixels':sum(map(sum,matrix)),'confusion':matrix,'classes':rows}


def fixture(home):
    p=template();p['data_root']=str(home/'data')
    runtime=home/p['runtime_relative'];runtime.mkdir(parents=True)
    for name in RUNTIME_FILES:(runtime/name).write_bytes(('frozen '+name+'\n').encode())
    hashes={name:subject.sha256_file(runtime/name) for name in RUNTIME_FILES}
    runs={}
    for spec in p['models']:
        directory=home/spec['run_relative'];(directory/'training').mkdir(parents=True)
        checkpoint=directory/'training/best.pt';checkpoint.write_bytes(spec['architecture'].encode())
        spec['checkpoint_sha256']=subject.sha256_file(checkpoint)
        model_config={'in_channels':1,'num_classes':6,'base_channels':16}
        if spec['architecture']=='ImprovedUNet2D':model_config['dropout']=0.3
        runs[spec['architecture']]={
            'directory':str(directory),'provenance':{'source_commit':p['source_commit'],
                'source_directory':str(runtime),'source_sha256':hashes},
            'protocol':{'data':{'root':p['data_root']}},'data':{'matched':'fixture'},
            'config':{'splits':{'train_patients':[4],'validation_patients':[36,37,38,39]},'source_sha256':hashes},
            'validation':{'best_epoch':spec['checkpoint_epoch'],'checkpoint_sha256':spec['checkpoint_sha256']},
            'result':{'training_gpu_profile':{'model_config':model_config}}}
    images=home/'data/keras_slices_test';masks=home/'data/keras_slices_seg_test'
    images.mkdir(parents=True);masks.mkdir(parents=True)
    for patient in (40,41,42):
        for index in range(180):
            suffix=f'{patient:03d}_week_0_slice_{index}.nii.gz'
            (images/('case_'+suffix)).write_bytes(b'image fixture')
            (masks/('seg_'+suffix)).write_bytes(b'mask fixture')
    return p,runtime,runs


def synthetic_report(p,runs,data,name,trial):
    slices=[]
    for index,entry in enumerate(data['manifest']):
        target=index%2;prediction=target if name=='UNet2D' else 1
        matrix=[[0]*6 for _ in range(6)];matrix[target][prediction]=1
        slices.append({'index':index,**{k:entry[k] for k in ('image','mask','patient_id','week','slice_id')},
                       'shape':[1,1],'model_forward_ms':(2 if name=='UNet2D' else 5)+trial/10+index%3/100,
                       **metrics(matrix)})
    patients=[];cases=[]
    for patient in p['test_patients']:
        members=[row for row in slices if row['patient_id']==patient]
        matrix=[[sum(row['confusion'][i][j] for row in members) for j in range(6)] for i in range(6)]
        group={'patient_id':patient,'samples':len(members),'processing_seconds':0.5+trial/100,**metrics(matrix)}
        patients.append(group);cases.append({**group,'week':0})
    matrix=[[sum(row['confusion'][i][j] for row in slices) for j in range(6)] for i in range(6)]
    spec=next(row for row in p['models'] if row['architecture']==name)
    hashes=runs[name]['provenance']['source_sha256']
    return {'architecture':name,'split':'test','smoke_only':False,'partial_checkpoint':False,
            'limited_evaluation':False,'max_samples':0,'checkpoint_sha256':spec['checkpoint_sha256'],
            'checkpoint_epoch':spec['checkpoint_epoch'],'model_config':runs[name]['result']['training_gpu_profile']['model_config'],
            'processed_samples':540,'split_samples':540,'split_patient_ids':[40,41,42],'processed_patient_ids':[40,41,42],
            'device':'cuda','batch_size':1,'warmup_steps_per_shape':3,'normalization':'per-slice z-score',
            'checkpoint_source_sha256':runs[name]['config']['source_sha256'],
            'evaluation_source_sha256':{key:hashes[key] for key in ('evaluate.py','predict.py','model_factory.py','dataset.py','metrics.py')},
            'device_name':'Synthetic A100 record','torch_version':'synthetic','torch_cuda_build':'synthetic',
            'pooled':metrics(matrix),'patients':patients,'cases':cases,'slices':slices,
            'median_model_forward_ms':statistics.median(row['model_forward_ms'] for row in slices),
            'latency_scope':'Synthetic model-only timing record','case_time_scope':'Synthetic case-processing record'}


class HeldoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.home=Path(self.temporary.name)

    def test_protocol_refuses_partial_wrong_order_and_anatomical_selection(self):
        subject.validate_test_protocol(template())
        variants=[]
        p=template();p['evaluation']['max_samples']=2;variants.append(p)
        p=template();p['evaluation']['device']='cpu';variants.append(p)
        p=template();p['evaluation']['order'].reverse();variants.append(p)
        p=template();p['test_samples']=2;variants.append(p)
        p=template();p['reporting']['label_ids']=[1];variants.append(p)
        p=template();p['models'][0]['checkpoint_sha256']='invalid';variants.append(p)
        for p in variants:
            with self.assertRaises(ValueError):subject.validate_test_protocol(p)

    def test_plan_is_six_interleaved_test_calls_without_training_or_partial_flags(self):
        with patch.object(subject.Path,'home',return_value=self.home):
            plan=subject.build_test_plan(template(),self.home/'output',self.home/'runtime','python')
        self.assertEqual([stage for stage,_ in plan],
             ['UNet2D_test_1','ImprovedUNet2D_test_1','UNet2D_test_2','ImprovedUNet2D_test_2','UNet2D_test_3','ImprovedUNet2D_test_3'])
        for _,command in plan:
            self.assertEqual(command[command.index('--split')+1],'test')
            self.assertEqual(command[command.index('--max-samples')+1],'0')
            self.assertNotIn('--allow-partial',command)
            self.assertNotIn('train.py',' '.join(command))

    def test_dry_run_has_no_data_reads_children_or_outputs(self):
        destination=self.home/'dry'
        with patch.object(subject,'prepare_models') as prepare, patch.object(subject,'audit_test_files') as audit, \
             patch.object(subject.subprocess,'run') as child,contextlib.redirect_stdout(io.StringIO()):
            subject.main(['--output-dir',str(destination),'--runner-commit','f'*40,'--dry-run'])
        prepare.assert_not_called();audit.assert_not_called();child.assert_not_called()
        self.assertFalse(destination.exists())

    def test_preflight_guards_masks_overlap_and_fixed_checkpoint_bytes(self):
        p,runtime,runs=fixture(self.home)
        with patch.object(subject.Path,'home',return_value=self.home), \
             patch.object(subject,'inspect_run',side_effect=lambda directory,name:runs[name]):
            subject.prepare_models(p)
            data=subject.audit_test_files(p,runs)
            self.assertEqual(len(data['manifest']),540)
            mask=Path(p['data_root'])/'keras_slices_seg_test'/data['manifest'][0]['mask'];original=mask.read_bytes();mask.unlink()
            with self.assertRaisesRegex(ValueError,'Missing test mask'):subject.audit_test_files(p,runs)
            mask.write_bytes(original)
            runs['UNet2D']['config']['splits']['validation_patients'].append(40)
            with self.assertRaisesRegex(ValueError,'overlap'):subject.audit_test_files(p,runs)
            runs['UNet2D']['config']['splits']['validation_patients'].remove(40)
            checkpoint=self.home/p['models'][0]['run_relative']/'training/best.pt';checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint bytes'):subject.prepare_models(p)

    def run_mocked(self,mutate_source=False):
        p,runtime,runs=fixture(self.home)
        data=subject.audit_test_files(p,runs)
        actual_read=subject.read_json
        def read(path):
            return copy.deepcopy(p) if Path(path).name=='heldout_protocol.json' else actual_read(path)
        calls=[]
        def child(command,**kwargs):
            destination=Path(command[command.index('--output-dir')+1]);destination.mkdir(parents=True)
            name=destination.parent.name;trial=int(destination.name.rsplit('_',1)[1])
            subject.write_json(destination/'evaluation.json',synthetic_report(p,runs,data,name,trial))
            calls.append(name)
            if mutate_source:(runtime/'evaluate.py').write_bytes(b'mutated source')
        destination=self.home/'output'
        with patch.object(subject.Path,'home',return_value=self.home), \
             patch.object(subject,'read_json',side_effect=read), \
             patch.object(subject,'inspect_run',side_effect=lambda directory,name:runs[name]), \
             patch.object(subject.subprocess,'run',side_effect=child), \
             patch.object(subject,'render_test_plots') as plots, \
             patch.dict(os.environ,{'SLURM_JOB_ID':'999'}),contextlib.redirect_stdout(io.StringIO()):
            if mutate_source:
                with self.assertRaisesRegex(RuntimeError,'snapshot'):subject.main(['--output-dir',str(destination),'--runner-commit','f'*40])
                self.assertEqual(len(calls),1)
                self.assertEqual(json.loads((destination/'test_provenance.json').read_text())['status'],'failed')
                self.assertFalse((destination/'test_comparison.json').exists())
                return None
            report=subject.main(['--output-dir',str(destination),'--runner-commit','f'*40])
            plots.assert_called_once()
        return report,destination,p,runs,data

    def test_complete_mocked_execution_exports_full_scope_nulls_and_provenance(self):
        report,destination,p,runs,data=self.run_mocked()
        self.assertEqual(report['status'],'completed')
        self.assertEqual(len(report['per_class']),6);self.assertEqual(len(report['per_patient']),18)
        self.assertEqual(len(report['case_times']),18)
        self.assertAlmostEqual(report['per_class'][0]['baseline_dice'],1)
        self.assertEqual(report['per_class'][0]['improved_dice'],0)
        self.assertIsNone(report['per_class'][5]['baseline_dice'])
        self.assertEqual(report['node'],json.loads((destination/'test_provenance.json').read_text())['node'])
        self.assertEqual(report['runner_commit'],'f'*40)
        for path in destination.glob('*.csv'):
            self.assertNotIn(b'\r',path.read_bytes())
            self.assertIn('full_heldout_test_numeric_labels',path.read_text())
        self.assertNotIn('NaN',(destination/'test_comparison.json').read_text())

    def test_partial_or_changed_test_records_are_rejected(self):
        p,runtime,runs=fixture(self.home);data=subject.audit_test_files(p,runs)
        for spec in p['models']:
            for trial in range(1,4):
                folder=self.home/'output'/spec['architecture']/f'test_{trial}';folder.mkdir(parents=True)
                subject.write_json(folder/'evaluation.json',synthetic_report(p,runs,data,spec['architecture'],trial))
        path=self.home/'output/UNet2D/test_1/evaluation.json';original=json.loads(path.read_text())
        for key,value in [('partial_checkpoint',True),('split','validate'),('checkpoint_sha256','a'*64),('processed_samples',2)]:
            record=copy.deepcopy(original);record[key]=value;subject.write_json(path,record)
            with self.assertRaises(ValueError):subject.summarize_test_reports(self.home/'output',p,runs,data)
        subject.write_json(path,original)
        self.assertEqual(len(subject.summarize_test_reports(self.home/'output',p,runs,data)['per_class']),6)

    def test_source_mutation_stops_after_first_child_without_completed_result(self):
        self.run_mocked(mutate_source=True)

    def test_existing_output_is_preserved(self):
        destination=self.home/'existing';destination.mkdir();marker=destination/'keep';marker.write_text('preserve')
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(FileExistsError):
            subject.main(['--output-dir',str(destination),'--runner-commit','f'*40])
        self.assertEqual(marker.read_text(),'preserve')
        self.assertEqual(len(list(destination.iterdir())),1)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib') is not None,'Matplotlib required')
    def test_two_test_figures_render_with_absent_labels_and_patient_differences(self):
        p,runtime,runs=fixture(self.home);data=subject.audit_test_files(p,runs)
        for spec in p['models']:
            for trial in range(1,4):
                folder=self.home/'output'/spec['architecture']/f'test_{trial}';folder.mkdir(parents=True)
                subject.write_json(folder/'evaluation.json',synthetic_report(p,runs,data,spec['architecture'],trial))
        summary=subject.summarize_test_reports(self.home/'output',p,runs,data);summary['protocol']=p
        subject.render_test_plots(summary,self.home/'output')
        for name in ('heldout_metrics_comparison.png','heldout_patients_and_timing.png'):
            self.assertTrue((self.home/'output'/name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))


if __name__=='__main__':unittest.main()
