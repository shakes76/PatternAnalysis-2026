import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn

class EfficiencyTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_r4_efficiency'), 'R4 implementation missing')
        import run_r4_efficiency
        return run_r4_efficiency

    def test_protocol_common_fp32_and_fixed_repetitions(self):
        p=self.api().protocol()
        self.assertEqual((p['batch_size'],p['precision'],p['num_workers']),(4,'float32',0))
        self.assertEqual((p['forward_warmup'],p['forward_steps'],p['forward_repeats'],p['end_to_end_repeats']),(20,100,5,3))
        self.assertFalse(p['tf32']);self.assertEqual(p['split'],'validation')

    def test_summary_units_and_invalid_durations(self):
        a=self.api();s=a.summarize([1.,2.,3.],samples=100,patients=5)
        self.assertEqual(s['median_seconds'],2.)
        self.assertEqual(s['median_ms_per_slice'],20.)
        self.assertEqual(s['median_ms_per_patient_amortized'],400.)
        self.assertEqual(s['median_slices_per_second'],50.)
        self.assertEqual(s['sample_sd_seconds'],1.)
        for durations in ([],[0.],[-1.],[math.nan],[math.inf]):
            with self.assertRaises(ValueError):a.summarize(durations,100,5)

    def test_timing_synchronizes_both_boundaries_and_returns_value(self):
        a=self.api();events=[];ticks=iter([10.,12.])
        def clock():events.append('clock');return next(ticks)
        seconds,value=a.timed(lambda:events.append('operation') or 7,lambda:events.append('sync'),clock)
        self.assertEqual((seconds,value),(2.,7))
        self.assertEqual(events,['sync','clock','operation','sync','clock'])

    def test_forward_eval_no_grad_and_warmup_excluded(self):
        a=self.api()
        class Probe(nn.Module):
            def __init__(self):super().__init__();self.layer=nn.Linear(2,2);self.calls=0
            def forward(self,x):
                self.calls+=1
                assert not self.training and not torch.is_grad_enabled()
                return self.layer(x)
        model=Probe();before={k:v.clone() for k,v in model.state_dict().items()}
        result=a.measure_forward(model,torch.ones(4,2),'cpu',warmup=2,steps=3,repeats=2)
        self.assertEqual(model.calls,8);self.assertEqual(len(result['seconds']),2)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertIsNone(result['peak_memory'])

    def test_end_to_end_includes_patient_mean_and_tail_batch(self):
        a=self.api()
        from torch.utils.data import Dataset,DataLoader
        class Samples(Dataset):
            def __init__(self):self.read_calls=0
            def __len__(self):return 6
            def __getitem__(self,i):
                self.read_calls+=1
                return dict(image=torch.tensor([float(i),0.]),label=i//3,
                    patient_id=str(i//3),image_id=str(i//3),file_path=str(i))
        ds=Samples();model=nn.Identity().eval()
        rows,patients=a.end_to_end_pass(model,DataLoader(ds,batch_size=4),'cpu',expected_slices=3)
        self.assertEqual(ds.read_calls,6);self.assertEqual(len(rows),6);self.assertEqual(len(patients),2)
        self.assertAlmostEqual(patients[0]['p_ad'],sum(r['p_ad'] for r in rows[:3])/3)

    def test_prediction_identity_and_numeric_drift_rejected(self):
        a=self.api();rows=[dict(patient_id='p',image_id='1',label=1,slice_count=20,p_ad=.6,prediction=1)]
        a.compare_patients(rows,rows)
        for change in (dict(patient_id='q'),dict(p_ad=.61),dict(label=0),dict(slice_count=19)):
            with self.assertRaises(ValueError):a.compare_patients(rows,[dict(rows[0],**change)])
        with self.assertRaises(ValueError):a.compare_patients(rows,rows*2)

    def test_path_cannot_escape_cluster_root(self):
        a=self.api()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);((root/'x').resolve()).mkdir()
            self.assertEqual(a.source_path(root,'x'),(root/'x').resolve())
            for bad in ('../outside','/absolute'):
                with self.assertRaises(ValueError):a.source_path(root,bad)

    def test_frozen_lock_has_nine_primary_checkpoints(self):
        a=self.api();lock=json.loads(Path(a.__file__).with_name('r4_frozen_bests.json').read_text())
        a.validate_lock(lock)
        self.assertEqual(len(lock['runs']),9)
        self.assertEqual([r['epoch'] for r in lock['runs'] if r['model']=='ConvNeXt-F'],[4,1,30])
        for mutate in ('duplicate','epoch','hash','model'):
            changed=json.loads(json.dumps(lock))
            if mutate=='duplicate':changed['runs'][-1]=changed['runs'][0]
            if mutate=='epoch':changed['runs'][0]['epoch']=0
            if mutate=='hash':changed['runs'][0]['checkpoint_sha256']='bad'
            if mutate=='model':changed['runs'][0]['model']='wrong'
            with self.assertRaises(ValueError):a.validate_lock(changed)

    def test_input_hash_failure_before_model_or_images(self):
        a=self.api()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'training_summary.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash'):
                a.verify_files(p,{'training_summary.json':'0'*64})


    def test_load_real_cnn_primary_checkpoint_and_reject_tamper(self):
        a=self.api()
        from modules import SimpleCNN
        from dual_checkpoint import DualCheckpointSelector
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'run';source.mkdir()
            config=dict(model='SimpleCNN',arm='A',seed=42,epochs=30,early_stopping=False)
            provenance=dict(input_sha256={'manifest':'a','summary':'b'},code_sha256={n:a.sha256(Path(a.__file__).with_name(n)) for n in ('modules.py','dataset.py')})
            records=[dict(file_path=str(i),patient_id=str(i//3),image_id=str(i//3),label='NC' if i<3 else 'AD',split='validation') for i in range(6)]
            rows=[dict(r,label=int(i>=3),p_ad=.2 if i<3 else .8) for i,r in enumerate(records)]
            model=SimpleCNN().eval();selector=DualCheckpointSelector(source/'selection',config,provenance,3)
            selector.consider(1,model,rows,records,.5)
            training=dict(status='passed',config=config,provenance=provenance)
            (source/'training_summary.json').write_text(json.dumps(training))
            pred=source/'reload/patient_nll';pred.mkdir(parents=True)
            a.write_csv(pred/'patient_predictions.csv',a.aggregate_patients(rows,3))
            files={n:a.sha256(source/n) for n in ('training_summary.json','selection/selection_summary.json','selection/best_patient_nll.pt','reload/patient_nll/patient_predictions.csv')}
            entry=dict(model='CNN-A',seed=42,epoch=1,source='run',files=files)
            loaded,expected=a.load_frozen(root,entry,{'data_files':{'x':'a','y':'b'}})
            self.assertEqual(a.state_sha256(model),a.state_sha256(loaded));self.assertEqual(len(expected),2)
            (source/'selection/best_patient_nll.pt').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'hash'):a.load_frozen(root,entry,{'data_files':{'x':'a','y':'b'}})

    def test_consolidate_separates_seed_variation_and_rejects_mixed_conditions(self):
        a=self.api();results=[]
        for model in ('CNN-A','CNN-B','ConvNeXt-F'):
            for seed in (42,43,44):
                result=dict(status='passed',historical_predictions_match_R3=True,unified_repeats_match_reference=True,unified_vs_history={'patient_drift':{'r4_exceed_count':0}},entry=dict(model=model,seed=seed,epoch=1,checkpoint_sha256='x'),
                    parameter_count=4,protocol=a.protocol(),environment={'gpu':'one'},lock_sha256='lock',code_sha256={'x':'x'})
                for mode in ('forward','end_to_end'):
                    result[mode]=dict(a.summarize([1.,2.,3.],100,5),peak_memory=dict(allocated_bytes=100,reserved_bytes=200))
                results.append(result)
        rows,aggregate=a.consolidate(results)
        self.assertEqual(len(rows),9);self.assertTrue(all(r['sample_sd']==0 for r in aggregate))
        results[-1]['environment']={'gpu':'other'}
        with self.assertRaisesRegex(ValueError,'conditions'):a.consolidate(results)
        with self.assertRaises(ValueError):a.consolidate(results[:-1])

    def test_worker_refuses_cpu_and_records_failure_without_images(self):
        a=self.api()
        with tempfile.TemporaryDirectory() as d,patch('torch.cuda.is_available',return_value=False):
            out=Path(d)/'output'
            with self.assertRaisesRegex(ValueError,'CUDA'):
                a.run_worker(Path(d),Path(a.__file__).with_name('r4_frozen_bests.json'),0,out)
            result=json.loads((out/'efficiency_summary.json').read_text())
            self.assertEqual(result['status'],'failed');self.assertEqual(result['read_counts'],{})
            with self.assertRaises(FileExistsError):a.run_worker(Path(d),None,0,out)


    def test_run_all_nine_fresh_processes_and_output(self):
        a=self.api();lock_path=Path(a.__file__).with_name('r4_frozen_bests.json')
        lock=json.loads(lock_path.read_text());seen=[]
        def child(command,**kwargs):
            index=int(command[-1]);entry=lock['runs'][index];seen.append((entry['model'],entry['seed']))
            folder=Path(command[command.index('--output-dir')+1]);folder.mkdir()
            result=dict(status='passed',historical_predictions_match_R3=True,unified_repeats_match_reference=True,unified_vs_history={'patient_drift':{'r4_exceed_count':0}},entry=entry,parameter_count=4,protocol=a.protocol(),
                environment={'gpu':'one'},lock_sha256=a.sha256(lock_path),code_sha256={'x':'x'},read_counts={'test':0})
            for mode in ('forward','end_to_end'):
                result[mode]=dict(a.summarize([1.,2.,3.],100,5),peak_memory=dict(allocated_bytes=100,reserved_bytes=200))
            a.write_json(folder/'efficiency_summary.json',result)
        with tempfile.TemporaryDirectory() as d,patch.object(a,'verify_files'),patch.object(a.subprocess,'run',side_effect=child) as run:
            out=Path(d)/'results';result=a.run_all(Path(d),lock_path,out)
            self.assertEqual(run.call_count,9);self.assertEqual(result['status'],'passed')
            self.assertEqual(seen,[('CNN-A',42),('CNN-B',42),('ConvNeXt-F',42),('CNN-B',43),('ConvNeXt-F',43),('CNN-A',43),('ConvNeXt-F',44),('CNN-A',44),('CNN-B',44)])
            self.assertTrue((out/'nine_run_efficiency.csv').exists())
            self.assertEqual(result['test_image_reads'],0)

    def test_run_all_failed_child_cannot_report_success(self):
        a=self.api();lock_path=Path(a.__file__).with_name('r4_frozen_bests.json')
        with tempfile.TemporaryDirectory() as d,patch.object(a,'verify_files'),patch.object(a.subprocess,'run',side_effect=RuntimeError('child failed')) as run:
            out=Path(d)/'results'
            with self.assertRaisesRegex(RuntimeError,'child failed'):a.run_all(Path(d),lock_path,out)
            result=json.loads((out/'r4_summary.json').read_text())
            self.assertEqual(result['status'],'failed');self.assertEqual(run.call_count,1)
            self.assertFalse((out/'nine_run_efficiency.csv').exists())

if __name__=='__main__':unittest.main()
