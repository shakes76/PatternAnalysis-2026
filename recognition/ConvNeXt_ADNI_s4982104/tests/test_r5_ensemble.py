import copy
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


class EnsembleTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_r5_ensemble'), 'R5a implementation missing')
        import run_r5_ensemble
        return run_r5_ensemble

    def rows(self, p):
        return [dict(patient_id='a', image_id='i', label='1', slice_count='20', p_ad=p, prediction=int(p>=.5)),
                dict(patient_id='b', image_id='j', label='0', slice_count='20', p_ad=.2, prediction=0)]

    def test_equal_mean_and_reordering(self):
        a=self.api(); inputs={42:self.rows(.6),43:self.rows(.8)[::-1],44:self.rows(.7)}
        saved=copy.deepcopy(inputs); result=a.ensemble(inputs)
        self.assertAlmostEqual(result[0]['p_ad'],.7)
        self.assertEqual(result[0]['p_ad_seed43'],.8)
        self.assertEqual(inputs,saved)
        self.assertEqual(result,a.ensemble({s:r[::-1] for s,r in inputs.items()}))

    def test_bad_seed_missing_duplicate_and_metadata(self):
        a=self.api()
        for kind in ('seed','missing','duplicate','label','image','count','prediction','nan','range'):
            inputs={s:self.rows(.6) for s in (42,43,44)}
            if kind=='seed': inputs[45]=inputs.pop(44)
            elif kind=='missing': inputs[43].pop()
            elif kind=='duplicate': inputs[43].append(inputs[43][0])
            else:
                field,value={'label':('label','0'),'image':('image_id','other'),'count':('slice_count',19),
                    'prediction':('prediction',0),'nan':('p_ad',float('nan')),'range':('p_ad',1.2)}[kind]
                inputs[43][0][field]=value
            with self.subTest(kind=kind), self.assertRaises(ValueError): a.ensemble(inputs)

    def test_slice_reconstruction_and_bad_identity(self):
        a=self.api(); patients=self.rows(.6)
        slices=[dict(file_path=f'{r["patient_id"]}/{i}', **{k:v for k,v in r.items() if k!='slice_count'}) for r in patients for i in range(20)]
        manifest=[dict(r,split='validation',label='AD' if r['label']=='1' else 'NC') for r in slices]
        a.validate_slices(patients,slices,manifest)
        for kind in ('probability','duplicate','wrong_split','image'):
            ss=copy.deepcopy(slices); mm=copy.deepcopy(manifest)
            if kind=='probability':ss[0]['p_ad']=.9
            if kind=='duplicate':ss[-1]=ss[0]
            if kind=='wrong_split':mm[0]['split']='test'
            if kind=='image':ss[0]['image_id']='wrong'
            with self.subTest(kind=kind),self.assertRaises(ValueError):a.validate_slices(patients,ss,mm)

    def test_metrics_boundary_ties_extremes(self):
        a=self.api(); r=self.rows(.5); r[1]['p_ad']=.5
        m=a.metrics(r)
        self.assertEqual(m['tp'],1);self.assertEqual(m['fp'],1)
        self.assertEqual(m['auroc'],.5);self.assertAlmostEqual(m['nll'],math.log(2))
        self.assertEqual(m['brier'],.25)
        r[0]['p_ad']=1;r[1]['p_ad']=0
        self.assertEqual(a.metrics(r)['accuracy'],1)
        self.assertTrue(math.isfinite(a.metrics(r)['nll']))

    def test_hash_rejects_changed_input(self):
        a=self.api()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.csv';p.write_text('original')
            h=a.sha(p);a.checked(p,h)
            p.write_text('changed')
            with self.assertRaises(ValueError):a.checked(p,h)

    def test_summary_identity_and_protocol(self):
        a=self.api();entry=dict(model='CNN-A',seed=42,epoch=28,checkpoint_sha256='x')
        summary=dict(status='passed',entry=entry,lock_sha256='lock',protocol=dict(split='validation',batch_size=4,precision='float32',tf32=False,amp=False),environment=dict(slurm_job_id='706895'),historical_predictions_match_R3=True,unified_repeats_match_reference=True,parameters_unchanged=True,read_counts=dict(train=0,test=0))
        a.validate_summary(summary,entry,'lock')
        for key,value in [('seed',43),('epoch',1),('checkpoint_sha256','bad')]:
            bad=copy.deepcopy(summary);bad['entry'][key]=value
            with self.assertRaises(ValueError):a.validate_summary(bad,entry,'lock')
        summary['protocol']['split']='test'
        with self.assertRaises(ValueError):a.validate_summary(summary,entry,'lock')

if __name__=='__main__':unittest.main()


class PipelineTests(unittest.TestCase):
    """Synthetic full 136-patient fixture; no private cohort or images required."""
    def setUp(self):
        import json
        from unittest.mock import patch
        import run_r5_ensemble as a
        self.a=a; self.json=json;self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source';self.source.mkdir()
        self.split=self.root/'split.csv';self.out=self.root/'out'
        patients=[dict(patient_id=f'p{i:03}',image_id=f'i{i}',label=int(i<44),slice_count=20,p_ad=.8 if i<44 else .2,prediction=int(i<44)) for i in range(136)]
        slices=[dict(file_path=f'/not-real/{r["patient_id"]}/{j}.jpeg',**{k:v for k,v in r.items() if k!='slice_count'}) for r in patients for j in range(20)]
        a.write_csv(self.split,[dict(r,label='AD' if r['label'] else 'NC',split='validation') for r in slices])
        entries=[dict(model=m,seed=s,epoch=1,checkpoint_sha256='checkpoint') for m in a.MODELS for s in a.SEEDS]
        lock=dict(runs=entries,data_files={'split_seed42_60_20_20/split_samples.csv':a.sha(self.split)})
        self.lock=self.root/'r4_frozen_bests.json';self.lock.write_text(json.dumps(lock))
        protocol=dict(split='validation',batch_size=4,precision='float32',tf32=False,amp=False)
        top=dict(status='passed',lock_sha256=a.sha(self.lock),test_image_reads=0,completed=[],output_sha256={})
        for entry in entries:
            name=f'{entry["model"]}_{entry["seed"]}';folder=self.source/name;folder.mkdir();top['completed'].append(name)
            summary=dict(status='passed',entry=entry,lock_sha256=a.sha(self.lock),protocol=protocol,environment=dict(slurm_job_id='706895'),historical_predictions_match_R3=True,unified_repeats_match_reference=True,parameters_unchanged=True,read_counts=dict(train=0,test=0))
            (folder/'efficiency_summary.json').write_text(json.dumps(summary))
            a.write_csv(folder/'unified_reference_patient_predictions.csv',patients)
            a.write_csv(folder/'unified_reference_slice_predictions.csv',slices)
            for p in folder.iterdir():top['output_sha256'][str(p.relative_to(self.source))]=a.sha(p)
        self.top=self.source/'r4_summary.json';self.top.write_text(json.dumps(top))
        for target,value in [('__file__',str(self.root/'fake_script.py')),('LOCK_SHA',a.sha(self.lock)),('SUMMARY_SHA',a.sha(self.top))]:
            patcher=patch.object(a,target,value);patcher.start();self.addCleanup(patcher.stop)
        (self.root/'fake_script.py').write_text('synthetic code identity')

    def test_complete_pipeline_outputs_and_inputs_unchanged(self):
        a=self.a
        before={p:a.sha(p) for p in self.source.rglob('*') if p.is_file()}
        result=a.run(self.source,self.split,self.out)
        self.assertEqual(result['status'],'passed');self.assertEqual(len(result['source_runs']),9)
        self.assertEqual(result['image_reads'],dict(train=0,validation=0,test=0))
        self.assertEqual(len(result['output_sha256']),6)
        self.assertEqual(len(a.read_csv(self.out/'CNN-A_patient_predictions.csv')),136)
        self.assertTrue(all(m['accuracy']==1 for m in result['metrics']))
        self.assertEqual(before,{p:a.sha(p) for p in before})
        with self.assertRaisesRegex(ValueError,'already exists'):a.run(self.source,self.split,self.out)

    def test_tampered_predictions_fail_before_output(self):
        p=self.source/'CNN-A_42/unified_reference_patient_predictions.csv';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'SHA'):self.a.run(self.source,self.split,self.out)
        self.assertFalse(self.out.exists())

    def test_wrong_top_summary_fails(self):
        self.top.write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA'):self.a.run(self.source,self.split,self.out)
        self.assertFalse(self.out.exists())

    def test_changed_split_fails(self):
        self.split.write_text(self.split.read_text().replace('validation','test'))
        with self.assertRaisesRegex(ValueError,'SHA'):self.a.run(self.source,self.split,self.out)
        self.assertFalse(self.out.exists())
