import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np


def fixture():
    manifest=[dict(file_path=f'p{i}/{j}.jpeg',patient_id=f'p{i}',image_id=f'i{i}',label=i%2,split='test') for i in range(6) for j in range(20)]
    entries=[dict(model=m,seed=s,epoch=1,checkpoint_sha256=f'{m}-{s}') for m in ('CNN-A','CNN-B','ConvNeXt-F') for s in (42,43,44)]
    contract=dict(schema=1,scope='frozen_final_evaluation',models=['CNN-A','CNN-B','ConvNeXt-F'],seeds=[42,43,44],
        checkpoints=entries,temperatures={m:1.2 for m in ('CNN-A','CNN-B','ConvNeXt-F')},
        supplemental_thresholds={m:.3 for m in ('CNN-A','CNN-B','ConvNeXt-F')},
        evaluation=dict(main_threshold=.5,nll_epsilon=1e-7,ece_bins=10,referral_rates=[0,.05,.1,.2],primary_referral_rate=.1,
            bootstrap_replicates=2000,bootstrap_seed=42,confidence_level=.95,quantile_method='linear'),
        aggregation=dict(slices_per_patient=20,seed_weights=[1/3]*3),
        cohort=dict(split='test',patients=6,ad=3,nc=3,slices=120),
        inference=dict(batch_size=4,precision='float32',tf32=False))
    def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    contract['cohort']['manifest_records_sha256']=digest(sorted(manifest,key=lambda r:r['file_path']))
    runs=[]
    for entry in entries:
        slices=[dict(r,p_ad=(.8 if r['label'] else .2),prediction=r['label']) for r in manifest]
        runs.append(dict(entry=entry,protocol_sha256=digest(contract),inference=contract['inference'],slices=slices))
    return contract,manifest,runs


class FinalEvaluationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('final_evaluation'),'Frozen evaluator missing')
        import final_evaluation
        return final_evaluation

    def test_aggregate_exact_cohort_and_order(self):
        a=self.api();c,m,r=fixture();rows=a.aggregate_slices(r[0]['slices'],m,20)
        self.assertEqual(len(rows),6);self.assertAlmostEqual(rows[0]['p_ad'],.2)
        self.assertEqual(rows,a.aggregate_slices(r[0]['slices'][::-1],m[::-1],20))
        for bad in [r[0]['slices'][:-20],r[0]['slices']+[r[0]['slices'][0]]]:
            with self.assertRaises(ValueError):a.aggregate_slices(bad,m,20)
        bad=copy.deepcopy(r[0]['slices']);bad[0]['label']=1
        with self.assertRaises(ValueError):a.aggregate_slices(bad,m,20)

    def test_same_stratified_indices_shared_and_deterministic(self):
        a=self.api();y=np.array([0,1,0,1,1,0])
        idx=a.bootstrap_indices(y,2000,42)
        self.assertEqual(idx.shape,(2000,6));self.assertTrue(np.all(y[idx].sum(axis=1)==3))
        np.testing.assert_array_equal(idx,a.bootstrap_indices(y,2000,42))
        v=a.bootstrap_metrics(y,np.array([.2,.8,.2,.8,.8,.2]),idx,.5)
        self.assertTrue(np.all(v['accuracy']==1));self.assertTrue(np.all(v['auroc']==1))
        self.assertEqual(a.interval(v['accuracy']),[1.,1.])
        self.assertEqual(a.interval(v['nll']-v['nll']),[0.,0.])

    def test_pipeline_applies_only_frozen_parameters(self):
        a=self.api();c,m,r=fixture()
        import run_r5_temperature
        with patch.object(run_r5_temperature,'fit_temperature',side_effect=AssertionError('Must not fit')):
            report=a.evaluate(c,m,r)
        self.assertEqual(len(report['metrics']),18)
        self.assertEqual(len(report['seed_summary']),3)
        self.assertEqual(len(report['referrals']),12)
        self.assertEqual(len(report['reliability_bins']),60)
        self.assertEqual(report['patient_count'],6)
        self.assertTrue(all(x['accuracy']==1 for x in report['metrics']))
        self.assertTrue(all(x['intervals']['accuracy']==[1.,1.] for x in report['metrics']))
        self.assertEqual(len(report['paired_differences']),6)
        self.assertEqual(report['bootstrap']['replicates'],2000)

    def test_wrong_identity_protocol_and_manifest_rejected(self):
        a=self.api();c,m,r=fixture()
        for kind in ['weight','seed','missing','protocol','inference','split']:
            rr=copy.deepcopy(r);mm=copy.deepcopy(m)
            if kind=='weight':rr[0]['entry']['checkpoint_sha256']='bad'
            if kind=='seed':rr[0]['entry']['seed']=99
            if kind=='missing':rr.pop()
            if kind=='protocol':rr[0]['protocol_sha256']='bad'
            if kind=='inference':rr[0]['inference']['batch_size']=16
            if kind=='split':mm[0]['split']='validation'
            with self.subTest(kind=kind),self.assertRaises(ValueError):a.evaluate(c,mm,rr)

    def test_reordering_and_pairwise_identity(self):
        a=self.api();c,m,r=fixture();r[0]['slices'].reverse()
        report=a.evaluate(c,m[::-1],r[::-1])
        for pair in report['paired_differences'][:3]:
            for value in pair['metrics'].values():
                self.assertEqual(value['estimate'],0);self.assertEqual(value['interval'],[0.,0.])

    def test_clipping_and_undefined_referral_denominators(self):
        a=self.api();c,m,r=fixture()
        for run in r:
            for row in run['slices']:row['p_ad']=float(row['label'])
        report=a.evaluate(c,m,r)
        self.assertTrue(all(np.isfinite(x['nll']) for x in report['metrics']))
        self.assertTrue(all(x['error_capture_rate'] is None for x in report['referrals']))

    def test_load_protocol_digest_and_code_checks(self):
        a=self.api();c,_,_=fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'worker.py').write_text('stable')
            c['code_sha256']={'worker.py':hashlib.sha256(b'stable').hexdigest()}
            f=p/'protocol.json';f.write_text(json.dumps(c))
            digest=hashlib.sha256(f.read_bytes()).hexdigest()
            self.assertEqual(a.load_protocol(f,digest,p),c)
            with self.assertRaises(ValueError):a.load_protocol(f,'bad',p)
            (p/'worker.py').write_text('changed')
            with self.assertRaises(ValueError):a.load_protocol(f,digest,p)

    def test_bad_probability_and_scan_rejected(self):
        a=self.api();c,m,r=fixture()
        for field,value in [('p_ad',float('nan')),('image_id','other'),('patient_id','missing'),('prediction',1)]:
            rr=copy.deepcopy(r);rr[0]['slices'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):a.evaluate(c,m,rr)


class IndependentMetricTests(unittest.TestCase):
    def test_mixed_confusion_tied_auc_and_threshold(self):
        from final_evaluation import bootstrap_metrics
        y=np.array([0,1,0,1]);p=np.array([.2,.4,.4,.8]);idx=np.arange(4)[None,:]
        m=bootstrap_metrics(y,p,idx,.5)
        self.assertAlmostEqual(m['accuracy'][0],.75)
        self.assertAlmostEqual(m['ad_recall'][0],.5)
        self.assertAlmostEqual(m['nc_specificity'][0],1)
        self.assertAlmostEqual(m['macro_f1'][0],(2/3+4/5)/2)
        self.assertAlmostEqual(m['auroc'][0],.875)
        m=bootstrap_metrics(y,p,idx,.4)
        self.assertEqual(m['ad_recall'][0],1);self.assertEqual(m['nc_specificity'][0],.5)
    def test_probability_average_and_nonzero_paired_difference(self):
        from final_evaluation import evaluate
        c,m,runs=fixture()
        for run in runs:
            if run['entry']['model']=='CNN-A':
                value={42:.1,43:.2,44:.9}[run['entry']['seed']]
                for row in run['slices']:row.update(p_ad=value,prediction=int(value>=.5))
        report=evaluate(c,m,runs)
        self.assertAlmostEqual(report['patient_tables']['CNN-A/ensemble'][0]['p_ad_raw'],.4)
        delta=report['paired_differences'][0]['metrics']['accuracy']
        self.assertEqual(delta['estimate'],.5);self.assertEqual(delta['interval'],[.5,.5])

if __name__=='__main__':unittest.main()
