import copy
import importlib.util
import math
import unittest


class TemperatureTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_r5_temperature'), 'R5b implementation missing')
        import run_r5_temperature
        return run_r5_temperature

    def rows(self, p=.8):
        # 75% correct in both predicted classes, a known calibration optimum.
        return [dict(patient_id=str(i), image_id=str(i), slice_count=20, split='validation',
                     label=y, p_ad=p if i<4 else 1-p, prediction=int(i<4))
                for i,y in enumerate([1,1,1,0,0,0,0,1])]

    def test_temperature_one_and_monotonic_class_preservation(self):
        a=self.api(); ps=[0,.001,.1,.499,.5,.501,.9,.999,1]
        for p in ps:self.assertAlmostEqual(a.scale(p,1),max(1e-7,min(1-1e-7,p)),places=14)
        for t in [.05,.5,1,2,20]:
            qs=[a.scale(p,t) for p in ps]
            self.assertEqual(qs,sorted(qs))
            self.assertEqual([p>=.5 for p in ps],[q>=.5 for q in qs])
            self.assertTrue(all(math.isfinite(q) and 0<=q<=1 for q in qs))

    def test_invalid_probability_temperature(self):
        a=self.api()
        for p,t in [(float('nan'),1),(-.1,1),(1.1,1),(.5,0),(.5,-1),(.5,float('inf'))]:
            with self.subTest(p=p,t=t),self.assertRaises(ValueError):a.scale(p,t)

    def test_analytic_optimum_and_determinism(self):
        a=self.api();rows=self.rows();saved=copy.deepcopy(rows)
        fit=a.fit_temperature(rows)
        expected=math.log(4)/math.log(3)
        self.assertAlmostEqual(fit['temperature'],expected,places=6)
        self.assertTrue(fit['converged']);self.assertEqual(fit['status'],'passed')
        self.assertLess(fit['objective_nll'],fit['baseline_nll'])
        self.assertEqual(fit,a.fit_temperature(rows[::-1]));self.assertEqual(rows,saved)

    def test_endpoint_and_uninformative_need_review(self):
        a=self.api();rows=self.rows()
        for r in rows:r['label']=r['prediction']
        fit=a.fit_temperature(rows)
        self.assertEqual(fit['status'],'needs_review');self.assertTrue(fit['at_boundary'])
        self.assertEqual(fit['temperature'],.05)
        for r in rows:r['p_ad']=.5;r['prediction']=1
        # Preserve both classes but zero logits provide no identifiable temperature.
        rows[0]['label']=0
        fit=a.fit_temperature(rows)
        self.assertEqual(fit['status'],'needs_review');self.assertEqual(fit['temperature'],1)

    def test_nonconvergence_not_accepted(self):
        a=self.api();fit=a.fit_temperature(self.rows(),max_iterations=1)
        self.assertFalse(fit['converged']);self.assertEqual(fit['status'],'needs_review')

    def test_fit_refuses_test_duplicate_metadata_and_single_class(self):
        a=self.api()
        for kind in ['split','duplicate','label','count','prediction','one_class']:
            rows=self.rows()
            if kind=='split':rows[0]['split']='test'
            elif kind=='duplicate':rows[-1]['patient_id']=rows[0]['patient_id']
            elif kind=='label':rows[0]['label']=2
            elif kind=='count':rows[0]['slice_count']=19
            elif kind=='prediction':rows[0]['prediction']=0
            else:
                for r in rows:r['label']=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):a.fit_temperature(rows)

    def test_confidence_ece_counts_boundary_empty_bins(self):
        a=self.api()
        rows=[dict(p_ad=.6,label=1),dict(p_ad=.4,label=0),dict(p_ad=.9,label=0),dict(p_ad=1,label=1)]
        bins,ece=a.reliability(rows)
        self.assertEqual(len(bins),10);self.assertEqual(sum(b['count'] for b in bins),4)
        self.assertEqual(bins[6]['count'],2);self.assertEqual(bins[9]['count'],2)
        self.assertAlmostEqual(ece,.425)
        self.assertIsNone(bins[0]['accuracy']);self.assertIsNone(bins[0]['mean_confidence'])
        self.assertIsNone(bins[0]['absolute_gap'])

    def test_stable_objective_on_extreme_probabilities(self):
        a=self.api();rows=self.rows()
        for i,r in enumerate(rows):r['p_ad']=1 if i<4 else 0
        loss=a.objective(rows,.05)
        self.assertTrue(math.isfinite(loss));self.assertGreater(loss,50)
        # Stable NLL must not silently cap saturated wrong predictions at -log(1e-7).



class TemperaturePipelineTests(unittest.TestCase):
    def setUp(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import run_r5_temperature as a
        from run_r5_ensemble import write_csv,sha
        self.a=a;self.json=json;self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source';self.source.mkdir();self.out=self.root/'output'
        summary=dict(status='passed',split='validation',source_job='706895',calibrated=False,
                     seeds=[42,43,44],patient_count=136,class_counts={'AD':44,'NC':92},output_sha256={})
        for model in a.MODELS:
            rows=[]
            for i in range(136):
                p=.8 if i<33 or 44<=i<55 else .2
                rows.append(dict(patient_id=f'p{i:03}',image_id=str(i),label=int(i<44),slice_count=20,
                                 split='validation',p_ad=p,prediction=int(p>=.5),
                                 p_ad_seed42=p,p_ad_seed43=p,p_ad_seed44=p))
            path=self.source/f'{model}_patient_predictions.csv';write_csv(path,rows)
            summary['output_sha256'][path.name]=sha(path)
        path=self.source/'ensemble_summary.json';path.write_text(json.dumps(summary))
        patcher=patch.object(a,'ENSEMBLE_SUMMARY_SHA',sha(path));patcher.start();self.addCleanup(patcher.stop)

    def test_complete_fit_and_figures_preserve_inputs(self):
        from run_r5_ensemble import sha,read_csv
        self.assertIsNotNone(importlib.util.find_spec('plot_r5_reliability'),'reliability plot implementation missing')
        before={p:sha(p) for p in self.source.iterdir()}
        result=self.a.run(self.source,self.out)
        self.assertEqual(result['status'],'passed');self.assertEqual(len(result['temperatures']),3)
        self.assertEqual(len(result['metrics']),6)
        self.assertEqual(result['image_reads'],dict(train=0,validation=0,test=0))
        rows=read_csv(self.out/'CNN-A_patient_predictions.csv');self.assertEqual(len(rows),136)
        self.assertTrue(all(r['prediction']==r['prediction_raw'] for r in rows))
        for ext in ['png','svg','pdf']:
            self.assertGreater((self.out/f'reliability_diagram.{ext}').stat().st_size,1000)
        for p,h in result['output_sha256'].items():self.assertEqual(sha(self.out/p),h)
        self.assertEqual(before,{p:sha(p) for p in before})
        with self.assertRaisesRegex(ValueError,'already exists'):self.a.run(self.source,self.out)

    def test_tampered_patient_source_rejected_before_output(self):
        p=self.source/'CNN-A_patient_predictions.csv';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'SHA'):self.a.run(self.source,self.out)
        self.assertFalse(self.out.exists())

    def test_tampered_summary_rejected(self):
        (self.source/'ensemble_summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA'):self.a.run(self.source,self.out)
        self.assertFalse(self.out.exists())

    def test_fit_rows_and_calibrated_rows_keep_original_probabilities(self):
        from run_r5_ensemble import read_csv
        rows=read_csv(self.source/'CNN-A_patient_predictions.csv');saved=copy.deepcopy(rows)
        changed=self.a.calibrated_rows(rows,2)
        self.assertEqual(rows,saved)
        self.assertEqual(changed[0]['p_ad_raw'],.8)
        self.assertAlmostEqual(changed[0]['p_ad'],2/3)

if __name__=='__main__':unittest.main()
