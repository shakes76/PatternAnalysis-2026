import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def fixture(pos,neg):
    return [dict(patient_id=f'p{i}',image_id=f'i{i}',label=y,p_ad=p,slice_count=20,
                 split='validation',prediction=int(p>=.5))
            for i,(y,p) in enumerate([(1,p) for p in pos]+[(0,p) for p in neg])]


class EnsembleThresholdTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_r5_threshold'),'R5c implementation missing')
        import run_r5_threshold
        return run_r5_threshold

    def test_44_ad_requires_36_and_cost(self):
        r=self.api().select(fixture([i/100 for i in range(1,45)],[.05,.1,.5]))
        self.assertEqual(r['required_tp'],36);self.assertEqual(r['selected']['threshold'],.09)
        self.assertEqual(r['selected']['tp'],36);self.assertEqual(r['selected']['fp'],2)
        self.assertEqual(r['delta']['fn'],-36);self.assertEqual(r['delta']['fp'],1)

    def test_ties_inclusive_and_highest_threshold(self):
        a=self.api();r=a.select(fixture([.2]*5,[.2,.1]))
        self.assertEqual(r['selected']['threshold'],.2);self.assertEqual(r['selected']['tp'],5)
        self.assertEqual(r['selected']['fp'],1)
        r=a.select(fixture([.9,.8,.7,.6,.4],[.1,.2]))
        self.assertEqual(r['selected']['threshold'],.6)

    def test_reordering_unchanged_and_no_mutation(self):
        a=self.api();rows=fixture([.1,.3,.4,.6,.9],[.2,.5]);saved=copy.deepcopy(rows)
        self.assertEqual(a.select(rows),a.select(rows[::-1]));self.assertEqual(rows,saved)

    def test_invalid_rows_rejected(self):
        a=self.api()
        for field,value in [('split','test'),('p_ad',float('nan')),('label',2),('slice_count',19),('prediction',0)]:
            rows=fixture([.8,.9],[.1,.3]);rows[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):a.select(rows)
        rows=fixture([.8],[.1]);rows.append(rows[0])
        with self.assertRaises(ValueError):a.select(rows)

    def test_all_equal_zero_and_one_candidates(self):
        a=self.api()
        for p in [0.,1.]:
            r=a.select(fixture([p]*5,[p]*3));self.assertEqual(r['selected']['threshold'],p)
            self.assertEqual(r['selected']['tp'],5);self.assertEqual(r['selected']['fp'],3)


class ThresholdPipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('run_r5_threshold'),'R5c implementation missing')
        import run_r5_threshold as a
        from run_r5_ensemble import write_csv,sha
        from run_r5_temperature import scale
        self.a=a;self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source';self.source.mkdir();self.out=self.root/'out'
        summary=dict(status='passed',calibration_accepted=True,split='validation',source_job='706895',
                     patient_count=136,class_counts={'AD':44,'NC':92},temperatures={},output_sha256={})
        fits={}
        for model in a.MODELS:
            t=1.2;summary['temperatures'][model]=t
            fits[model]=dict(status='passed',temperature=t,converged=True,at_boundary=False,review_reasons=[])
            rows=fixture([.05+i/60 for i in range(44)],[.02+i/100 for i in range(92)])
            for r in rows:
                raw=r['p_ad'];r.update(p_ad_raw=raw,p_ad=scale(raw,t),temperature=t,
                    p_ad_seed42=raw,p_ad_seed43=raw,p_ad_seed44=raw,prediction_raw=r['prediction'])
            p=self.source/f'{model}_patient_predictions.csv';write_csv(p,rows);summary['output_sha256'][p.name]=sha(p)
        p=self.source/'temperature_fits.json';p.write_text(json.dumps(fits));summary['output_sha256'][p.name]=sha(p)
        p=self.source/'temperature_summary.json';p.write_text(json.dumps(summary))
        pt=patch.object(a,'TEMPERATURE_SUMMARY_SHA',sha(p));pt.start();self.addCleanup(pt.stop)

    def test_complete_pipeline_counts_hashes_and_no_overwrite(self):
        from run_r5_ensemble import sha,read_csv
        before={p:sha(p) for p in self.source.iterdir()}
        r=self.a.run(self.source,self.out)
        self.assertEqual(r['status'],'passed');self.assertEqual(len(r['models']),3)
        self.assertEqual(r['image_reads'],dict(train=0,validation=0,test=0))
        for m in r['models']:
            self.assertEqual(m['required_tp'],36);self.assertGreaterEqual(m['selected']['tp'],36)
            patients=read_csv(self.out/f'{m["model"]}_patient_decisions.csv')
            self.assertEqual(len(patients),136)
            self.assertEqual(sum(int(p['label'])==1 and int(p['prediction_ad80'])==1 for p in patients),m['selected']['tp'])
        for p,h in r['output_sha256'].items():self.assertEqual(sha(self.out/p),h)
        self.assertEqual(before,{p:sha(p) for p in before})
        with self.assertRaisesRegex(ValueError,'already exists'):self.a.run(self.source,self.out)

    def test_tampered_csv_rejected(self):
        p=self.source/'CNN-A_patient_predictions.csv';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'SHA'):self.a.run(self.source,self.out)
        self.assertFalse(self.out.exists())

    def test_tampered_summary_rejected(self):
        (self.source/'temperature_summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA'):self.a.run(self.source,self.out)
        self.assertFalse(self.out.exists())

    def test_calibration_and_temperature_mismatch_rejected(self):
        from run_r5_ensemble import read_csv
        rows=read_csv(self.source/'CNN-A_patient_predictions.csv')
        for field,value in [('temperature',2),('p_ad_raw',.7),('p_ad',.9),('p_ad_seed42',.99)]:
            bad=copy.deepcopy(rows);bad[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.a.validate_calibrated(bad,1.2)

if __name__=='__main__':unittest.main()
