import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def rows():
    # Two misses, one false alarm, one correct NC. Three errors in total.
    return [dict(patient_id=p,p_ad_raw=q,p_ad=q,label=y,prediction=int(q>=.5),
                 image_id=p,slice_count=20,split='validation')
            for p,q,y in [('d',.1,1),('b',.4,1),('a',.6,0),('c',.2,0)]]


class ReferralTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_r5_referral'),'R5d implementation missing')
        import run_r5_referral
        return run_r5_referral

    def test_exact_ceil_counts(self):
        a=self.api();r=[dict(patient_id=str(i),p_ad_raw=i/136) for i in range(136)]
        self.assertEqual([len(a.select_referrals(r,x)) for x in [0,.05,.1,.2]],[0,7,14,28])
        self.assertEqual(len(a.select_referrals(r[:10],.1)),1)

    def test_rank_ties_reorder_no_mutation(self):
        a=self.api();r=rows();saved=copy.deepcopy(r)
        self.assertEqual(a.select_referrals(r,.25),['a'])
        self.assertEqual(a.select_referrals(r,.5),['a','b'])
        self.assertEqual(a.select_referrals(r,.5),a.select_referrals(r[::-1],.5));self.assertEqual(r,saved)

    def test_selector_never_reads_labels_or_calibrated_probability(self):
        a=self.api()
        class ScoresOnly(dict):
            def __getitem__(self,key):
                if key not in ('patient_id','p_ad_raw'):raise AssertionError('Selector read forbidden field')
                return super().__getitem__(key)
        r=[ScoresOnly(patient_id='a',p_ad_raw=.51),ScoresOnly(patient_id='b',p_ad_raw=.9)]
        self.assertEqual(a.select_referrals(r,.5),['a'])

    def test_capture_remaining_errors_and_conservation(self):
        r=self.api().evaluate(rows(),.5)
        self.assertEqual(r['referred_n'],2);self.assertEqual(r['retained_n'],2)
        self.assertEqual(r['referred_errors'],2);self.assertEqual(r['retained_errors'],1)
        self.assertEqual(r['referred_fn'],1);self.assertEqual(r['retained_fn'],1)
        self.assertAlmostEqual(r['error_capture_rate'],2/3)
        self.assertEqual(r['retained_error_rate'],.5)
        for k in ['tp','fn','tn','fp']:
            self.assertEqual(r['full_'+k],r['referred_'+k]+r['retained_'+k])
        self.assertEqual(r['full_errors'],3)  # Referring does not correct errors.

    def test_zero_and_full_referral_missing_denominators(self):
        a=self.api();zero=a.evaluate(rows(),0);full=a.evaluate(rows(),1)
        self.assertIsNone(zero['referred_error_rate']);self.assertEqual(zero['error_capture_rate'],0)
        self.assertIsNone(full['retained_error_rate']);self.assertEqual(full['retained_fn'],0)
        self.assertIsNone(full['retained_ad_recall'])
        good=rows()
        for r in good:r['label']=r['prediction']
        result=a.evaluate(good,.5)
        self.assertIsNone(result['error_capture_rate']);self.assertIsNone(result['ad_miss_capture_rate'])

    def test_bad_scores_rates_identity_and_split(self):
        a=self.api()
        for rate in [-.1,1.1,float('nan'),True]:
            with self.subTest(rate=rate),self.assertRaises(ValueError):a.select_referrals(rows(),rate)
        for kind in ['duplicate','nan','range','empty_id']:
            r=rows()
            if kind=='duplicate':r[0]['patient_id']=r[1]['patient_id']
            if kind=='nan':r[0]['p_ad_raw']=float('nan')
            if kind=='range':r[0]['p_ad_raw']=1.1
            if kind=='empty_id':r[0]['patient_id']=''
            with self.subTest(kind=kind),self.assertRaises(ValueError):a.select_referrals(r,.1)
        r=rows();r[0]['split']='test'
        with self.assertRaises(ValueError):a.evaluate(r,.1)


class ReferralPipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('run_r5_referral'),'R5d implementation missing')
        import run_r5_referral as a
        import run_r5_temperature as temp
        import run_r5_threshold as threshold
        from run_r5_ensemble import sha,write_csv
        self.a=a;self.temp=temp;self.threshold=threshold
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.raw=self.root/'raw';self.raw.mkdir();self.cal=self.root/'cal';self.cal.mkdir();self.out=self.root/'out'
        raw_summary=dict(status='passed',split='validation',source_job='706895',calibrated=False,seeds=[42,43,44],patient_count=136,class_counts={'AD':44,'NC':92},output_sha256={})
        cal_summary=dict(status='passed',split='validation',source_job='706895',calibration_accepted=True,patient_count=136,class_counts={'AD':44,'NC':92},temperatures={},output_sha256={})
        fits={}
        for model in a.MODELS:
            r=[];c=[];t=1.2
            for i in range(136):
                p=.05+i/150
                row=dict(patient_id=f'p{i:03}',image_id=str(i),label=int(i<44),split='validation',slice_count=20,p_ad=p,prediction=int(p>=.5),p_ad_seed42=p,p_ad_seed43=p,p_ad_seed44=p)
                r.append(row);q=temp.scale(p,t)
                c.append(dict(row,p_ad_raw=p,p_ad=q,temperature=t,prediction=int(q>=.5),prediction_raw=int(p>=.5)))
            for folder,table,summ in [(self.raw,r,raw_summary),(self.cal,c,cal_summary)]:
                path=folder/f'{model}_patient_predictions.csv';write_csv(path,table);summ['output_sha256'][path.name]=sha(path)
            cal_summary['temperatures'][model]=t
            fits[model]=dict(status='passed',temperature=t,converged=True,at_boundary=False,review_reasons=[])
        path=self.cal/'temperature_fits.json';path.write_text(json.dumps(fits));cal_summary['output_sha256'][path.name]=sha(path)
        for module,attr,path,data in [(temp,'ENSEMBLE_SUMMARY_SHA',self.raw/'ensemble_summary.json',raw_summary),(threshold,'TEMPERATURE_SUMMARY_SHA',self.cal/'temperature_summary.json',cal_summary)]:
            path.write_text(json.dumps(data));p=patch.object(module,attr,sha(path));p.start();self.addCleanup(p.stop)

    def test_complete_pipeline_preserves_inputs_and_outputs(self):
        from run_r5_ensemble import sha,read_csv
        self.assertIsNotNone(importlib.util.find_spec('plot_r5_referral'),'R5d plot implementation missing')
        before={p:sha(p) for folder in [self.raw,self.cal] for p in folder.iterdir()}
        result=self.a.run(self.raw,self.cal,self.out)
        self.assertEqual(result['status'],'passed');self.assertEqual(len(result['results']),12)
        for m in result['primary_10_percent']:self.assertEqual(m['referred_n'],14)
        self.assertEqual(result['image_reads'],dict(train=0,validation=0,test=0))
        for model in self.a.MODELS:
            r=read_csv(self.out/f'{model}_referral_flags.csv');self.assertEqual(len(r),136)
            self.assertEqual(sum(int(x['refer_10_percent']) for x in r),14)
        for name,h in result['output_sha256'].items():self.assertEqual(sha(self.out/name),h)
        self.assertEqual(before,{p:sha(p) for p in before})
        with self.assertRaises(ValueError):self.a.run(self.raw,self.cal,self.out)

    def test_tampered_raw_or_calibrated_source_rejected(self):
        for folder in [self.raw,self.cal]:
            path=folder/'CNN-A_patient_predictions.csv';text=path.read_text();path.write_text(text+'\n')
            with self.subTest(folder=folder),self.assertRaisesRegex(ValueError,'SHA'):self.a.run(self.raw,self.cal,self.out)
            self.assertFalse(self.out.exists());path.write_text(text)

    def test_matching_patients_but_wrong_raw_probability_rejected(self):
        from run_r5_ensemble import read_csv
        r=read_csv(self.raw/'CNN-A_patient_predictions.csv');c=read_csv(self.cal/'CNN-A_patient_predictions.csv')
        c[0]['p_ad_raw']=.6
        with self.assertRaises(ValueError):self.a.match_raw(r,c)

if __name__=='__main__':unittest.main()
