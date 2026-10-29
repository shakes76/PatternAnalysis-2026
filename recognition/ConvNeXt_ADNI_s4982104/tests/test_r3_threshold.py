import importlib.util
import math
from pathlib import Path
import tempfile
import unittest


def fixture(positives,negatives):
    rows=[dict(patient_id=f'p{i}',image_id=f'i{i}',label=y,p_ad=p,slice_count=20)
          for i,(y,p) in enumerate([(1,p) for p in positives]+[(0,p) for p in negatives])]
    expected=[dict(r,split='validation') for r in rows]
    return rows,expected


class ThresholdTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_r3_threshold'),'threshold implementation missing')
        import run_r3_threshold
        return run_r3_threshold

    def test_44_ad_requires_36_detected_and_counts_cost(self):
        rows,expected=fixture([i/100 for i in range(1,45)],[.05,.1,.5])
        r=self.api().choose_threshold(rows,expected)
        self.assertEqual(r['required_tp'],36);self.assertEqual(r['selected']['threshold'],.09)
        self.assertEqual(r['selected']['tp'],36);self.assertEqual(r['selected']['fp'],2)
        self.assertEqual(r['baseline']['tp'],0);self.assertEqual(r['baseline']['fp'],1)
        self.assertEqual(r['delta']['tp'],36);self.assertEqual(r['delta']['fp'],1)

    def test_equal_probabilities_cannot_split_and_boundary_is_inclusive(self):
        rows,expected=fixture([.2]*5,[.2,.1])
        r=self.api().choose_threshold(rows,expected)
        self.assertEqual(r['selected']['threshold'],.2)
        self.assertEqual(r['selected']['tp'],5);self.assertEqual(r['selected']['fp'],1)

    def test_equal_specificity_prefers_larger_threshold_can_exceed_half(self):
        rows,expected=fixture([.9,.8,.7,.6,.4],[.1,.2])
        r=self.api().choose_threshold(rows,expected)
        self.assertEqual(r['selected']['threshold'],.6)
        self.assertEqual(r['selected']['tn'],2)
        self.assertEqual(r['delta']['tp'],0);self.assertEqual(r['delta']['fp'],0)

    def test_all_zero_scores_and_single_positive(self):
        for pos,neg in [([0]*5,[0]*3),([1],[0])]:
            rows,expected=fixture(pos,neg);r=self.api().choose_threshold(rows,expected)
            self.assertEqual(r['selected']['tp'],len(pos))
        rows,expected=fixture([0]*5,[0]*3);r=self.api().choose_threshold(rows,expected)
        self.assertEqual(r['selected']['threshold'],0);self.assertEqual(r['selected']['fp'],3)

    def test_order_invariance_and_finite_scores(self):
        rows,expected=fixture([.2,.3,.4,.5,.6],[.1,.7])
        self.assertEqual(self.api().choose_threshold(rows,expected),self.api().choose_threshold(rows[::-1],expected[::-1]))
        for value in [float('nan'),float('inf'),-.01,1.01,True]:
            bad=[dict(r) for r in rows];bad[0]['p_ad']=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.api().choose_threshold(bad,expected)

    def test_wrong_patient_label_scan_split_or_missing_class_rejected(self):
        rows,expected=fixture([.4,.6],[.1,.5]);api=self.api()
        for key,value in [('patient_id','unknown'),('image_id','wrong'),('label',0),('label',True),('slice_count',19)]:
            bad=[dict(r) for r in rows];bad[0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):api.choose_threshold(bad,expected)
        for bad in [rows[:-1],rows+[rows[0]],[]]:
            with self.assertRaises(ValueError):api.choose_threshold(bad,expected)
        bad=[dict(r,split='test') for r in expected]
        with self.assertRaises(ValueError):api.choose_threshold(rows,bad)
        with self.assertRaises(ValueError):api.choose_threshold(rows[:2],expected[:2])
        with self.assertRaises(ValueError):api.choose_threshold(rows,expected+[expected[0]])

    def test_invalid_recall_target_rejected(self):
        rows,expected=fixture([.3],[.4])
        for target in [0,1.1,float('nan'),True,'0.8']:
            with self.subTest(target=target),self.assertRaises(ValueError):self.api().choose_threshold(rows,expected,target)

    def test_frozen_input_hash_rejects_tamper_and_unrecorded_file(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json';p.write_text('{}');hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()}
            self.assertEqual(self.api().verified_bytes(p,hashes),b'{}')
            p.write_text('{"changed":1}')
            with self.assertRaises(ValueError):self.api().verified_bytes(p,hashes)
            with self.assertRaises(ValueError):self.api().verified_bytes(p,{})


class ThresholdPipelineTests(unittest.TestCase):
    """Synthetic full nine-run evidence chain; contains CSVs only, no images."""
    def setUp(self):
        import csv,json,hashlib
        from patient_evaluation import evaluate_patients,POLICY
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.hashes={}
        def save_json(path,data):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));self.hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        def save_csv(path,rows):
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            self.hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.save_csv=save_csv
        manifest=[];slices=[]
        for patient in range(136):
            label=int(patient<44)
            for index in range(20):
                record=dict(file_path=f'/not-real/{patient}/{index}.jpeg',patient_id=f'p{patient}',image_id=f'i{patient}',label=label,split='validation')
                manifest.append(record);slices.append(dict(record,p_ad=.4 if label else .2))
        self.manifest=self.root/'split_samples.csv';save_csv(self.manifest,[dict(r,label='AD' if r['label'] else 'NC') for r in manifest])
        save_json(self.root/'split_summary.json',{})
        evaluation=evaluate_patients(slices,manifest);metrics=evaluation['metrics'];main=[]
        for model in ('CNN-A','CNN-B','ConvNeXt-F'):
            for seed in (42,43,44):
                source=self.root/f'{model}_{seed}';folder=source/'reload/patient_nll'
                save_csv(folder/'slice_predictions.csv',slices);save_csv(folder/'patient_predictions.csv',evaluation['patients'])
                best=dict(epoch=1,value=metrics['nll'],sha256='synthetic-checkpoint')
                selection=dict(status='passed',policy=POLICY,best=dict(patient_nll=best))
                cfg=dict(seed=seed,epochs=30,early_stopping=False,threshold=.5,model='convnext_tiny' if model=='ConvNeXt-F' else 'SimpleCNN',arm=model[-1])
                training=dict(status='passed',config=cfg,selection=selection,provenance=dict(input_sha256={k:self.hashes[str(self.root/k)] for k in ('split_samples.csv','split_summary.json')}),history=[dict(epoch=e,patient_metrics=dict(nll=metrics['nll'])) for e in range(1,31)])
                save_json(source/'training_summary.json',training);save_json(source/'selection/selection_summary.json',selection)
                save_json(folder/'summary.json',dict(epoch=1,checkpoint_sha256='synthetic-checkpoint',repeat_equal=True,selected_epoch_equal=True,output_sha256={n:self.hashes[str(folder/n)] for n in ('slice_predictions.csv','patient_predictions.csv')}))
                main.append(dict(model=model,seed=seed,source=str(source),epoch=1,patients=136,threshold=.5,tn=92,fp=0,fn=44,tp=0,accuracy=92/136,ad_recall=0,ad_precision=0,nc_specificity=1,balanced_accuracy=.5,macro_f1=(184/228)/2,**{k:metrics[k] for k in ('nll','brier','auroc')}))
        for name in ('patient_evaluation.py','train.py'):
            path=Path(__file__).resolve().parents[1]/name;self.hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.main=self.root/'nine_run_main.csv';save_csv(self.main,main);self.rows=main
        save_json(self.root/'audit_summary.json',dict(status='passed',runs=9,inputs_sha256=dict(self.hashes)))

    def test_pipeline_nine_frozen_bests_without_images_and_no_overwrite(self):
        import csv
        from run_r3_threshold import analyze
        result=analyze(self.main,self.manifest,self.root/'out')
        self.assertEqual(result['status'],'passed');self.assertEqual(result['mri_reads'],0)
        rows=list(csv.DictReader((self.root/'out/nine_thresholds.csv').open()))
        self.assertEqual(len(rows),9)
        self.assertTrue(all(float(r['selected_threshold'])==.4 and int(r['selected_tp'])==44 and int(r['selected_fp'])==0 for r in rows))
        with self.assertRaises(ValueError):analyze(self.main,self.manifest,self.root/'out')

    def test_pipeline_rejects_switched_checkpoint_and_tampered_predictions(self):
        from run_r3_threshold import analyze
        self.rows[0]['epoch']=2;self.save_csv(self.main,self.rows)
        with self.assertRaisesRegex(ValueError,'best epoch'):analyze(self.main,self.manifest,self.root/'bad_epoch')
        self.rows[0]['epoch']=1;self.save_csv(self.main,self.rows)
        path=Path(self.rows[0]['source'])/'reload/patient_nll/patient_predictions.csv';path.write_text('tampered')
        with self.assertRaisesRegex(ValueError,'Frozen input changed'):analyze(self.main,self.manifest,self.root/'bad_hash')

    def test_pipeline_rejects_changed_baseline_and_duplicate_run(self):
        from run_r3_threshold import analyze
        original=self.rows[0]['accuracy'];self.rows[0]['accuracy']=.99;self.save_csv(self.main,self.rows)
        with self.assertRaisesRegex(ValueError,'baseline mismatch'):analyze(self.main,self.manifest,self.root/'bad_baseline')
        self.rows[0]['accuracy']=original;self.rows[1]=dict(self.rows[0]);self.save_csv(self.main,self.rows)
        with self.assertRaisesRegex(ValueError,'nine unique'):analyze(self.main,self.manifest,self.root/'duplicate')

    def test_pipeline_rejects_changed_provenance_summaries(self):
        from run_r3_threshold import analyze
        source=Path(self.rows[0]['source'])
        for relative in ('training_summary.json','selection/selection_summary.json','reload/patient_nll/summary.json'):
            path=source/relative;original=path.read_bytes();path.write_text('{}')
            try:
                with self.subTest(file=relative),self.assertRaisesRegex(ValueError,'Frozen input changed'):
                    analyze(self.main,self.manifest,self.root/'bad_summary')
            finally:path.write_bytes(original)
