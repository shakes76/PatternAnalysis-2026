"""Synthetic tests only: no private dataset, GPU or Internet access."""
from __future__ import annotations

import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import confidence_analysis as ca
import analyse_confidence as runner
from dataset import COLUMNS, FINAL_STATUS, SliceRecord, counts_for_records, preprocessing_spec
from training_utils import aggregate_predictions


class MathsTests(unittest.TestCase):
    def test_empty_input_rejected(self):
        with self.assertRaises(ca.ConfidenceError): ca.arrays([], [])

    def test_dimensions_rejected(self):
        with self.assertRaises(ca.ConfidenceError): ca.arrays([0,1], [.1])

    def test_invalid_label_rejected(self):
        with self.assertRaises(ca.ConfidenceError): ca.arrays([2], [.1])

    def test_invalid_probabilities_rejected(self):
        for p in (-.01, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ca.ConfidenceError): ca.arrays([0], [p])

    def test_one_class_fit_rejected(self):
        with self.assertRaises(ca.ConfidenceError): ca.fit_temperature([0,0], [.1,.4])

    def test_temperature_positive_finite(self):
        for t in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ca.ConfidenceError): ca.scaled_probabilities([.2,.8], t)

    def test_temperature_identity(self):
        p=np.array([.001,.2,.5,.7,.999])
        np.testing.assert_allclose(ca.scaled_probabilities(p,1),p,atol=1e-15)

    def test_extreme_probabilities_finite(self):
        for t in (.5,1,10):
            q=ca.scaled_probabilities([0,.5,1],t)
            self.assertTrue(np.isfinite(q).all()); self.assertTrue(((q>0)&(q<1)).all())

    def test_constant_half_scores_prefer_identity(self):
        self.assertEqual(ca.fit_temperature([0,1,0,1],[.5]*4)['temperature'],1)

    def test_overconfidence_softened(self):
        r=ca.fit_temperature([0,0,1,1], [.01,.99,.01,.99])
        self.assertGreater(r['temperature'],1)
        self.assertLessEqual(r['temperature_fit_nll'],r['raw_fit_nll'])

    def test_underconfidence_can_sharpen(self):
        r=ca.fit_temperature([0,0,1,1], [.4,.3,.6,.7])
        self.assertLess(r['temperature'],1)
        self.assertLess(r['temperature_fit_nll'],r['raw_fit_nll'])

    def test_temperature_decisions_and_confidence_order_unchanged(self):
        p=np.array([.02,.12,.35,.49,.5,.59,.7,.85,.98])
        for t in (.5,1,5,10):
            q=ca.scaled_probabilities(p,t)
            np.testing.assert_array_equal(q>=.5,p>=.5)
            # Ignore numerical near-ties by comparing pairwise margins >1e-6.
            a=np.maximum(p,1-p); b=np.maximum(q,1-q)
            for i in range(len(p)):
                for j in range(len(p)):
                    if a[i]>a[j]+1e-6: self.assertGreater(b[i],b[j])

    def test_bounded_fit_beats_dense_grid(self):
        rng=np.random.default_rng(9); p=rng.uniform(.001,.999,60); y=rng.integers(0,2,60)
        fit=ca.fit_temperature(y,p)
        losses=[ca.probability_scores(y,ca.scaled_probabilities(p,t))['nll'] for t in np.linspace(.5,10,1000)]
        self.assertLessEqual(fit['temperature_fit_nll'],min(losses)+1e-8)

    def test_scores_for_half_predictions(self):
        m=ca.probability_scores([0,1],[.5,.5])
        self.assertAlmostEqual(m['nll'],np.log(2)); self.assertEqual(m['brier'],.25)
        self.assertEqual(m['ece_ad_probability_5_equal_width_bins'],0)

    def test_reliability_bins_cover_endpoints_once(self):
        m=ca.probability_scores([0,0,0,1,1,1],[0,.2,.4,.6,.8,1])
        self.assertEqual(sum(b['n'] for b in m['reliability_bins']),6)
        self.assertEqual(m['reliability_bins'][-1]['n'],2)

    def test_no_errors_error_auroc_undefined(self):
        m=ca.confidence_diagnostics([0,1],[.1,.9])
        self.assertIsNone(m['error_detection_auroc']); self.assertIsNone(m['mean_confidence_wrong'])

    def test_error_detection_ranking(self):
        m=ca.confidence_diagnostics([0,1,1,0],[.01,.99,.4,.6])
        self.assertEqual(m['error_detection_auroc'],1)


class ReferralTests(unittest.TestCase):
    def test_refer_all_does_not_claim_perfect_accuracy(self):
        m=ca.triage_metrics([0,1],[.1,.9],[False,False])
        self.assertIsNone(m['accepted_accuracy']); self.assertIsNone(m['accepted_error_rate'])
        self.assertEqual(m['n_ad_referred'],1)
        self.assertEqual(m['automatic_ad_detection_fraction_of_all_ad'],0)

    def test_confidence_ties_accepted_together(self):
        np.testing.assert_array_equal(ca.apply_referral([.25,.75,.5],.75),[True,True,False])

    def test_none_threshold_rejects_everyone(self):
        self.assertFalse(ca.apply_referral([0,1],None).any())

    def test_mask_must_be_boolean(self):
        with self.assertRaises(ca.ConfidenceError):ca.triage_metrics([0,1],[.1,.9],[0,1])

    def test_invalid_cutoff(self):
        with self.assertRaises(ca.ConfidenceError):ca.apply_referral([.1],.4)

    def test_triage_counts(self):
        m=ca.triage_metrics([0,0,1,1,1],[.1,.9,.1,.9,.6],[True,True,True,True,False])
        self.assertEqual(m['accepted_confusion_matrix_rows_true_columns_predicted_NC_AD'],[[1,1],[1,1]])
        self.assertEqual(m['n_ad_referred'],1); self.assertAlmostEqual(m['ad_automatically_cleared_fraction_of_all_ad'],1/3)
        self.assertEqual(m['accepted_error_rate'],.5)

    def test_wilson_empty(self):
        self.assertIsNone(ca.wilson_interval(0,0)['upper'])

    def test_wilson_zero_error_still_uncertain(self):
        m=ca.wilson_interval(0,20); self.assertAlmostEqual(m['upper'],.161125,places=5)

    def test_uninformative_scores_have_no_feasible_policy(self):
        r=ca.select_referral([0]*20+[1]*20,[.5]*40)
        self.assertIsNone(r['confidence_threshold'])
        self.assertEqual(r['development_fit_triage']['n_accepted'],0)

    def test_perfect_scores_select_maximum_coverage(self):
        r=ca.select_referral([0]*20+[1]*20,[.01]*20+[.99]*20)
        self.assertEqual(r['confidence_threshold'],.5)
        self.assertEqual(r['development_fit_triage']['coverage'],1)

    def test_ad_clearance_constraint_prevents_majority_only_policy(self):
        r=ca.select_referral([0]*90+[1]*10,[.01]*100)
        self.assertIsNone(r['confidence_threshold'])

    def test_no_acceptance_below_minimum_sample_requirement(self):
        r=ca.select_referral([0]*5+[1]*5,[.01]*5+[.99]*5)
        self.assertIsNone(r['confidence_threshold'])


class CrossfitTests(unittest.TestCase):
    def setUp(self):
        self.ids=[f'SYNTH_{i:03}' for i in range(40)]
        self.y=np.array([0]*20+[1]*20)
        self.p=np.linspace(.02,.98,40)

    def test_folds_deterministic_and_stratified(self):
        f=ca.fold_assignments(self.ids,self.y)
        np.testing.assert_array_equal(f,ca.fold_assignments(self.ids,self.y))
        for k in range(5):
            self.assertEqual(int(((f==k)&(self.y==0)).sum()),4)
            self.assertEqual(int(((f==k)&(self.y==1)).sum()),4)

    def test_folds_independent_of_input_order(self):
        f=ca.fold_assignments(self.ids,self.y)
        order=np.random.default_rng(3).permutation(40)
        other=ca.fold_assignments([self.ids[i] for i in order],self.y[order])
        np.testing.assert_array_equal(other,f[order])

    def test_duplicate_subject_rejected(self):
        ids=self.ids.copy(); ids[1]=ids[0]
        with self.assertRaises(ca.ConfidenceError):ca.fold_assignments(ids,self.y)

    def test_insufficient_class_for_five_folds_rejected(self):
        with self.assertRaises(ca.ConfidenceError):ca.fold_assignments(['a','b'],[0,1])

    def test_crossfit_never_fits_held_fold(self):
        original=ca.fit_temperature; recorded=[]
        def spy(y,p):
            recorded.append(list(p)); return original(y,p)
        folds=ca.fold_assignments(self.ids,self.y)
        with patch.object(ca,'fit_temperature',side_effect=spy):
            result,private=ca.crossfit(self.ids,self.y,self.p)
        self.assertEqual(len(recorded),5)
        for k,fit_p in enumerate(recorded):
            np.testing.assert_array_equal(fit_p,self.p[folds!=k])
        self.assertEqual(result['pooled_triage']['n_total'],40)
        self.assertEqual(len(private['fold_index']),40)

    def test_analysis_public_output_has_no_subject_ids(self):
        r,private=ca.analyse_patients(self.ids,self.y,self.p)
        text=json.dumps(r,allow_nan=False)
        self.assertTrue(all(s not in text for s in self.ids))
        self.assertIn(self.ids[0],json.dumps(private))
        self.assertEqual(r['diagnosis_threshold'],.5)


class SyntheticRun:
    """Complete aggregate/prediction fixture; creates NO images or real weights."""
    def __init__(self, private: Path):
        self.private=private; private.mkdir()
        self.project=Path(runner.__file__).resolve().parent
        (private/'project_path.txt').write_text(str(self.project),encoding='utf-8')
        root=private/'data'/'ADNI'; root.mkdir(parents=True)
        self.splits=private/'splits'; self.splits.mkdir()
        self.records={}; total_subjects=0; hashes={}; counts={}
        for split in ('train','validation','test'):
            records=[]; per_class=20 if split=='validation' else 1
            for label,name in enumerate(('NC','AD')):
                for patient in range(per_class):
                    total_subjects+=1; scan=str(100000+total_subjects); subject=f'SYNTH_{split}_{label}_{patient}'
                    for z in range(80,100):
                        records.append(SliceRecord(f'AD_NC/train/{name}/{scan}_{z}.jpeg',subject,scan,z,'train',name,2*label,label,split))
            self.records[split]=tuple(records)
            stream=io.StringIO(newline=''); w=csv.DictWriter(stream,fieldnames=COLUMNS,lineterminator='\n'); w.writeheader()
            w.writerows(asdict(r) for r in records)
            path=self.splits/f'{split}.csv'; path.write_bytes(stream.getvalue().encode())
            hashes[path.name]=runner.sha256(path); counts[split]=counts_for_records(tuple(records))
        summary={"format_version":1,"status":FINAL_STATUS,"mapping_structural_checks_passed":True,
            "patient_identifier_semantics_confirmed_by_user":True,"tutor_split_change_approved_as_reported_by_user":True,
            "zero_group_overlap_verified":True,"splits_finalized":True,"new_pairwise_candidate_subject_overlap":0,
            "model_label_map":{"NC":0,"AD":1},"metadata_to_model_labels":{"0":0,"2":1},
            "manifest_sha256":hashes,"proposed_or_final_counts":counts,
            "evidence":{"images":total_subjects*20,"scan_prefixes":total_subjects,"candidate_subjects":total_subjects}}
        runner.write_json(self.splits/'split_summary.json',summary)
        (private/'latest_patient_splits.txt').write_text(str(self.splits),encoding='utf-8')
        self.paths={}; reports={}
        source_names=('dataset.py','modules.py','training_utils.py','train.py','train_experiment.py','check_experiment.py')
        sources={n:runner.sha256(self.project/n) for n in source_names}
        for key in ('initial','baseline','convnext'):
            path=private/key; path.mkdir(); self.paths[key]=path
            probabilities=[]
            for r in self.records['validation']:
                index=int(r.subject_id.rsplit('_',1)[1]); correct=(index%7!=0)
                p=(.90 if r.label else .10) if correct else (.10 if r.label else .90)
                if key=='initial':p=.5
                probabilities.append(p)
            metrics,predictions=aggregate_predictions(self.records['validation'],probabilities)
            runner.write_json(path/'PRIVATE_validation_predictions.json',predictions)
            cp=path/('PRIVATE_best_baseline.pt' if key=='initial' else 'PRIVATE_best_model.pt')
            cp.write_bytes(b'SYNTHETIC_HASH_FIXTURE_NOT_A_REAL_CHECKPOINT')
            report={'baseline_training_completed':True,'experiment_completed':True,'model_name':key,
                'final_test_model_evaluation_performed':False,'final_test_images_decoded':0,'selected_epoch':1,
                'manifest_sha256':hashes,'preprocessing':preprocessing_spec(0.0),
                'split_counts':{s:counts[s] for s in ('train','validation')},'configuration':{'model':key},
                'source_code_sha256':sources,'selected_validation_metrics':metrics,
                'selected_checkpoint_reloaded_and_checked':True,'selected_checkpoint_sha256':runner.sha256(cp)}
            runner.write_json(path/('baseline_summary.json' if key=='initial' else 'experiment_summary.json'),report)
            reports[key]=report
            pointer={'initial':'latest_baseline_run.txt','baseline':'latest_step8_baseline.txt','convnext':'latest_step8_convnext.txt'}[key]
            (private/pointer).write_text(str(path),encoding='utf-8')
        compare=private/'comparison'; compare.mkdir()
        c={'matched_step8_configuration_verified':True,'manifest_sha256':hashes,
           'preferred_cnn_by_prespecified_validation_rule':'Regularized CNN','models':{}}
        for key,name in [('initial','Initial CNN'),('baseline','Regularized CNN'),('convnext','ConvNeXt-Tiny')]:
            c['models'][name]={'selected_epoch':1,'patient_metrics':reports[key]['selected_validation_metrics']['patient']}
        runner.write_json(compare/'comparison_summary.json',c)
        (private/'latest_step8_comparison.txt').write_text(str(compare),encoding='utf-8')
        self.reports=reports


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.f=SyntheticRun(Path(self.temp.name)/'private')

    def test_cached_predictions_reproduce_reports(self):
        m=runner.check_inputs(self.f.private)
        self.assertEqual(len(m['models']['cnn']['subjects']),40)
        self.assertEqual(m['models']['cnn']['name'],'Regularized CNN')

    def test_changed_checkpoint_rejected(self):
        (self.f.paths['convnext']/'PRIVATE_best_model.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ca.ConfidenceError,'checkpoint'):runner.check_inputs(self.f.private)

    def test_changed_predicted_probability_rejected(self):
        path=self.f.paths['convnext']/'PRIVATE_validation_predictions.json';d=runner.read_json(path)
        d['slices'][0]['p_ad']=.42;runner.write_json(path,d)
        with self.assertRaises((ca.ConfidenceError,ValueError)):runner.check_inputs(self.f.private)

    def test_duplicate_prediction_rejected(self):
        path=self.f.paths['convnext']/'PRIVATE_validation_predictions.json';d=runner.read_json(path)
        d['slices'][1]=d['slices'][0];runner.write_json(path,d)
        with self.assertRaisesRegex(ca.ConfidenceError,'Duplicate'):runner.check_inputs(self.f.private)

    def test_wrong_partition_prediction_rejected(self):
        path=self.f.paths['convnext']/'PRIVATE_validation_predictions.json';d=runner.read_json(path)
        d['slices'][0]['image_path']=self.f.records['test'][0].image_path;runner.write_json(path,d)
        with self.assertRaisesRegex(ca.ConfidenceError,'exactly'):runner.check_inputs(self.f.private)

    def test_changed_patient_prediction_rejected(self):
        path=self.f.paths['convnext']/'PRIVATE_validation_predictions.json';d=runner.read_json(path)
        d['patients'][0]['p_ad']=.42;runner.write_json(path,d)
        with self.assertRaises(ca.ConfidenceError):runner.check_inputs(self.f.private)

    def test_changed_source_guard(self):
        report=self.f.reports['convnext'];report['source_code_sha256']['modules.py']='0'*64
        # Change both new reports, so matched-recipe verification still passes.
        for key in ('baseline','convnext'):
            r=self.f.reports[key];r['source_code_sha256']['modules.py']='0'*64
            runner.write_json(self.f.paths[key]/'experiment_summary.json',r)
        with self.assertRaisesRegex(ca.ConfidenceError,'source'):runner.check_inputs(self.f.private)

    def test_missing_pointer_clear_error(self):
        (self.f.private/'latest_step8_comparison.txt').unlink()
        with self.assertRaisesRegex(ca.ConfidenceError,'pointer'):runner.check_inputs(self.f.private)

    def test_empty_pointer_rejected(self):
        (self.f.private/'latest_baseline_run.txt').write_text('  ',encoding='utf-8')
        with self.assertRaisesRegex(ca.ConfidenceError,'Empty'):runner.check_inputs(self.f.private)

    def test_finished_run_preserves_inputs_and_bundles_only_allowlist(self):
        before={str(p):runner.sha256(p) for p in self.f.private.rglob('*') if p.is_file()}
        # There are no JPEG files anywhere; any image access would fail.
        output=runner.run(self.f.private)
        for p,h in before.items():self.assertEqual(runner.sha256(Path(p)),h)
        report=runner.read_json(output/'confidence_summary.json')
        self.assertTrue(report['step9_completed']);self.assertFalse(report['final_test_model_evaluation_performed'])
        self.assertEqual(report['mri_images_opened'],0)
        with ZipFile(output/'COMP3710_Step9_Results.zip') as z:
            self.assertEqual(set(z.namelist()),set(runner.BUNDLE_FILES))
            for name in z.namelist():
                if not name.endswith('.png'):self.assertNotIn('SYNTH_validation',z.read(name).decode())
        self.assertEqual((self.f.private/'latest_step9_confidence.txt').read_text().strip(),str(output))

    def test_failure_does_not_replace_success_pointer(self):
        target=self.f.private/'latest_step9_confidence.txt';target.write_text('keep-me',encoding='utf-8')
        with patch.object(runner,'plot_results',side_effect=RuntimeError('synthetic plotting failure')):
            with self.assertRaises(RuntimeError):runner.run(self.f.private)
        self.assertEqual(target.read_text(),'keep-me')


if __name__=='__main__':unittest.main()
