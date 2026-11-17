"""Synthetic tests only; never read the user's medical images or private runs."""
from __future__ import annotations
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import negative_policy as n


class PolicyTests(unittest.TestCase):
    def test_reference_matches_existing(self):
        from confidence_analysis import apply_referral
        p = [0., .05, .15, .25, .49, .5, .75, .85, .95, 1.]
        for t in (.5, .75, .85, .95, 1., None):
            accepted = apply_referral(p,t).tolist()
            actual = n.actions(p,t,'reference')
            self.assertEqual([a!='refer' for a in actual],accepted)
    def test_strict_cn_boundary(self):
        self.assertEqual(n.actions([.05,.050000001],.75,'strict_cn'),['auto_CN','refer'])
    def test_ad_branch_unchanged(self):
        for t in (.5,.75,.85,None):
            p=[0,.02,.25,.49,.5,.75,.9,1]
            reference=n.actions(p,t,'reference')
            for mode in n.POLICIES:
                self.assertEqual([a=='AD_flag' for a in n.actions(p,t,mode)],
                                 [a=='AD_flag' for a in reference])
    def test_no_auto_cn_at_zero(self):
        self.assertEqual(n.actions([0,.01,.49],.75,'no_auto_cn'),['refer']*3)
    def test_half_is_never_cn(self):
        for mode in n.POLICIES:
            self.assertEqual(n.actions([.5],.5,mode),['AD_flag'])
    def test_refer_all_propagates(self):
        for mode in n.POLICIES:
            self.assertEqual(n.actions([0,.2,.8,1],None,mode),['refer']*4)
    def test_original_stricter_than_new_cap(self):
        self.assertEqual(n.actions([.02,.005],.99,'strict_cn'),['refer','auto_CN'])
    def test_nonfinite_rejected(self):
        for x in (float('nan'),float('inf'),-.01,1.01):
            with self.assertRaises(ValueError):n.actions([x],.75,'reference')
    def test_invalid_threshold_rejected(self):
        for x in (.49,1.01,float('nan'),True):
            with self.assertRaises(ValueError):n.actions([.5],x,'reference')
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):n.actions([],.75,'reference')
    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):n.actions([.5],.75,'new_search')
    def test_counts_and_denominators(self):
        m=n.metrics([0,1,0,1,0,1],['auto_CN','auto_CN','AD_flag','AD_flag','refer','refer'])
        self.assertEqual(m['action_table_rows_true_CN_AD_columns_auto_CN_AD_flag_review'],[[1,1,1],[1,1,1]])
        self.assertAlmostEqual(m['AD_auto_CN_fraction_of_all_AD']['estimate'],1/3)
        self.assertAlmostEqual(m['AD_fraction_among_auto_CN']['estimate'],1/2)
        self.assertEqual(m['n_clinical_reviews_if_all_AD_flags_also_reviewed'],4)
    def test_no_accepted_is_not_perfect(self):
        m=n.metrics([0,1],['refer','refer'])
        self.assertIsNone(m['automated_research_label_accuracy'])
        self.assertIsNone(m['automated_research_label_error']['estimate'])
        self.assertIsNone(m['negative_predictive_value'])
    def test_no_auto_cn_still_requires_clinical_review(self):
        m=n.metrics([0,1],['refer','AD_flag'])
        self.assertEqual(m['n_clinical_reviews_if_all_AD_flags_also_reviewed'],2)
        self.assertEqual(m['AD_auto_CN_fraction_of_all_AD']['estimate'],0.)
    def test_no_ad_denominator(self):
        m=n.metrics([0,0],['auto_CN','refer'])
        self.assertIsNone(m['AD_auto_CN_fraction_of_all_AD']['estimate'])
    def test_zero_errors_nonzero_upper_interval(self):
        self.assertGreater(n.interval(0,10)['upper'],.2)
    def test_monotone_false_negative_count(self):
        y=[0,1]*101
        p=[i/201 for i in range(202)]
        result,_=n.compare_actions(y,p,[.75]*len(y))
        self.assertLessEqual(result['strict_cn']['n_AD_auto_CN'],result['reference']['n_AD_auto_CN'])
        self.assertEqual(result['no_auto_cn']['n_AD_auto_CN'],0)
        self.assertGreaterEqual(result['strict_cn']['n_unresolved_referrals'],result['reference']['n_unresolved_referrals'])
    def test_accepted_error_fraction_can_worsen(self):
        result,_=n.compare_actions([0,0,1,0],[.1,.2,.01,.99],[.75]*4)
        self.assertGreater(result['strict_cn']['automated_research_label_error']['estimate'],
                           result['reference']['automated_research_label_error']['estimate'])
    def test_new_referral_counts_by_class(self):
        result,_=n.compare_actions([0,1,1],[.2,.2,.9],[.75]*3)
        changes=result['strict_cn']['relative_to_reference']
        self.assertEqual(changes['additional_referrals'],2)
        self.assertEqual(changes['additional_AD_referrals_previously_auto_CN'],1)
        self.assertEqual(changes['additional_CN_referrals_previously_auto_CN'],1)
    def test_labels_and_action_validation(self):
        for y,a in [([2],['refer']),([0],['bad']),([],[]),([0],['refer','refer']),([False],['refer'])]:
            with self.assertRaises(ValueError):n.metrics(y,a)
    def test_threshold_vector_length(self):
        with self.assertRaises(ValueError):n.compare_actions([0,1],[.1,.9],[.75])


def fixture():
    from confidence_analysis import analyse_patients
    subjects=[f'SYNTHETIC_ONLY_{i}' for i in range(30)]
    labels=[0]*15+[1]*15
    p=[.01,.03,.05,.08,.1,.12,.18,.2,.3,.4,.45,.55,.7,.85,.98,
       .01,.02,.1,.2,.3,.4,.5,.6,.65,.7,.8,.85,.9,.95,.99]
    report,private=analyse_patients(subjects,labels,p)
    report['original_validation_metrics']={'patient':{'n':30,'accuracy':sum((v>=.5)==y for v,y in zip(p,labels))/30}}
    rule={'temperature':report['full_validation_temperature_fit']['temperature'],
          'confidence_threshold':report['operating_rule']['confidence_threshold']}
    candidate=SimpleNamespace(subjects=subjects,labels=labels,p=p,name='Synthetic model',report={'selected_epoch':13})
    return candidate,private['patients'],report,rule


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=fixture()
    def data_copy(self):return copy.deepcopy(self.data)
    def test_saved_rows_and_original_reference_reproduce(self):
        c,rows,report,rule=self.data_copy()
        result=n.inspect_patients(c,rows,report,rule)
        self.assertEqual(result['primary_classifier_metrics_unchanged']['n'],30)
        self.assertEqual(len(result['crossfit_diagnostic']['fold_results']),5)
    def test_raw_probability_tamper_rejected(self):
        c,rows,r,t=self.data_copy();rows[0]['raw_p_ad']+=.001
        with self.assertRaises(ValueError):n.inspect_patients(c,rows,r,t)
    def test_oof_tamper_rejected(self):
        c,rows,r,t=self.data_copy();rows[0]['oof_p_ad']+=.001
        with self.assertRaises(ValueError):n.inspect_patients(c,rows,r,t)
    def test_full_calibration_tamper_rejected(self):
        c,rows,r,t=self.data_copy();rows[0]['full_fit_calibrated_p_ad']+=.001
        with self.assertRaises(ValueError):n.inspect_patients(c,rows,r,t)
    def test_fold_tamper_rejected(self):
        c,rows,r,t=self.data_copy();rows[0]['fold_index']=(rows[0]['fold_index']+1)%5
        with self.assertRaises(ValueError):n.inspect_patients(c,rows,r,t)
    def test_acceptance_tamper_rejected(self):
        c,rows,r,t=self.data_copy();rows[0]['oof_accepted']=not rows[0]['oof_accepted']
        with self.assertRaises(ValueError):n.inspect_patients(c,rows,r,t)
    def test_duplicate_patient_rejected(self):
        c,rows,r,t=self.data_copy();rows[1]['subject_id']=rows[0]['subject_id']
        with self.assertRaises(ValueError):n.inspect_patients(c,rows,r,t)
    def test_missing_patient_rejected(self):
        c,rows,r,t=self.data_copy();rows.pop()
        with self.assertRaises(ValueError):n.inspect_patients(c,rows,r,t)
    def test_label_mismatch_rejected(self):
        c,rows,r,t=self.data_copy();rows[0]['label']=1
        with self.assertRaises(ValueError):n.inspect_patients(c,rows,r,t)
    def test_new_rule_changes_protected(self):
        c,rows,r,t=self.data_copy();t['confidence_threshold']=.95
        with self.assertRaises(ValueError):n.inspect_patients(c,rows,r,t)
    def test_no_new_fit_called(self):
        c,rows,r,t=self.data_copy()
        with patch('confidence_analysis.fit_temperature',side_effect=AssertionError('Unexpected fit')):
            with patch('confidence_analysis.select_referral',side_effect=AssertionError('Unexpected search')):
                n.inspect_patients(c,rows,r,t)
    def test_report_has_no_individual_identifiers(self):
        c,rows,r,t=self.data_copy()
        result=n.inspect_patients(c,rows,r,t)
        encoded=json.dumps(result)
        self.assertNotIn('SYNTHETIC_ONLY',encoded)
        self.assertNotIn('subject_id',encoded)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='negative policy tests ')
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.project=self.base/'repo'/'recognition'/'project'
        self.project.mkdir(parents=True)
        self.private=self.base/'private';self.private.mkdir()
        for name in n.CODE_FILES:
            p=self.project/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes((PROJECT/name).read_bytes())
        (self.project/'README.md').write_bytes(b'# Existing report\r\nKeep original.\r\n')
        c,rows,m,r=fixture()
        rules={'models':{}}
        summary={'models':{}}
        privaterows={}
        candidates={}
        for alias in ('cnn','a1'):
            a=copy.deepcopy(c);a.name=alias;candidates[alias]=a
            rule=copy.deepcopy(r);rule.update(name=alias,selected_epoch=13,checkpoint_sha256='abc')
            rules['models'][alias]=rule;summary['models'][alias]=copy.deepcopy(m)
            privaterows[alias]={'patients':copy.deepcopy(rows)}
        self.ctx={'private':self.private,'project':self.project,'candidates':candidates}
        self.inputs={'synthetic_input_sha256':'example'}
        self.value=(self.ctx,summary,rules,privaterows,self.inputs)
        self.patcher=patch.object(n,'load_inputs',return_value=self.value);self.patcher.start()
        self.addCleanup(self.patcher.stop)
        n.prepare(self.private)
    def test_complete_synthetic_workflow_no_images(self):
        out=n.run(self.private,enforce_git=False)
        report=n.read_json(out/'negative_policy_summary.json')
        self.assertTrue(report['comparison_completed'])
        self.assertFalse(report['new_inference_performed'])
        self.assertFalse(report['new_threshold_search_performed'])
        with ZipFile(out/n.BUNDLE) as z:
            self.assertEqual(set(z.namelist()),set(n.PUBLIC_FILES))
            self.assertNotIn(b'SYNTHETIC_ONLY',b''.join(z.read(k) for k in z.namelist()))
    def test_completed_repeat_is_idempotent(self):
        out=n.run(self.private,enforce_git=False);old=(out/'negative_policy_summary.json').read_bytes()
        n.run(self.private,enforce_git=False)
        self.assertEqual(old,(out/'negative_policy_summary.json').read_bytes())
    def test_report_publish_preserves_old_bytes(self):
        old=(self.project/'README.md').read_bytes();n.run(self.private,enforce_git=False)
        n.publish(self.private);new=(self.project/'README.md').read_bytes()
        self.assertTrue(new.startswith(old))
        n.publish(self.private);self.assertEqual(new,(self.project/'README.md').read_bytes())
    def test_plan_conflict_rejected(self):
        plan=n.read_json(self.project/n.PLAN_FILE);plan['strict_CN_maximum_calibrated_p_AD']=.2
        n.write_json(self.project/n.PLAN_FILE,plan)
        with self.assertRaises(ValueError):n.prepare(self.private)
        with self.assertRaises(ValueError):n.run(self.private,enforce_git=False)
    def test_output_conflict_rejected(self):
        out=n.run(self.private,enforce_git=False)
        (out/'negative_policy_rules.json').write_text('{}')
        with self.assertRaises(ValueError):n.run(self.private,enforce_git=False)
    def test_rules_retain_both_secondary_policies(self):
        out=n.run(self.private,enforce_git=False)
        r=n.read_json(out/'negative_policy_rules.json')
        self.assertTrue(r['policies_all_retained_no_winner_selected'])
        for model in r['models'].values():self.assertEqual(set(model['policies']),set(n.POLICIES))
    def test_input_change_midrun_rejected_before_completion(self):
        changed=copy.copy(self.value);changed=list(changed);changed[-1]={'changed':'yes'}
        with patch.object(n,'load_inputs',side_effect=[self.value,tuple(changed)]):
            with self.assertRaises(ValueError):n.run(self.private,enforce_git=False)
        self.assertFalse((self.private/n.OUTPUT_FOLDER/'negative_policy_summary.json').exists())
    def test_duplicate_json_key_rejected(self):
        f=self.private/'x.json';f.write_text('{"a":1,"a":2}')
        with self.assertRaises(ValueError):n.read_json(f)
    def test_nonfinite_json_rejected(self):
        f=self.private/'x.json';f.write_text('{"a":NaN}')
        with self.assertRaises(ValueError):n.read_json(f)
    def test_requires_commit(self):
        repo=self.project.parents[1]
        def g(*args):
            r=subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True)
            if r.returncode:self.fail(r.stderr)
        g('init');g('checkout','-b','topic-recognition')
        g('config','user.name','Synthetic tester');g('config','user.email','synthetic@example.invalid')
        with self.assertRaises(ValueError):n.require_committed(self.project)
        g('add','.');g('commit','-m','Synthetic study')
        n.require_committed(self.project)
        with (self.project/'negative_policy.py').open('a') as f:f.write('\n# changed\n')
        with self.assertRaises(ValueError):n.require_committed(self.project)


if __name__=='__main__':
    unittest.main()
