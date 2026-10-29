import json
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import test_weighting
from run_e3 import run_baseline
from run_e4 import run_weighted
from run_e5 import summarize_runs, run_repeats

class E5Tests(unittest.TestCase):
    def test_known_paired_statistics_and_rule(self):
        rows=[]
        for seed,a,b in [(42,0.2,0.3),(43,0.4,0.6),(44,0.6,0.5)]:
            for arm,x in [('A',a),('B',b)]:
                rows.append(dict(seed=seed,arm=arm,patient_ad_recall=x,patient_balanced_accuracy=x,
                    patient_macro_f1=x,patient_auroc=None))
        s=summarize_runs(rows)
        self.assertAlmostEqual(s['statistics']['A']['patient_ad_recall']['mean'],0.4)
        self.assertAlmostEqual(s['statistics']['A']['patient_ad_recall']['sample_std'],0.2)
        self.assertAlmostEqual(s['statistics']['B_minus_A']['patient_ad_recall']['mean'],.2/3)
        self.assertEqual(s['ad_recall_improved_pairs'],2)
        self.assertTrue(s['candidate_for_further_validation'])
        self.assertIsNone(s['statistics']['A']['patient_auroc']['mean'])
        with self.assertRaises(ValueError):summarize_runs(rows[:-1])
        with self.assertRaises(ValueError):summarize_runs(rows+[rows[0]])
    def setUp(self):
        self.fixture=test_weighting.WeightingTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.f=self.fixture.f
    def refs(self):
        args=dict(device='cpu',epochs=3,expected_lengths=None,expected_slices=5,expected_patients=1)
        run_baseline(self.f.manifest,self.f.summary,self.f.root/'A42',**args)
        run_weighted(self.f.manifest,self.f.summary,self.f.root/'A42',self.f.root/'B42',**args)
    def run_small(self):
        return run_repeats(self.f.manifest,self.f.summary,self.f.root/'A42',self.f.root/'B42',self.f.root/'e5',
            device='cpu',epochs=3,expected_lengths=None,expected_slices=5,expected_patients=1)
    def test_end_to_end_four_new_runs_six_rows_and_readonly_refs(self):
        self.refs();before={str(p):p.read_bytes() for d in ['A42','B42'] for p in (self.f.root/d).rglob('*') if p.is_file()}
        s=self.run_small();self.assertEqual(s['status'],'passed');self.assertEqual(len(s['runs']),6)
        self.assertEqual(s['new_run_read_counts'],dict(train=60,validation=100,test=0))
        self.assertEqual(set(s['paired_checks']),{'42','43','44'})
        self.assertTrue(all(Path(p).read_bytes()==v for p,v in before.items()))
        self.assertTrue((self.f.root/'e5/aggregate.csv').exists())
        with self.assertRaises(FileExistsError):self.run_small()
    def test_missing_reference_fails_before_training(self):
        with self.assertRaises(FileNotFoundError):self.run_small()
        self.assertFalse((self.f.root/'e5/A43').exists())
        self.assertEqual(json.loads((self.f.root/'e5/e5_summary.json').read_text())['status'],'failed')
    def test_tampered_B_reference_rejected_before_training(self):
        self.refs();p=self.f.root/'B42/prediction_full/patient_predictions.csv';p.write_text(p.read_text()+'\n')
        with self.assertRaises(ValueError):self.run_small()
        self.assertFalse((self.f.root/'e5/A43').exists())
