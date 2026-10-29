import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

class DiagnosticTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_r4_diagnostic'))
        import run_r4_diagnostic
        return run_r4_diagnostic
    def rows(self):
        return [dict(patient_id='p',image_id='1',label=1,slice_count=20,p_ad=.5,prediction=1)]
    def test_conditions_change_one_factor_and_bridge_mode(self):
        c=self.api().conditions()
        self.assertEqual([(r['batch_size'],r['tf32']) for r in c[:4]],[(16,True),(16,False),(4,True),(4,False)])
        self.assertTrue(all(r['mode']=='no_grad' for r in c[:4]));self.assertEqual(c[4]['mode'],'inference_mode')
        self.assertEqual(c[3]['batch_size'],c[4]['batch_size']);self.assertEqual(c[3]['tf32'],c[4]['tf32'])
    def test_original_tolerance_formula_not_loosened(self):
        a=self.api();ref=self.rows();actual=[dict(ref[0],p_ad=.5000055)]
        rows,s=a.differences(actual,ref,'patient')
        self.assertFalse(rows[0]['r4_probability_close']);self.assertTrue(rows[0]['r3_probability_close'])
        self.assertEqual(s['r4_exceed_count'],1);self.assertEqual(s['r3_exceed_count'],0)
    def test_class_change_is_recorded_not_abort(self):
        r=self.rows();rows,s=self.api().differences([dict(r[0],p_ad=.49,prediction=0)],r,'patient')
        self.assertEqual(s['class_change_count'],1);self.assertEqual(s['r4_exceed_count'],1)
    def test_wrong_identity_duplicate_missing_nonfinite_rejected(self):
        a=self.api();r=self.rows()
        for bad in ([dict(r[0],patient_id='wrong')],r*2,[],[dict(r[0],p_ad=math.nan)],[dict(r[0],label=0)]):
            with self.assertRaises(ValueError):a.differences(bad,r,'patient')
    def test_slice_differences_and_signed_error(self):
        r=[dict(self.rows()[0],file_path='/a')];v=[dict(r[0],p_ad=.6)]
        rows,s=self.api().differences(v,r,'slice');self.assertAlmostEqual(rows[0]['signed_difference'],.1)
        self.assertAlmostEqual(s['max_absolute_difference'],.1)
    def test_save_all_predictions_before_reporting_mismatch(self):
        a=self.api();r=self.rows();v=[dict(r[0],p_ad=.6)]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);s=a.save_comparison(p,'patients',v,r,'patient')
            self.assertEqual(s['r4_exceed_count'],1)
            self.assertTrue((p/'patients_actual.csv').exists());self.assertTrue((p/'patients_differences.csv').exists())
    def test_fixed_lock_and_scope(self):
        a=self.api();import json
        lock=json.loads(Path(a.__file__).with_name('r4_frozen_bests.json').read_text())
        r=a.select_entry(lock);self.assertEqual((r['model'],r['seed'],r['epoch']),('CNN-A',42,28))
        lock['runs'][0]['checkpoint_sha256']='0'*64
        with self.assertRaises(ValueError):a.select_entry(lock)

    def test_orchestration_collects_all_conditions_despite_probability_mismatch(self):
        a=self.api();from unittest.mock import patch
        lock=Path(a.__file__).with_name('r4_frozen_bests.json');seen=[]
        def child(cmd,**kwargs):
            i=int(cmd[-1]);seen.append(i);out=Path(cmd[cmd.index('--output-dir')+1]);out.mkdir()
            patients=[dict(self.rows()[0],p_ad=.5+i*.01)];slices=[dict(patients[0],file_path='/a')]
            a.write_csv(out/'pass1_patients_actual.csv',patients);a.write_csv(out/'pass1_slices_actual.csv',slices)
            a.write_json(out/'diagnostic_summary.json',dict(status='completed',environment={},code_sha256={},read_counts={'test':0},passes=[{'patients':{'r4_exceed_count':i,'class_change_count':0}}]))
        with tempfile.TemporaryDirectory() as d,patch.object(a,'verify_files'),patch.object(a.subprocess,'run',side_effect=child):
            r=a.run_all(Path(d),lock,Path(d)/'out')
            self.assertEqual(seen,list(range(5)));self.assertEqual(r['status'],'completed')
            self.assertTrue((Path(d)/'out/factor_comparisons.csv').exists())
            self.assertEqual(r['test_image_reads'],0)
    def test_worker_cpu_failure_keeps_summary_and_no_overwrite(self):
        a=self.api();from unittest.mock import patch
        import json
        with tempfile.TemporaryDirectory() as d,patch('torch.cuda.is_available',return_value=False):
            out=Path(d)/'out';lock=Path(a.__file__).with_name('r4_frozen_bests.json')
            with self.assertRaisesRegex(ValueError,'CUDA'):a.worker(Path(d),lock,0,out)
            self.assertEqual(json.loads((out/'diagnostic_summary.json').read_text())['status'],'failed')
            with self.assertRaises(FileExistsError):a.worker(Path(d),lock,0,out)
