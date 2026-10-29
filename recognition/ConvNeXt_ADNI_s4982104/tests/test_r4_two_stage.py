import importlib.util
import unittest
from pathlib import Path
import tempfile

class TwoStageTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('r4_prediction_audit'))
        import r4_prediction_audit
        return r4_prediction_audit
    def test_historical_settings_use_recorded_cnn_flags(self):
        a=self.api();t=dict(config=dict(model='SimpleCNN',batch_size=16),backend_flags=dict(cudnn_allow_tf32=True,matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True))
        s=a.historical_settings(t);self.assertEqual(s['batch_size'],16);self.assertTrue(s['cudnn_allow_tf32'])
        t['backend_flags'].pop('cudnn_allow_tf32')
        with self.assertRaises(ValueError):a.historical_settings(t)
    def test_convnext_historical_settings(self):
        s=self.api().historical_settings({'config':{'model':'convnext_tiny','batch_size':4}})
        self.assertFalse(s['cudnn_allow_tf32']);self.assertEqual(s['mode'],'no_grad')
    def test_drift_saved_and_not_accepted_as_history(self):
        a=self.api();r=[dict(patient_id='p',image_id='1',label=1,slice_count=20,p_ad=.5,prediction=1)]
        v=[dict(r[0],p_ad=.6)]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);s=a.record_predictions(p,'unified',[dict(v[0],file_path='/a')],v,[dict(r[0],file_path='/a')],r)
            self.assertEqual(s['patient_drift']['r4_exceed_count'],1)
            self.assertTrue((p/'unified_patient_predictions.csv').exists())
    def test_unchanged_math_isclose_boundary_and_class_flips(self):
        a=self.api();r=[dict(patient_id='p',image_id='1',label=1,slice_count=20,p_ad=.5,prediction=1)]
        rows,s=a.differences([dict(r[0],p_ad=.5000055)],r,'patient');self.assertEqual(s['r4_exceed_count'],1)
        rows,s=a.differences([dict(r[0],p_ad=.49,prediction=0)],r,'patient');self.assertEqual(s['class_change_count'],1)

    def test_historical_replay_fails_but_saves_full_predictions(self):
        import run_r4_efficiency as r
        import torch
        from torch import nn
        from unittest.mock import patch
        rows=[dict(file_path=str(i),patient_id=str(i//20),image_id=str(i//20),label=i//20,p_ad=.6,prediction=1) for i in range(40)]
        expected=[dict(x,p_ad=.7) for x in rows];patients=r.aggregate_patients(expected)
        self.assertTrue(hasattr(r,'replay_history'),'historical replay helper missing')
        with tempfile.TemporaryDirectory() as d,patch.object(r,'predict_pass',return_value=rows):
            with self.assertRaisesRegex(ValueError,'probability'):
                r.replay_history(nn.Identity(),None,'cpu',Path(d),expected,patients,{})
            self.assertTrue((Path(d)/'historical_replay_slice_predictions.csv').exists())
            self.assertTrue((Path(d)/'historical_replay_patient_differences.csv').exists())
    def test_unified_reference_is_distinct_from_history_and_repeat_is_strict(self):
        import run_r4_efficiency as r
        self.assertTrue(hasattr(r,'check_repeat'),'unified repeat helper missing')
        slices=[dict(file_path=str(i),patient_id='p',image_id='1',label=1,p_ad=.6,prediction=1) for i in range(20)]
        patients=r.aggregate_patients(slices)
        with tempfile.TemporaryDirectory() as d:
            r.check_repeat(Path(d),1,slices,patients,slices,patients,{})
            bad=[dict(x,p_ad=.61) for x in slices]
            with self.assertRaisesRegex(ValueError,'probability'):
                r.check_repeat(Path(d),2,bad,r.aggregate_patients(bad),slices,patients,{})
            self.assertTrue((Path(d)/'unified_repeat_2_patient_predictions.csv').exists())
