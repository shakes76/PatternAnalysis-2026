"""Synthetic alignment/metric checks; no real test cases or training."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
import modules as components_2d
from evaluate import (align_shared, native_dhw, make_reference_model,
                      load_reference_module, scope_summary, freeze_checkpoint, digest)
from support.evaluation_metrics import class_metrics, hd95


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shared_grid_matches_full_3d_nearest_with_both_Z_sizes(self):
        for depth in (128,144):
            x=np.arange(256,dtype=np.int16)[:,None,None]
            y=np.arange(256,dtype=np.int16)[None,:,None]
            z=np.arange(depth,dtype=np.int16)[None,None,:]
            volume=((x+3*y+z)%6).astype(np.uint8)
            slices=np.stack([volume[::-1,98+i,:] for i in range(60)])
            actual=align_shared(slices)
            dhw=torch.from_numpy(volume.transpose(2,0,1).copy())
            expected=F.interpolate(dhw[None,None].float(),size=(64,128,128),
                                   mode='nearest')[0,0,:,:,49:79].to(torch.uint8).numpy()
            np.testing.assert_array_equal(actual,expected)
            # This fixture detects flipping after nearest resize (parity shift).
            wrong=F.interpolate(torch.from_numpy(slices.transpose(2,1,0).copy())[None,None].float(),
                                size=(64,128,30),mode='nearest')[0,0].flip(1).numpy()
            self.assertTrue(np.any(actual!=wrong))

    def test_alignment_rejects_padding_and_missing_slices(self):
        with self.assertRaises(ValueError):
            native_dhw(np.zeros((59,256,128),np.uint8))
        with self.assertRaises(ValueError):
            native_dhw(np.zeros((60,256,140),np.uint8))

    def test_both_empty_undefined_and_single_empty_zero(self):
        target=np.zeros((3,4,5),dtype=np.uint8)
        self.assertIsNone(class_metrics(target,target)[5]['dice'])
        pred=target.copy();pred[0,0,0]=5
        self.assertEqual(class_metrics(pred,target)[5]['dice'],0.)

    def test_frozen_copy_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.pt';source.write_bytes(b'synthetic weights')
            target=Path(folder)/'frozen.pt'
            try:
                expected=digest(source)
                freeze_checkpoint(source,target,expected)
                self.assertEqual(digest(target),expected)
                freeze_checkpoint(source,target,expected)
                with self.assertRaisesRegex(ValueError,'source hash'):
                    freeze_checkpoint(source,target,'0'*64)
                target.chmod(0o644);target.write_bytes(b'different weights')
                with self.assertRaisesRegex(ValueError,'Frozen checkpoint'):
                    freeze_checkpoint(source,target,expected)
            finally:
                if target.exists():target.chmod(0o644)

    def test_aggregation_retains_scan_patient_and_pooled_definitions(self):
        rows=[]
        for case,patient,miss in [('a','P1',False),('b','P1',True),('c','P2',True)]:
            target=np.zeros((2,2,2),np.uint8);target[0,0,0]=5
            prediction=np.zeros_like(target) if miss else target.copy()
            for r in class_metrics(prediction,target):
                r.update(case=case,patient_id=patient,hd95_mm=None,hd95_status='prediction_empty',
                         prostate_fp_in_bladder=0,prostate_fp_in_rectum=0)
                rows.append(r)
        summary=scope_summary(rows)
        c=summary['class_metrics'][5]
        self.assertAlmostEqual(c['mean_case_dice'],1/3)
        self.assertAlmostEqual(c['mean_patient_dice'],.25)
        self.assertAlmostEqual(c['pooled_dice'],.5)
        self.assertEqual(summary['prostate_hd95_invalid_cases'],3)

    def test_reference_architecture_does_not_shadow_2d_modules(self):
        source=next(p/'UNet3D_s4984160' for p in PROJECT.parents
                    if (p/'UNet3D_s4984160/modules.py').is_file())
        ref=load_reference_module(source/'modules.py','test_3d_components')
        # BN affine parameters can initialize IN; discard only BN running stats.
        state={k:v for k,v in ref.UNet3D().state_dict().items()
               if 'running_' not in k and 'num_batches_tracked' not in k}
        checkpoint=dict(model_config=dict(base_channels=8,normalization='instance'),
                        model_state_dict=state)
        model=make_reference_model(source,checkpoint,device='cpu')
        with torch.inference_mode():
            self.assertEqual(tuple(model(torch.randn(1,1,16,32,32)).shape),(1,6,16,32,32))
        self.assertTrue(hasattr(components_2d,'UNet2D'))
        self.assertEqual(sum(p.numel() for p in model.parameters()),351206)

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy required for HD95')
    def test_hd95_physical_spacing_and_empty_status(self):
        target=np.zeros((9,9,9),bool);target[4,4,4]=True
        prediction=np.zeros_like(target);prediction[5,4,4]=True
        distance,status=hd95(prediction,target,(2.,1.,1.))
        self.assertEqual(status,'ok');self.assertEqual(distance,2.)
        self.assertEqual(hd95(np.zeros_like(target),target,(2.,1.,1.)),(None,'prediction_empty'))


def render_smoke(path):
    from predict import render
    shape=(64,128,128)
    image=np.random.default_rng(42).normal(size=shape).astype(np.float32)
    target=np.zeros(shape,np.uint8)
    target[20:40,55:75,55:70]=5;target[40:50,55:75,55:70]=3
    reference=target.copy()
    prediction=np.full(shape,255,np.uint8)
    prediction[:,:,49:79]=np.roll(target,2,axis=1)[:,:,49:79]
    meta=dict(case='SYNTHETIC_LAYOUT_CHECK',split='synthetic only',
              original_axcodes=['L','P','S'],native_2d_dice=.8,
              shared_2d_dice=.75,shared_3d_dice=1.)
    render(image,target,prediction,reference,meta,path)
    print(f'Synthetic rendering saved: {path}')


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--render-smoke':
        render_smoke(sys.argv[2])
    else:
        unittest.main()
