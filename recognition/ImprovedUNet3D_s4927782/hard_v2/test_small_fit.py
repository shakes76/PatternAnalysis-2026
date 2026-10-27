"""Preflight checks for the fixed TRAIN memorisation gate and confusion counts."""
import importlib.util
import unittest
import numpy as np

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import small_fit as fit


@unittest.skipUnless(HAS_TORCH,'Actual execution environment requires Torch')
class FitGateTests(unittest.TestCase):
    def test_all_three_final_conditions_are_required(self):
        qa=dict(minimum_fit_prostate_dice=.9,minimum_fit_foreground_dice=.9,maximum_final_initial_loss_ratio=.25)
        good=dict(prostate_dice=.95,foreground_dice=.96,loss=.2)
        self.assertTrue(fit.fit_passed(good,1.,qa))
        for name,value in [('prostate_dice',.89),('foreground_dice',.89),('loss',.26)]:
            self.assertFalse(fit.fit_passed(dict(good,**{name:value}),1.,qa))
        self.assertFalse(fit.fit_passed(dict(good,prostate_dice=None),1.,qa))

    def test_confusion_and_false_positive_absent_organ(self):
        gt=np.array([0,1,1,2,5,5]).reshape(1,2,3)
        pred=np.array([0,1,5,2,5,3]).reshape(1,2,3)
        result=fit.metrics(pred,gt)
        self.assertEqual(result['per_class'][5],dict(class_index=5,tp=1,fp=1,fn=1,dice=.5))
        self.assertEqual(result['per_class'][3]['dice'],0.)
        self.assertIsNone(result['per_class'][4]['dice'])
        self.assertAlmostEqual(result['foreground_dice'],(2/3+1+0+.5)/4)

    def test_bad_ground_truth_fails_before_confusion_reshape(self):
        with self.assertRaises(ValueError):
            fit.metrics(np.zeros((1,1,1),dtype=int),np.full((1,1,1),6,dtype=int))


if __name__=='__main__':unittest.main()
