import importlib.util
import unittest
import math
from test_patient_evaluation import fixture

class AuditTests(unittest.TestCase):
    def test_independent_decimal_scores(self):
        self.assertIsNotNone(importlib.util.find_spec('audit_patient_selection'),'R1 audit missing')
        from audit_patient_selection import independent_scores
        rows,_=fixture();score=independent_scores(rows)
        self.assertAlmostEqual(score['nll'],(-math.log(.8)-math.log(.7))/2,places=14)
        self.assertAlmostEqual(score['brier'],.065,places=14)
        self.assertEqual(score['confusion_matrix'],[[1,0],[0,1]])
        self.assertEqual(score['auroc'],1.)
        self.assertEqual(score['balanced_accuracy'],1.)
