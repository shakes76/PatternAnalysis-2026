import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_boundaries import mean_defined, select_failures, summarize


def row(index, case, area, dice, fp, fn, boundary):
    values = {'prostate_dice': dice, 'false_positive_pixels': fp,
              'false_negative_pixels': fn, 'boundary_f1': [None]*4+[boundary],
              'on_bladder': 0, 'on_rectum': 0}
    return {'index': index, 'case_alias': case, 'prostate_truth_pixels': area,
            'truth_pixels': [1000,0,0,0,0,area], 'annotation_endpoint': False,
            'models': {'unet': copy.deepcopy(values), 'improved_unet': values}}


class BoundaryAnalysisTests(unittest.TestCase):
    def test_defined_zero_and_empty_are_distinct(self):
        self.assertEqual(mean_defined([None, 0., 1.]),
                         {'mean': .5, 'defined_slices': 2, 'undefined_slices': 1})
        self.assertIsNone(mean_defined([])['mean'])

    def test_same_truth_population_does_not_hide_absent_false_positives(self):
        rows = [row(0,'A',10,.5,1,2,.5), row(1,'A',0,0,4,0,0), row(2,'B',0,None,0,0,None)]
        rows[1]['models']['unet']['boundary_f1'][4] = None
        rows[1]['models']['unet']['false_positive_pixels'] = 0
        result = summarize(rows)
        self.assertEqual(result['unet']['per_class_boundary_f1'][4]['truth_positive_slices']['mean'], .5)
        self.assertEqual(result['improved_unet']['per_class_boundary_f1'][4]['truth_positive_slices']['mean'], .5)
        self.assertEqual(result['improved_unet']['per_class_boundary_f1'][4]['all_slices']['mean'], .25)
        self.assertEqual(result['improved_unet']['prostate_absent_with_false_positive'], 1)
        self.assertEqual(result['improved_unet']['prostate_absent_slices'], 2)

    def test_selection_covers_cases_and_distinct_error_extremes(self):
        rows = [row(0,'A',10,0,0,10,0), row(1,'B',20,.2,0,15,0),
                row(2,'C',30,.3,0,20,0), row(3,'A',200,.4,1,150,.3),
                row(4,'B',0,0,99,0,0), row(5,'A',300,.9,2,5,.9)]
        chosen = select_failures(rows)
        self.assertEqual([x[0]['index'] for x in chosen], [0,1,2,3,4])
        self.assertEqual(len({x[0]['index'] for x in chosen}),5)


if __name__ == '__main__':
    unittest.main()
