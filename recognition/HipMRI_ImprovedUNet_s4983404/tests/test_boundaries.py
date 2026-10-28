"""Boundary conventions on precisely known synthetic masks."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyse_failures import boundary_errors


class BoundaryTests(unittest.TestCase):
    def test_known_distances_and_empty_masks(self) -> None:
        actual = np.zeros((10, 10), dtype=bool)
        predicted = actual.copy()
        self.assertEqual(boundary_errors(predicted, actual)['empty_case'], 'both_empty')
        actual[3, 3] = True
        self.assertIsNone(boundary_errors(predicted, actual)['hd95_pixels'])
        self.assertEqual(boundary_errors(predicted, actual)['dice'], 0)
        predicted[3, 6] = True
        result = boundary_errors(predicted, actual)
        self.assertEqual(result['hd95_pixels'], 3)
        self.assertEqual(result['fp_pixels'], 1)
        self.assertEqual(result['fn_pixels'], 1)
        self.assertEqual(boundary_errors(actual, actual)['hd95_pixels'], 0)
        self.assertEqual(boundary_errors(actual, actual)['dice'], 1)
        predicted[0, 0] = True
        self.assertEqual(boundary_errors(predicted, actual)['predicted_components_4connected'], 2)


if __name__ == '__main__':
    unittest.main()
