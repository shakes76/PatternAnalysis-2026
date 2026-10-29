"""Small synthetic tests of leakage prevention and graph normalization inputs.

These are software checks, not performance evidence on the course dataset.
Run with: python -m unittest test_data_pipeline -v
"""

import unittest

import numpy as np

from dataset import (
    canonical_edges, standardize_features, stratified_split, validate_arrays,
    verify_split,
)


class DataPipelineTests(unittest.TestCase):
    def test_split_reproducibility_class_coverage_and_no_overlap(self):
        labels = np.repeat(np.arange(4), 25)
        splits = stratified_split(labels, 42)
        repeated = stratified_split(labels, 42)
        verify_split(splits, len(labels))
        for name, ids in splits.items():
            np.testing.assert_array_equal(ids, repeated[name])
            np.testing.assert_array_equal(np.unique(labels[ids]), np.arange(4))
        corrupted = {name: ids.copy() for name, ids in splits.items()}
        corrupted["test"][0] = corrupted["train"][0]
        with self.assertRaises(ValueError):
            verify_split(corrupted, len(labels))

    def test_scaling_does_not_fit_on_held_out_nodes(self):
        features = np.array([[1., 5.], [3., 5.], [10000., -9000.]], dtype=np.float32)
        _, mean, scale = standardize_features(features, np.array([0, 1]))
        np.testing.assert_allclose(mean, [2., 5.])
        np.testing.assert_allclose(scale, [1., 1.])

    def test_duplicate_self_loops_are_removed_before_gcn_normalization(self):
        edges = np.array([[0, 0, 1, 1, 2, 2], [0, 0, 0, 2, 1, 2]], dtype=np.int64)
        cleaned = canonical_edges(edges, 3)
        expected = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
        np.testing.assert_array_equal(cleaned, expected)

    def test_invalid_endpoints_and_nonfinite_features_are_rejected(self):
        features = np.ones((20, 128), dtype=np.float32)
        labels = np.repeat(np.arange(4), 5)
        edges = np.array([[0, 1], [1, 20]], dtype=np.int64)
        with self.assertRaises(ValueError):
            validate_arrays(features, labels, edges, expected_nodes=20)
        edges[1, 1] = 2
        features[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_arrays(features, labels, edges, expected_nodes=20)


if __name__ == "__main__":
    unittest.main()
