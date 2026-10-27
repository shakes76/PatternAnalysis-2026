import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_pipeline_check as check


class PipelineVerifierTests(unittest.TestCase):
    def test_fixed_first_visit_of_two_distinct_train_patients(self):
        manifest = {"records": [
            {"patient": "B006", "visit": "Week1", "split": "train"},
            {"patient": "B038", "visit": "Week0", "split": "train"},
            {"patient": "B006", "visit": "Week0", "split": "train"},
            {"patient": "A001", "visit": "Week0", "split": "test"},
        ]}
        actual = check.select_cases(manifest)
        self.assertEqual([(r["patient"], r["visit"]) for r in actual], list(check.CASE_KEYS))
        manifest["records"][2]["split"] = "test"
        with self.assertRaises(ValueError):
            check.select_cases(manifest)

    def test_independent_confusion_uses_target_rows_and_prediction_columns(self):
        target = np.array([0, 1, 1, 1, 2, 2], dtype=np.uint8)
        prediction = np.array([0, 1, 1, 2, 2, 0], dtype=np.uint8)
        metrics, confusion = check.confusion_metrics(prediction, target, 4)
        np.testing.assert_array_equal(confusion, [[1, 0, 0, 0], [0, 2, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
        self.assertEqual([r["dice"] for r in metrics], [2/3, 4/5, 1/2, None])
        self.assertEqual([r["iou"] for r in metrics], [1/2, 2/3, 1/3, None])

    def test_array_identity_includes_dtype_shape_and_c_order_values(self):
        value = np.arange(12, dtype=np.uint8).reshape(3, 4)
        header = b'{"dtype":"|u1","shape":[3,4]}'
        expected = hashlib.sha256(header + b"\0" + bytes(range(12))).hexdigest()
        self.assertEqual(check.array_digest(value), expected)
        self.assertEqual(check.array_digest(np.asfortranarray(value)), expected)
        self.assertNotEqual(check.array_digest(value.reshape(4, 3)), expected)
        self.assertNotEqual(check.array_digest(value.astype(np.int16)), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
