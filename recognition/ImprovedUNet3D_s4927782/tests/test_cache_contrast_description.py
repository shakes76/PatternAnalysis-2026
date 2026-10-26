"""Light checks for diagnostic scope and descriptor/identity semantics."""

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "tools/describe_cache_contrast.py"
SPEC = importlib.util.spec_from_file_location("cache_contrast", SOURCE)
contrast = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contrast)


class CacheContrastTests(unittest.TestCase):
    def test_fixed_scope_excludes_other_test_and_requires_acknowledgement(self):
        rows = [{"patient": f"T{i}", "visit": "Week0", "split": "train", "cache_path": f"volumes/T{i}.npz"} for i in range(143)]
        rows += [{"patient": f"V{i}", "visit": "Week0", "split": "validation", "cache_path": f"volumes/V{i}.npz"} for i in range(38)]
        rows += [{"patient": "M015", "visit": f"Week{i}", "split": "test", "cache_path": f"volumes/M015_{i}.npz"} for i in range(8)]
        rows.append({"patient": "OTHER", "visit": "Week0", "split": "test", "cache_path": "volumes/OTHER.npz"})
        selected = contrast.select_records({"records": rows})
        self.assertEqual(len(selected), 189)
        self.assertFalse(any(r["patient"] == "OTHER" for r in selected))
        with self.assertRaisesRegex(ValueError, "diagnostic-test-ack"):
            contrast.describe_archive("absent", "absent", "absent", "absent")
        rows[0]["cache_path"] = "../unsafe.npz"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            contrast.select_records({"records": rows})
        rows.pop(143)
        with self.assertRaisesRegex(ValueError, "fixed"):
            contrast.select_records({"records": rows})

    def test_descriptors_keep_zero_and_undefined_iqr_and_hash_dtype_shape(self):
        image = np.array([0, 0, -1, -1, 0, 0], dtype=np.float32)
        mask = np.array([1, 1, 3, 3, 5, 5], dtype=np.uint8)
        row = contrast.describe_case({"patient": "M015", "visit": "Week2", "split": "test"}, image, mask, np.ones(3, dtype=np.float32))
        self.assertEqual(row["contrasts"]["bladder_median_minus_body_median"], -1)
        self.assertIsNone(row["contrasts"]["bladder_body_iqr_standardized_median_gap"])
        self.assertEqual(row["class_intensity"]["1"]["exact_zero_fraction"], 1)
        self.assertEqual(row["class_intensity"]["0"], {"n": 0})
        header = {"dtype": image.dtype.str, "shape": list(image.shape)}
        expected = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\0" + image.tobytes()).hexdigest()
        self.assertEqual(contrast.array_identity(image), expected)
        self.assertNotEqual(contrast.array_identity(image), contrast.array_identity(image.reshape(2, 3)))


if __name__ == "__main__":
    unittest.main()
