"""Leakage checks must reject invalid metadata before any model is trained."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from runtime import select_device
from split_utils import define_splits, load_splits, validate_splits


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.frames = {
            name: pd.DataFrame({"image_name": [f"{name}_0", f"{name}_1", f"{name}_2"],
                                "patient_id": [f"p_{name}_0", f"p_{name}_1", f"p_{name}_2"],
                                "lesion_id": [f"l_{name}_0", f"l_{name}_1", f"l_{name}_2"],
                                "target": [0, 1, 0]})
            for name in ("train", "val", "test")
        }
        self.duplicates = pd.DataFrame({"image_name_1": ["train_0"], "image_name_2": ["train_2"]})

    def test_valid_audit_preserves_order_and_normalizes_ids(self):
        self.frames["train"].loc[0, "image_name"] = " train_0.jpg "
        result = validate_splits(self.frames, self.duplicates)
        self.assertEqual(result["train"].image_name.tolist(), ["train_0", "train_1", "train_2"])
        self.assertEqual(self.frames["train"].image_name.iloc[0], " train_0.jpg ")

    def test_all_overlap_types_and_partition_pairs(self):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            for column in ("image_name", "patient_id", "lesion_id"):
                with self.subTest(a=a, b=b, column=column):
                    frames = {k: v.copy() for k, v in self.frames.items()}
                    frames[b].loc[0, column] = frames[a].loc[0, column]
                    with self.assertRaisesRegex(ValueError, f"{column} overlap"):
                        validate_splits(frames, self.duplicates)

    def test_invalid_metadata(self):
        mutations = [
            ("image_name", 1, "train_0.jpg"),
            ("patient_id", 0, None),
            ("lesion_id", 0, " "),
            ("target", 0, 2),
            ("lesion_id", 1, "l_train_0"),
        ]
        for column, index, value in mutations:
            with self.subTest(column=column, value=value):
                frames = {k: v.copy() for k, v in self.frames.items()}
                frames["train"].loc[index, column] = value
                with self.assertRaises(ValueError):
                    validate_splits(frames, self.duplicates)
        for column in ("patient_id", "lesion_id", "target", "image_name"):
            frames = {k: v.copy() for k, v in self.frames.items()}
            frames["test"] = frames["test"].drop(columns=column)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                validate_splits(frames, self.duplicates)

    def test_duplicate_metadata_cannot_be_skipped_or_cross_partitions(self):
        for endpoint, message in (("test_0", "crosses splits"), ("unknown", "absent"),
                                  ("train_1", "conflicting labels"), (None, "missing image")):
            duplicates = self.duplicates.copy()
            duplicates.loc[0, "image_name_2"] = endpoint
            with self.assertRaisesRegex(ValueError, message):
                validate_splits(self.frames, duplicates)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, frame in self.frames.items():
                frame.to_csv(root / f"{name}.csv", index=False)
            with self.assertRaises(FileNotFoundError):
                load_splits(root, root / "absent.csv")
            self.duplicates.to_csv(root / "duplicates.csv", index=False)
            self.assertEqual(set(load_splits(root, root / "duplicates.csv")), set(self.frames))

    def test_split_definition_is_deterministic_and_covers_cohort(self):
        metadata = pd.DataFrame([
            {"image_name": f"img_{patient}_{label}", "patient_id": f"p{patient:03}",
             "lesion_id": f"l{patient}_{label}", "target": label}
            for patient in range(50) for label in (0, 1)
        ])
        frames, definition = define_splits(metadata)
        repeated, repeated_definition = define_splits(metadata)
        self.assertEqual(definition, repeated_definition)
        self.assertEqual(sum(map(len, frames.values())), len(metadata))
        empty_duplicates = pd.DataFrame(columns=["image_name_1", "image_name_2"])
        validate_splits(frames, empty_duplicates)
        for name in frames:
            pd.testing.assert_frame_equal(frames[name], repeated[name])

    def test_cuda_default_never_silently_falls_back(self):
        with patch("runtime.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                select_device("cuda")
            self.assertEqual(select_device("cpu").type, "cpu")


if __name__ == "__main__":
    unittest.main()
