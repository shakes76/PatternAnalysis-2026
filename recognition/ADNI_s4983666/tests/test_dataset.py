"""Exercise deterministic slice loading and mandatory manifest verification."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image, ImageOps
import torch

import adni_splits
from dataset import ADNISliceDataset, load_fold, manifest_sha256
import test_adni_splits as split_tests


class SliceDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="adni_dataset_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for index, value in enumerate((0, 255)):
            relative = f"slices/100_{index}.jpeg"
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (6, 4), color=value).save(path, quality=100)
            self.rows.append({"relative_path": relative, "label": "1",
                              "patient_id": "001_S_4000", "image_id": "100",
                              "slice_index": str(index),
                              "file_sha256": adni_splits.digest(path.read_bytes())})

    def test_fixed_normalization_shape_and_metadata(self):
        dataset = ADNISliceDataset(self.rows, self.root, image_size=(4, 6))
        self.assertEqual(len(dataset), 2)
        for index, expected_value in enumerate((-1.0, 1.0)):
            item = dataset[index]
            self.assertEqual(tuple(item["image"].shape), (1, 4, 6))
            self.assertEqual(item["image"].dtype, torch.float32)
            self.assertTrue(torch.equal(item["image"], torch.full((1, 4, 6), expected_value)))
            self.assertEqual(item["label"].dtype, torch.float32)
            self.assertEqual(item["label"].ndim, 0)
            self.assertEqual(item["label"].item(), 1.0)
            self.assertEqual(item["slice_index"], index)
            self.assertEqual(item["patient_id"], self.rows[index]["patient_id"])
            self.assertEqual(item["image_id"], "100")

    def test_rgb_grayscale_resize_and_determinism(self):
        path = self.root / self.rows[0]["relative_path"]
        image = Image.new("RGB", (6, 4))
        image.putdata([(i * 10, 255 - i * 7, i * 3) for i in range(24)])
        image.save(path, quality=95)
        self.rows[0]["file_sha256"] = adni_splits.digest(path.read_bytes())
        dataset = ADNISliceDataset(self.rows, self.root, image_size=(8, 9))
        first, second = dataset[0]["image"], dataset[0]["image"]
        self.assertEqual(tuple(first.shape), (1, 8, 9))
        self.assertTrue(torch.equal(first, second))
        self.assertGreaterEqual(first.min().item(), -1.0)
        self.assertLessEqual(first.max().item(), 1.0)
        with Image.open(path) as source:
            expected_image = ImageOps.exif_transpose(source).convert("L").resize(
                (9, 8), Image.Resampling.BILINEAR)
            expected = torch.tensor(list(expected_image.tobytes()), dtype=torch.float32)
            expected = ((expected / 255.0 - 0.5) / 0.5).reshape(1, 8, 9)
        self.assertTrue(torch.equal(first, expected))

    def test_input_rows_are_copied(self):
        dataset = ADNISliceDataset(self.rows, self.root)
        self.rows[0]["label"] = "0"
        self.assertEqual(dataset[0]["label"].item(), 1.0)

    def test_source_change_after_dataset_construction_is_rejected(self):
        dataset = ADNISliceDataset(self.rows, self.root)
        Image.new("L", (6, 4), color=127).save(self.root / self.rows[0]["relative_path"])
        with self.assertRaisesRegex(adni_splits.AuditError, "Source image changed"):
            dataset[0]

    def test_rejects_path_traversal_and_absolute_paths(self):
        for relative in ("../outside.jpeg", str(self.root / "slices/100_0.jpeg")):
            with self.subTest(relative=relative):
                row = dict(self.rows[0], relative_path=relative)
                with self.assertRaisesRegex(adni_splits.AuditError, "inside the data directory"):
                    ADNISliceDataset([row], self.root)

    def test_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory(prefix="adni_outside_") as outside:
            external = Path(outside) / "outside.jpeg"
            Image.new("L", (6, 4)).save(external)
            (self.root / "escape.jpeg").symlink_to(external)
            with self.assertRaisesRegex(adni_splits.AuditError, "outside the data directory"):
                ADNISliceDataset([dict(self.rows[0], relative_path="escape.jpeg")], self.root)

    def test_rejects_duplicates_and_inconsistent_scan_identities(self):
        mutations = (
            dict(self.rows[1], relative_path=self.rows[0]["relative_path"]),
            dict(self.rows[1], slice_index="0"),
            dict(self.rows[1], patient_id="002_S_5000"),
            dict(self.rows[1], label="0"),
        )
        for second in mutations:
            with self.subTest(second=second):
                with self.assertRaises(adni_splits.AuditError):
                    ADNISliceDataset([self.rows[0], second], self.root)

    def test_longitudinal_patient_can_have_different_scan_labels(self):
        rows = [self.rows[0], dict(self.rows[1], image_id="101", label="0")]
        dataset = ADNISliceDataset(rows, self.root)
        self.assertEqual([dataset[i]["label"].item() for i in range(2)], [1.0, 0.0])

    def test_rejects_empty_data_invalid_labels_and_dimensions(self):
        with self.assertRaises(adni_splits.AuditError):
            ADNISliceDataset([], self.root)
        with self.assertRaises(adni_splits.AuditError):
            ADNISliceDataset([dict(self.rows[0], label="2")], self.root)
        with self.assertRaises(adni_splits.AuditError):
            ADNISliceDataset(self.rows, self.root, image_size=(0, 256))


class FoldLoadingTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing synthetic fixture without inheriting its test
        # class, which would execute the full split suite a second time.
        self.fixture = split_tests.ADNISplitTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare()

    def load(self, fold=1):
        with contextlib.redirect_stdout(io.StringIO()):
            return load_fold(self.fixture.root, self.fixture.out, fold)

    def test_loader_runs_full_audit_once_and_excludes_holdouts(self):
        with mock.patch.object(adni_splits, "verify", wraps=adni_splits.verify) as verify:
            loaded = self.load()
        self.assertEqual(verify.call_count, 1)
        self.assertEqual(set(loaded), {"train", "early_stop", "val", "report", "manifest_sha256"})
        self.assertEqual(loaded["manifest_sha256"], manifest_sha256(self.fixture.out))
        roles = [loaded[role] for role in ("train", "early_stop", "val")]
        for rows in roles:
            self.assertTrue(rows)
            self.assertEqual({row["partition"] for row in rows}, {"development"})
        for index, rows in enumerate(roles):
            for other in roles[index + 1:]:
                self.assertFalse({row["patient_id"] for row in rows}
                                 & {row["patient_id"] for row in other})
        self.assertEqual(loaded["report"]["config"]["expected_slices"], 2)

    def test_corrupted_source_stops_loader(self):
        source = next(self.fixture.root.rglob("*.jpeg"))
        source.write_bytes(b"corrupted image")
        with self.assertRaises(adni_splits.AuditError):
            self.load()

    def test_invalid_fold_cannot_load_arbitrary_manifest(self):
        for fold in (0, 6, "../test", True):
            with self.subTest(fold=fold):
                with self.assertRaisesRegex(adni_splits.AuditError, "fold must be an integer"):
                    self.load(fold)

    def test_audit_failure_prevents_manifest_reading(self):
        with mock.patch.object(adni_splits, "verify", side_effect=adni_splits.AuditError("blocked")):
            with mock.patch.object(adni_splits, "read_csv") as read_csv:
                with self.assertRaisesRegex(adni_splits.AuditError, "blocked"):
                    self.load()
                read_csv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
