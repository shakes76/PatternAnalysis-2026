"""Synthetic, non-patient tests for the Step 6 data pipeline."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import (ADNISliceDataset, COLUMNS, DataError, FINAL_STATUS, SliceRecord,
                     counts_for_records, image_to_tensor, load_finalized_splits,
                     make_loader, preprocessing_spec)
from check_data_pipeline import run_check

torch.set_num_threads(1)


class Fixture:
    """Six artificial subjects, one scan each, 20 slices per scan."""
    def __init__(self, base: Path) -> None:
        self.base = base
        self.root = base / "data"
        self.splits = base / "splits"
        self.root.mkdir()
        self.splits.mkdir()
        self.rows = {}
        self.summary = {
            "format_version": 1, "status": FINAL_STATUS,
            "mapping_structural_checks_passed": True,
            "patient_identifier_semantics_confirmed_by_user": True,
            "tutor_split_change_approved_as_reported_by_user": True,
            "zero_group_overlap_verified": True, "splits_finalized": True,
            "new_pairwise_candidate_subject_overlap": 0,
            "model_label_map": {"NC": 0, "AD": 1},
            "metadata_to_model_labels": {"0": 0, "2": 1},
            "evidence": {"images": 120, "scan_prefixes": 6, "candidate_subjects": 6},
            "manifest_sha256": {}, "proposed_or_final_counts": {},
        }
        for split_index, split in enumerate(("train", "validation", "test")):
            self.rows[split] = []
            for label, name in enumerate(("NC", "AD")):
                subject = f"000_S_{9000 + split_index * 2 + label}"
                scan = str(1000000 + split_index * 2 + label)
                # Deliberately train on original test-folder files; use the CSV!
                source = "test" if split == "train" else "train"
                for z in range(80, 100):
                    relative = f"AD_NC/{source}/{name}/{scan}_{z}.jpeg"
                    path = self.root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pixels = np.tile(np.arange(256, dtype=np.uint8), (240, 1))
                    pixels[80:160, 70:190] = (z + 30 * label) % 255
                    Image.fromarray(pixels).save(path)
                    self.rows[split].append(dict(
                        image_path=relative, subject_id=subject, scan_id=scan,
                        slice_index=z, source_split=source, source_class=name,
                        metadata_label=2 * label, label=label, split=split))
            rows = tuple(SliceRecord(**row) for row in self.rows[split])
            self.summary["proposed_or_final_counts"][split] = counts_for_records(rows)
            self.write_csv(split)
        self.write_summary()

    def write_csv(self, split: str, *, columns=COLUMNS) -> None:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.rows[split])
        content = stream.getvalue().encode()
        (self.splits / f"{split}.csv").write_bytes(content)
        self.summary["manifest_sha256"][f"{split}.csv"] = hashlib.sha256(content).hexdigest()
        self.write_summary()

    def write_summary(self) -> None:
        (self.splits / "split_summary.json").write_text(json.dumps(self.summary), encoding="utf-8")

    def bundle(self):
        return load_finalized_splits(self.root, self.splits)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(Path(self.temp.name))

    def test_valid_finalized_bundle(self):
        b = self.f.bundle()
        self.assertEqual(len(b.records["train"]), 40)
        self.assertEqual(len(b.records["test"]), 40)

    def test_uses_final_manifest_not_original_directory(self):
        ds = ADNISliceDataset(self.f.bundle(), "train")
        self.assertTrue(all(r.source_split == "test" for r in ds.records))
        self.assertEqual(ds[0]["split"], "train")

    def test_hash_change_rejected(self):
        path = self.f.splits / "train.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(DataError, "SHA-256"):
            self.f.bundle()

    def test_unfinalized_summary_rejected(self):
        self.f.summary["status"] = "DRAFT"
        self.f.write_summary()
        with self.assertRaisesRegex(DataError, "not been finalized"):
            self.f.bundle()

    def test_missing_approval_flag_rejected(self):
        del self.f.summary["tutor_split_change_approved_as_reported_by_user"]
        self.f.write_summary()
        with self.assertRaises(DataError):
            self.f.bundle()

    def test_model_label_mapping_rejected(self):
        self.f.summary["model_label_map"] = {"NC": 1, "AD": 0}
        self.f.write_summary()
        with self.assertRaisesRegex(DataError, "encoding"):
            self.f.bundle()

    def test_csv_columns_rejected(self):
        self.f.write_csv("train", columns=COLUMNS[:-1])
        with self.assertRaisesRegex(DataError, "columns"):
            self.f.bundle()

    def test_path_traversal_rejected(self):
        self.f.rows["train"][0]["image_path"] = "../outside.jpeg"
        self.f.write_csv("train")
        with self.assertRaisesRegex(DataError, "safe canonical"):
            self.f.bundle()

    def test_absolute_windows_path_rejected(self):
        self.f.rows["train"][0]["image_path"] = "C:/Users/private/image.jpeg"
        self.f.write_csv("train")
        with self.assertRaises(DataError):
            self.f.bundle()

    def test_metadata_label_mismatch_rejected(self):
        self.f.rows["train"][0]["metadata_label"] = 2
        self.f.write_csv("train")
        with self.assertRaisesRegex(DataError, "Metadata"):
            self.f.bundle()

    def test_wrong_row_split_rejected(self):
        self.f.rows["train"][0]["split"] = "test"
        self.f.write_csv("train")
        with self.assertRaisesRegex(DataError, "wrong final split"):
            self.f.bundle()

    def test_scan_slice_filename_mismatch_rejected(self):
        self.f.rows["train"][0]["slice_index"] = 5
        self.f.write_csv("train")
        with self.assertRaisesRegex(DataError, "scan/slice"):
            self.f.bundle()

    def test_subject_overlap_rejected(self):
        train_subject = self.f.rows["train"][0]["subject_id"]
        for row in self.f.rows["validation"][:20]:
            row["subject_id"] = train_subject
        self.f.write_csv("validation")
        with self.assertRaisesRegex(DataError, "subject spans"):
            self.f.bundle()

    def test_conflicting_subject_labels_rejected(self):
        subject = self.f.rows["train"][0]["subject_id"]
        for row in self.f.rows["train"][20:]:
            row["subject_id"] = subject
        self.f.write_csv("train")
        with self.assertRaisesRegex(DataError, "subject spans"):
            self.f.bundle()

    def test_duplicate_image_row_rejected(self):
        self.f.rows["train"][1] = dict(self.f.rows["train"][0])
        self.f.write_csv("train")
        with self.assertRaisesRegex(DataError, "repeated"):
            self.f.bundle()

    def test_summary_count_mismatch_rejected(self):
        self.f.summary["proposed_or_final_counts"]["train"]["images"] += 1
        self.f.write_summary()
        with self.assertRaisesRegex(DataError, "counts disagree"):
            self.f.bundle()

    def test_twenty_slices_per_scan_enforced(self):
        self.f.rows["train"].pop()
        self.f.summary["proposed_or_final_counts"]["train"] = counts_for_records(
            tuple(SliceRecord(**row) for row in self.f.rows["train"]))
        self.f.write_csv("train")
        with self.assertRaisesRegex(DataError, "20 distinct"):
            self.f.bundle()

    def test_missing_manifest_rejected(self):
        (self.f.splits / "test.csv").unlink()
        with self.assertRaisesRegex(DataError, "could not be read"):
            self.f.bundle()

    def test_duplicate_json_key_rejected(self):
        p = self.f.splits / "split_summary.json"
        p.write_text('{"status": "a", "status": "b"}')
        with self.assertRaisesRegex(DataError, "duplicate JSON"):
            self.f.bundle()

    def test_tensor_shape_dtype_and_range(self):
        item = ADNISliceDataset(self.f.bundle(), "train")[0]
        self.assertEqual(item["image"].shape, (1, 224, 224))
        self.assertEqual(item["image"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(item["image"]).all())
        self.assertGreaterEqual(item["image"].min().item(), -1)
        self.assertLessEqual(item["image"].max().item(), 1)

    def test_padding_and_fixed_scaling(self):
        tensor = image_to_tensor(Image.new("L", (256, 240), 255))[0]
        self.assertTrue(torch.equal(tensor[:7], torch.full((7, 224), -1.0)))
        self.assertTrue(torch.equal(tensor[7:-7], torch.ones(210, 224)))
        self.assertTrue(torch.equal(tensor[-7:], torch.full((7, 224), -1.0)))

    def test_source_mode_and_size_rejected(self):
        for image in [Image.new("RGB", (256, 240)), Image.new("L", (224, 224))]:
            with self.assertRaises(DataError):
                image_to_tensor(image)

    def test_corrupt_image_rejected(self):
        ds = ADNISliceDataset(self.f.bundle(), "train")
        ds.source_path(0).write_bytes(b"not an image")
        with self.assertRaisesRegex(DataError, "could not be loaded"):
            ds[0]

    def test_returned_metadata_preserves_aggregation_keys(self):
        ds = ADNISliceDataset(self.f.bundle(), "train")
        item, row = ds[0], ds.records[0]
        self.assertEqual(item["subject_id"], row.subject_id)
        self.assertEqual(item["scan_id"], row.scan_id)
        self.assertEqual(item["slice_index"], row.slice_index)
        self.assertEqual(item["label"], row.label)

    def test_test_pixels_blocked_by_default(self):
        with self.assertRaisesRegex(DataError, "test image access"):
            ADNISliceDataset(self.f.bundle(), "test")

    def test_test_pixels_can_be_explicitly_enabled_later(self):
        ds = ADNISliceDataset(self.f.bundle(), "test", allow_test=True)
        self.assertEqual(ds[0]["image"].shape, (1, 224, 224))

    def test_validation_augmentation_blocked(self):
        with self.assertRaisesRegex(DataError, "augmentation"):
            ADNISliceDataset(self.f.bundle(), "validation", rotation_degrees=5)

    def test_train_optional_rotation_is_finite(self):
        ds = ADNISliceDataset(self.f.bundle(), "train", rotation_degrees=5)
        self.assertTrue(torch.isfinite(ds[0]["image"]).all())

    def test_evaluation_rotation_and_shuffle_blocked(self):
        with self.assertRaises(DataError):
            ADNISliceDataset(self.f.bundle(), "test", allow_test=True, rotation_degrees=1)
        val = ADNISliceDataset(self.f.bundle(), "validation")
        with self.assertRaises(DataError):
            make_loader(val, shuffle=True)

    def test_validation_repeat_identical(self):
        ds = ADNISliceDataset(self.f.bundle(), "validation")
        self.assertTrue(torch.equal(ds[0]["image"], ds[0]["image"]))

    def test_loader_keeps_last_partial_batch(self):
        ds = ADNISliceDataset(self.f.bundle(), "train")
        batches = list(make_loader(ds, batch_size=16))
        self.assertEqual([len(b["label"]) for b in batches], [16, 16, 8])
        self.assertEqual(batches[0]["label"].dtype, torch.int64)

    def test_same_seed_first_batch(self):
        ds = ADNISliceDataset(self.f.bundle(), "train")
        a = next(iter(make_loader(ds, seed=42)))
        b = next(iter(make_loader(ds, seed=42)))
        self.assertEqual(a["image_path"], b["image_path"])
        self.assertTrue(torch.equal(a["image"], b["image"]))

    def test_default_preprocessing_has_no_augmentation(self):
        self.assertEqual(preprocessing_spec()["training_rotation_degrees"], 0)
        self.assertFalse(preprocessing_spec()["fitted_dataset_statistics"])

    def test_complete_check_never_opens_final_test_images(self):
        # Removing all final-test images must NOT prevent the loader check.
        for row in self.f.rows["test"]:
            (self.f.root / row["image_path"]).unlink()
        out = self.f.base / "check"
        report = run_check(self.f.root, self.f.splits, out)
        self.assertTrue(report["data_pipeline_check_passed"])
        self.assertEqual(report["final_test_images_decoded"], 0)
        self.assertEqual(report["train_loader"]["images_checked"], 40)
        self.assertTrue((out / "PRIVATE_training_preview.png").is_file())
        self.assertTrue((out / "README_data_pipeline_section.md").is_file())
        text = (out / "data_pipeline_summary.json").read_text()
        for row in self.f.rows["train"]:
            self.assertNotIn(row["subject_id"], text)
            self.assertNotIn(row["image_path"], text)

    def test_failed_check_writes_false_status(self):
        path = self.f.splits / "train.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        out = self.f.base / "failed"
        with self.assertRaises(DataError):
            run_check(self.f.root, self.f.splits, out)
        report = json.loads((out / "data_pipeline_summary.json").read_text())
        self.assertFalse(report["data_pipeline_check_passed"])
        self.assertTrue(report["errors"])

    def test_existing_output_not_overwritten(self):
        out = self.f.base / "existing"
        out.mkdir()
        with self.assertRaisesRegex(DataError, "not overwritten"):
            run_check(self.f.root, self.f.splits, out)

    def test_output_inside_source_rejected(self):
        with self.assertRaises(DataError):
            run_check(self.f.root, self.f.splits, self.f.root / "output")

    def test_cli_smoke(self):
        out = self.f.base / "cli"
        script = Path(__file__).resolve().parents[1] / "check_data_pipeline.py"
        result = subprocess.run([sys.executable, str(script), "--data-root", str(self.f.root),
                                 "--splits-dir", str(self.f.splits), "--output-dir", str(out)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("DATA_PIPELINE_CHECK_PASSED", result.stdout)


if __name__ == "__main__":
    unittest.main()
