"""Verify scan normalization/cropping, provenance, role isolation and checkpoint replay."""

import argparse
import contextlib
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image
import torch

import audit_preprocessing
from adni_data.preprocessing import (PreprocessingConfig, apply_preprocessing,
                                   checkpoint_preprocessing, crop_box, fit_training_crop, prepare_scans)
from adni_data.slices import ADNISliceDataset
from engine import prediction, training
from utils.preprocessing_artifacts import export_preprocessing_audit
import test_adni_splits as split_fixture


class ScanPreprocessingTests(unittest.TestCase):
    """Use lossless asymmetric inputs to detect changes to content and scan grouping."""

    def setUp(self) -> None:
        """Create complete synthetic scans with known positions and brightness differences."""
        self.temp = tempfile.TemporaryDirectory(prefix="adni_scan_preprocessing_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for scan, shift, offset in (("100", 0, 0), ("101", 9, 40)):
            for index in range(2):
                image = Image.new("L", (80, 60), 0)
                for y in range(20, 32):
                    for x in range(22 + index * 6, 40 + index * 6):
                        image.putpixel((x + shift, y), 40 + offset + 20 * ((x + y) % 3))
                path = self.root / f"{scan}_{index}.png"
                image.save(path)
                self.rows.append({"relative_path": path.name, "image_id": scan,
                                  "patient_id": f"patient_{scan}", "slice_index": index, "label": 0,
                                  "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.config = PreprocessingConfig("scan_intensity_crop", crop_margin=2,
                                          crop_height=32, crop_width=32)

    def test_scan_brightness_and_translation_invariance_without_resizing(self) -> None:
        """Matched brightness/position variants retain identical standardized pixels."""
        scans = prepare_scans(self.rows, self.root, self.config)
        dataset = ADNISliceDataset(self.rows, self.root, (32, 32), preprocessing=self.config,
                                   scan_parameters=scans)
        self.assertTrue(torch.equal(dataset[0]["image"], dataset[2]["image"]))
        self.assertTrue(torch.equal(dataset[1]["image"], dataset[3]["image"]))
        # Two slices of one scan share a box, so their genuine relative positions remain.
        self.assertFalse(torch.equal(dataset[0]["image"], dataset[1]["image"]))
        self.assertEqual(scans["100"]["intensity_low"], 40)
        self.assertEqual(scans["101"]["intensity_low"], 80)
        original_bytes = {row["relative_path"]: (self.root / row["relative_path"]).read_bytes() for row in self.rows}
        export_preprocessing_audit(self.root, dataset, "synthetic", max_images=2)
        self.assertEqual(original_bytes, {r["relative_path"]: (self.root / r["relative_path"]).read_bytes() for r in self.rows})
        self.assertTrue((self.root / "synthetic_preprocessing_preview.png").is_file())

    def test_geometry_fit_uses_training_only_and_rejects_clipping(self) -> None:
        """Held-out extents cannot silently enlarge a trained model's input shape."""
        config = replace(self.config, crop_height=0, crop_width=0)
        scans = prepare_scans(self.rows, self.root, config)
        fitted, record = fit_training_crop(scans, config, role="train")
        self.assertEqual((fitted.crop_height, fitted.crop_width), (32, 32))
        self.assertEqual(record["fit_role"], "train")
        with self.assertRaisesRegex(ValueError, "training scans only"):
            fit_training_crop(scans, config, role="early_stop")
        with self.assertRaisesRegex(ValueError, "requires at least"):
            fit_training_crop(scans, replace(config, crop_height=16, crop_width=16), role="train")
        wide = dict(scans["100"], bbox=[0, 0, 79, 59])
        with self.assertRaisesRegex(ValueError, "does not fit"):
            crop_box(wide, fitted)

    def test_failure_subset_uses_full_scan_statistics_and_source_bindings(self) -> None:
        """Exporting a single failure must not refit its normalization or centering."""
        scans = prepare_scans(self.rows, self.root, self.config)
        full = ADNISliceDataset(self.rows, self.root, (32, 32), preprocessing=self.config, scan_parameters=scans)
        subset = ADNISliceDataset([self.rows[1]], self.root, (32, 32), preprocessing=self.config, scan_parameters=scans)
        self.assertTrue(torch.equal(full[1]["image"], subset[0]["image"]))
        with self.assertRaisesRegex(ValueError, "different settings"):
            ADNISliceDataset(self.rows, self.root, (32, 32),
                             preprocessing=replace(self.config, foreground_threshold=20), scan_parameters=scans)
        with self.assertRaisesRegex(ValueError, "source identities"):
            ADNISliceDataset([dict(self.rows[1], patient_id="wrong")], self.root, (32, 32),
                             preprocessing=self.config, scan_parameters=scans)
        Image.new("L", (80, 60)).save(self.root / self.rows[0]["relative_path"])
        with self.assertRaisesRegex(ValueError, "Source image changed"):
            prepare_scans(self.rows, self.root, self.config)

    def test_crop_only_preserves_native_values_and_black_padding(self) -> None:
        """Integer crop/pad keeps foreground intensities and dimensions exactly."""
        config = replace(self.config, name="scan_crop", crop_height=96, crop_width=96)
        scans = prepare_scans(self.rows, self.root, config)
        image = Image.open(self.root / self.rows[0]["relative_path"]).copy()
        result = apply_preprocessing(image, scans["100"], config, (96, 96))
        self.assertEqual(result.size, (96, 96))
        for value in (40, 60, 80):
            self.assertEqual(result.histogram()[value], image.histogram()[value])
        self.assertEqual(result.getpixel((0, 0)), 0)

    def test_empty_and_constant_foregrounds_are_finite_and_flagged(self) -> None:
        """Degenerate scans use declared identity behavior rather than division by zero."""
        for value, status in ((0, "empty_foreground_identity"), (90, "constant_foreground_identity")):
            path = self.root / f"flat_{value}.png"
            Image.new("L", (12, 10), value).save(path)
            row = {"relative_path": path.name, "image_id": str(value), "patient_id": "flat",
                   "slice_index": 0, "label": 1}
            config = replace(self.config, crop_height=32, crop_width=32)
            scans = prepare_scans([row], self.root, config)
            self.assertEqual(scans[str(value)]["intensity_status"], status)
            image = ADNISliceDataset([row], self.root, (32, 32), preprocessing=config)[0]["image"]
            self.assertTrue(torch.isfinite(image).all())

    def test_parameters_do_not_use_labels_and_are_order_independent(self) -> None:
        """Input preprocessing is independent of diagnosis and row ordering."""
        original = prepare_scans(self.rows, self.root, self.config)
        changed = prepare_scans([dict(row, label=1) for row in reversed(self.rows)], self.root, self.config)
        self.assertEqual(original, changed)

    def test_strict_checkpoint_contract_and_configuration_validation(self) -> None:
        """Malformed/new algorithms fail before loading any evaluation data."""
        config = {"image_size": [32, 32], "preprocessing_config": self.config.to_dict()}
        self.assertEqual(checkpoint_preprocessing(config), self.config)
        for field, value in (("algorithm", "changed"), ("foreground_threshold", 255),
                             ("lower_percentile", float("nan")), ("crop_height", True)):
            bad = dict(config, preprocessing_config={**self.config.to_dict(), field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                checkpoint_preprocessing(bad)
        with self.assertRaisesRegex(ValueError, "disagree"):
            checkpoint_preprocessing(dict(config, image_size=[64, 64]))
        with self.assertRaises(ValueError):
            PreprocessingConfig("scan_crop", crop_height=32)


class PreprocessedTrainingTests(unittest.TestCase):
    """Exercise the existing audited pipeline and real CPU checkpoint inference."""

    def setUp(self) -> None:
        """Build disjoint synthetic patient manifests without loading real images."""
        self.fixture = split_fixture.ADNISplitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare()
        self.args = training.build_parser().parse_args([
            "--data-root", str(self.fixture.root), "--splits-dir", str(self.fixture.out),
            "--output", str(self.fixture.base / "preprocessed_run"), "--model", "cnn", "--epochs", "1",
            "--warmup-epochs", "0", "--batch-size", "16", "--workers", "0", "--threads", "1",
            "--image-height", "32", "--image-width", "32", "--device", "cpu",
            "--preprocessing", "scan_intensity_crop", "--crop-margin", "2", "--skip-inference-profile"])

    def test_train_predict_and_failure_export_use_identical_preprocessing(self) -> None:
        """Training only builds train and val loaders (never holdouts); replay is exact."""
        real_loader = training.make_loader
        seen = []

        def observed_loader(rows: list[dict], *args: object, **kwargs: object) -> object:
            """Check deterministic preprocessing is enabled for both development roles."""
            seen.append(kwargs["role"])
            self.assertIn(kwargs["role"], ("train", "val"))
            self.assertEqual(kwargs["preprocessing"].name, "scan_intensity_crop")
            return real_loader(rows, *args, **kwargs)

        with mock.patch("engine.training.make_loader", side_effect=observed_loader), contextlib.redirect_stdout(io.StringIO()):
            result = training.run(self.args)
        self.assertEqual(seen, ["train", "val"])
        saved = torch.load(self.args.output / "final.pt", map_location="cpu", weights_only=True)
        self.assertEqual(saved["config"]["checkpoint_format_version"], 6)
        self.assertEqual(saved["config"]["preprocessing_crop_fit"]["fit_role"], "train")
        self.assertEqual(saved["config"]["image_size"], [32, 32])
        self.assertEqual(result["preprocessing_audits"]["val"]["status"], "exported")
        # Failure grids retain all source slices per scan even if only one was wrong.
        import utils.evaluation_artifacts as failure_module
        actual_dataset = failure_module.ADNISliceDataset
        prediction_args = prediction.build_parser().parse_args([
            "--checkpoint", str(self.args.output / "final.pt"), "--data-root", str(self.fixture.root),
            "--splits-dir", str(self.fixture.out), "--output", str(self.fixture.base / "prediction"),
            "--role", "val", "--batch-size", "16", "--workers", "0", "--threads", "1",
            "--device", "cpu", "--skip-inference-profile"])
        with mock.patch.object(failure_module, "ADNISliceDataset", wraps=actual_dataset) as exported, \
                contextlib.redirect_stdout(io.StringIO()):
            prediction.run(prediction_args)
        reproduced = json.loads((prediction_args.output / "metrics.json").read_text())
        for unit in ("slice", "scan", "patient"):
            self.assertEqual(result["metrics"][unit], reproduced["metrics"][unit])
        self.assertEqual((self.args.output / "val_slice_predictions.csv").read_bytes(),
                         (prediction_args.output / "slice_predictions.csv").read_bytes())
        if exported.call_args:
            params = exported.call_args.kwargs["scan_parameters"]
            self.assertTrue(all(len(scan["source_files"]) == 2 for scan in params.values()))
        broken = {**saved, "config": {**saved["config"], "preprocessing_config": {
            **saved["config"]["preprocessing_config"], "algorithm": "unknown"}}}
        bad_path = self.fixture.base / "bad.pt"
        torch.save(broken, bad_path)
        prediction_args.checkpoint, prediction_args.output = [bad_path], self.fixture.base / "rejected"
        with mock.patch("engine.prediction.load_fold") as load_fold, self.assertRaises(ValueError), \
                contextlib.redirect_stdout(io.StringIO()):
            prediction.run(prediction_args)
        load_fold.assert_not_called()
        self.assertFalse(prediction_args.output.exists())

    def test_audit_has_no_model_and_only_exports_development_inputs(self) -> None:
        """The review entry point freezes dimensions without training or scoring."""
        args = argparse.Namespace(**{**vars(self.args), "output": self.fixture.base / "input_audit", "max_images": 2})
        with contextlib.redirect_stdout(io.StringIO()):
            report = audit_preprocessing.run(args)
        self.assertEqual(set(report["roles"]), {"train", "val"})
        self.assertEqual(report["image_size"], [32, 32])
        self.assertFalse((args.output / "final.pt").exists())
        self.assertFalse((args.output / "metrics.json").exists())
        self.assertTrue((args.output / "train_preprocessing_preview.png").exists())


if __name__ == "__main__":
    unittest.main()
