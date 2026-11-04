"""Grad-CAM maps and the validation-only explanation CLI on synthetic data."""

import contextlib
import io
import json
import unittest

import torch

import explain_gradcam
from engine import training as train
from evaluation.gradcam import grad_cam, outside_share, target_layer
from models.cnn import SmallCNN
from models.convnext import ConvNeXtLite
import test_adni_splits as split_fixture


class GradCamTests(unittest.TestCase):
    def test_maps_are_normalized_and_explain_the_predicted_class(self):
        torch.manual_seed(0)
        for model in (SmallCNN(3), ConvNeXtLite(3)):
            for depth in ("last", "penultimate"):
                images = torch.randn(2, 3, 64, 64)
                maps, logits = grad_cam(model, images, target_layer(model, depth))
                self.assertEqual(tuple(maps.shape), (2, 64, 64))
                self.assertGreaterEqual(maps.min().item(), 0.0)
                self.assertLessEqual(maps.max().item(), 1.0 + 1e-6)
                self.assertTrue(torch.allclose(logits, model.eval()(images), atol=1e-5))

    def test_outside_share_counts_background_mass(self):
        maps = torch.zeros(1, 4, 4)
        maps[0, 0, 0], maps[0, 2, 2] = 1.0, 3.0
        head = torch.zeros(1, 4, 4, dtype=torch.bool)
        head[0, 1:, 1:] = True
        self.assertAlmostEqual(outside_share(maps, head).item(), 0.25)

    def test_cli_explains_validation_scans_only(self):
        fixture = split_fixture.ADNISplitTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.prepare()
        run = fixture.base / "run"
        args = train.build_parser().parse_args([
            "--data-root", str(fixture.root), "--splits-dir", str(fixture.out), "--output", str(run),
            "--model", "cnn", "--epochs", "1", "--warmup-epochs", "0", "--batch-size", "8", "--workers", "0",
            "--threads", "1", "--image-height", "32", "--image-width", "32", "--device", "cpu",
            "--skip-inference-profile"])
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(args)
            summary = explain_gradcam.run(explain_gradcam.build_parser().parse_args([
                "--checkpoint", str(run / "final.pt"), "--data-root", str(fixture.root),
                "--splits-dir", str(fixture.out), "--output", str(fixture.base / "cam"),
                "--batch-size", "4", "--workers", "0", "--threads", "1", "--device", "cpu"]))
        val_scans = {row["image_id"] for row in split_fixture.read_csv(fixture.out / "fold_01/val.csv")}
        explained = {row["image_id"] for row in split_fixture.read_csv(fixture.base / "cam/gradcam_scans.csv")}
        self.assertEqual(explained, val_scans)
        self.assertEqual(summary["role"], "val")
        for name in ("gradcam_cases.png", "gradcam_group_means.png"):
            self.assertTrue((fixture.base / "cam" / name).is_file())
        self.assertEqual(json.loads((fixture.base / "cam/gradcam_summary.json").read_text())["groups"].keys(),
                         {"AD unknown", "NC unknown"})


if __name__ == "__main__":
    unittest.main()
