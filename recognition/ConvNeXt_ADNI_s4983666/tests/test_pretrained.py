"""Loading the authors' ImageNet ConvNeXt-Tiny weights: name mapping, checks and a CPU run."""

import contextlib
import io
import unittest

import torch

from engine import prediction as predict
from engine import training as train
from models.convnext import ConvNeXtTiny, ConvNeXtTinyImageNet
from models.pretrained import load_imagenet_weights, official_name
import test_adni_splits as split_fixture


def fake_official_checkpoint(path) -> dict:
    """Write a file with the official key layout (random values, 1000-class head)."""
    torch.manual_seed(0)
    source = ConvNeXtTiny(3)
    state = {official_name(name): tensor.clone() for name, tensor in source.state_dict().items()
             if official_name(name) is not None}
    state["head.weight"], state["head.bias"] = torch.randn(1000, 768), torch.randn(1000)
    torch.save({"model": state}, path)
    return state


class PretrainedWeightTests(unittest.TestCase):
    def setUp(self):
        self.fixture = split_fixture.ADNISplitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.weights = self.fixture.base / "convnext_tiny_1k_224_ema.pth"
        self.official = fake_official_checkpoint(self.weights)
        torch.set_num_threads(1)

    def test_every_backbone_tensor_is_loaded_and_the_classifier_stays_random(self):
        model = ConvNeXtTinyImageNet()
        classifier = model.classifier.weight.clone()
        provenance = load_imagenet_weights(model, self.weights)
        state = model.state_dict()
        for name, tensor in state.items():
            source = official_name(name)
            if source is not None:
                self.assertTrue(torch.equal(tensor, self.official[source]), name)
        self.assertTrue(torch.equal(model.classifier.weight, classifier))
        self.assertEqual(provenance["loaded_tensors"], len(self.official) - 2)
        self.assertEqual(set(provenance["random_tensors"]),
                         {"classifier.weight", "classifier.bias", "input_normalization.mean", "input_normalization.std"})
        self.assertEqual(len(provenance["sha256"]), 64)
        # [-1, 1] input becomes ImageNet-normalized: mid-grey 0 maps to (0.5 - mean) / std.
        normalized = model.input_normalization(torch.zeros(1, 3, 1, 1)).flatten()
        self.assertAlmostEqual(normalized[0].item(), (0.5 - 0.485) / 0.229, places=5)

    def test_mismatched_or_incomplete_files_are_rejected(self):
        for change in ("shape", "missing", "extra"):
            state = dict(self.official)
            if change == "shape":
                state["stages.0.0.gamma"] = torch.zeros(5)
            elif change == "missing":
                del state["norm.weight"]
            else:
                state["stages.9.0.gamma"] = torch.zeros(96)
            path = self.fixture.base / f"{change}.pth"
            torch.save({"model": state}, path)
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_imagenet_weights(ConvNeXtTinyImageNet(), path)

    def test_training_records_the_weights_and_prediction_replays_them(self):
        self.fixture.prepare()
        output = self.fixture.base / "run"
        base = ["--data-root", str(self.fixture.root), "--splits-dir", str(self.fixture.out),
                "--epochs", "1", "--warmup-epochs", "0", "--batch-size", "8", "--workers", "0",
                "--threads", "1", "--image-height", "32", "--image-width", "32", "--device", "cpu",
                "--skip-inference-profile"]
        with self.assertRaisesRegex(ValueError, "pretrained-weights"):
            train.run(train.build_parser().parse_args(base + ["--output", str(output), "--model", "convnext_tiny_in1k"]))
        with self.assertRaisesRegex(ValueError, "pretrained-weights"):
            train.run(train.build_parser().parse_args(base + ["--output", str(output), "--model", "cnn",
                                                              "--pretrained-weights", str(self.weights)]))
        args = train.build_parser().parse_args(base + ["--output", str(output), "--model", "convnext_tiny_in1k",
                                                       "--pretrained-weights", str(self.weights)])
        with contextlib.redirect_stdout(io.StringIO()):
            result = train.run(args)
        saved = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
        self.assertEqual(saved["config"]["initialization"], "imagenet_1k")
        self.assertEqual(saved["config"]["pretrained_weights"]["file"], "convnext_tiny_1k_224_ema.pth")
        replay = predict.build_parser().parse_args([
            "--checkpoint", str(output / "final.pt"), "--data-root", str(self.fixture.root),
            "--splits-dir", str(self.fixture.out), "--output", str(self.fixture.base / "replay"),
            "--role", "val", "--batch-size", "8", "--workers", "0", "--threads", "1", "--device", "cpu",
            "--skip-inference-profile"])
        with contextlib.redirect_stdout(io.StringIO()):
            replayed = predict.run(replay)
        self.assertEqual(replayed["metrics"]["scan"]["accuracy"], result["metrics"]["scan"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
