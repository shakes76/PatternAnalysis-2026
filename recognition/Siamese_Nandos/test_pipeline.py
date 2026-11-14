"""CPU regression tests; run with python -m unittest discover -s recognition/Siamese_Nandos."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from dataset import ImageCache, ISICImageDataset, PairDataset, index_images, make_transform
from modules import ResNet18Baseline, SiameseNetWithClassifier, contrastive_loss
from predict import choose_threshold, load_model


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.splits = self.root / "splits"
        self.splits.mkdir()
        rng = np.random.default_rng(3710)
        for split, count in (("train", 8), ("val", 4), ("test", 4)):
            rows = []
            for i in range(count):
                name = f"{split}_{i}"
                Image.fromarray(rng.integers(0, 256, (39, 45, 3), dtype=np.uint8)).save(self.images / f"{name}.jpg")
                rows.append({"image_name": f" {name}.jpg ", "target": i % 2, "patient_id": name, "lesion_id": name})
            pd.DataFrame(rows).to_csv(self.splits / f"{split}.csv", index=False)
        pd.DataFrame(columns=["image_name_1", "image_name_2"]).to_csv(
            self.root / "ISIC_2020_Training_Duplicates.csv", index=False)
        self.frame = pd.read_csv(self.splits / "train.csv")
        self.paths = index_images(self.images)

    def run_script(self, script, *args):
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name(script)), *map(str, args)],
            capture_output=True, text=True, env=env, timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cache_matches_original_preprocessing(self):
        cache = ImageCache(self.frame.image_name, self.paths, 32)
        direct = ISICImageDataset(self.frame, self.paths, make_transform(32, train=True))
        cached = ISICImageDataset(self.frame, self.paths, make_transform(32, train=True, resize=False), cache)
        for i in range(len(direct)):
            torch.manual_seed(i)
            expected = direct[i]
            torch.manual_seed(i)
            actual = cached[i]
            self.assertTrue(torch.equal(expected[0], actual[0]))
            self.assertEqual(expected[1:], actual[1:])

    def test_models_and_checkpoint_compatibility(self):
        torch.manual_seed(7)
        model = ResNet18Baseline().eval()
        reference = resnet18(weights=None)
        reference.fc = torch.nn.Linear(512, 1)
        reference.load_state_dict(model.state_dict(), strict=True)
        reference.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), reference(x).squeeze(1)))
        siamese = SiameseNetWithClassifier(16, pretrained=False).eval()
        with torch.no_grad():
            a, b, la, lb = siamese(x, x)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(la, lb))
            self.assertTrue(torch.allclose(a.norm(dim=1), torch.ones(2)))
            self.assertTrue(torch.equal(la, siamese.classify(x)))
            self.assertLess(contrastive_loss(a, b, torch.ones(2)).item(), 1e-8)
        for name, net, key in (("baseline", model, "state_dict"), ("siamese_aux", siamese, "model")):
            path = self.root / f"{name}.pt"
            torch.save({key: net.state_dict(), "config": {"embedding_dim": 16}}, path)
            loaded, _ = load_model(name, path, torch.device("cpu"))
            for k, v in net.state_dict().items():
                self.assertTrue(torch.equal(v, loaded.state_dict()[k]))

    def test_persistent_workers_receive_epoch(self):
        dataset = PairDataset(self.frame, self.paths, 16, transform=make_transform(32))
        loader = DataLoader(dataset, batch_size=16, num_workers=1, persistent_workers=True)
        first = next(iter(loader))
        dataset.set_epoch(1)
        second = next(iter(loader))
        expected = next(iter(DataLoader(dataset, batch_size=16, num_workers=0)))
        for a, b in zip(second, expected):
            self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(first[0], second[0]))
        self.assertEqual(second[2].tolist(), [1., 1., 0., 0.] * 4)
        del loader

    def test_threshold_policies(self):
        labels = np.array([0, 0, 0, 0, 1, 1])
        scores = np.array([.1, .2, .3, .4, .5, .6])
        for policy in ("val_specificity80_max_sensitivity", "val_recall70_max_precision"):
            cutoff, _ = choose_threshold(labels, scores, policy)
            self.assertEqual(cutoff, .5)

    def test_training_resume_and_evaluation(self):
        for name in ("baseline", "siamese_aux"):
            output = self.root / name
            args = ["--model", name, "--splits", self.splits, "--images", self.images,
                    "--output", output, "--epochs", 1, "--batch-size", 4,
                    "--image-size", 32, "--workers", 0, "--device", "cpu", "--no-pretrained"]
            if name == "siamese_aux":
                args += ["--pairs-per-epoch", 4, "--embedding-dim", 16, "--cache-images"]
            self.run_script("train.py", *args)
            if name == "siamese_aux":
                self.run_script("train.py", *args, "--resume", "--epochs", 2)
                self.assertEqual(len(pd.read_csv(output / "history.csv")), 2)
                load_model(name, output / "last.pt", torch.device("cpu"))
            evaluation = output / "evaluation"
            self.run_script("predict.py", "--model", name, "--checkpoint", output / "best.pt",
                            "--splits", self.splits, "--images", self.images, "--output", evaluation,
                            "--image-size", 32, "--workers", 0, "--device", "cpu", "--threshold", .5)
            self.assertEqual(len(pd.read_csv(evaluation / "test_predictions.csv")), 4)
            metrics = pd.read_csv(evaluation / "test_metrics.csv")
            self.assertFalse(metrics.validation_recall_at_threshold.isna().any())
            self.assertEqual(len(list(evaluation.glob("*.png"))), 3)


if __name__ == "__main__":
    unittest.main()
