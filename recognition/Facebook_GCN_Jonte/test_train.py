"""MLP checks and short training tests using the approved real Facebook data.

These short fits verify software behavior; the separately logged baseline run
provides the project's measured training/selection results. No test targets are
evaluated here.
"""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dataset import load_facebook
from modules import MLP
from train import TrainConfig, evaluate, train_mlp


class RealDataMLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_facebook()

    def fit(self, output_dir: Path, **overrides):
        config = replace(
            TrainConfig(device="cpu", max_epochs=2, patience=2), **overrides,
        )
        with redirect_stdout(io.StringIO()):
            summary = train_mlp(self.dataset, config, output_dir)
        checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
        return summary, checkpoint

    def test_architecture_parameter_count_logits_and_row_independence(self):
        model = MLP().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8516)
        features = self.dataset.graph.x[:32]
        with torch.no_grad():
            logits = model(features)
            reordered = model(features.flip(0))
        self.assertEqual(logits.shape, (32, 4))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.allclose(reordered, logits.flip(0)))
        self.assertEqual(sum(isinstance(m, torch.nn.Linear) for m in model.modules()), 2)
        self.assertFalse(any(isinstance(m, torch.nn.Softmax) for m in model.modules()))

    def test_checkpoint_reloads_and_reproduces_selection_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary, checkpoint = self.fit(output, max_epochs=1, patience=1)
            model = MLP(**checkpoint["model_config"])
            model.load_state_dict(checkpoint["model_state_dict"])
            measured = evaluate(
                model, self.dataset.graph.x[self.dataset.masks["selection"]],
                self.dataset.targets("selection"),
            )
            self.assertEqual(measured, summary["restored_selection_metrics"])
            self.assertEqual(summary["reserved_splits_evaluated"], [])
            self.assertEqual(summary["epochs_completed"], 1)
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(len(json.loads((output / "history.json").read_text())), 1)
            self.assertGreater((output / "learning_curves.png").stat().st_size, 0)
            self.assertEqual(checkpoint["data_provenance"]["sha256"], self.dataset.provenance["sha256"])

    def test_reserved_labels_cannot_change_weights_or_checkpoint_selection(self):
        poisoned_targets = {
            name: targets.clone() for name, targets in self.dataset._targets.items()
        }
        for name in ("test", "calibration", "policy"):
            poisoned_targets[name].fill_(-999)
        poisoned = replace(self.dataset, _targets=poisoned_targets)
        target_access = poisoned.targets

        def permitted_targets(name, **kwargs):
            if name not in ("train", "selection"):
                raise AssertionError(f"Training accessed reserved labels: {name}")
            return target_access(name, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "original"
            second = Path(directory) / "reserved-labels-poisoned"
            summary, checkpoint = self.fit(first)
            with patch.object(poisoned, "targets", side_effect=permitted_targets) as calls:
                with redirect_stdout(io.StringIO()):
                    other = train_mlp(
                        poisoned, TrainConfig(device="cpu", max_epochs=2, patience=2), second,
                    )
            changed = torch.load(second / "best.pt", map_location="cpu", weights_only=True)
            self.assertEqual([call.args[0] for call in calls.call_args_list], ["train", "selection"])
            self.assertEqual(summary["best_epoch"], other["best_epoch"])
            self.assertEqual(summary["restored_selection_metrics"], other["restored_selection_metrics"])
            for name, tensor in checkpoint["model_state_dict"].items():
                self.assertTrue(torch.equal(tensor, changed["model_state_dict"][name]), name)

    def test_early_stopping_restores_first_checkpoint_when_improvement_is_impossible(self):
        # Macro-F1 cannot improve by more than 1.0. This tests real early stopping
        # without inventing or mocking validation scores.
        with tempfile.TemporaryDirectory() as directory:
            summary, checkpoint = self.fit(
                Path(directory), max_epochs=10, patience=2, min_delta=1.0,
            )
        self.assertEqual(summary["epochs_completed"], 3)
        self.assertEqual(summary["best_epoch"], 1)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertTrue(summary["stopped_early"])
        self.assertEqual(summary["restored_selection_metrics"], checkpoint["selection_metrics"])

    def test_existing_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / "keep.txt"
            sentinel.write_text("existing run")
            with self.assertRaises(FileExistsError):
                train_mlp(self.dataset, TrainConfig(), output)
            self.assertEqual(sentinel.read_text(), "existing run")

    def test_split_seed_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "split seeds"):
                train_mlp(self.dataset, TrainConfig(split_seed=43), Path(directory))


if __name__ == "__main__":
    unittest.main()
