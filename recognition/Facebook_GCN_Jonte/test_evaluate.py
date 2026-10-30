"""Hand-checkable metric fixtures and real-checkpoint selection evaluation."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset import load_facebook
from evaluate import (PROJECT, benchmark_forward, classification_metrics,
                      evaluate_checkpoint, load_model, run_evaluation)
from train import TrainConfig, configure_reproducibility


class MetricTests(unittest.TestCase):
    def test_counts_precision_recall_f1_and_pairwise_auc(self):
        # Synthetic metric fixture, never used as project data or training data.
        targets = np.repeat(np.arange(4), 2)
        predictions = np.array([0, 1, 1, 1, 2, 3, 3, 0])
        probabilities = np.full((8, 4), .1)
        probabilities[np.arange(8), predictions] = .7
        report = classification_metrics(targets, probabilities)
        self.assertEqual(report["confusion_matrix"],
                         [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]])
        self.assertEqual(report["accuracy"], 5 / 8)
        precision = [.5, 2 / 3, 1, .5]
        recall = [.5, 1, .5, .5]
        f1 = [.5, .8, 2 / 3, .5]
        aucs = []
        for i, row in enumerate(report["per_class"]):
            self.assertEqual(row["support"], 2)
            for key, expected in (("precision", precision[i]), ("recall", recall[i]), ("f1", f1[i])):
                self.assertAlmostEqual(row[key], expected)
            positive = probabilities[targets == i, i]
            negative = probabilities[targets != i, i]
            # AUROC independently defined as positive-negative ranking probability.
            auc = np.mean([float(a > b) + .5 * float(a == b)
                           for a in positive for b in negative])
            aucs.append(auc)
            self.assertAlmostEqual(row["ovr_auroc"], auc)
        self.assertAlmostEqual(report["macro_f1"], np.mean(f1))
        self.assertAlmostEqual(report["macro_precision"], np.mean(precision))
        self.assertAlmostEqual(report["macro_recall"], np.mean(recall))
        self.assertAlmostEqual(report["macro_ovr_auroc"], np.mean(aucs))

    def test_perfect_predictions_and_undefined_precision(self):
        targets = np.arange(4)
        perfect = classification_metrics(targets, np.eye(4))
        self.assertEqual(perfect["accuracy"], 1)
        self.assertEqual(perfect["macro_ovr_auroc"], 1)
        constant = classification_metrics(targets, np.tile([1., 0, 0, 0], (4, 1)))
        self.assertEqual(constant["per_class"][1]["precision"], 0)
        self.assertEqual(constant["macro_ovr_auroc"], .5)

    def test_invalid_probabilities_and_missing_classes_fail(self):
        for targets, scores in ((np.arange(4), np.ones((4, 4))),
                                (np.arange(4), np.full((4, 4), np.nan)),
                                (np.arange(3), np.eye(4)[:3]),
                                (np.arange(4).astype(float), np.eye(4))):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                classification_metrics(targets, scores)

    def test_benchmark_calls_are_real_and_gradients_disabled(self):
        calls = []

        def forward():
            self.assertFalse(torch.is_grad_enabled())
            calls.append(1)
            return torch.ones(4).sum()

        report, samples = benchmark_forward(forward, torch.device("cpu"), warmup=2, repeats=3)
        self.assertEqual(len(calls), 6)  # first forward + warmup + timed calls
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(value > 0 for value in samples))
        self.assertAlmostEqual(report["mean_ms"], np.mean(samples))
        with self.assertRaises(ValueError):
            benchmark_forward(forward, torch.device("cpu"), warmup=0)


class RealCheckpointEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_facebook()
        cls.mlp = PROJECT / "outputs/mlp_seed42/best.pt"
        cls.gcn = PROJECT / "outputs/gcn_seed42/best.pt"
        for checkpoint in (cls.mlp, cls.gcn):
            if not checkpoint.exists():
                raise FileNotFoundError("Run the documented genuine MLP and GCN training first")
        configure_reproducibility(TrainConfig(device="cpu"))

    def test_wrong_checkpoint_type_data_or_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model type"):
            load_model(self.mlp, self.dataset, "cpu", "GCN")
        wrong = replace(self.dataset, provenance={**self.dataset.provenance, "sha256": "bad"})
        with self.assertRaisesRegex(ValueError, "dataset hash"):
            load_model(self.mlp, wrong, "cpu", "MLP")
        wrong_audit = {**self.dataset.audit, "splits": {
            name: {**row, "mask_sha256": "bad"}
            for name, row in self.dataset.audit["splits"].items()}}
        wrong = replace(self.dataset, audit=wrong_audit)
        with self.assertRaisesRegex(ValueError, "split hashes"):
            load_model(self.mlp, wrong, "cpu", "MLP")

    def test_pipeline_exports_only_selection_evidence(self):
        poisoned = replace(self.dataset, _targets={
            name: (values.clone() if name == "selection" else torch.full_like(values, -999))
            for name, values in self.dataset._targets.items()})
        access = poisoned.targets

        def selection_only(name, **kwargs):
            if name != "selection":
                raise AssertionError(f"Evaluation accessed {name} labels")
            return access(name, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(poisoned, "targets", side_effect=selection_only) as calls:
                with redirect_stdout(io.StringIO()):
                    report = run_evaluation(poisoned, self.mlp, self.gcn, output,
                                            device="cpu", warmup=1, repeats=2)
            self.assertTrue(all(call.args[0] == "selection" for call in calls.call_args_list))
            self.assertEqual(report["partition"], "selection")
            self.assertEqual(report["reserved_partitions_evaluated"], [])
            self.assertEqual(report["tsne"]["nodes"], 2247)
            self.assertEqual(report["tsne"]["input_dimensions"], 64)
            saved = json.loads((output / "selection_metrics.json").read_text())
            self.assertEqual(saved["models"], report["models"])
            for model in report["models"]:
                self.assertEqual(model["metrics"]["nodes"], 2247)
                self.assertEqual(model["resources"]["parameter_count"], 8516)
                self.assertEqual(sum(map(sum, model["metrics"]["confusion_matrix"])), 2247)
                self.assertIsNone(model["resources"]["peak_cuda_allocated_mib"])
            for stem in ("mlp_confusion_selection", "gcn_confusion_selection", "gcn_tsne_selection"):
                for extension in ("png", "svg"):
                    self.assertGreater((output / f"{stem}.{extension}").stat().st_size, 1000)
            with np.load(output / "node_outputs.npz", allow_pickle=False) as raw:
                self.assertEqual(raw["gcn_hidden"].shape, (2247, 64))
                self.assertEqual(raw["tsne_coordinates"].shape, (2247, 2))
                self.assertTrue(np.isfinite(raw["tsne_coordinates"]).all())
                expected = np.flatnonzero(self.dataset.masks["selection"].numpy())
                np.testing.assert_array_equal(raw["selection_node_ids"], expected)
            with self.assertRaises(FileExistsError):
                run_evaluation(self.dataset, self.mlp, self.gcn, output)

    def test_reserved_targets_do_not_change_gcn_metrics_or_embeddings(self):
        poisoned = replace(self.dataset, _targets={
            name: (values.clone() if name == "selection" else torch.full_like(values, -999))
            for name, values in self.dataset._targets.items()})
        first = evaluate_checkpoint(self.dataset, self.gcn, "GCN", torch.device("cpu"), 1, 2)
        second = evaluate_checkpoint(poisoned, self.gcn, "GCN", torch.device("cpu"), 1, 2)
        self.assertEqual(first[0]["metrics"], second[0]["metrics"])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])


if __name__ == "__main__":
    unittest.main()
