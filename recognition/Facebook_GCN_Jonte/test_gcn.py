"""Real-data GCN checks; reserved targets are never evaluated."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import coalesce, remove_self_loops

from dataset import load_facebook
from modules import GCN
from train import TrainConfig, configure_reproducibility, evaluate, train_model


class RealDataGCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_facebook()
        configure_reproducibility(TrainConfig(device="cpu"))

    def fit(self, dataset, output, **overrides):
        config = replace(TrainConfig(model_type="GCN", device="cpu", max_epochs=2,
                                     patience=2), **overrides)
        with redirect_stdout(io.StringIO()):
            summary = train_model(dataset, config, output)
        checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        return summary, checkpoint

    def test_architecture_and_real_graph_dependence(self):
        model = GCN().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8516)
        self.assertEqual(sum(isinstance(m, GCNConv) for m in model.modules()), 2)
        graph = self.dataset.graph
        with torch.no_grad():
            logits = model(graph.x, graph.edge_index)
            # A fresh model is essential: cached normalization binds each model
            # instance to its first graph. Empty-edge control retains real features.
            control = GCN().eval()
            control.load_state_dict(model.state_dict())
            no_neighbors = control(graph.x, graph.edge_index[:, :0])
        self.assertEqual(tuple(logits.shape), (22470, 4))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(torch.allclose(logits, no_neighbors))
        for layer in (model.conv1, model.conv2):
            self.assertTrue(layer.cached and layer.normalize and layer.add_self_loops)
        self.assertEqual(set(model.state_dict()),
                         {"conv1.bias", "conv1.lin.weight", "conv2.bias", "conv2.lin.weight"})

    def test_duplicate_loops_match_clean_graph_outputs(self):
        graph = self.dataset.graph
        clean_edges = coalesce(remove_self_loops(graph.edge_index)[0], num_nodes=graph.num_nodes)
        first, second = GCN().eval(), GCN().eval()
        second.load_state_dict(first.state_dict())
        with torch.no_grad():
            raw = first(graph.x, graph.edge_index)
            cleaned = second(graph.x, clean_edges)
        torch.testing.assert_close(raw, cleaned, atol=1e-6, rtol=1e-5)
        for layer in (first.conv1, first.conv2):
            edges, weights = layer._cached_edge_index
            self.assertEqual(int((edges[0] == edges[1]).sum()), 22470)
            self.assertEqual(edges.size(1), 364116)
            self.assertTrue(torch.isfinite(weights).all())

    def test_checkpoint_rebuilds_cache_and_early_stopping_restores_best(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, checkpoint = self.fit(self.dataset, Path(directory), max_epochs=10,
                                           patience=2, min_delta=1.0)
            model = GCN(**checkpoint["model_config"])
            self.assertIsNone(model.conv1._cached_edge_index)
            model.load_state_dict(checkpoint["model_state_dict"])
            measured = evaluate(model, self.dataset.graph.x, self.dataset.targets("selection"),
                                self.dataset.graph.edge_index, self.dataset.masks["selection"])
        self.assertEqual(summary["epochs_completed"], 3)
        self.assertEqual(summary["best_epoch"], 1)
        self.assertEqual(measured, checkpoint["selection_metrics"])
        self.assertEqual(measured, summary["restored_selection_metrics"])
        self.assertEqual(summary["reserved_splits_evaluated"], [])
        self.assertEqual(checkpoint["model_type"], "GCN")

    def test_reserved_labels_cannot_change_training(self):
        targets = {name: values.clone() for name, values in self.dataset._targets.items()}
        for name in ("test", "calibration", "policy"):
            targets[name].fill_(-999)
        poisoned = replace(self.dataset, _targets=targets)
        original_access = poisoned.targets

        def allowed(name, **kwargs):
            if name not in ("train", "selection"):
                raise AssertionError("Reserved labels accessed")
            return original_access(name, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            summary, checkpoint = self.fit(self.dataset, Path(directory) / "original")
            with patch.object(poisoned, "targets", side_effect=allowed) as calls:
                other, restored = self.fit(poisoned, Path(directory) / "poisoned")
        self.assertEqual([call.args[0] for call in calls.call_args_list], ["train", "selection"])
        self.assertEqual(summary["restored_selection_metrics"], other["restored_selection_metrics"])
        self.assertEqual(summary["best_epoch"], other["best_epoch"])
        for name, tensor in checkpoint["model_state_dict"].items():
            self.assertTrue(torch.equal(tensor, restored["model_state_dict"][name]), name)


if __name__ == "__main__":
    unittest.main()
