"""Check GCN normalization, permutation equivariance and differentiability."""

import unittest

import torch

from modules import GCN, normalized_adjacency


class GCNTests(unittest.TestCase):
    def test_binary_graph_normalization_with_duplicate_loops(self):
        edges = torch.tensor([[0, 1, 1, 2, 2, 2], [1, 0, 2, 1, 2, 2]])
        actual = normalized_adjacency(edges, 4).to_dense()
        binary = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 0.],
                               [0., 1., 1., 0.], [0., 0., 0., 1.]])
        inverse = binary.sum(1).rsqrt()
        expected = inverse[:, None] * binary * inverse[None, :]
        torch.testing.assert_close(actual, expected)

    def test_permuting_nodes_permutes_predictions_and_gradients_are_finite(self):
        torch.manual_seed(7)
        features = torch.randn(4, 3)
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        adjacency = normalized_adjacency(edges, 4)
        model = GCN(inputs=3, hidden=5, classes=2, dropout=0.)
        model.eval()
        logits, _ = model(features, adjacency)
        order = torch.tensor([2, 0, 3, 1])
        permuted_adjacency = adjacency.to_dense()[order][:, order].to_sparse()
        permuted_logits, _ = model(features[order], permuted_adjacency)
        torch.testing.assert_close(permuted_logits, logits[order])
        loss = torch.nn.functional.cross_entropy(logits[:3], torch.tensor([0, 1, 0]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
