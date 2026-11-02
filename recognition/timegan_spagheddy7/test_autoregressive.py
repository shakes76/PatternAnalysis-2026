"""Check AR estimation, innovation dependence, rollout indexing and persistence."""

import io
import unittest

import torch

from modules import VARFit, VectorAR, lagged_pairs


class AutoregressiveChecks(unittest.TestCase):
    def test_varx_recovers_known_exogenous_coefficients_and_recursive_rollout(self):
        rng = torch.Generator().manual_seed(77)
        previous, context = torch.randn(300, 1, generator=rng, dtype=torch.float64), torch.randn(300, 1, generator=rng, dtype=torch.float64)
        following = .5 * previous + 2 * context + .3
        model = VARFit.from_pairs(previous, following, context).fit(0)
        torch.testing.assert_close(model.weight, torch.tensor([[.5]], dtype=torch.float64), rtol=0, atol=1e-12)
        torch.testing.assert_close(model.context_weight, torch.tensor([[2.]], dtype=torch.float64), rtol=0, atol=1e-12)
        actual = model.generate(torch.zeros(1, 1), 3, context=torch.tensor([[2.]]))
        torch.testing.assert_close(actual, torch.tensor([[[4.3],[6.45],[7.525]]],dtype=torch.float64), rtol=0, atol=1e-11)
        restored = VectorAR(**model.state_dict())
        torch.testing.assert_close(actual, restored.generate(torch.zeros(1,1),3,context=torch.tensor([[2.]])), rtol=0,atol=0)
        with self.assertRaises(ValueError):
            model.generate(torch.zeros(1,1))

    def test_zero_context_has_same_ridge_solution_as_absent_context(self):
        values = torch.randn(200, 3, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
        previous, following = lagged_pairs(values, 2)
        reference = VARFit(values, 2).fit(.01)
        zero = VARFit.from_pairs(previous, following, torch.zeros(len(previous), 19)).fit(.01)
        for name in ("weight","bias","residuals"):
            torch.testing.assert_close(getattr(zero,name),getattr(reference,name),rtol=0,atol=1e-12)
        self.assertEqual(float(zero.context_weight.abs().max()),0.)

    def test_shared_svd_matches_augmented_least_squares(self):
        values = torch.randn((300, 3), generator=torch.Generator().manual_seed(71), dtype=torch.float64)
        values[:, 2] = values[:, 0]  # Include rank deficiency, also at zero regularisation.
        previous, following = lagged_pairs(values, lags=2)
        x, y = previous - previous.mean(0), following - following.mean(0)
        fitter = VARFit(values, lags=2)
        for ridge in (0., 1e-6, 1e-4, 1e-2):
            penalty = (len(x) * ridge) ** 0.5 * torch.eye(x.shape[1], dtype=x.dtype)
            design, target = torch.cat((x, penalty)), torch.cat((y, y.new_zeros((x.shape[1], y.shape[1]))))
            expected = torch.linalg.lstsq(design, target, driver="gelsd").solution.T
            model = fitter.fit(ridge)
            torch.testing.assert_close(model.weight, expected, rtol=1e-8, atol=1e-10)
            torch.testing.assert_close(model.bias, following.mean(0) - previous.mean(0) @ expected.T, rtol=1e-8, atol=1e-10)

    def test_lag_blocks_and_rollout_are_in_the_correct_order(self):
        values = torch.tensor([[1.], [2.], [4.], [8.]], dtype=torch.float64)
        x, y = lagged_pairs(values, lags=2)
        torch.testing.assert_close(x, torch.tensor([[2., 1.], [4., 2.]], dtype=torch.float64))
        torch.testing.assert_close(y, values[2:])
        model = VectorAR(torch.tensor([[2., 3.]]), torch.tensor([0.]), torch.zeros((1, 1)))
        generated = model.generate(torch.tensor([[[1.], [2.]]]), sequence_length=3)
        torch.testing.assert_close(generated, torch.tensor([[[7.], [20.], [61.]]], dtype=torch.float64))

    def test_fit_recovers_two_lag_process(self):
        rng = torch.Generator().manual_seed(101)
        values = torch.zeros((16000, 1), dtype=torch.float64)
        noise = torch.randn(values.shape, generator=rng, dtype=torch.float64) * 0.2
        for index in range(2, len(values)):
            values[index] = 0.5 * values[index - 1] - 0.3 * values[index - 2] + 0.1 + noise[index]
        model = VectorAR.fit(values[200:], ridge=0, lags=2)
        torch.testing.assert_close(model.weight, torch.tensor([[0.5, -0.3]], dtype=torch.float64), atol=0.025, rtol=0)
        torch.testing.assert_close(model.bias, torch.tensor([0.1], dtype=torch.float64), atol=0.01, rtol=0)
        self.assertAlmostEqual(model.spectral_radius, float(torch.linalg.eigvals(
            torch.tensor([[float(model.weight[0, 0]), float(model.weight[0, 1])], [1., 0.]], dtype=torch.float64)
        ).abs().max()))

    def test_fit_recovers_known_vector_process(self):
        rng = torch.Generator().manual_seed(17)
        weight = torch.tensor([[0.65, 0.20], [-0.10, 0.45]], dtype=torch.float64)
        bias = torch.tensor([0.10, -0.05], dtype=torch.float64)
        values = torch.zeros((12000, 2), dtype=torch.float64)
        noise = torch.randn(values.shape, generator=rng, dtype=torch.float64) * 0.1
        for index in range(1, len(values)):
            values[index] = values[index - 1] @ weight.T + bias + noise[index]
        model = VectorAR.fit(values[100:], ridge=0)
        torch.testing.assert_close(model.weight, weight, rtol=0, atol=0.025)
        torch.testing.assert_close(model.bias, bias, rtol=0, atol=0.01)

    def test_rollout_starts_after_the_supplied_state(self):
        model = VectorAR(torch.eye(2), torch.tensor([1., 2.]), torch.zeros((1, 2)))
        actual = model.generate(torch.tensor([[10., 20.]]), sequence_length=3)
        expected = torch.tensor([[[11., 22.], [12., 24.], [13., 26.]]], dtype=torch.float64)
        torch.testing.assert_close(actual, expected)

    def test_joint_residuals_preserve_cross_feature_dependence(self):
        residuals = torch.tensor([[1., 2.], [-1., -2.]], dtype=torch.float64)
        model = VectorAR(torch.zeros((2, 2)), torch.zeros(2), residuals)
        generated = model.generate(torch.zeros((8, 2)), sequence_length=32)
        torch.testing.assert_close(generated[..., 1], 2 * generated[..., 0])
        self.assertEqual(set(generated[..., 0].unique().tolist()), {-1., 1.})

    def test_save_reload_and_local_random_seed(self):
        rng = torch.Generator().manual_seed(23)
        model = VectorAR.fit(torch.randn((400, 3), generator=rng, dtype=torch.float64))
        initial = torch.zeros((4, 3))
        global_state = torch.random.get_rng_state().clone()
        expected = model.generate(initial, seed=9)
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = VectorAR(**torch.load(stream, weights_only=True))
        torch.testing.assert_close(restored.generate(initial, seed=9), expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(restored.generate(initial, seed=10), expected))

    def test_constant_features_fit_without_singular_matrix_failure(self):
        values = torch.tensor([[1., 2.]], dtype=torch.float64).repeat(100, 1)
        model = VectorAR.fit(values, ridge=0)
        expected = values[0].expand(2, 32, 2)
        torch.testing.assert_close(model.generate(values[:2]), expected)


if __name__ == "__main__":
    unittest.main()
