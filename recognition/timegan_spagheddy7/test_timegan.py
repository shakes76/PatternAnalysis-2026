"""TimeGAN loss alignment, gradient routes and reproducible generation."""

import io
import unittest
from unittest.mock import patch

import torch

from modules import TimeGAN, ReferenceGRU, temporal_loss, moment_loss, order_penalty
from train import TimeGANTrainer, reconstruction_price_weight


class TimeGANChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        self.model = TimeGAN(features=4, hidden=8, layers=2, noise_dim=4, cell="reference")
        self.real, self.noise = torch.rand(8, 10, 4), torch.rand(8, 10, 4)
        self.trainer = TimeGANTrainer(self.model)

    def changed_networks(self, operation):
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        operation()
        return {name.split('.')[0] for name, value in self.model.named_parameters() if not torch.equal(value, before[name])}

    def test_temporal_loss_uses_next_update_and_ignores_last_prediction(self):
        latent = torch.arange(10.).reshape(1, 5, 2)
        prediction = torch.cat((latent[:, 1:], torch.full((1, 1, 2), 999.)), dim=1)
        self.assertEqual(float(temporal_loss(latent, prediction)), 0)
        self.assertGreater(float(temporal_loss(latent, latent)), 0)
        with self.assertRaises(ValueError):
            temporal_loss(latent[:, :1], prediction[:, :1])

    def test_moments_use_batch_population_statistics_per_time_and_feature(self):
        real = torch.tensor([[[0., 2.], [2., 4.]], [[2., 4.], [4., 6.]]])
        torch.testing.assert_close(moment_loss(real, real + 2), torch.tensor(2.))
        self.assertEqual(float(moment_loss(real, real.flip(0))), 0)
        self.assertGreater(float(moment_loss(real, real.mean(0, keepdim=True).expand_as(real))), .9)

    def test_pretraining_updates_only_embedder_and_recovery(self):
        self.assertEqual(self.changed_networks(lambda: self.trainer.embedding_step(self.real)), {"embedder", "recovery"})

    def test_joint_embedding_starts_with_fresh_adam_state(self):
        self.trainer.embedding_step(self.real)
        self.trainer.embedding_step(self.real)
        warmup, joint = self.trainer.optimizers["embedding_pretrain"], self.trainer.optimizers["embedding"]
        self.assertTrue(all(float(state["step"]) == 2 for state in warmup.state.values()))
        self.assertFalse(joint.state)
        self.trainer.embedding_step(self.real, joint=True)
        self.assertTrue(all(float(state["step"]) == 1 for state in joint.state.values()))
        self.assertTrue(all(float(state["step"]) == 2 for state in warmup.state.values()))

    def test_reference_gru_matches_concat_equations_and_gradients(self):
        gru = ReferenceGRU(3, 4, 1).double()
        cell = gru.cells[0]
        values = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
        state, expected = torch.zeros(2, 4, dtype=torch.float64), []
        for x in values.unbind(1):
            reset, update = torch.sigmoid(torch.cat((x, state), 1) @ cell["gate_kernel"] + cell["gate_bias"]).chunk(2, 1)
            candidate = torch.tanh(torch.cat((x, reset * state), 1) @ cell["candidate_kernel"] + cell["candidate_bias"])
            state = update * state + (1 - update) * candidate
            expected.append(state)
        expected, actual = torch.stack(expected, 1), gru(values)[0]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        inputs = (values, *gru.parameters())
        for left, right in zip(torch.autograd.grad(actual.square().sum(), inputs), torch.autograd.grad(expected.square().sum(), inputs)):
            torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-11)

    def test_joint_losses_match_reference_weights_and_roots(self):
        with torch.no_grad():
            latent = self.model.embedder(self.real)
            raw_fake = self.model.generator(self.noise)
            fake = self.model.supervisor(raw_fake)
            supervision = temporal_loss(latent, self.model.supervisor(latent))
            reconstruction = (self.model.recovery(latent) - self.real).square().mean()
            generator = (self.trainer.classify(self.model.discriminator(fake), True)
                         + self.trainer.classify(self.model.discriminator(raw_fake), True)
                         + 100 * supervision.sqrt() + 100 * moment_loss(self.real, self.model.recovery(fake)))
        with patch.object(self.trainer, "update", return_value=torch.tensor(0.)) as update:
            self.trainer.embedding_step(self.real, joint=True)
            torch.testing.assert_close(update.call_args.args[0], 10 * reconstruction.sqrt() + .1 * supervision)
            self.trainer.generator_step(self.real, self.noise)
            torch.testing.assert_close(update.call_args.args[0], generator)

    def test_skipping_diagnostics_preserves_parameter_updates(self):
        quiet = TimeGAN(**self.model.config)
        quiet.load_state_dict(self.model.state_dict())
        trainer = TimeGANTrainer(quiet)
        for collect, runner in ((True, self.trainer), (False, trainer)):
            runner.embedding_step(self.real, collect=collect)
            runner.supervisor_step(self.real, collect=collect)
            runner.generator_step(self.real, self.noise, collect=collect)
            runner.embedding_step(self.real, joint=True, collect=collect)
            runner.discriminator_step(self.real, self.noise, collect=collect)
        for left, right in zip(self.model.parameters(), quiet.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_supervision_updates_only_supervisor(self):
        self.assertEqual(self.changed_networks(lambda: self.trainer.supervisor_step(self.real)), {"supervisor"})

    def test_adversarial_gradients_reach_generator_through_frozen_networks(self):
        trainer = TimeGANTrainer(self.model, supervised_weight=0, moment_weight=0)
        self.assertEqual(self.changed_networks(lambda: trainer.generator_step(self.real, self.noise)), {"generator", "supervisor"})
        for name in ("embedder", "recovery", "discriminator"):
            self.assertTrue(all(p.grad is None for p in getattr(self.model, name).parameters()))

    def test_joint_embedding_and_discriminator_have_separate_updates(self):
        self.assertEqual(self.changed_networks(lambda: self.trainer.embedding_step(self.real, joint=True)), {"embedder", "recovery"})
        self.assertEqual(self.changed_networks(lambda: self.trainer.discriminator_step(self.real, self.noise)), {"discriminator"})
        self.assertEqual(self.changed_networks(lambda: self.trainer.discriminator_step(self.real, self.noise, threshold=100)), set())

    def test_reconstruction_learns_a_small_batch(self):
        with torch.no_grad():
            before = float((self.model.reconstruct(self.real) - self.real).square().mean())
        for _ in range(400):
            self.trainer.embedding_step(self.real)
        with torch.no_grad():
            after = float((self.model.reconstruct(self.real) - self.real).square().mean())
        self.assertLess(after, .8 * before)

    def test_noise_only_generation_reload_local_seed_and_causality(self):
        state = torch.get_rng_state()
        first = self.model.generate(5, 10, 42, batch_size=2)
        torch.testing.assert_close(torch.get_rng_state(), state)
        self.assertEqual(first.shape, (5, 10, 4))
        self.assertTrue(torch.isfinite(first).all() and (first >= 0).all() and (first <= 1).all())
        self.assertFalse(torch.equal(first, self.model.generate(5, 10, 43, batch_size=2)))
        payload = io.BytesIO()
        torch.save(self.model.state_dict(), payload)
        payload.seek(0)
        reloaded = TimeGAN(**self.model.config)
        reloaded.load_state_dict(torch.load(payload, weights_only=True))
        torch.testing.assert_close(first, reloaded.generate(5, 10, 42, batch_size=2), rtol=0, atol=0)
        changed = self.noise.clone()
        changed[:, 5:] = 100
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.noise)[:, :5], self.model(changed)[:, :5])

    def test_legacy_pytorch_checkpoint_architecture_still_loads(self):
        legacy = TimeGAN(features=4, hidden=8, layers=2, noise_dim=4)
        reloaded = TimeGAN(**{key: value for key, value in legacy.config.items() if key != "cell"})
        reloaded.load_state_dict(legacy.state_dict())
        torch.testing.assert_close(legacy.generate(3), reloaded.generate(3), rtol=0, atol=0)


class SupervisionAblationChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        self.model = TimeGAN(features=4, hidden=8, layers=2, noise_dim=4, cell="reference")
        self.real, self.noise = torch.rand(8, 10, 4), torch.rand(8, 10, 4)

    def test_no_supervision_uses_no_temporal_targets_but_still_learns(self):
        trainer = TimeGANTrainer(self.model, no_supervision=True)
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        with patch("train.temporal_loss", side_effect=AssertionError("Temporal targets must not be used")):
            with patch.object(self.model.embedder, "forward", side_effect=AssertionError("No real latent targets in G step")):
                generated = trainer.generator_step(self.real, self.noise)
            embedded = trainer.embedding_step(self.real, joint=True)
        self.assertEqual(float(generated["supervised_mse"]), 0.)
        self.assertEqual(float(embedded["embedding_temporal_mse"]), 0.)
        changed = {name.split('.')[0] for name, value in self.model.named_parameters() if not torch.equal(value, before[name])}
        self.assertEqual(changed, {"generator", "supervisor", "embedder", "recovery"})
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))
        with self.assertRaisesRegex(ValueError, "skip supervisor"):
            trainer.supervisor_step(self.real)

    def test_omitted_supervisor_updates_keep_joint_batch_draws_paired(self):
        from train import skip_supervisor_draws
        control, ablated = torch.Generator().manual_seed(12), torch.Generator().manual_seed(12)
        for _ in range(9):
            torch.randperm(57, generator=control)[:8]
        skip_supervisor_draws(ablated, 57, 9)
        for _ in range(3):
            torch.testing.assert_close(torch.randperm(57, generator=control)[:8], torch.randperm(57, generator=ablated)[:8])

    def test_explicit_reference_keeps_default_updates_identical(self):
        copy = TimeGAN(**self.model.config)
        copy.load_state_dict(self.model.state_dict())
        with torch.jit.optimized_execution(False):
            for trainer in (TimeGANTrainer(self.model), TimeGANTrainer(copy, no_supervision=False)):
                trainer.embedding_step(self.real)
                trainer.supervisor_step(self.real)
                trainer.generator_step(self.real, self.noise)
                trainer.embedding_step(self.real, joint=True)
                trainer.discriminator_step(self.real, self.noise)
        for left, right in zip(self.model.parameters(), copy.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


class ConditionalTimeGANChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)
        self.model = TimeGAN(features=4, hidden=8, layers=2, noise_dim=4, cell="reference", context_dim=3)
        self.context = torch.randn(8, 3)
        self.real, self.noise = torch.rand(8, 10, 4), torch.rand(8, 10, 4)

    def test_conditioning_changes_outputs_and_keeps_seed_and_batch_reproducibility(self):
        state = torch.get_rng_state()
        first = self.model.generate(8, 10, 7, 8, self.context)
        torch.testing.assert_close(first, self.model.generate(8, 10, 7, 3, self.context), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(state, torch.get_rng_state())
        self.assertFalse(torch.equal(first, self.model.generate(8, 10, 7, 8, -self.context)))
        self.assertFalse(torch.equal(first, self.model.generate(8, 10, 8, 8, self.context)))
        with self.assertRaises(ValueError):
            self.model.generate(8, 10)
        with self.assertRaises(ValueError):
            self.model.generate(8, 10, context=self.context[:3])

    def test_context_has_a_gradient_path_through_each_network(self):
        for name, network in self.model.named_children():
            context = self.context.clone().requires_grad_()
            values = self.real if name == "embedder" else self.noise if name == "generator" else torch.rand(8, 10, 8)
            network(values, context).sum().backward()
            self.assertTrue(torch.isfinite(context.grad).all())
            self.assertGreater(float(context.grad.abs().sum()), 0.)

    def test_conditional_training_and_checkpoint_roundtrip(self):
        trainer = TimeGANTrainer(self.model)
        trainer.embedding_step(self.real, context=self.context)
        trainer.supervisor_step(self.real, context=self.context)
        trainer.generator_step(self.real, self.noise, context=self.context)
        trainer.embedding_step(self.real, joint=True, context=self.context)
        trainer.discriminator_step(self.real, self.noise, context=self.context)
        buffer = io.BytesIO()
        torch.save(dict(architecture=self.model.config, state_dict=self.model.state_dict()), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = TimeGAN(**saved["architecture"])
        restored.load_state_dict(saved["state_dict"])
        torch.testing.assert_close(self.model.generate(8, 10, context=self.context), restored.generate(8, 10, context=self.context), rtol=0, atol=0)


class StructuralLossChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.anchors = torch.tensor([[value for level in range(10) for value in (101. + level * .05, 100. - level * .05)]], dtype=torch.float64)
        self.minimum, self.span = torch.zeros(40), torch.ones(40)

    def test_valid_books_have_zero_penalty_and_volume_changes_do_not_affect_it(self):
        features = torch.zeros(1, 3, 40, requires_grad=True)
        with torch.no_grad():
            features[..., 1::2] = 100
        loss = order_penalty(features, self.anchors, self.minimum, self.span)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertEqual(float(features.grad.abs().sum()), 0.)

    def test_violations_at_every_level_and_crossings_have_correct_gradient_direction(self):
        # Each deeper ask must rise and each deeper bid must fall to repair its violation.
        cases = [(4 * level, 99., -1) for level in range(1, 10)] + [(4 * level + 2, 102., 1) for level in range(1, 10)] + [(0, 99., -1)]
        for column, target, sign in cases:
            features = torch.zeros(1, 2, 40, dtype=torch.float64, requires_grad=True)
            with torch.no_grad():
                features[0, 0, column] = torch.log(torch.tensor(target) / self.anchors[0, column // 2])
            loss = order_penalty(features, self.anchors, self.minimum, self.span)
            self.assertGreater(float(loss.detach()), 0.)
            loss.backward()
            self.assertGreater(float(features.grad[0, 0, column]) * sign, 0.)
            self.assertTrue(torch.isfinite(features.grad).all())
            self.assertEqual(float(features.grad[..., 1::2].abs().sum()), 0.)

    def test_penalty_reaches_generator_through_frozen_recovery(self):
        torch.manual_seed(7)
        model = TimeGAN(hidden=8, layers=2, cell="reference")
        trainer = TimeGANTrainer(model, supervised_weight=0, moment_weight=0, structure_weight=.1, minimum=self.minimum, span=self.span)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        with patch.object(trainer, "classify", side_effect=lambda logits, real: logits.sum() * 0):
            log = trainer.generator_step(torch.rand(1, 3, 40), torch.rand(1, 3, 40), anchors=self.anchors)
        self.assertGreater(float(log["generator_order_penalty"]), 0.)
        changed = {name.split('.')[0] for name, value in model.named_parameters() if not torch.equal(value, before[name])}
        self.assertEqual(changed, {"generator", "supervisor"})
        for name in ("embedder", "recovery", "discriminator"):
            self.assertTrue(all(p.grad is None for p in getattr(model, name).parameters()))


class ReconstructionBalanceChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = TimeGAN(features=4, hidden=8, layers=2, noise_dim=4, cell="reference")

    def test_weighted_loss_and_gradient_match_known_fixture(self):
        trainer = TimeGANTrainer(self.model, price_reconstruction_weight=9.)
        recovered = torch.tensor([[[.1, .2, .3, .4]]], requires_grad=True)
        with patch.object(self.model.recovery, "forward", return_value=recovered), patch.object(trainer, "update", return_value=torch.tensor(0.)) as update:
            log = trainer.embedding_step(torch.zeros_like(recovered))
        loss = update.call_args.args[0]
        torch.testing.assert_close(log["reconstruction_mse"], torch.tensor(.075))
        torch.testing.assert_close(log["weighted_reconstruction_mse"], torch.tensor(.275))
        torch.testing.assert_close(loss, 10 * torch.tensor(.275).sqrt())
        loss.backward()
        torch.testing.assert_close(recovered.grad[0, 0, 0] / recovered.grad[0, 0, 1], torch.tensor(4.5))

    def test_explicit_unit_weight_preserves_all_default_updates(self):
        copy = TimeGAN(**self.model.config)
        copy.load_state_dict(self.model.state_dict())
        real, noise = torch.rand(8, 10, 4), torch.rand(8, 10, 4)
        # Keep both sides on the same JIT path before its profiling warm-up finishes.
        with torch.jit.optimized_execution(False):
            for trainer in (TimeGANTrainer(self.model), TimeGANTrainer(copy, price_reconstruction_weight=1.)):
                self.assertIsNone(trainer.reconstruction_weights)
                trainer.embedding_step(real)
                trainer.supervisor_step(real)
                trainer.generator_step(real, noise)
                trainer.embedding_step(real, joint=True)
                trainer.discriminator_step(real, noise)
        for left, right in zip(self.model.parameters(), copy.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_weight_uses_mean_per_feature_training_variance(self):
        # Price variances 1 and 4; volume variances 9 and 36: ratio is 9.
        values = torch.tensor([[[0., 10., 100., 20.], [2., 16., 104., 32.]]])
        original = values.clone()
        self.assertEqual(reconstruction_price_weight(values), 9.)
        self.assertEqual(reconstruction_price_weight(values + torch.tensor([99., 40., -30., 500.])), 9.)
        torch.testing.assert_close(values, original, rtol=0, atol=0)
        for invalid in (torch.zeros(1, 2, 4), torch.ones(2, 4), torch.ones(2, 3, 3), values * float("nan")):
            with self.assertRaises(ValueError):
                reconstruction_price_weight(invalid)

    def test_invalid_price_weights_rejected_before_training(self):
        for weight in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                TimeGANTrainer(self.model, price_reconstruction_weight=weight)

    def test_partial_balance_interpolates_without_changing_the_default(self):
        from train import fit_timegan
        values = torch.tensor([[[0., 10., 100., 20.], [2., 16., 104., 32.]]])
        self.assertEqual(reconstruction_price_weight(values), 9.)
        self.assertEqual(reconstruction_price_weight(values, .5), 3.)
        self.assertEqual(reconstruction_price_weight(values, 0.), 1.)
        for invalid in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                reconstruction_price_weight(values, invalid)
        with self.assertRaisesRegex(ValueError, "balance-reconstruction"):
            fit_timegan(None, balance_power=.5)


class TuningChecks(unittest.TestCase):
    def test_atomic_report_retries_a_temporary_windows_file_lock(self):
        import json
        import tempfile
        from pathlib import Path
        from tune_timegan import write_json
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "results.json"
            target.write_text('{"old": 1}')
            original, calls = Path.replace, []
            def locked_then_replace(source, destination):
                calls.append(1)
                self.assertEqual(json.loads(target.read_text()), {"old": 1})
                if len(calls) < 3:
                    raise PermissionError("temporary file-sharing lock")
                return original(source, destination)
            with patch.object(Path, "replace", locked_then_replace), patch("tune_timegan.time.sleep"):
                write_json(target, {"new": 2})
            self.assertEqual(json.loads(target.read_text()), {"new": 2})
            self.assertEqual(len(calls), 3)
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_conditioning_screen_keeps_distribution_and_temporal_failures_visible(self):
        from timegan_report import conditioning_screen
        zero = dict(invalid_rate_mean=.51, kl_spread_mean=.166, kl_mid_return_mean=.647,
                    abs_return_acf_mae_mean=.176, depth_ssim_mean=.128)
        matched = dict(invalid_rate_mean=.417, kl_spread_mean=.206, kl_mid_return_mean=.607,
                       abs_return_acf_mae_mean=.306, depth_ssim_mean=.145)
        noise = [{"Price SD / real": .12, "Volume SD / real": .28} for _ in range(3)]
        rows = conditioning_screen(matched, zero, noise)
        self.assertEqual([r["Passed"] for r in rows], [True, False, True, False, True, True])

    def test_conditioning_screen_requires_price_and_volume_noise_in_every_case(self):
        from timegan_report import conditioning_screen
        score = dict(invalid_rate_mean=.5, kl_spread_mean=.2, kl_mid_return_mean=.6,
                     abs_return_acf_mae_mean=.2, depth_ssim_mean=.15)
        noise = [{"Price SD / real": .12, "Volume SD / real": .28} for _ in range(3)]
        noise[-1]["Volume SD / real"] = .09
        self.assertFalse(conditioning_screen(score, score, noise)[-1]["Passed"])
        with self.assertRaisesRegex(ValueError, "three"):
            conditioning_screen(score, score, noise[:2])
        noise[-1]["Volume SD / real"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            conditioning_screen(score, score, noise)

    def test_conditioning_pair_rejects_unequal_budgets_sources_or_test_scoring(self):
        import copy
        from timegan_report import check_conditioning_pair
        runs = {mode: dict(config=dict(anchor_conditioning=mode, training_seed=2026, joint_steps=5000),
                          test_evaluated=False, comparison_signature="same", metric_specification={"bins": 50},
                          source_files={"modules.py": "same"}, parameter_count=61361) for mode in ("zero", "matched")}
        check_conditioning_pair(runs)
        for key, value in (("source_files", {"modules.py": "changed"}), ("test_evaluated", True),
                           ("metric_specification", {"bins": 25}), ("parameter_count", 54521)):
            changed = copy.deepcopy(runs)
            changed["matched"][key] = value
            with self.assertRaises(ValueError):
                check_conditioning_pair(changed)
        changed = copy.deepcopy(runs)
        changed["matched"]["config"]["joint_steps"] = 1000
        with self.assertRaisesRegex(ValueError, "budgets"):
            check_conditioning_pair(changed)

    def test_conditioning_report_command_does_not_train_or_rewrite_summary(self):
        import sys
        from tune_timegan import main
        with patch.object(sys, "argv", ["tune_timegan.py", "--data-dir", "data", "--conditioning-report"]), \
             patch("tune_timegan.analyse_conditioning", return_value={"conclusion": "pilot"}) as report, \
             patch("tune_timegan.run_candidate") as train, patch("tune_timegan.collect_summary") as collect, \
             patch("builtins.print"):
            main()
        report.assert_called_once()
        train.assert_not_called()
        collect.assert_not_called()

    def test_shifted_anchor_control_changes_indices_without_mutating_inputs(self):
        import numpy as np
        from timegan_report import shifted_anchors
        anchors = np.arange(100).reshape(5, 20)
        before = anchors.copy()
        moved, shift = shifted_anchors(anchors, 42)
        np.testing.assert_array_equal(anchors, before)
        np.testing.assert_array_equal(moved, shifted_anchors(anchors, 42)[0])
        self.assertTrue(0 < shift < len(anchors))
        self.assertFalse(np.any(np.all(moved == anchors, axis=1)))
        np.testing.assert_array_equal(np.sort(moved[:, 0]), np.sort(anchors[:, 0]))


if __name__ == "__main__":
    unittest.main()
