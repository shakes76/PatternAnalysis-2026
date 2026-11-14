"""CPU tests of the B2 GAN pieces and the B3 scratch UNet, with tiny inputs."""

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("diffusers") is None:
    raise unittest.SkipTest("needs diffusers and peft (training environment)")

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tools"))
from baseline_gan import (  # noqa: E402
    CollapseRule,
    ConditionalDiscriminator,
    ConditionalGenerator,
    generate,
    mean_pairwise_distance,
    r1_penalty,
    recalibrate_batchnorm,
)
from modules import build_unet, load_weights, save_weights  # noqa: E402


class GanTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.generator = ConditionalGenerator(resolution=32, z_dim=16, width=8)
        self.discriminator = ConditionalDiscriminator(resolution=32, width=8)

    def test_shapes_and_range(self):
        images = self.generator(torch.randn(4, 16), torch.tensor([0, 1, 0, 1]))
        self.assertEqual(tuple(images.shape), (4, 1, 32, 32))
        self.assertLessEqual(images.abs().max().item(), 1.0)
        scores = self.discriminator(images, torch.tensor([0, 1, 0, 1]))
        self.assertEqual(tuple(scores.shape), (4,))

    def test_label_changes_both_networks(self):
        self.generator.eval()
        z = torch.randn(4, 16)
        ad = self.generator(z, torch.zeros(4, dtype=torch.long))
        nc = self.generator(z, torch.ones(4, dtype=torch.long))
        self.assertFalse(torch.allclose(ad, nc))
        self.assertFalse(
            torch.allclose(
                self.discriminator(ad, torch.zeros(4, dtype=torch.long)),
                self.discriminator(ad, torch.ones(4, dtype=torch.long)),
            )
        )

    def test_r1_penalty_is_finite_and_differentiable(self):
        real = torch.rand(4, 1, 32, 32).mul(2).sub(1).requires_grad_(True)
        scores = self.discriminator(real, torch.tensor([0, 1, 0, 1]))
        penalty = r1_penalty(scores, real)
        self.assertTrue(torch.isfinite(penalty))
        penalty.backward()
        self.assertTrue(
            any(p.grad is not None for p in self.discriminator.parameters())
        )

    def test_diversity_measure(self):
        same = torch.zeros(8, 1, 32, 32)
        self.assertEqual(mean_pairwise_distance(same), 0.0)
        self.assertGreater(mean_pairwise_distance(torch.rand(8, 1, 32, 32)), 0.0)

    def test_collapse_rule_follows_the_protocol(self):
        rule = CollapseRule(after=5000)
        self.assertEqual(rule.observe(1000, 0.1), (False, False))  # ignored early
        self.assertEqual(rule.observe(5000, 0.9), (True, False))  # healthy snapshot
        self.assertEqual(rule.observe(6000, 0.5), (False, False))  # first low
        self.assertEqual(rule.observe(7000, 0.7), (False, False))  # resets the count
        self.assertEqual(rule.observe(8000, 0.5), (False, False))
        self.assertEqual(rule.observe(9000, 0.4), (False, True))  # second in a row
        self.assertEqual((rule.snapshot_step, rule.stopped_at), (5000, 9000))

    def test_recalibration_is_deterministic(self):
        first = recalibrate_batchnorm(
            self.generator, torch.device("cpu"), seed=3, batches=2, size=8
        )
        second = recalibrate_batchnorm(
            self.generator, torch.device("cpu"), seed=3, batches=2, size=8
        )
        for a, b in zip(first.state_dict().values(), second.state_dict().values()):
            self.assertTrue(torch.equal(a, b))

    def test_per_class_recalibration_uses_only_that_label(self):
        # B2b's first runs calibrated each per-class GAN with random labels, so
        # half the statistics came from the untrained label embedding.
        def running_means(model):
            return torch.cat(
                [b for n, b in model.named_buffers() if n.endswith("running_mean")]
            )

        cpu = torch.device("cpu")
        with torch.no_grad():
            self.generator.label_embedding.weight[0].fill_(5.0)  # AD, "untrained"
        nc_only = recalibrate_batchnorm(
            self.generator, cpu, seed=3, batches=2, size=8, label=1
        )
        mixed = recalibrate_batchnorm(self.generator, cpu, seed=3, batches=2, size=8)
        self.assertFalse(torch.allclose(running_means(nc_only), running_means(mixed)))
        # The NC statistics must not depend on the AD embedding at all.
        with torch.no_grad():
            self.generator.label_embedding.weight[0].fill_(-5.0)
        changed_ad = recalibrate_batchnorm(
            self.generator, cpu, seed=3, batches=2, size=8, label=1
        )
        self.assertTrue(torch.equal(running_means(nc_only), running_means(changed_ad)))

    def save_checkpoint(self, generator, path, step):
        torch.save(
            {
                "generator": generator.state_dict(),
                "resolution": 32,
                "z_dim": 16,
                "width": 8,
                "step": step,
                "diversity_ratio": None,
            },
            path,
        )

    def run_generate(
        self, folder, checkpoint=None, class_checkpoint=(), allow_partial=False
    ):
        args = argparse.Namespace(
            checkpoint=checkpoint,
            class_checkpoint=list(class_checkpoint),
            allow_partial=allow_partial,
            output=folder / "cohort",
            first_seed=37200,
            count=2,
            device="cpu",
        )
        generate(args)
        return json.loads((folder / "cohort" / "report.json").read_text())

    def test_generate_writes_hashed_cohort(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            self.save_checkpoint(self.generator, folder / "final.pt", 7)
            report = self.run_generate(folder, checkpoint=folder / "final.pt")
            self.assertEqual(len(report["records"]), 4)
            for record in report["records"]:
                data = (folder / "cohort" / record["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), record["png_sha256"])

    def test_generate_uses_each_class_checkpoint(self):
        torch.manual_seed(5)
        other = ConditionalGenerator(resolution=32, z_dim=16, width=8)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            self.save_checkpoint(self.generator, folder / "ad.pt", 11)
            self.save_checkpoint(other, folder / "nc.pt", 22)
            report = self.run_generate(
                folder,
                class_checkpoint=[f"AD={folder / 'ad.pt'}", f"NC={folder / 'nc.pt'}"],
            )
            steps = {k: v["checkpoint_step"] for k, v in report["checkpoints"].items()}
            self.assertEqual(steps, {"AD": 11, "NC": 22})
            self.assertEqual(
                [r["label"] for r in report["records"]], ["AD", "AD", "NC", "NC"]
            )
            with self.assertRaises(SystemExit):
                self.run_generate(
                    folder / "again", class_checkpoint=[f"AD={folder / 'ad.pt'}"]
                )
            # B2b: a class without a result is left out only when asked.
            partial = self.run_generate(
                folder / "partial",
                class_checkpoint=[f"NC={folder / 'nc.pt'}"],
                allow_partial=True,
            )
            self.assertEqual([r["label"] for r in partial["records"]], ["NC", "NC"])
            self.assertEqual(set(partial["checkpoints"]), {"NC"})


class ScratchUnetTest(unittest.TestCase):
    def test_size_forward_and_weight_round_trip(self):
        torch.manual_seed(0)
        unet = build_unet(Path("unused"), torch.device("cpu"), "scratch")
        count = sum(p.numel() for p in unet.parameters())
        self.assertAlmostEqual(count / 1e6, 92.2, delta=0.1)
        with torch.no_grad():
            out = unet(
                torch.randn(1, 4, 32, 32), torch.tensor([500]), torch.randn(1, 77, 768)
            ).sample
        self.assertEqual(tuple(out.shape), (1, 4, 32, 32))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scratch.safetensors"
            save_weights(dict(unet.named_parameters()), path)
            torch.manual_seed(1)
            fresh = build_unet(Path("unused"), torch.device("cpu"), "scratch")
            self.assertEqual(load_weights(fresh, path), "unet")
            first = next(iter(fresh.parameters()))
            self.assertTrue(
                torch.allclose(first, next(iter(unet.parameters())).half().float())
            )


if __name__ == "__main__":
    unittest.main()
