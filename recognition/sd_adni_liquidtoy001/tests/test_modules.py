"""CPU tests of modules.py with tiny random models.

They check the mechanics (noise, dropout, EMA, sampler, weight files), not
image quality. The tiny UNet keeps SD1.5's block types and
cross-attention but is a few hundred thousand parameters.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

if importlib.util.find_spec("diffusers") is None:
    raise unittest.SkipTest("needs diffusers and peft (training environment)")

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from modules import (  # noqa: E402
    ExponentialMovingAverage,
    condition_batch,
    configure_trainable,
    ddim_invert,
    diffusion_loss,
    guided_ddim,
    load_weights,
    save_weights,
    slerp,
    training_noise,
)

CONTEXT = 16  # text embedding width of the tiny UNet


def tiny_unet(seed=0):
    torch.manual_seed(seed)
    return UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=CONTEXT,
        attention_head_dim=8,
        norm_num_groups=8,
    )


def tiny_vae():
    torch.manual_seed(1)
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16),
        latent_channels=4,
        norm_num_groups=8,
        sample_size=16,
    )


def sd_schedule():
    """SD1.5's scheduler_config.json, read the way load_noise_schedule reads it."""
    return DDPMScheduler.from_config(
        {
            "_class_name": "PNDMScheduler",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
        }
    )


def tiny_embeddings():
    generator = torch.Generator().manual_seed(2)
    return {
        k: torch.randn((1, 5, CONTEXT), generator=generator)
        for k in ("AD", "NC", "empty")
    }


class NoiseAndConditioningTest(unittest.TestCase):
    def test_offset_adds_a_constant_per_image_and_channel(self):
        latents = torch.zeros(3, 4, 8, 8)
        plain = training_noise(latents, 0.0, torch.Generator().manual_seed(5))
        offset = training_noise(latents, 0.1, torch.Generator().manual_seed(5))
        difference = offset - plain
        # Same first draw, so the difference is exactly the constant term.
        self.assertTrue(
            torch.allclose(difference, difference[:, :, :1, :1].expand_as(difference))
        )
        self.assertGreater(difference.abs().max().item(), 0)

    def test_dropout_extremes(self):
        embeddings = tiny_embeddings()
        labels = torch.tensor([0, 1, 1, 0])
        kept = condition_batch(labels, embeddings, 0.0)
        self.assertTrue(torch.equal(kept[0], embeddings["AD"][0]))
        self.assertTrue(torch.equal(kept[1], embeddings["NC"][0]))
        dropped = condition_batch(
            labels, embeddings, 1.0, torch.Generator().manual_seed(0)
        )
        self.assertTrue(torch.equal(dropped, embeddings["empty"].expand(4, -1, -1)))


class AveragingTest(unittest.TestCase):
    def test_warmup_decay_and_temporary_swap(self):
        parameter = torch.nn.Parameter(torch.zeros(3))
        ema = ExponentialMovingAverage({"w": parameter}, decay=0.999)
        with torch.no_grad():
            parameter.fill_(1.0)
        ema.update({"w": parameter})
        # First update uses decay min(0.999, 2 / 11), so the shadow moves 9/11.
        self.assertTrue(torch.allclose(ema.shadow["w"], torch.full((3,), 9 / 11)))
        with ema.applied_to({"w": parameter}):
            self.assertTrue(torch.allclose(parameter, torch.full((3,), 9 / 11)))
        self.assertTrue(torch.equal(parameter, torch.ones(3)))


class TrainableTest(unittest.TestCase):
    def test_full_unet_trains_everything(self):
        unet = tiny_unet()
        trainable = configure_trainable(unet, "unet")
        self.assertEqual(len(trainable), len(list(unet.parameters())))

    def test_lora_only_trains_adapters_and_gradients_respect_that(self):
        unet = tiny_unet()
        trainable = configure_trainable(unet, "lora", lora_rank=2)
        self.assertTrue(all("lora_" in name for name in trainable))
        latents = torch.randn(2, 4, 8, 8)
        conditioning = condition_batch(torch.tensor([0, 1]), tiny_embeddings(), 0.0)
        loss = diffusion_loss(
            unet, latents, conditioning, sd_schedule(), torch.randn_like(latents)
        )
        loss.backward()
        frozen = [p for n, p in unet.named_parameters() if n not in trainable]
        self.assertTrue(all(p.grad is None for p in frozen))
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in trainable.values()
            )
        )


class SamplerTest(unittest.TestCase):
    def test_guided_ddim_matches_the_official_pipeline(self):
        unet, vae, schedule = tiny_unet().eval(), tiny_vae().eval(), sd_schedule()
        embeddings = tiny_embeddings()
        start = torch.randn((1, 4, 8, 8), generator=torch.Generator().manual_seed(9))
        ours = guided_ddim(
            unet,
            schedule.config,
            start,
            embeddings["AD"],
            embeddings["empty"],
            7.5,
            steps=5,
        )
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            unet=unet,
            scheduler=DDIMScheduler.from_config(schedule.config),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipeline.set_progress_bar_config(disable=True)
        official = pipeline(
            prompt_embeds=embeddings["AD"],
            negative_prompt_embeds=embeddings["empty"],
            height=16,
            width=16,
            num_inference_steps=5,
            guidance_scale=7.5,
            eta=0.0,
            latents=start.clone(),
            output_type="latent",
        ).images
        self.assertTrue(
            torch.allclose(ours, official, atol=1e-5), (ours - official).abs().max()
        )

    def test_a_batch_gives_the_same_latents_as_one_at_a_time(self):
        unet, schedule = tiny_unet().eval(), sd_schedule()
        embeddings = tiny_embeddings()
        starts = torch.randn((3, 4, 8, 8), generator=torch.Generator().manual_seed(5))

        def sample(start):
            return guided_ddim(
                unet,
                schedule.config,
                start,
                embeddings["NC"],
                embeddings["empty"],
                7.5,
                steps=4,
            )

        together = sample(starts)
        alone = torch.cat([sample(s.unsqueeze(0)) for s in starts])
        # Only float rounding differs between batch shapes (about 1e-6 per
        # forward pass, amplified by guidance 7.5 over the steps).
        self.assertTrue(torch.allclose(together, alone, atol=1e-4))

    def test_inversion_reproduces_its_input_when_the_noise_ignores_the_image(self):
        # With a noise prediction that depends on the timestep only, DDIM
        # inversion is exact, so sampling from the inverted noise must give the
        # input back. A wrong timestep order or alpha would break this.
        pattern = torch.randn((1, 4, 8, 8), generator=torch.Generator().manual_seed(3))

        class TimestepOnly(torch.nn.Module):
            def forward(self, latents, timestep, context):
                scale = 1 + float(timestep) / 1000
                return SimpleNamespace(sample=(scale * pattern).expand_as(latents))

        stub, schedule = TimestepOnly(), sd_schedule()
        clean = torch.randn((2, 4, 8, 8), generator=torch.Generator().manual_seed(4))
        context = tiny_embeddings()["AD"]
        noise = ddim_invert(stub, schedule.config, clean, context, steps=10)
        back = guided_ddim(
            stub, schedule.config, noise, context, context, 1.0, steps=10
        )
        self.assertTrue(
            torch.allclose(back, clean, atol=1e-4), (back - clean).abs().max()
        )
        self.assertGreater((noise - clean).abs().max().item(), 0.1)

    def test_slerp_endpoints_and_norm(self):
        generator = torch.Generator().manual_seed(6)
        a = torch.randn((1, 4, 32, 32), generator=generator)
        b = torch.randn((1, 4, 32, 32), generator=generator)
        self.assertTrue(torch.allclose(slerp(a, b, 0.0), a, atol=1e-5))
        self.assertTrue(torch.allclose(slerp(a, b, 1.0), b, atol=1e-5))
        middle = slerp(a, b, 0.5)
        linear = 0.5 * (a + b)
        # Spherical mixing keeps a Gaussian sample's norm; linear mixing shrinks it.
        self.assertAlmostEqual(middle.norm().item() / a.norm().item(), 1.0, delta=0.05)
        self.assertLess(linear.norm().item() / a.norm().item(), 0.8)


class WeightFileTest(unittest.TestCase):
    def test_partial_full_unet_file_is_rejected(self):
        unet = tiny_unet()
        weights = dict(list(unet.named_parameters())[:3])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "partial.safetensors"
            save_weights(weights, path)
            with self.assertRaises(KeyError):
                load_weights(tiny_unet(), path)


if __name__ == "__main__":
    unittest.main()
