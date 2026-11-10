"""Uncurated, paired DDIM probes; no clinical or distribution-quality claim.

Scheduler API: https://huggingface.co/docs/diffusers/api/schedulers/ddim
"""

import hashlib
import json
import time

import torch
from diffusers import DDIMScheduler
from PIL import Image, ImageDraw


@torch.no_grad()
def sample_grid(unet, vae, embeddings, noise_schedule, output, step):
    """Preserve training RNG and mode; use identical latent seeds at every stage."""
    was_training = unet.training
    unet.eval()
    started = time.perf_counter()
    records = []
    grid = Image.new("RGB", (4 * 256, 2 * 280), "white")
    draw = ImageDraw.Draw(grid)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for row, label in enumerate(("AD", "NC")):
                for col, seed in enumerate((37100, 37101, 37102, 37103)):
                    scheduler = DDIMScheduler.from_config(noise_schedule.config)
                    scheduler.set_timesteps(30, device="cuda")
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    latent = torch.randn(
                        (1, 4, 32, 32), generator=generator, device="cuda"
                    )
                    noise_hash = hashlib.sha256(
                        latent.cpu().numpy().tobytes()
                    ).hexdigest()
                    latent *= scheduler.init_noise_sigma
                    for timestep in scheduler.timesteps:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            prediction = unet(
                                scheduler.scale_model_input(latent, timestep),
                                timestep,
                                embeddings[label],
                            ).sample
                        latent = scheduler.step(
                            prediction.float(), timestep, latent, eta=0
                        ).prev_sample
                    decoded = vae.decode(latent / vae.config.scaling_factor).sample
                    if not torch.isfinite(decoded).all():
                        raise RuntimeError("Non-finite generated image")
                    pixels = (
                        (decoded / 2 + 0.5)
                        .clamp(0, 1)[0]
                        .permute(1, 2, 0)
                        .mul(255)
                        .round()
                        .byte()
                        .cpu()
                        .numpy()
                    )
                    image = Image.fromarray(pixels)
                    name = f"step-{step:03d}-{label}-seed-{seed}.png"
                    image.save(output / name)
                    grid.paste(image, (col * 256, row * 280))
                    draw.text(
                        (col * 256 + 5, row * 280 + 258),
                        f"{label} | seed {seed} | step {step}",
                        fill="black",
                    )
                    records.append(
                        {
                            "file": name,
                            "label": label,
                            "seed": seed,
                            "initial_noise_sha256": noise_hash,
                            "png_sha256": hashlib.sha256(
                                (output / name).read_bytes()
                            ).hexdigest(),
                        }
                    )
    finally:
        unet.train(was_training)
    grid.save(output / f"grid-{step:03d}.png")
    (output / f"sampling-{step:03d}.json").write_text(
        json.dumps(
            {
                "step": step,
                "scheduler": "DDIM",
                "scheduler_config": dict(scheduler.config),
                "inference_steps": 30,
                "eta": 0,
                "guidance_scale": 1.0,
                "image_size": 256,
                "seconds": time.perf_counter() - started,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
