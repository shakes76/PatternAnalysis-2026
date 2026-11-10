"""Compare historical sampler output with official pipeline and fixed settings.

API reference: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img
All outputs are private diagnostics, not medical-quality validation.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig
from PIL import Image, ImageDraw
from transformers import CLIPTextModel, CLIPTokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--align-scheduler-precision", action="store_true")
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(3710)
    torch.cuda.manual_seed_all(3710)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    model = Path("models/sd15-smoke")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert saved["step"] == 20000
    assert saved["contract"]["model"] == digest(model / "provenance.json")
    assert saved["contract"]["vae"] == digest("models/sd15-vae/provenance.json")
    tokenizer = CLIPTokenizer.from_pretrained(
        model, subfolder="tokenizer", local_files_only=True
    )
    text = (
        CLIPTextModel.from_pretrained(
            model,
            subfolder="text_encoder",
            variant="fp16",
            local_files_only=True,
            use_safetensors=True,
        )
        .eval()
        .requires_grad_(False)
        .to("cuda", dtype=torch.float32)
    )
    prompts = {
        "AD": "a brain MRI slice, Alzheimer disease",
        "NC": "a brain MRI slice, normal control",
        "empty": "",
    }
    embeddings = {}
    for key, prompt in prompts.items():
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        embeddings[key] = text(tokens.input_ids.to("cuda"))[0]
    vae = (
        AutoencoderKL.from_pretrained(
            "models/sd15-vae",
            subfolder="vae",
            local_files_only=True,
            use_safetensors=True,
        )
        .eval()
        .requires_grad_(False)
        .to("cuda", dtype=torch.float32)
    )
    unet = (
        UNet2DConditionModel.from_pretrained(
            model,
            subfolder="unet",
            variant="fp16",
            local_files_only=True,
            use_safetensors=True,
        )
        .to("cuda", dtype=torch.float32)
        .requires_grad_(False)
    )
    unet.add_adapter(
        LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0.0,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    adapter = {k: p for k, p in unet.named_parameters() if p.requires_grad}
    assert set(adapter) == set(saved["adapter"])
    for key, param in adapter.items():
        param.copy_(saved["adapter"][key])
    unet.eval().requires_grad_(False)
    if args.align_scheduler_precision:
        # Pipeline requests return_dict=False. Match the historical explicit
        # prediction.float() boundary without changing learned weights.
        def scheduler_input_float32(module, inputs, output):
            if not isinstance(output, tuple):
                raise TypeError("Expected pipeline UNet tuple output")
            return tuple(value.float() for value in output)

        unet.register_forward_hook(scheduler_input_float32)
    noise_schedule = DDPMScheduler.from_pretrained(
        model, subfolder="scheduler", local_files_only=True
    )
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_config(noise_schedule.config),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.set_progress_bar_config(disable=True)
    records = []
    parity = []
    started = time.perf_counter()
    for size in (256,) if args.parity_only else (256, 512):
        for guidance in (1.0,) if args.parity_only else (1.0, 7.5):
            grid = Image.new("RGB", (size * 2, (size + 24) * 2), "white")
            draw = ImageDraw.Draw(grid)
            for row, label in enumerate(("AD", "NC")):
                for col, seed in enumerate((37100, 37101)):
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    latent = torch.randn(
                        (1, 4, size // 8, size // 8), generator=generator, device="cuda"
                    )
                    initial_hash = hashlib.sha256(
                        latent.cpu().numpy().tobytes()
                    ).hexdigest()
                    pipeline.scheduler = DDIMScheduler.from_config(
                        noise_schedule.config
                    )
                    torch.cuda.reset_peak_memory_stats()
                    # Only denoising is autocast, as in the historical sampler.
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = pipeline(
                            prompt_embeds=embeddings[label],
                            negative_prompt_embeds=embeddings["empty"],
                            height=size,
                            width=size,
                            num_inference_steps=30,
                            guidance_scale=guidance,
                            eta=0.0,
                            latents=latent.clone(),
                            output_type="latent",
                        ).images
                    decoded = vae.decode(
                        result.float() / vae.config.scaling_factor
                    ).sample
                    if not torch.isfinite(decoded).all():
                        raise RuntimeError("Non-finite pipeline output")
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
                    name = f"official-{size}-cfg{guidance}-{label}-{seed}.png"
                    image.save(args.output / name)
                    grid.paste(image, (col * size, row * (size + 24)))
                    draw.text(
                        (col * size + 4, row * (size + 24) + size + 4),
                        f"{label} seed{seed} cfg{guidance}",
                        fill="black",
                    )
                    if size == 256 and guidance == 1.0:
                        historical_path = (
                            args.checkpoint.parent
                            / f"step-20000-{label}-seed-{seed}.png"
                        )
                        historical = np.asarray(
                            Image.open(historical_path).convert("RGB")
                        )
                        delta = np.abs(
                            pixels.astype(np.int16) - historical.astype(np.int16)
                        )
                        parity.append(
                            {
                                "label": label,
                                "seed": seed,
                                "max_pixel_difference": int(delta.max()),
                                "mean_pixel_difference": float(delta.mean()),
                                "exact_pixels": bool(
                                    np.array_equal(pixels, historical)
                                ),
                                "historical_png_sha256": digest(historical_path),
                            }
                        )
                    records.append(
                        {
                            "file": name,
                            "size": size,
                            "guidance": guidance,
                            "label": label,
                            "seed": seed,
                            "initial_noise_sha256": initial_hash,
                            "png_sha256": digest(args.output / name),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        }
                    )
                    print(json.dumps(records[-1]), flush=True)
            grid.save(args.output / f"grid-{size}-cfg{guidance}.png")
    report = {
        "aligned_scheduler_precision": args.align_scheduler_precision,
        "parity_only": args.parity_only,
        "checkpoint_sha256": digest(args.checkpoint),
        "code_sha256": digest(__file__),
        "prompts": prompts,
        "parity": parity,
        "records": records,
        "scheduler_config": dict(pipeline.scheduler.config),
        "inference_steps": 30,
        "eta": 0.0,
        "seconds": time.perf_counter() - started,
        "torch": str(torch.__version__),
        "scope": "4 historical pairs; 16 fixed factorial samples; pixel comparison is not latent-level proof",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
