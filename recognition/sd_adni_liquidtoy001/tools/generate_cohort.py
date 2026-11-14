"""Compare historical sampler output with official pipeline and fixed settings.

API reference: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img
All outputs are private diagnostics, not medical-quality validation.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

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
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--guidance", type=float, nargs="+", default=[7.5])
    parser.add_argument("--interpolate", action="store_true")
    args = parser.parse_args()
    if len(set(args.guidance)) != len(args.guidance) or any(
        not 1.0 <= value <= 20.0 for value in args.guidance
    ):
        parser.error("Guidance values must be distinct and between1 and20")
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
    if not args.base:
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
    started = time.perf_counter()
    for size in (256,):
        for guidance in args.guidance:
            columns = 5 if args.interpolate else 8
            label_rows = 2 if args.interpolate else 4
            grid = Image.new(
                "RGB", (size * columns, (size + 24) * label_rows * 2), "white"
            )
            draw = ImageDraw.Draw(grid)
            for row, label in enumerate(("AD", "NC")):
                samples = (
                    [
                        (a, b, t)
                        for a, b in ((37200, 37201), (37216, 37217))
                        for t in (0.0, 0.25, 0.5, 0.75, 1.0)
                    ]
                    if args.interpolate
                    else [(s, None, 0.0) for s in range(37200, 37232)]
                )
                for col, (seed, end_seed, alpha) in enumerate(samples):
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    latent = torch.randn(
                        (1, 4, size // 8, size // 8), generator=generator, device="cuda"
                    )
                    if end_seed is not None:
                        end = torch.randn(
                            latent.shape,
                            device="cuda",
                            generator=torch.Generator(device="cuda").manual_seed(
                                end_seed
                            ),
                        )
                        # Preserve endpoints exactly. Interior variance changes under
                        # linear interpolation and is reported as a limitation.
                        latent = (
                            latent
                            if alpha == 0
                            else end
                            if alpha == 1
                            else (1 - alpha) * latent + alpha * end
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
                    if end_seed is not None:
                        name = f"interp-{size}-cfg{guidance}-{label}-{seed}-{end_seed}-a{alpha}.png"
                    image.save(args.output / name)
                    grid.paste(
                        image,
                        (
                            (col % columns) * size,
                            (row * label_rows + col // columns) * (size + 24),
                        ),
                    )
                    draw.text(
                        (
                            (col % columns) * size + 4,
                            (row * label_rows + col // columns) * (size + 24)
                            + size
                            + 4,
                        ),
                        (
                            f"{label} {seed}->{end_seed} a{alpha}"
                            if end_seed is not None
                            else f"{label} seed{seed} cfg{guidance}"
                        ),
                        fill="black",
                    )
                    records.append(
                        {
                            "file": name,
                            "size": size,
                            "guidance": guidance,
                            "label": label,
                            "seed": seed,
                            "end_seed": end_seed,
                            "alpha": alpha,
                            "initial_noise_norm": latent.norm().item(),
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
        "base_model": args.base,
        "records": records,
        "scheduler_config": dict(pipeline.scheduler.config),
        "inference_steps": 30,
        "eta": 0.0,
        "seconds": time.perf_counter() - started,
        "torch": str(torch.__version__),
        "guidance_values": args.guidance,
        "interpolation": args.interpolate,
        "scope": f"{len(records)} predeclared images; no test evaluation",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
