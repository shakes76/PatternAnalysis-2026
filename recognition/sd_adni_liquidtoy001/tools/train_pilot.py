"""Bounded 200-step SD LoRA pilot with fixed paired generation probes.

Adapter setup follows https://huggingface.co/docs/diffusers/training/lora .
This diagnostic is not the final general-purpose coursework trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from pilot_sampling import sample_grid
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import ADNISliceDataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_indices(rows):
    selected = []
    for label in ("AD", "NC"):
        patients = sorted(
            {r["patient_id"] for r in rows if r["class_name"] == label},
            key=lambda p: hashlib.sha256(f"3710:{p}".encode()).hexdigest(),
        )[:16]
        for patient in patients:
            candidates = sorted(
                [i for i, r in enumerate(rows) if r["patient_id"] == patient],
                key=lambda i: (int(rows[i]["scan_id"]), int(rows[i]["slice_index"])),
            )
            selected.append(candidates[len(candidates) // 2])
    if len(selected) != 32:
        raise ValueError("Need sixteen training patients per class")
    return selected


def validate_contract(saved, expected):
    if saved != expected:
        raise ValueError("Checkpoint contract mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, choices=(100, 200), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA allocation required")
    torch.manual_seed(3710)
    torch.cuda.manual_seed_all(3710)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    model = Path("models/sd15-smoke")
    manifest = Path("data/manifest-seed3710.csv")
    dataset = ADNISliceDataset(
        Path("/home/groups/comp3710/ADNI"), manifest, "train", 256
    )
    indices = training_indices(dataset.rows)
    order_generator = torch.Generator().manual_seed(3710)
    order = []
    while len(order) < 200:
        order.extend(
            indices[i]
            for i in torch.randperm(len(indices), generator=order_generator).tolist()
        )
    order = order[:200]
    contract = {
        "schema": 2,
        "experiment": "pilot-32patients-200steps",
        "sampling_code": digest("tools/pilot_sampling.py"),
        "order": order,
        "seed": 3710,
        "image_size": 256,
        "rank": 4,
        "lr": 1e-4,
        "batch": 1,
        "precision": "fp32 weights, bf16 autocast",
        "manifest": digest(manifest),
        "model": digest(model / "provenance.json"),
        "vae": digest("models/sd15-vae/provenance.json"),
        "code": digest(__file__),
        "dataset_code": digest("dataset.py"),
        "indices": indices,
        "torch": str(torch.__version__),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = CLIPTokenizer.from_pretrained(
        model, subfolder="tokenizer", local_files_only=True
    )
    text_encoder = (
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
    embeddings = {}
    with torch.no_grad():
        for label, description in (
            ("AD", "Alzheimer disease"),
            ("NC", "normal control"),
        ):
            tokens = tokenizer(
                f"a brain MRI slice, {description}",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            embeddings[label] = text_encoder(tokens.input_ids.to("cuda"))[0].detach()
    del text_encoder
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
    unet.train()
    trainable = {k: p for k, p in unet.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in k for k in trainable):
        raise RuntimeError("Unexpected trainable parameter set")
    initial = {k: p.detach().cpu().clone() for k, p in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-4, foreach=False)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    noise_schedule = DDPMScheduler.from_pretrained(
        model, subfolder="scheduler", local_files_only=True
    )
    step, traces = 0, []
    if args.resume:
        # Only locally generated checkpoints from this diagnostic are accepted.
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        validate_contract(saved["contract"], contract)
        if set(saved["adapter"]) != set(trainable):
            raise ValueError("Adapter parameter names differ")
        with torch.no_grad():
            for name, param in trainable.items():
                param.copy_(saved["adapter"][name])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["scheduler"])
        step, traces = saved["step"], saved["traces"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if not 0 <= step < args.steps:
        raise ValueError("Invalid resume step")
    if not args.resume:
        sample_grid(unet, vae, embeddings, noise_schedule, args.output, 0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    start_step = step
    for step in range(start_step, args.steps):
        item = dataset[order[step]]
        if (
            digest(dataset.root / item["relative_path"])
            != dataset.rows[order[step]]["sha256"]
        ):
            raise ValueError("Source image hash changed")
        pixels = item["pixel_values"].unsqueeze(0).to("cuda")
        with torch.no_grad():
            latent = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latent)
        timestep = torch.randint(
            0, noise_schedule.config.num_train_timesteps, (1,), device="cuda"
        )
        noisy = noise_schedule.add_noise(latent, noise, timestep)
        prediction_type = noise_schedule.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = noise_schedule.get_velocity(latent, noise, timestep)
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = unet(noisy, timestep, embeddings[item["class_name"]]).sample
            loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            list(trainable.values()), 1.0, error_if_nonfinite=True
        )
        if not norm > 0:
            raise RuntimeError("Zero gradient")
        optimizer.step()
        schedule.step()
        traces.append(
            {
                "step": step + 1,
                "index": order[step],
                "timestep": timestep.item(),
                "loss": loss.item(),
                "gradient_norm": norm.item(),
                "noise_sha256": hashlib.sha256(
                    noise.cpu().numpy().tobytes()
                ).hexdigest(),
            }
        )
        print(json.dumps(traces[-1]), flush=True)
    torch.cuda.synchronize()
    adapter = {k: p.detach().cpu().clone() for k, p in trainable.items()}
    changed = max((adapter[k] - initial[k]).abs().max().item() for k in adapter)
    if not changed > 0 or not all(torch.isfinite(p).all() for p in adapter.values()):
        raise RuntimeError("Adapter failed to update finitely")
    state = {
        "contract": contract,
        "adapter": adapter,
        "optimizer": optimizer.state_dict(),
        "scheduler": schedule.state_dict(),
        "step": args.steps,
        "traces": traces,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    checkpoint = args.output / "checkpoint.pt"
    temporary = args.output / "checkpoint.tmp"
    torch.save(state, temporary)
    os.replace(temporary, checkpoint)
    report = {
        "steps": args.steps,
        "resumed": args.resume is not None,
        "traces": traces,
        "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "max_adapter_change": changed,
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "gpu": torch.cuda.get_device_name(),
        "contract": contract,
    }
    sample_grid(unet, vae, embeddings, noise_schedule, args.output, args.steps)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
