"""Sampling speed and memory of each model (GPU job); the cost measures of
docs/final-evaluation-protocol.md.

For each model and batch size (1 and 16): 3 warm-up batches, then 20 timed
batches with CUDA events; the median gives seconds per image. Diffusion models
use the cohort setting (DDIM 30, guidance 7.5, decoding included), timed twice:
with the deterministic settings the cohorts were generated with, and with the
fast settings a deployment would use (TF32, cuDNN autotuning, fused attention).

    python tools/profile_sampling.py --e2 PATH --e3 PATH --b3 PATH \
        --b1 PATH --b2 PATH --output outputs/final-profile-<job>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules import (  # noqa: E402
    build_unet,
    decode,
    encode_prompts,
    guided_ddim,
    load_noise_schedule,
    load_vae,
    load_weights,
)

WARMUP, TIMED, BATCHES = 3, 20, (1, 16)


def settings(deterministic: bool) -> None:
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cuda.enable_flash_sdp(not deterministic)
    torch.backends.cuda.enable_mem_efficient_sdp(not deterministic)


@torch.no_grad()
def time_batches(run, batch: int) -> dict:
    for _ in range(WARMUP):
        run(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    seconds = []
    for _ in range(TIMED):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        run(batch)
        end.record()
        torch.cuda.synchronize()
        seconds.append(start.elapsed_time(end) / 1000)
    median = statistics.median(seconds)
    return {
        "batch": batch,
        "seconds_per_batch_median": median,
        "seconds_per_image": median / batch,
        "images_per_second": batch / median,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def diffusion_runner(unet, vae, schedule, embeddings, device):
    def run(batch):
        start = torch.randn((batch, 4, 32, 32), device=device)
        latents = guided_ddim(
            unet, schedule.config, start, embeddings["AD"], embeddings["empty"], 7.5
        )
        return decode(vae, latents)

    return run


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--e2", type=Path, required=True, help="full-UNet weights")
    parser.add_argument("--e3", type=Path, required=True, help="adapter weights")
    parser.add_argument("--b3", type=Path, required=True, help="scratch-UNet weights")
    parser.add_argument("--b1", type=Path, required=True, help="VAE checkpoint")
    parser.add_argument("--b2", type=Path, required=True, help="GAN checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/sd15-smoke"))
    parser.add_argument("--vae-dir", type=Path, default=Path("models/sd15-vae"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["e2", "e3", "b3", "b1", "b2"],
        help="a subset, so the profile can be split across jobs",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    started = time.perf_counter()
    embeddings = encode_prompts(args.model_dir, device)
    vae = load_vae(args.vae_dir, device)
    schedule = load_noise_schedule(args.model_dir)
    results = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}

    def save():
        # Written after every model, so a job that hits its time limit keeps
        # the models it finished.
        results["seconds"] = time.perf_counter() - started
        (args.output / "profile.json").write_text(json.dumps(results, indent=2) + "\n")

    for name, weights, init in (
        ("e2", args.e2, "pretrained"),
        ("e3", args.e3, "pretrained"),
        ("b3", args.b3, "scratch"),
    ):
        if name not in args.models:
            continue
        unet = build_unet(args.model_dir, device, init)
        mode = load_weights(unet, weights)
        unet.eval().requires_grad_(False)
        run = diffusion_runner(unet, vae, schedule, embeddings, device)
        results[name] = {"weights_mode": mode}
        for label, deterministic in (("as_generated", True), ("fast", False)):
            settings(deterministic)
            results[name][label] = [time_batches(run, b) for b in BATCHES]
        del unet, run
        torch.cuda.empty_cache()
        print(name, json.dumps(results[name]), flush=True)
        save()

    settings(True)
    from baseline_gan import load_generator
    from baseline_vae import SmallVAE

    b1 = SmallVAE().to(device).eval()
    b1.load_state_dict(
        torch.load(args.b1, map_location="cpu", weights_only=True)["model"]
    )
    b2, state = load_generator(args.b2, device)
    runners = {
        "b1": lambda batch: b1.decode(
            torch.randn((batch, 64), device=device),
            torch.zeros(batch, dtype=torch.long, device=device),
        ),
        "b2": lambda batch: b2(
            torch.randn((batch, state["z_dim"]), device=device),
            torch.zeros(batch, dtype=torch.long, device=device),
        ),
    }
    for name, run in runners.items():
        if name not in args.models:
            continue
        results[name] = {"as_generated": [time_batches(run, b) for b in BATCHES]}
        print(name, json.dumps(results[name]), flush=True)
        save()


if __name__ == "__main__":
    main()
