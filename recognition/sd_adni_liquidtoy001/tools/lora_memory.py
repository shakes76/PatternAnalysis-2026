"""Where does E3's training memory go? (post-hoc check, GPU job)

E3 trains a 0.8 M-parameter LoRA adapter yet peaked at 36.8 GiB, above full
fine-tuning's 23.8 GiB (docs/experiments.md, 2026-10-16). This script repeats
one training step of each configuration on random latents of the real shapes
(batch 16, 4 x 32 x 32 latents, 77 x 768 prompt embeddings) with the UNet,
loss, bfloat16 autocast and AdamW of the run, and reports:

- peak allocated memory over a forward, backward and optimiser step;
- memory held at the end of the forward pass, i.e. what autograd saved;
- the saved tensors by dtype, and the largest saved shapes;
- the attention processor classes in use.

Variants (one per process, so peaks do not mix):

- full:          every UNet weight trained (E2).
- lora:          rank-4 adapter on the attention projections (E3 as trained).
- lora-bf16base: the same adapter, with the frozen UNet stored in bfloat16.
- frozen:        no trainable weight; gradients flow only to the input
                 latents, as a floor for the activations a frozen UNet keeps.

Data content does not change memory, so no ADNI image is read.

Two peaks are reported: over the first step, which includes cuDNN's
autotuning trials as train.py's peak does, and over the third step, after
the algorithms are chosen. --no-benchmark turns autotuning off, to test
whether its trials explain the recorded peaks.

    python tools/lora_memory.py --variant lora --output outputs/lora-memory-<job>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import (  # noqa: E402
    autocast,
    configure_trainable,
    load_noise_schedule,
    load_unet,
)

VARIANTS = ("full", "lora", "lora-bf16base", "frozen")
GIB = 2**30


def build(variant: str, model_dir: Path, device: torch.device, rank: int):
    """UNet and the dict of trained parameters for one variant."""
    unet = load_unet(model_dir, device)
    if variant == "full":
        parameters = configure_trainable(unet, "unet")
    elif variant in ("lora", "lora-bf16base"):
        if variant == "lora-bf16base":
            unet.to(torch.bfloat16)
        parameters = configure_trainable(unet, "lora", rank)
        for p in parameters.values():
            p.data = p.data.float()  # adapter weights stay float32, as in E3
    else:
        unet.requires_grad_(False)
        parameters = {}
    unet.train()
    return unet, parameters


class SavedTensorCensus:
    """Records every tensor autograd saves for backward, counted once per storage."""

    def __init__(self):
        self.seen = {}

    def pack(self, tensor):
        try:
            key = tensor.untyped_storage().data_ptr()
            size = tensor.untyped_storage().nbytes()
        except (RuntimeError, NotImplementedError):
            return tensor
        if key not in self.seen or self.seen[key][2] < size:
            self.seen[key] = (str(tensor.dtype), tuple(tensor.shape), size)
        return tensor

    @staticmethod
    def unpack(tensor):
        return tensor

    def summary(self, top: int = 12) -> dict:
        by_dtype = collections.Counter()
        by_shape = collections.Counter()
        count_shape = collections.Counter()
        for dtype, shape, size in self.seen.values():
            by_dtype[dtype] += size
            by_shape[(dtype, shape)] += size
            count_shape[(dtype, shape)] += 1
        return {
            "saved_gib_by_dtype": {k: v / GIB for k, v in by_dtype.most_common()},
            "saved_gib_total": sum(by_dtype.values()) / GIB,
            "largest_saved_shapes": [
                {
                    "dtype": dtype,
                    "shape": list(shape),
                    "count": count_shape[(dtype, shape)],
                    "gib": size / GIB,
                }
                for (dtype, shape), size in by_shape.most_common(top)
            ],
        }


def step(unet, parameters, optimizer, schedule, batch, device, census=None):
    """One training step on random data; returns memory after the forward pass."""
    latents = torch.randn(batch, 4, 32, 32, device=device)
    conditioning = torch.randn(batch, 77, 768, device=device)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, 1000, (batch,), device=device)
    noisy = schedule.add_noise(latents, noise, timesteps)
    if not parameters:
        noisy.requires_grad_(True)
    hooks = (
        torch.autograd.graph.saved_tensors_hooks(census.pack, census.unpack)
        if census
        else torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t)
    )
    with hooks, autocast(device):
        prediction = unet(noisy, timesteps, conditioning).sample
    loss = F.mse_loss(prediction.float(), noise)
    after_forward = torch.cuda.memory_allocated(device)
    loss.backward()
    if optimizer:
        torch.nn.utils.clip_grad_norm_(parameters.values(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return after_forward


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--model-dir", type=Path, default=Path("models/sd15-smoke"))
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    # The training run's throughput settings.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = not args.no_benchmark
    torch.manual_seed(3710)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    unet, parameters = build(args.variant, args.model_dir, device, args.lora_rank)
    schedule = load_noise_schedule(args.model_dir)
    optimizer = (
        torch.optim.AdamW(parameters.values(), lr=1e-5, fused=True)
        if parameters
        else None
    )
    weights_gib = torch.cuda.memory_allocated(device) / GIB
    torch.cuda.reset_peak_memory_stats(device)
    first = step(unet, parameters, optimizer, schedule, args.batch_size, device)
    torch.cuda.synchronize()
    peak_first_step = torch.cuda.max_memory_allocated(device) / GIB
    del first
    for _ in range(1):  # warm-up: optimiser state and cuDNN autotuning
        step(unet, parameters, optimizer, schedule, args.batch_size, device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    census = SavedTensorCensus()
    after_forward = step(
        unet, parameters, optimizer, schedule, args.batch_size, device, census
    )
    torch.cuda.synchronize()
    report = {
        "variant": args.variant,
        "batch_size": args.batch_size,
        "trainable_parameters": sum(p.numel() for p in parameters.values()),
        "weights_gib": weights_gib,
        "resident_before_step_gib": before / GIB,
        "held_after_forward_gib": (after_forward - before) / GIB,
        "peak_step_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "peak_first_step_gib": peak_first_step,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        **census.summary(),
        "attention_processors": dict(
            collections.Counter(
                type(p).__name__ for p in unet.attn_processors.values()
            )
        ),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "seconds": time.perf_counter() - started,
    }
    tag = "-nobench" if args.no_benchmark else ""
    path = args.output / f"{args.variant}{tag}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
