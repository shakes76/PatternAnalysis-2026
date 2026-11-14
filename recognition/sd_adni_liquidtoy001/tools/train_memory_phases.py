"""Peak GPU memory of each phase of train.py (post-hoc check, GPU job).

tools/lora_memory.py showed that one LoRA training step peaks at 9.2 GiB,
so E3's recorded 36.8 GiB must arise elsewhere in train.py. This wrapper runs
train.py unchanged, with the arguments given after "--", and records the peak
allocated memory inside each phase it wraps:

- sample_grid      (the fixed-seed grid at epoch 0 and after each epoch)
- validation_loss  (live and EMA weights)
- forward          (diffusion_loss during training)
- backward         (loss.backward)
- clip_grad_norm, optimizer_step, ema_update

The peak between phases is recorded as "other". Each phase resets the CUDA
peak counter, so train.py's own peak_allocated_gib is not meaningful here;
the phase peaks are written to --report.

    python tools/train_memory_phases.py --report outputs/phases-lora.json -- \
        --output $TMPDIR/run --trainable lora --lora-rank 4 --lr 1e-4 --max-steps 50
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modules  # noqa: E402
import train  # noqa: E402

GIB = 2**30
peaks = collections.defaultdict(float)
calls = collections.Counter()
depth = 0


def tracked(name, function):
    """Wrap function so the CUDA peak inside its outermost call is credited to name."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        global depth
        if depth or not torch.cuda.is_available():
            return function(*args, **kwargs)
        peaks["other"] = max(peaks["other"], torch.cuda.max_memory_allocated() / GIB)
        torch.cuda.reset_peak_memory_stats()
        depth += 1
        try:
            return function(*args, **kwargs)
        finally:
            depth -= 1
            torch.cuda.synchronize()
            peaks[name] = max(peaks[name], torch.cuda.max_memory_allocated() / GIB)
            calls[name] += 1
            torch.cuda.reset_peak_memory_stats()

    return wrapper


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--report", type=Path, required=True)
    args, rest = parser.parse_known_args()
    if rest[:1] == ["--"]:
        rest = rest[1:]

    train.sample_grid = tracked("sample_grid", train.sample_grid)
    train.validation_loss = tracked("validation_loss", train.validation_loss)
    train.diffusion_loss = tracked("forward", train.diffusion_loss)
    torch.Tensor.backward = tracked("backward", torch.Tensor.backward)
    torch.nn.utils.clip_grad_norm_ = tracked(
        "clip_grad_norm", torch.nn.utils.clip_grad_norm_
    )
    torch.optim.AdamW.step = tracked("optimizer_step", torch.optim.AdamW.step)
    modules.ExponentialMovingAverage.update = tracked(
        "ema_update", modules.ExponentialMovingAverage.update
    )

    sys.argv = ["train.py", *rest]
    train.main()
    peaks["other"] = max(peaks["other"], torch.cuda.max_memory_allocated() / GIB)
    report = {
        "train_args": rest,
        "peak_gib_by_phase": dict(sorted(peaks.items(), key=lambda kv: -kv[1])),
        "calls": dict(calls),
        "gpu": torch.cuda.get_device_name(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
