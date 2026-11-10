"""Export E0's rank-4 adapter to the weight file predict.py loads.

E0 is Codex's 20,000-step LoRA run (job 629931). Its checkpoint stores the
adapter under the same parameter names that modules.add_lora creates, so the
values are copied unchanged, in float32 as trained.

    python tools/export_e0_adapter.py \
        --checkpoint ../runs/full20k-20261004/outputs/full-629931/step-20000/checkpoint.pt \
        --output outputs/e0-adapter/lora-e0-step-20000.safetensors
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import save_weights  # noqa: E402


def export(checkpoint: Path, output: Path) -> dict:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    adapter = saved["adapter"]
    if not adapter or not all("lora_" in name for name in adapter):
        raise ValueError("Checkpoint holds no LoRA adapter")
    save_weights(adapter, output, dtype=torch.float32)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "step": saved.get("step"),
        "tensors": len(adapter),
        "parameters": sum(t.numel() for t in adapter.values()),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = export(args.checkpoint, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
