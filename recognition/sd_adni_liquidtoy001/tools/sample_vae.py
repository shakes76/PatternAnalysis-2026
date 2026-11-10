"""Draw a larger cohort from the trained B1 conditional VAE.

Uses baseline_vae.generate, so seeds 37200-37231 reproduce the original B1
cohort and later seeds extend it.

    python tools/sample_vae.py --checkpoint ../runs/baseline128-20261005/outputs/baseline-vae-632570/epoch-5.pt \
        --output outputs/final-b1 --count 512
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_vae import SmallVAE, generate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32, help="seeds per class")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        # The settings B1's own generation ran with.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    model = SmallVAE().to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    records = generate(model, args.output, count=args.count, device=args.device)
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "epoch": state.get("epoch"),
        "count_per_class": args.count,
        "records": records,
        "scope": f"{len(records)} predeclared images; no test-split data read",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))


if __name__ == "__main__":
    main()
