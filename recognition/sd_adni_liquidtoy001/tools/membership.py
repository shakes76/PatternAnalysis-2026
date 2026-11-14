"""Loss-based membership inference for the final evaluation (GPU job).

Measure M3 of docs/final-evaluation-protocol.md: does a diffusion model fit
the slices it trained on better than slices of unseen patients? Members are
1,024 training slices the models trained on (512 per class, the two defective
scans excluded); non-members are 1,024 held-out slices (test in the final run,
validation in the rehearsal). Each image's score is its noise-prediction loss
with the true-class prompt on the VAE posterior mean:

- primary: the mean over t = 100, 200, 300 with 4 noise draws each;
- secondary: one draw at each t from 50 to 950 in steps of 50.

Noise depends only on the set, the image's position and the draw, so every
model sees the same draws. SD (no weights) is the control. The loss-threshold
attack follows Carlini et al. (USENIX Security 2023) and Duan et al. (ICML 2023).

    python tools/membership.py --model e2 --weights PATH --mode rehearsal \
        --exclude-scans ../runs/e2-20261013/outputs/e2-656467/private-excluded-scans.json \
        --output outputs/final-mia-e2-<job>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ADNISliceDataset  # noqa: E402
from final_common import ADNI_ROOT, MANIFEST, balanced_sample, roles  # noqa: E402
from modules import (  # noqa: E402
    LABELS,
    autocast,
    build_unet,
    encode_prompts,
    load_noise_schedule,
    load_vae,
    load_weights,
)

PRIMARY_TIMESTEPS = (100, 200, 300)
PRIMARY_DRAWS = 4
SECONDARY_TIMESTEPS = tuple(range(50, 1000, 50))
PER_CLASS = 512
NOISE_SEED = {"member": 10_000_000, "nonmember": 20_000_000}


def member_rows(rows: list[dict], excluded_scans: set[str]) -> list[int]:
    """Training rows the models trained on: the defective scans are removed."""
    return [i for i, r in enumerate(rows) if r["scan_id"] not in excluded_scans]


def fixed_noise(kind: str, position: int, draw: int, shape, device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(
        NOISE_SEED[kind] + 100 * position + draw
    )
    return torch.randn(shape, generator=generator, device=device)


@torch.no_grad()
def per_image_loss(unet, latents, conditioning, schedule, noise, timestep):
    timesteps = torch.full((len(latents),), timestep, device=latents.device)
    noisy = schedule.add_noise(latents, noise, timesteps)
    with autocast(latents.device):
        prediction = unet(noisy, timesteps, conditioning).sample
    return F.mse_loss(prediction.float(), noise, reduction="none").mean(dim=(1, 2, 3))


@torch.no_grad()
def score_set(kind, dataset, indices, unet, vae, embeddings, schedule, args):
    """Primary scores [N] and per-timestep scores [N, len(SECONDARY_TIMESTEPS)]."""
    device = torch.device(args.device)
    primary, secondary = [], []
    for start in range(0, len(indices), args.batch_size):
        part = indices[start : start + args.batch_size]
        items = [dataset[i] for i in part]
        pixels = torch.stack([x["pixel_values"] for x in items]).to(device)
        latents = vae.encode(pixels).latent_dist.mean * vae.config.scaling_factor
        conditioning = torch.cat([embeddings[x["class_name"]] for x in items])
        positions = range(start, start + len(part))

        def noise(draw):
            return torch.cat(
                [fixed_noise(kind, p, draw, (1, 4, 32, 32), device) for p in positions]
            )

        losses = [
            per_image_loss(unet, latents, conditioning, schedule, noise(d), t)
            for t in PRIMARY_TIMESTEPS
            for d in range(PRIMARY_DRAWS)
        ]
        primary.append(torch.stack(losses).mean(0).cpu())
        first = noise(0)
        secondary.append(
            torch.stack(
                [
                    per_image_loss(unet, latents, conditioning, schedule, first, t)
                    for t in SECONDARY_TIMESTEPS
                ],
                dim=1,
            ).cpu()
        )
    return torch.cat(primary).numpy(), torch.cat(secondary).numpy()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="name used in the results")
    parser.add_argument("--weights", type=Path, help="omit for the SD control")
    parser.add_argument(
        "--unet-init", choices=("pretrained", "scratch"), default="pretrained"
    )
    parser.add_argument("--exclude-scans", type=Path, required=True)
    parser.add_argument("--mode", choices=("rehearsal", "test"), required=True)
    parser.add_argument("--frozen-commit", help="required with --mode test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/sd15-smoke"))
    parser.add_argument("--vae-dir", type=Path, default=Path("models/sd15-vae"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.mode == "test" and not args.frozen_commit:
        parser.error("--mode test reads the sealed split; give --frozen-commit")
    if args.unet_init == "scratch" and not args.weights:
        parser.error("a from-scratch UNet needs its weights")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    role = roles(args.mode)

    train = ADNISliceDataset(ADNI_ROOT, MANIFEST, "train", 256, verify_hash=True)
    heldout = ADNISliceDataset(
        ADNI_ROOT, MANIFEST, role["heldout"], 256, verify_hash=True
    )
    excluded = set(json.loads(args.exclude_scans.read_text()))
    eligible = member_rows(train.rows, excluded)
    chosen = balanced_sample(
        [train.rows[i]["class_name"] for i in eligible], PER_CLASS, seed=3710
    )
    members = [eligible[i] for i in chosen]
    nonmembers = balanced_sample(
        [r["class_name"] for r in heldout.rows], PER_CLASS, seed=3710
    )

    embeddings = encode_prompts(args.model_dir, device)
    vae = load_vae(args.vae_dir, device)
    unet = build_unet(args.model_dir, device, args.unet_init)
    mode = load_weights(unet, args.weights) if args.weights else "base"
    unet.eval().requires_grad_(False)
    schedule = load_noise_schedule(args.model_dir)
    member_primary, member_by_t = score_set(
        "member", train, members, unet, vae, embeddings, schedule, args
    )
    nonmember_primary, nonmember_by_t = score_set(
        "nonmember", heldout, nonmembers, unet, vae, embeddings, schedule, args
    )
    np.savez(
        args.output / "private-scores.npz",
        member_primary=member_primary,
        nonmember_primary=nonmember_primary,
        member_by_t=member_by_t,
        nonmember_by_t=nonmember_by_t,
        timesteps=np.array(SECONDARY_TIMESTEPS),
        member_is_nc=np.array(
            [train.rows[i]["class_name"] == "NC" for i in members], dtype=np.int8
        ),
        nonmember_is_nc=np.array(
            [heldout.rows[i]["class_name"] == "NC" for i in nonmembers], dtype=np.int8
        ),
    )
    script = Path(__file__).resolve()
    report = {
        "protocol": "docs/final-evaluation-protocol.md",
        "model": args.model,
        "weights": str(args.weights) if args.weights else None,
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest()
        if args.weights
        else None,
        "mode": args.mode,
        "weights_mode": mode,
        "unet_init": args.unet_init,
        "heldout_split": role["heldout"],
        "test_split_read": role["heldout"] == "test",
        "frozen_commit": args.frozen_commit,
        "excluded_scans": len(excluded),
        "members": len(members),
        "nonmembers": len(nonmembers),
        "primary_timesteps": PRIMARY_TIMESTEPS,
        "primary_draws": PRIMARY_DRAWS,
        "labels": LABELS,
        "mean_primary_loss": {
            "member": float(member_primary.mean()),
            "nonmember": float(nonmember_primary.mean()),
        },
        "provenance_sha256": {
            name: hashlib.sha256((script.parent / name).read_bytes()).hexdigest()
            for name in ("final_common.py", "membership.py")
        },
        "seconds": time.perf_counter() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
