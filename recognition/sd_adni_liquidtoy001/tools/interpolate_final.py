"""Latent interpolation for the final evaluation (GPU job).

I1 and I2 of docs/final-evaluation-protocol.md, for one model:

- I1, synthetic to synthetic: 4 seed pairs (2 AD, 2 NC), 9 spherical
  interpolation frames between the starting noises, sampled as the cohorts
  were (DDIM 30, guidance 7.5). Public panel.
- I2, training slice to synthetic: 4 training slices (2 per class) inverted to
  starting noise by DDIM inversion without guidance, then interpolated towards
  a synthetic seed's noise and sampled without guidance (inversion is only
  accurate then). Panels and images private.

Each frame gets its nearest-training and nearest-held-out pixel distance and
its distance to the previous frame. An interior frame closer to a training
slice than the closest real-to-real pair is flagged as a jump towards it.

    python tools/interpolate_final.py --model e2 --weights PATH --mode rehearsal \
        --real outputs/final-real-<job> --output outputs/final-interp-e2-<job>
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
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ADNISliceDataset  # noqa: E402
from final_common import ADNI_ROOT, MANIFEST, balanced_sample, nearest  # noqa: E402
from modules import (  # noqa: E402
    build_unet,
    ddim_invert,
    decode,
    encode_prompts,
    guided_ddim,
    initial_noise,
    load_noise_schedule,
    load_vae,
    load_weights,
    slerp,
)

FRAMES = 9
PAIRS = (
    ("AD", 37200, 37201),
    ("AD", 37202, 37203),
    ("NC", 37200, 37201),
    ("NC", 37202, 37203),
)
SYNTHETIC_PARTNER = 37204  # the I2 end point, a seed outside the I1 pairs


def gray64(images: torch.Tensor) -> torch.Tensor:
    """[B, 3, 256, 256] in [0, 1] -> grayscale 64 px rows, via 8-bit PNG values
    and PIL's bilinear resize, the path every generated image took before."""
    rows = []
    for image in images:
        array = (image.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        small = Image.fromarray(array).convert("L").resize((64, 64), Image.BILINEAR)
        rows.append(torch.from_numpy(np.asarray(small, dtype=np.float32) / 255))
    return torch.stack(rows).flatten(1)


def frame_measures(images, train_gray, heldout_gray, closest_real, flag=True):
    rows = gray64(images).to(train_gray.device)
    to_train, _ = nearest(rows, train_gray)
    to_heldout, _ = nearest(rows, heldout_gray)
    steps = (rows[1:] - rows[:-1]).pow(2).mean(1).sqrt()
    interior = to_train[1:-1]
    return {
        "nearest_training_rmse": to_train.tolist(),
        "nearest_heldout_rmse": to_heldout.tolist(),
        "step_rmse": steps.tolist(),
        # Only meaningful between two synthetic end points: in I2 the first
        # frames are meant to be close to the training slice they start from.
        "jump_towards_training": bool((interior < closest_real).any())
        if flag
        else None,
    }


def strip(images: torch.Tensor, caption: str) -> Image.Image:
    tiles = [
        Image.fromarray(
            (x.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        )
        .convert("L")
        .resize((128, 128))
        for x in images
    ]
    canvas = Image.new("L", (128 * len(tiles), 148), 255)
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (128 * i, 0))
    ImageDraw.Draw(canvas).text((4, 132), caption, fill=0)
    return canvas


def stack(strips: list[Image.Image]) -> Image.Image:
    canvas = Image.new("L", (strips[0].width, sum(s.height for s in strips)), 255)
    y = 0
    for s in strips:
        canvas.paste(s, (0, y))
        y += s.height
    return canvas


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument(
        "--exclude-scans",
        type=Path,
        required=True,
        help="E2's private list of defective scans, never used as end points",
    )
    parser.add_argument("--mode", choices=("rehearsal", "test"), required=True)
    parser.add_argument("--frozen-commit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/sd15-smoke"))
    parser.add_argument("--vae-dir", type=Path, default=Path("models/sd15-vae"))
    args = parser.parse_args()
    real_report = json.loads((args.real / "report.json").read_text())
    if real_report["mode"] != args.mode:
        parser.error("the real features were made in the other mode")
    if args.mode == "test" and not args.frozen_commit:
        parser.error("--mode test reads the sealed split; give --frozen-commit")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    arrays = np.load(args.real / "private-real-arrays.npz")
    train_gray = torch.from_numpy(arrays["train_gray64"]).to(device).float() / 255
    heldout_gray = torch.from_numpy(arrays["heldout_gray64"]).to(device).float() / 255
    closest_real = real_report["heldout_to_train_pixel_rmse"]["min"]
    embeddings = encode_prompts(args.model_dir, device)
    vae = load_vae(args.vae_dir, device)
    unet = build_unet(args.model_dir, device, "pretrained")
    load_weights(unet, args.weights)
    unet.eval().requires_grad_(False)
    config = load_noise_schedule(args.model_dir).config
    weights = torch.linspace(0, 1, FRAMES).tolist()

    def sample(noises, label, guidance):
        latents = guided_ddim(
            unet,
            config,
            torch.cat(noises),
            embeddings[label],
            embeddings["empty"],
            guidance,
        )
        return decode(vae, latents)

    results = {
        "model": args.model,
        "mode": args.mode,
        "frames": FRAMES,
        "i1": [],
        "i2": [],
    }
    strips = []
    for label, a, b in PAIRS:
        start, end = initial_noise(a, 256, device), initial_noise(b, 256, device)
        images = sample([slerp(start, end, w) for w in weights], label, 7.5)
        results["i1"].append(
            {
                "label": label,
                "seeds": [a, b],
                **frame_measures(images, train_gray, heldout_gray, closest_real),
            }
        )
        strips.append(strip(images, f"{args.model} {label} seed {a} -> {b}"))
    stack(strips).save(args.output / "i1-synthetic-to-synthetic.png")

    train = ADNISliceDataset(ADNI_ROOT, MANIFEST, "train", 256, verify_hash=True)
    excluded = set(json.loads(args.exclude_scans.read_text()))
    eligible = [i for i, r in enumerate(train.rows) if r["scan_id"] not in excluded]
    picks = balanced_sample(
        [train.rows[i]["class_name"] for i in eligible], 2, seed=3710
    )
    chosen = [eligible[i] for i in picks]
    private_strips, originals, reconstructions = [], [], []
    for index in chosen:
        item = train[index]
        label = item["class_name"]
        pixels = item["pixel_values"].unsqueeze(0).to(device)
        latent = vae.encode(pixels).latent_dist.mean * vae.config.scaling_factor
        inverted = ddim_invert(unet, config, latent, embeddings[label])
        partner = initial_noise(SYNTHETIC_PARTNER, 256, device)
        images = sample([slerp(inverted, partner, w) for w in weights], label, 1.0)
        original = (pixels[0] + 1) / 2
        reconstruction_rmse = float(F.mse_loss(images[0], original).sqrt())
        measures = frame_measures(
            images, train_gray, heldout_gray, closest_real, flag=False
        )
        results["i2"].append(
            {"label": label, "reconstruction_rmse_256": reconstruction_rmse, **measures}
        )
        originals.append((original[0] * 255).round().byte().cpu().numpy())
        reconstructions.append((images[0].mean(0) * 255).round().byte().cpu().numpy())
        private_strips.append(
            strip(
                torch.cat([original.unsqueeze(0), images]),
                f"training slice ({label}) | inverted -> seed {SYNTHETIC_PARTNER}",
            )
        )
    stack(private_strips).save(args.output / "private-i2-training-to-synthetic.png")
    np.savez(
        args.output / "private-i2-endpoints.npz",
        originals=np.stack(originals),
        reconstructions=np.stack(reconstructions),
    )
    results["closest_real_to_real_rmse"] = closest_real
    results["weights_sha256"] = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    results["test_split_read"] = real_report["test_split_read"]
    results["frozen_commit"] = args.frozen_commit
    results["seconds"] = time.perf_counter() - started
    (args.output / "interpolation.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in results.items() if k not in ("i1", "i2")}))


if __name__ == "__main__":
    main()
