"""Generate fixed-seed ADNI slices from trained weights and summarise them.

    python predict.py --weights outputs/e1-<job>/weights/unet-ema-epoch-10.safetensors \
                      --output outputs/e1-cohort-<job>
    python predict.py --output outputs/base-cohort     # the untouched SD1.5 model
    python predict.py --weights ... --output outputs/ad-only --classes AD --count 100

Follows the protocol of the earlier cohorts: seeds 37200-37231 for each of AD
and NC, DDIM with 30 steps and eta 0, guidance 7.5, 256x256 pixels, and the
same deterministic GPU settings. --classes AD (or NC) generates one class
only; a seed gives the same image whichever classes are requested. Writes one
PNG per image, a labelled RGB grid, a grayscale grid, and report.json with the
SHA-256 of every file, in the layout tools/audit_*.py read. Prints two appearance statistics per image: channel
spread (MRI is single-channel, so any spread is an artefact) and outer-border
brightness (MRI background is black, so a bright border means a grey background).
"""

from __future__ import annotations

import os

# Deterministic cuBLAS must be configured before CUDA initialises.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from modules import (
    LABELS,
    PROMPTS,
    build_unet,
    decode,
    encode_prompts,
    guided_ddim,
    initial_noise,
    load_noise_schedule,
    load_vae,
    load_weights,
)

BORDER = 8  # pixels of the outer frame used for the background statistic


def appearance(image: np.ndarray) -> dict:
    """Channel spread and border brightness of one HxWx3 image in [0, 1]."""
    spread = float((image.max(axis=2) - image.min(axis=2)).mean())
    gray = image.mean(axis=2)
    frame = np.ones_like(gray, dtype=bool)
    frame[BORDER:-BORDER, BORDER:-BORDER] = False
    return {"channel_spread": spread, "border_mean": float(gray[frame].mean())}


def save_grids(tiles, columns, size, output, guidance):
    rows = (len(tiles) + columns - 1) // columns
    colour = Image.new("RGB", (columns * size, rows * (size + 24)), "white")
    gray = Image.new("L", (columns * size, rows * size), 0)
    draw = ImageDraw.Draw(colour)
    for i, (caption, image) in enumerate(tiles):
        x, y = (i % columns) * size, (i // columns)
        colour.paste(image, (x, y * (size + 24)))
        draw.text((x + 4, y * (size + 24) + size + 4), caption, fill="black")
        gray.paste(image.convert("L"), (x, y * size))
    colour.save(output / f"grid-cfg{guidance}.png")
    gray.save(output / f"grid-gray-cfg{guidance}.png")


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--weights", type=Path, help="trained UNet or adapter; omit for base SD1.5"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--unet-init",
        choices=("pretrained", "scratch"),
        default="pretrained",
        help="scratch: weights belong to the smaller B3 UNet",
    )
    parser.add_argument("--first-seed", type=int, default=37200)
    parser.add_argument("--count", type=int, default=32, help="seeds per class")
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=LABELS,
        default=list(LABELS),
        help="which classes to generate (default: both)",
    )
    parser.add_argument("--guidance", type=float, nargs="+", default=[7.5])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="images denoised together; each keeps its own seeded starting noise",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/sd15-smoke"))
    parser.add_argument("--vae-dir", type=Path, default=Path("models/sd15-vae"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.classes = [label for label in LABELS if label in args.classes]  # fixed order, no repeats
    device = torch.device(args.device)
    if device.type == "cuda":
        # The settings used for the E0 cohorts, so generation numerics match.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    embeddings = encode_prompts(args.model_dir, device)
    vae = load_vae(args.vae_dir, device)
    unet = build_unet(args.model_dir, device, args.unet_init)
    unet = unet.eval().requires_grad_(False)
    mode = load_weights(unet, args.weights) if args.weights else "base"
    unet.eval()
    schedule = load_noise_schedule(args.model_dir)

    records = []
    seeds = list(range(args.first_seed, args.first_seed + args.count))
    batches = [
        seeds[i : i + args.batch_size] for i in range(0, len(seeds), args.batch_size)
    ]
    for guidance in args.guidance:
        tiles = []
        for label in args.classes:
            for batch in batches:
                starts = [initial_noise(seed, 256, device) for seed in batch]
                latents = guided_ddim(
                    unet,
                    schedule.config,
                    torch.cat(starts),
                    embeddings[label],
                    embeddings["empty"],
                    guidance,
                    steps=args.steps,
                )
                images = decode(vae, latents).permute(0, 2, 3, 1).cpu().numpy()
                for seed, start, image in zip(batch, starts, images):
                    name = f"{label}-{seed}-cfg{guidance}.png"
                    tile = Image.fromarray((image * 255).round().astype(np.uint8))
                    tile.save(args.output / name)
                    records.append(
                        {
                            "file": name,
                            "label": label,
                            "seed": seed,
                            "guidance": guidance,
                            "initial_noise_sha256": hashlib.sha256(
                                start.cpu().numpy().tobytes()
                            ).hexdigest(),
                            "png_sha256": digest(args.output / name),
                            **appearance(image),
                        }
                    )
                    # The uncurated grid always shows the first 32 seeds per class.
                    if seed < args.first_seed + 32:
                        tiles.append((f"{label} {seed}", tile))
        save_grids(tiles, 8, 256, args.output, guidance)

    print(f"{'label':<6}{'cfg':>6}{'n':>4}{'channel spread':>17}{'border mean':>14}")
    summary = {}
    for guidance in args.guidance:
        for label in args.classes:
            group = [
                r for r in records if r["label"] == label and r["guidance"] == guidance
            ]
            spread = float(np.mean([r["channel_spread"] for r in group]))
            border = float(np.mean([r["border_mean"] for r in group]))
            summary[f"{label}-cfg{guidance}"] = {
                "n": len(group),
                "channel_spread": spread,
                "border_mean": border,
            }
            print(
                f"{label:<6}{guidance:>6}{len(group):>4}{spread:>17.4f}{border:>14.4f}"
            )
    report = {
        "weights": str(args.weights) if args.weights else None,
        "weights_sha256": digest(args.weights) if args.weights else None,
        "mode": mode,
        "unet_init": args.unet_init,
        "prompts": PROMPTS,
        "classes": args.classes,
        "scheduler": "DDIM",
        "inference_steps": args.steps,
        "batch_size": args.batch_size,
        "eta": 0.0,
        "guidance_values": args.guidance,
        "image_size": 256,
        "deterministic": device.type == "cuda",
        "code_sha256": {
            n: digest(Path(__file__).parent / n) for n in ("modules.py", "predict.py")
        },
        "summary": summary,
        "records": records,
        "seconds": time.perf_counter() - started,
        "torch": torch.__version__,
        "scope": f"{len(records)} predeclared images; no test-split data read",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
