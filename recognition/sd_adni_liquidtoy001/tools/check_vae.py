"""Bounded paired VAE diagnostic; never evaluates held-out test images.

Autoencoder usage follows Hugging Face's AutoencoderKL API documentation:
https://huggingface.co/docs/diffusers/api/models/autoencoderkl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import diffusers
import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import ADNISliceDataset


def select_indices(rows: list[dict[str, str]]) -> list[int]:
    """Choose two patients/class before inference, one median slice/patient."""
    selected = []
    for label in ("AD", "NC"):
        patients = sorted(
            {r["patient_id"] for r in rows if r["class_name"] == label},
            key=lambda p: hashlib.sha256(f"3710:{p}".encode()).hexdigest(),
        )
        if len(patients) < 2:
            raise ValueError("Need at least two patients in each class")
        for patient in patients[:2]:
            scan = min(int(r["scan_id"]) for r in rows if r["patient_id"] == patient)
            indices = sorted(
                [
                    i
                    for i, r in enumerate(rows)
                    if r["patient_id"] == patient and int(r["scan_id"]) == scan
                ],
                key=lambda i: int(rows[i]["slice_index"]),
            )
            selected.append(indices[len(indices) // 2])
    return selected


def save_triptych(source, reconstructed, path: Path, label: str):
    error = (source - reconstructed).abs().mul(5).clamp(0, 1)
    size = source.shape[-1]
    canvas = Image.new("RGB", (size * 3, size + 32), "#101820")
    draw = ImageDraw.Draw(canvas)
    for column, (tensor, title) in enumerate(
        zip(
            (source, reconstructed, error),
            ("Input", "VAE reconstruction", "Absolute error x5"),
        )
    ):
        pixels = tensor.permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
        canvas.paste(Image.fromarray(pixels), (column * size, 32))
        draw.text((column * size + 8, 8), f"{label} | {title}", fill="white")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires an allocated CUDA device")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(3710)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vae = (
        AutoencoderKL.from_pretrained(
            args.model,
            subfolder="vae",
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32,
        )
        .eval()
        .requires_grad_(False)
        .to("cuda")
    )
    results = []
    with torch.inference_mode():
        for split in ("train", "validation"):
            for size in (256, 512):
                dataset = ADNISliceDataset(args.root, args.manifest, split, size)
                for position, index in enumerate(select_indices(dataset.rows)):
                    sample = dataset[index]
                    source_path = args.root / sample["relative_path"]
                    actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
                    if actual_hash != dataset.rows[index]["sha256"]:
                        raise ValueError("Source image changed since manifest creation")
                    pixels = sample["pixel_values"].unsqueeze(0).to("cuda")
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    latent = vae.encode(pixels).latent_dist.mode()
                    decoded = vae.decode(latent).sample
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    if (
                        not torch.isfinite(decoded).all()
                        or not torch.isfinite(latent).all()
                    ):
                        raise RuntimeError("Non-finite reconstruction or latent")
                    original = (pixels[0] + 1) / 2
                    raw = (decoded[0] + 1) / 2
                    reconstruction = raw.clamp(0, 1)
                    # This intensity mask is a diagnostic, not anatomical segmentation.
                    foreground = original > 0.02
                    outside = (raw < 0) | (raw > 1)
                    raw_error = (original - raw).square()
                    region_stats = {}
                    for name, mask in (
                        ("foreground", foreground),
                        ("background", ~foreground),
                    ):
                        region_stats[name] = {
                            "pixels": int(mask.sum().item()),
                            "clipped_fraction": outside[mask].float().mean().item()
                            if mask.any()
                            else None,
                            "raw_mse": raw_error[mask].mean().item()
                            if mask.any()
                            else None,
                            "mean_clipping_magnitude": (raw - reconstruction)
                            .abs()[mask]
                            .mean()
                            .item()
                            if mask.any()
                            else None,
                        }
                    mse = (original - reconstruction).square().mean().item()
                    label = f"{split}-{position + 1}-{sample['class_name']}-{size}"
                    save_triptych(
                        original, reconstruction, args.output / f"{label}.png", label
                    )
                    results.append(
                        {
                            "sample": label,
                            "split": split,
                            "size": size,
                            "relative_path": sample["relative_path"],
                            "source_sha256": actual_hash,
                            "mse": mse,
                            "raw_mse": raw_error.mean().item(),
                            "raw_min": raw.min().item(),
                            "raw_max": raw.max().item(),
                            "intensity_threshold": 0.02,
                            "regions": region_stats,
                            "psnr_db": -10 * math.log10(mse) if mse else None,
                            "perfect_reconstruction": mse == 0,
                            "clipped_fraction": ((raw < 0) | (raw > 1))
                            .float()
                            .mean()
                            .item(),
                            "seconds": elapsed,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "latent_shape": list(latent.shape),
                        }
                    )
                    print(json.dumps(results[-1]), flush=True)
    record = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "model": json.loads((args.model / "provenance.json").read_text()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "posterior": "mode",
        "precision": "float32",
        "test_evaluated": False,
        "results": results,
    }
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
