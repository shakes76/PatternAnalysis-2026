"""Private encoded real/synthetic paths and reference export for local SSIM."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from audit_appearance import verify
from audit_cohort import ADNISliceDataset
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    feature_dir = Path("outputs/features-639765")
    records = json.loads((feature_dir / "private-rows.json").read_text())
    features = np.load(feature_dir / "private-features.npz")["features"]
    tensors = {}
    for split in ("train", "validation"):
        dataset = ADNISliceDataset(
            Path("/home/groups/comp3710/ADNI"),
            Path("data/manifest-seed3710.csv"),
            split,
            256,
        )
        lookup = {r["relative_path"]: i for i, r in enumerate(dataset.rows)}
        for i, row in enumerate(records):
            if row["group"] != split:
                continue
            index = lookup[row["file"]]
            verify(dataset.root / row["file"], dataset.rows[index]["sha256"])
            tensors[i] = dataset[index]["pixel_values"]
    order = sorted(tensors)
    np.savez_compressed(
        args.output / "private-reference-images.npz",
        indices=np.array(order),
        images=np.stack(
            [
                ((tensors[i][0].numpy() + 1) * 127.5).round().astype(np.uint8)
                for i in order
            ]
        ),
    )
    vae = (
        AutoencoderKL.from_pretrained(
            "models/sd15-vae",
            subfolder="vae",
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32,
        )
        .eval()
        .requires_grad_(False)
        .to("cuda")
    )
    root = Path("outputs/cohort-632557/adapted")
    generated = json.loads((root / "report.json").read_text())["records"]
    results = []
    for label, seed in (("NC", 37215), ("AD", 37220)):
        gen = next(r for r in generated if r["label"] == label and r["seed"] == seed)
        verify(root / gen["file"], gen["png_sha256"])
        qi = next(
            i
            for i, r in enumerate(records)
            if r["group"] == "adapted" and r["file"] == gen["file"]
        )
        candidates = [
            i
            for i, r in enumerate(records)
            if r["group"] == "train" and r["label"] == label
        ]
        ni = candidates[
            int(np.argmin(np.mean((features[candidates] - features[qi]) ** 2, axis=1)))
        ]
        with Image.open(root / gen["file"]) as image:
            x = (
                torch.from_numpy(np.array(image.convert("RGB")))
                .permute(2, 0, 1)
                .float()
                / 127.5
                - 1
            )
        y = tensors[ni]
        endpoints = torch.stack((x, y)).to("cuda")
        z = vae.encode(endpoints).latent_dist.mode()
        alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
        decoded = torch.cat(
            [
                vae.decode(((1 - t) * z[0] + t * z[1]).unsqueeze(0)).sample
                for t in alphas
            ]
        )
        if not torch.isfinite(decoded).all():
            raise ValueError("Nonfinite decoder output")
        decoded = (decoded / 2 + 0.5).clamp(0, 1).cpu()
        originals = torch.stack((x, y)) / 2 + 0.5
        canvas = Image.new("RGB", (256 * 7, 280), "white")
        draw = ImageDraw.Draw(canvas)
        for col, (tensor, title) in enumerate(
            [(originals[0], "original generated")]
            + list(zip(decoded, [f"alpha {t}" for t in alphas]))
            + [(originals[1], "original train")]
        ):
            pixels = tensor.permute(1, 2, 0).mul(255).round().byte().numpy()
            canvas.paste(Image.fromarray(pixels), (col * 256, 0))
            draw.text((col * 256 + 4, 260), title, fill="black")
        canvas.save(args.output / f"encoded-{label}-{seed}.png")
        results.append(
            {
                "label": label,
                "seed": seed,
                "generated_file": gen["file"],
                "reference_index": ni,
                "endpoint_mse": [
                    float(((decoded[0] - originals[0]) ** 2).mean()),
                    float(((decoded[-1] - originals[1]) ** 2).mean()),
                ],
            }
        )
    report = {
        "cases": results,
        "test_read": False,
        "reference_count": len(order),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
