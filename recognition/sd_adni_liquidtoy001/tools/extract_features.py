"""Private frozen SD VAE features; no test image or clinical quality claim."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from audit_appearance import verify
from audit_cohort import ADNISliceDataset, patient_selection
from diffusers import AutoencoderKL
from PIL import Image


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="additional generated cohort folder containing report.json",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = Path("models/sd15-vae")
    vae = (
        AutoencoderKL.from_pretrained(
            model,
            subfolder="vae",
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32,
        )
        .eval()
        .requires_grad_(False)
        .to("cuda")
    )
    rows, vectors = [], []

    def encode(tensor, metadata):
        latent = vae.encode(tensor.unsqueeze(0).to("cuda")).latent_dist.mode()
        feature = torch.nn.functional.adaptive_avg_pool2d(latent, (8, 8)).flatten()
        if not torch.isfinite(feature).all():
            raise ValueError("Nonfinite feature")
        vectors.append(feature.cpu().numpy())
        rows.append(metadata)

    for split in ("train", "validation"):
        data = ADNISliceDataset(
            Path("/home/groups/comp3710/ADNI"),
            Path("data/manifest-seed3710.csv"),
            split,
            256,
        )
        for index in patient_selection(data.rows, {"AD": 33, "NC": 69}):
            row = data.rows[index]
            verify(data.root / row["relative_path"], row["sha256"])
            encode(
                data[index]["pixel_values"],
                {
                    "group": split,
                    "label": row["class_name"],
                    "patient": row["patient_id"],
                    "file": row["relative_path"],
                },
            )
    for group, folder in [
        ("base", "outputs/cohort-632557/base"),
        ("adapted", "outputs/cohort-632557/adapted"),
        ("baseline", "../runs/baseline128-20261005/outputs/baseline-vae-632570"),
    ] + [tuple(item.split("=", 1)) for item in args.extra]:
        folder = Path(folder)
        report = json.loads((folder / "report.json").read_text())
        for row in report["records"]:
            verify(folder / row["file"], row["png_sha256"])
            with Image.open(folder / row["file"]) as im:
                array = np.array(
                    im.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                )
            tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1
            encode(tensor, {"group": group, "label": row["label"], "file": row["file"]})
    features = np.stack(vectors)
    real = sum(row["group"] in ("train", "validation") for row in rows)
    if real != 204 or features.shape != (len(rows), 256):
        raise ValueError(f"Unexpected reference/cohort coverage: {features.shape}")
    np.savez_compressed(args.output / "private-features.npz", features=features)
    (args.output / "private-rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    report = {
        "shape": list(features.shape),
        "encoder": "SD1.5 VAE posterior mean, avgpool8x8",
        "provenance_sha256": hashlib.sha256(
            (model / "provenance.json").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_read": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
