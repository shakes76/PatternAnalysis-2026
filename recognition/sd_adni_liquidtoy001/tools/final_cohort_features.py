"""Generated-cohort features and nearest neighbours for the final evaluation (GPU job).

Reads the real arrays written by final_real_features.py and, for each cohort,
writes Inception and VAE features, nearest-training and nearest-held-out
distances in pixel and VAE space, within-class diversity, and private panels of
the five closest generated-training pairs.

    python tools/final_cohort_features.py --real outputs/final-real-<job> \
        --cohort e2=PATH --cohort e3=PATH --output outputs/final-cohorts-<job>
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
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ADNISliceDataset  # noqa: E402
from final_common import (  # noqa: E402
    ADNI_ROOT,
    MANIFEST,
    cohort_records,
    generated_views,
    inception_input,
    load_inception,
    nearest,
    vae_features,
)


def summary(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def closest_panel(folder, records, distances, indices, train, output, count=5):
    """Generated sample beside its nearest training slice; private."""
    order = np.argsort(distances)[:count]
    panel = Image.new("L", (512, count * 276), 255)
    draw = ImageDraw.Draw(panel)
    for row, i in enumerate(order):
        with Image.open(folder / records[i]["file"]) as image:
            generated = image.convert("L").resize((256, 256))
        real = train[int(indices[i])]["pixel_values"][0]
        real = Image.fromarray(((real + 1) * 127.5).round().byte().numpy())
        panel.paste(generated, (0, row * 276))
        panel.paste(real, (256, row * 276))
        draw.text(
            (4, row * 276 + 258),
            f"{records[i]['file']} | nearest training slice, RMSE {distances[i]:.4f}",
            fill=0,
        )
    panel.save(output)
    return [{"sample": records[i]["file"], "rmse": float(distances[i])} for i in order]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--cohort", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inception-weights",
        type=Path,
        default=Path("models/inception/inception_v3_google-0cc3c7bd.pth"),
    )
    parser.add_argument("--vae-dir", type=Path, default=Path("models/sd15-vae"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    from diffusers import AutoencoderKL

    vae = (
        AutoencoderKL.from_pretrained(
            args.vae_dir,
            subfolder="vae",
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32,
        )
        .eval()
        .requires_grad_(False)
        .to(device)
    )
    inception = load_inception(args.inception_weights, device)
    real_report = json.loads((args.real / "report.json").read_text())
    arrays = np.load(args.real / "private-real-arrays.npz")
    train_gray = torch.from_numpy(arrays["train_gray64"]).to(device).float() / 255
    heldout_gray = torch.from_numpy(arrays["heldout_gray64"]).to(device).float() / 255
    train_vae = torch.from_numpy(arrays["train_vae"]).to(device)
    heldout_vae = torch.from_numpy(arrays["heldout_vae"]).to(device)
    train_images = ADNISliceDataset(ADNI_ROOT, MANIFEST, "train", 256)

    saved, listing, aggregate = {}, {}, {}
    for item in args.cohort:
        name, folder = item.split("=", 1)
        folder = Path(folder)
        records = cohort_records(folder)
        gray, vae_rows, inception_rows = [], [], []
        for start in range(0, len(records), args.batch_size):
            views = [
                generated_views(folder / r["file"])
                for r in records[start : start + args.batch_size]
            ]
            gray.append(torch.stack([v["gray64"] for v in views]))
            rgb256 = torch.stack([v["rgb256"] for v in views]).to(device)
            vae_rows.append(vae_features(vae, rgb256))
            # Stored sizes differ between models (128 or 256 px), so each
            # image is resized to 299 px on its own.
            inception_rows.append(
                torch.cat(
                    [
                        inception(inception_input(v["rgb"].unsqueeze(0).to(device)))
                        for v in views
                    ]
                )
            )
        gray = torch.cat(gray).to(device)
        features = torch.cat(vae_rows)
        pixel_train, pixel_train_index = nearest(gray, train_gray)
        pixel_heldout, _ = nearest(gray, heldout_gray)
        vae_train, _ = nearest(features, train_vae)
        vae_heldout, _ = nearest(features, heldout_vae)
        labels = [r["label"] for r in records]
        saved.update(
            {
                f"{name}_inception": torch.cat(inception_rows).cpu().numpy(),
                f"{name}_vae": features.cpu().numpy(),
                f"{name}_is_nc": np.array([x == "NC" for x in labels], dtype=np.int8),
                f"{name}_pixel_train": pixel_train.cpu().numpy(),
                f"{name}_pixel_heldout": pixel_heldout.cpu().numpy(),
                f"{name}_vae_train": vae_train.cpu().numpy(),
                f"{name}_vae_heldout": vae_heldout.cpu().numpy(),
            }
        )
        listing[name] = {
            "folder": str(folder),
            "files": [r["file"] for r in records],
            "labels": labels,
            "seeds": [r["seed"] for r in records],
        }
        within = {}
        for label in ("AD", "NC"):
            rows = [i for i, x in enumerate(labels) if x == label]
            if len(rows) < 2:  # B2b may have a result for one class only
                continue
            pairs = torch.pdist(gray[rows]) / gray.shape[1] ** 0.5
            within[label] = summary(pairs.cpu().numpy())
        panel = closest_panel(
            folder,
            records,
            pixel_train.cpu().numpy(),
            pixel_train_index.cpu().numpy(),
            train_images,
            args.output / f"private-closest-{name}.png",
        )
        aggregate[name] = {
            "samples": len(records),
            "pixel_nearest_training_rmse": summary(pixel_train.cpu().numpy()),
            "pixel_nearest_heldout_rmse": summary(pixel_heldout.cpu().numpy()),
            "within_class_pairwise_rmse": within,
        }
        listing[name]["closest"] = panel
        print(f"{name}: {len(records)} samples", flush=True)

    np.savez(args.output / "private-cohort-features.npz", **saved)
    (args.output / "private-cohorts.json").write_text(
        json.dumps(listing, indent=2) + "\n"
    )
    script = Path(__file__).resolve()
    report = {
        "protocol": "docs/final-evaluation-protocol.md",
        "real": str(args.real),
        "mode": real_report["mode"],
        "test_split_read": real_report["test_split_read"],
        "cohorts": aggregate,
        "provenance_sha256": {
            name: hashlib.sha256((script.parent / name).read_bytes()).hexdigest()
            for name in ("final_common.py", "final_cohort_features.py")
        },
        "seconds": time.perf_counter() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cohorts"}, indent=2))


if __name__ == "__main__":
    main()
