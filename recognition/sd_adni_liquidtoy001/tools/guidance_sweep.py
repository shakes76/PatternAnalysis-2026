"""Post-hoc: what does the guidance scale do to E2's samples? (GPU job)

Tests two explanations the README marked as untested:

- Section 7.3: E2's classes differ more than real slices do (probe AUC 0.95
  against 0.76) because guidance 7.5 extrapolates along the class direction.
  If so, the probe AUC should fall towards the real value as guidance falls.
- Section 8, autopsy 5: the over-bright worst samples come from guidance
  pushing intensities to extremes. If so, they should look normal at low
  guidance, and bright saturated tissue should grow with guidance overall.

For each guidance scale it generates the first --count seeds per class (from
37200) with the given weights, --batch-size images at a time, and writes the
Inception-v3 features of the saved PNGs (the probe is fitted locally by
tools/guidance_probe.py) and intensity statistics of every image. With
--autopsy it also regenerates the five autopsy seeds one at a time, the
setting of the final cohort, so at guidance 7.5 their PNGs must match the
final cohort's hashes (--reference-cohort). --real-references gives the
intensity statistics of the 102 one-per-patient validation slices of a
rehearsal-mode final_real_features run as the real reference. No test-split
data is read.

    python tools/guidance_sweep.py --weights PATH --guidance 1 3 --autopsy \
        --reference-cohort outputs/final-e2 \
        --real-references outputs/final-real-rehearsal-<job> \
        --output outputs/guidance-e2-<job>
"""

from __future__ import annotations

import os

# Deterministic cuBLAS must be configured before CUDA initialises.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_common import generated_views, inception_input, load_inception  # noqa: E402
from modules import (  # noqa: E402
    LABELS,
    build_unet,
    decode,
    encode_prompts,
    guided_ddim,
    initial_noise,
    load_noise_schedule,
    load_vae,
    load_weights,
)

# E2's five lowest-SSIM test samples (README Section 8), all NC.
AUTOPSY_SEEDS = (37263, 37595, 37407, 37489, 37341)
BORDER = 8  # outer frame for the background statistic, as in predict.py
FOREGROUND = 0.1  # grey level above which a pixel counts as tissue
BRIGHT = 0.9  # tissue brighter than this counts as saturated


def digest(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intensity(image: np.ndarray) -> dict:
    """Intensity statistics of one HxWx3 image in [0, 1].

    channel_spread and border_mean as in predict.py; foreground_mean and
    foreground_std are the brightness and contrast of the tissue (grey level
    above FOREGROUND); bright_fraction is the share of tissue above BRIGHT.
    """
    gray = image.mean(axis=2)
    frame = np.ones_like(gray, dtype=bool)
    frame[BORDER:-BORDER, BORDER:-BORDER] = False
    tissue = gray[gray > FOREGROUND]
    empty = tissue.size == 0
    return {
        "channel_spread": float((image.max(axis=2) - image.min(axis=2)).mean()),
        "border_mean": float(gray[frame].mean()),
        "foreground_fraction": float((gray > FOREGROUND).mean()),
        "foreground_mean": 0.0 if empty else float(tissue.mean()),
        "foreground_std": 0.0 if empty else float(tissue.std()),
        "bright_fraction": 0.0 if empty else float((tissue > BRIGHT).mean()),
        "p99": float(np.percentile(gray, 99)),
    }


STAT_KEYS = tuple(intensity(np.zeros((32, 32, 3))))


def mean_stats(records: list[dict]) -> dict:
    """Mean of each intensity statistic over a group of records."""
    return {
        "n": len(records),
        **{k: float(np.mean([r[k] for r in records])) for k in STAT_KEYS},
    }


def save_png(image: np.ndarray, path: Path) -> None:
    """Save as predict.py does, so hashes compare with the final cohort."""
    Image.fromarray((image * 255).round().astype(np.uint8)).save(path)


def real_reference_stats(folder: Path) -> dict:
    """Intensity statistics of the validation reference slices, by class."""
    report = json.loads((folder / "report.json").read_text())
    if report["test_split_read"] or report["roles"]["heldout"] != "validation":
        raise ValueError("--real-references must be a rehearsal-mode run")
    arrays = np.load(folder / "private-real-features.npz")
    images = arrays["heldout_ref_images"].astype(np.float32)
    if images.max() > 1.5:
        images = images / 255
    is_nc = arrays["heldout_ref_is_nc"]
    rows = [intensity(np.repeat(x[..., None], 3, axis=2)) for x in images]
    return {
        "source": "one validation slice per patient (rehearsal final_real_features)",
        "dtype": str(arrays["heldout_ref_images"].dtype),
        **{
            label: mean_stats([r for r, nc in zip(rows, is_nc) if nc == (label == "NC")])
            for label in LABELS
        },
    }


@torch.no_grad()
def main() -> None:
    """Generate the sweep, extract features and write report.json."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guidance", type=float, nargs="+", required=True)
    parser.add_argument("--count", type=int, default=128, help="seeds per class")
    parser.add_argument("--first-seed", type=int, default=37200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--autopsy", action="store_true")
    parser.add_argument(
        "--autopsy-guidance", type=float, nargs="+", default=[1.0, 3.0, 5.0, 7.5]
    )
    parser.add_argument("--reference-cohort", type=Path)
    parser.add_argument("--real-references", type=Path)
    parser.add_argument(
        "--inception-weights",
        type=Path,
        default=Path("models/inception/inception_v3_google-0cc3c7bd.pth"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/sd15-smoke"))
    parser.add_argument("--vae-dir", type=Path, default=Path("models/sd15-vae"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    # The settings of predict.py and every final cohort.
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
    unet = build_unet(args.model_dir, device, "pretrained").eval().requires_grad_(False)
    mode = load_weights(unet, args.weights)
    unet.eval()
    schedule = load_noise_schedule(args.model_dir)

    def sample(seeds, label, guidance):
        starts = torch.cat([initial_noise(s, 256, device) for s in seeds])
        latents = guided_ddim(
            unet,
            schedule.config,
            starts,
            embeddings[label],
            embeddings["empty"],
            guidance,
        )
        return decode(vae, latents).permute(0, 2, 3, 1).cpu().numpy()

    seeds = list(range(args.first_seed, args.first_seed + args.count))
    records = {}
    timing = {}
    for guidance in args.guidance:
        folder = args.output / f"cfg{guidance}"
        folder.mkdir()
        tic = time.perf_counter()
        rows = []
        for label in LABELS:
            for i in range(0, len(seeds), args.batch_size):
                batch = seeds[i : i + args.batch_size]
                for seed, image in zip(batch, sample(batch, label, guidance)):
                    name = f"{label}-{seed}.png"
                    save_png(image, folder / name)
                    rows.append(
                        {
                            "file": name,
                            "label": label,
                            "seed": seed,
                            "png_sha256": digest(folder / name),
                            **intensity(image),
                        }
                    )
        records[guidance] = rows
        timing[guidance] = time.perf_counter() - tic
        print(f"guidance {guidance}: {len(rows)} images, {timing[guidance]:.0f} s", flush=True)

    autopsy = []
    if args.autopsy:
        reference = {}
        if args.reference_cohort:
            listing = json.loads((args.reference_cohort / "report.json").read_text())
            reference = {(r["label"], r["seed"]): r for r in listing["records"]}
        folder = args.output / "autopsy"
        folder.mkdir()
        for seed in AUTOPSY_SEEDS:
            for guidance in args.autopsy_guidance:
                image = sample([seed], "NC", guidance)[0]
                name = f"NC-{seed}-cfg{guidance}.png"
                save_png(image, folder / name)
                row = {
                    "file": name,
                    "seed": seed,
                    "guidance": guidance,
                    "png_sha256": digest(folder / name),
                    **intensity(image),
                }
                if guidance == 7.5 and ("NC", seed) in reference:
                    row["matches_final_cohort"] = (
                        row["png_sha256"] == reference[("NC", seed)]["png_sha256"]
                    )
                autopsy.append(row)
        print(f"autopsy: {len(autopsy)} images", flush=True)

    # Inception features of the saved PNGs, one image at a time as in
    # tools/final_cohort_features.py.
    del unet
    torch.cuda.empty_cache()
    inception = load_inception(args.inception_weights, device)
    features = {}
    for guidance, rows in records.items():
        folder = args.output / f"cfg{guidance}"
        values = [
            inception(
                inception_input(generated_views(folder / r["file"])["rgb"][None].to(device))
            )
            for r in rows
        ]
        features[f"cfg{guidance}_inception"] = torch.cat(values).cpu().numpy()
        features[f"cfg{guidance}_is_nc"] = np.array(
            [r["label"] == "NC" for r in rows], dtype=np.int8
        )
        features[f"cfg{guidance}_seed"] = np.array([r["seed"] for r in rows])
    np.savez(args.output / "features.npz", **features)

    script = Path(__file__).resolve()
    report = {
        "post_hoc": True,
        "weights": str(args.weights),
        "weights_sha256": digest(args.weights),
        "weights_mode": mode,
        "guidance_values": args.guidance,
        "seeds_per_class": args.count,
        "first_seed": args.first_seed,
        "batch_size": args.batch_size,
        "inference_steps": 30,
        "thresholds": {"foreground": FOREGROUND, "bright": BRIGHT, "border": BORDER},
        "summary": {
            str(g): {
                label: mean_stats([r for r in rows if r["label"] == label])
                for label in LABELS
            }
            for g, rows in records.items()
        },
        "seconds_per_guidance": {str(g): s for g, s in timing.items()},
        "autopsy": autopsy,
        "real_references": real_reference_stats(args.real_references)
        if args.real_references
        else None,
        "records": {str(g): rows for g, rows in records.items()},
        "code_sha256": {
            name: digest(path)
            for name, path in (
                ("guidance_sweep.py", script),
                ("final_common.py", script.parent / "final_common.py"),
                ("modules.py", script.parents[1] / "modules.py"),
            )
        },
        "test_split_read": False,
        "seconds": time.perf_counter() - started,
        "torch": torch.__version__,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("summary", "autopsy", "real_references")}, indent=2))


if __name__ == "__main__":
    main()
