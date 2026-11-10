"""Course SSIM against the held-out split, its real floor, and the failure
autopsy (local; docs/final-evaluation-protocol.md).

The same SSIM as every earlier audit (audit_ssim.score): each sample against
its nearest held-out reference of the same class in VAE feature space, one
slice per patient, and against a fixed reference chosen by hashing label and
seed. The real floor is each floor reference against its nearest held-out
reference. The autopsy takes the named model's 5 lowest-SSIM samples and every
near-empty sample; the panel shows synthetic images only.

    python tools/final_ssim.py --real outputs/final-real-<job> \
        --cohorts outputs/final-cohorts-<job> --folder e2=outputs/final-e2 ... \
        --autopsy e2 --output outputs/final-ssim-<date>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_ssim import score, summarize  # noqa: E402
from final_metrics import bootstrap_mean_ci  # noqa: E402

AUTOPSY_LOWEST = 5
NEAR_EMPTY_BELOW, FOREGROUND = 0.05, 0.1  # as dataset.foreground_share


def load_gray256(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    return np.asarray(gray, dtype=np.float64) / 255


def nearest_reference(feature, candidates, reference_features) -> int:
    distances = np.mean((reference_features[candidates] - feature) ** 2, axis=1)
    return int(candidates[int(np.argmin(distances))])


def fixed_reference(label: str, seed: int, candidates) -> int:
    choice = int(hashlib.sha256(f"{label}:{seed}".encode()).hexdigest(), 16)
    return int(candidates[choice % len(candidates)])


def near_empty(path: Path) -> bool:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float64)
    return float((gray / 255 > FOREGROUND).mean()) < NEAR_EMPTY_BELOW


def autopsy_panel(rows, folder: Path, output: Path) -> None:
    panel = Image.new("L", (256 * len(rows), 280), 255)
    draw = ImageDraw.Draw(panel)
    for i, row in enumerate(rows):
        panel.paste(
            Image.fromarray(
                (load_gray256(folder / row["file"]) * 255).astype(np.uint8)
            ),
            (256 * i, 0),
        )
        draw.text(
            (256 * i + 4, 260),
            f"{row['file']}  SSIM {row['nearest_ssim']:.3f}",
            fill=0,
        )
    panel.save(output)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument(
        "--folder",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="local copy of a cohort folder, if not at the recorded path",
    )
    parser.add_argument("--autopsy", help="model whose failures are examined")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    real = np.load(args.real / "private-real-features.npz")
    real_report = json.loads((args.real / "report.json").read_text())
    cohorts = np.load(args.cohorts / "private-cohort-features.npz")
    listing = json.loads((args.cohorts / "private-cohorts.json").read_text())
    folders = {n: Path(v["folder"]) for n, v in listing.items()}
    folders.update({k: Path(v) for k, v in (x.split("=", 1) for x in args.folder)})

    ref_images = real["heldout_ref_images"].astype(np.float64) / 255
    ref_features, ref_is_nc = real["heldout_ref_vae"], real["heldout_ref_is_nc"]
    if abs(score(ref_images[0], ref_images[0]) - 1) > 1e-12:
        raise ValueError("SSIM identity control failed")
    candidates = {c: np.flatnonzero(ref_is_nc == c) for c in (0, 1)}

    floor = []
    for image, feature, is_nc in zip(
        real["floor_ref_images"].astype(np.float64) / 255,
        real["floor_ref_vae"],
        real["floor_ref_is_nc"],
    ):
        floor.append(
            score(
                image,
                ref_images[nearest_reference(feature, candidates[is_nc], ref_features)],
            )
        )

    private, report = (
        [],
        {
            "protocol": "docs/final-evaluation-protocol.md",
            "mode": real_report["mode"],
            "test_split_read": real_report["test_split_read"],
            "metric": "SSIM256 grayscale gaussian sigma1.5 population covariance",
            "references": "one held-out slice per patient, same class",
            "real_floor": summarize(floor),
            "groups": {},
        },
    )
    for name, entry in listing.items():
        rows = []
        for i, (file, label, seed) in enumerate(
            zip(entry["files"], entry["labels"], entry["seeds"])
        ):
            is_nc = int(label == "NC")
            image = load_gray256(folders[name] / file)
            near = nearest_reference(
                cohorts[f"{name}_vae"][i], candidates[is_nc], ref_features
            )
            fixed = fixed_reference(label, seed, candidates[is_nc])
            rows.append(
                {
                    "group": name,
                    "file": file,
                    "label": label,
                    "nearest_index": near,
                    "fixed_index": fixed,
                    "nearest_ssim": score(image, ref_images[near]),
                    "fixed_ssim": score(image, ref_images[fixed]),
                }
            )
        private.extend(rows)
        report["groups"][name] = {
            label: {
                metric: summarize(
                    [r[metric] for r in rows if label in ("all", r["label"])]
                )
                for metric in ("nearest_ssim", "fixed_ssim")
            }
            for label in ("all", *sorted({r["label"] for r in rows}))
        }
        report["groups"][name]["all"]["nearest_ssim"]["mean_ci95"] = bootstrap_mean_ci(
            np.array([r["nearest_ssim"] for r in rows])
        )
        print(name, round(report["groups"][name]["all"]["nearest_ssim"]["mean"], 4))

    if args.autopsy:
        rows = [r for r in private if r["group"] == args.autopsy]
        lowest = sorted(rows, key=lambda r: r["nearest_ssim"])[:AUTOPSY_LOWEST]
        empty = [r for r in rows if near_empty(folders[args.autopsy] / r["file"])]
        chosen = lowest + [r for r in empty if r not in lowest]
        autopsy_panel(
            chosen, folders[args.autopsy], args.output / f"autopsy-{args.autopsy}.png"
        )
        report["autopsy"] = {
            "model": args.autopsy,
            "rule": f"{AUTOPSY_LOWEST} lowest nearest SSIM plus every near-empty sample",
            "near_empty": len(empty),
            "samples": [
                {"file": r["file"], "nearest_ssim": r["nearest_ssim"]} for r in chosen
            ],
        }
    (args.output / "ssim.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "private-ssim.json").write_text(json.dumps(private, indent=2) + "\n")
    print("real floor", report["real_floor"])


if __name__ == "__main__":
    main()
