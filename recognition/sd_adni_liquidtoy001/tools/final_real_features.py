"""Real-slice features and distances for the final evaluation (GPU job).

Stage 2 of docs/final-evaluation-protocol.md. In --mode rehearsal no test slice
is read; --mode test needs --frozen-commit, the commit whose code was frozen
after the rehearsal. Outputs:

- private-real-arrays.npz: 64 px pixels and VAE features of every training and
  held-out slice, for the cohort job's nearest neighbours. Stays on Rangpur.
- private-real-features.npz: what the local metrics need: Inception and VAE
  features of the held-out and floor slices, the one-slice-per-patient SSIM
  references, and the calibration distances. No patient IDs.
- report.json: counts, provenance and the real reference values.
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_cohort import patient_selection  # noqa: E402
from dataset import ADNISliceDataset, read_manifest  # noqa: E402
from final_common import (  # noqa: E402
    ADNI_ROOT,
    FLOOR_PER_CLASS,
    MANIFEST,
    PATIENTS_PER_CLASS,
    balanced_sample,
    inception_input,
    load_inception,
    nearest,
    roles,
    subject_codes,
    vae_features,
)


class TwoSizes(Dataset):
    """Each slice at 64 px (as audit_cohort.py) and 256 px (as the VAE audits)."""

    def __init__(self, split: str):
        self.small = ADNISliceDataset(ADNI_ROOT, MANIFEST, split, 64)
        self.large = ADNISliceDataset(ADNI_ROOT, MANIFEST, split, 256, verify_hash=True)
        self.rows = self.large.rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        small = self.small[index]["pixel_values"][0]
        return {
            "gray64": ((small.flatten() + 1) / 2 * 255).round().to(torch.uint8),
            "rgb256": self.large[index]["pixel_values"],
            "index": index,
        }


def summary(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
    }


@torch.no_grad()
def encode_split(split, wanted_inception, wanted_images, vae, inception, args):
    data = TwoSizes(split)
    loader = DataLoader(
        data, batch_size=args.batch_size, num_workers=args.workers, shuffle=False
    )
    gray, vae_rows, inception_rows, images = [], [], {}, {}
    for batch in loader:
        rgb256 = batch["rgb256"].to(args.device)
        gray.append(batch["gray64"])
        vae_rows.append(vae_features(vae, rgb256).cpu())
        positions = [
            p for p, i in enumerate(batch["index"].tolist()) if i in wanted_inception
        ]
        if positions:
            pooled = inception(inception_input((rgb256[positions] + 1) / 2)).cpu()
            for p, feature in zip(positions, pooled):
                inception_rows[int(batch["index"][p])] = feature
        for p, i in enumerate(batch["index"].tolist()):
            if i in wanted_images:
                images[i] = ((rgb256[p, 0] + 1) * 127.5).round().byte().cpu().numpy()
    print(f"Encoded {split}: {len(data)} slices", flush=True)
    return {
        "rows": data.rows,
        "labels": [r["class_name"] for r in data.rows],
        "gray64": torch.cat(gray),
        "vae": torch.cat(vae_rows),
        "inception": inception_rows,
        "images": images,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("rehearsal", "test"), required=True)
    parser.add_argument("--frozen-commit", help="required with --mode test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inception-weights",
        type=Path,
        default=Path("models/inception/inception_v3_google-0cc3c7bd.pth"),
    )
    parser.add_argument("--vae-dir", type=Path, default=Path("models/sd15-vae"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.mode == "test" and not args.frozen_commit:
        parser.error("--mode test reads the sealed split; give --frozen-commit")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    role = roles(args.mode)
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
        .to(args.device)
    )
    inception = load_inception(args.inception_weights, torch.device(args.device))
    codes = dict(
        zip(
            [r["patient_id"] for r in read_manifest(MANIFEST)],
            subject_codes(read_manifest(MANIFEST)),
        )
    )

    splits = {}
    for split in dict.fromkeys(["train", *role.values()]):
        rows = ADNISliceDataset(ADNI_ROOT, MANIFEST, split, 64).rows
        labels = [r["class_name"] for r in rows]
        wanted_inception, wanted_images, extra = set(), set(), {}
        if split == role["heldout"]:
            wanted_inception |= set(range(len(rows)))
            extra["heldout_refs"] = patient_selection(rows, PATIENTS_PER_CLASS)
        if split == role["floor"]:
            extra["floor"] = balanced_sample(labels, FLOOR_PER_CLASS, seed=3710)
            extra["floor_refs"] = patient_selection(rows, PATIENTS_PER_CLASS)
            wanted_inception |= set(extra["floor"])
        if split == "train":
            extra["train_refs"] = patient_selection(rows, PATIENTS_PER_CLASS)
        for key in ("heldout_refs", "floor_refs"):
            wanted_images |= set(extra.get(key, []))
        splits[split] = encode_split(
            split, wanted_inception, wanted_images, vae, inception, args
        )
        splits[split].update(extra)
        splits[split]["subjects"] = torch.tensor([codes[r["patient_id"]] for r in rows])

    device = torch.device(args.device)
    train, heldout = splits["train"], splits[role["heldout"]]
    calibration, floor = splits[role["calibration"]], splits[role["floor"]]
    train_gray = train["gray64"].to(device).float() / 255
    heldout_gray = heldout["gray64"].to(device).float() / 255
    control, _ = nearest(train_gray[:2], train_gray)
    if control.max() > 1e-7:
        raise AssertionError("Exact reference-copy positive control failed")
    heldout_to_train, _ = nearest(heldout_gray, train_gray)
    calibration_gray = calibration["gray64"].to(device).float() / 255
    groups = (calibration["subjects"].to(device), heldout["subjects"].to(device))
    distances = {
        "pixel_train": nearest(calibration_gray, train_gray)[0],
        "pixel_heldout": nearest(calibration_gray, heldout_gray, *groups)[0],
        "vae_train": nearest(calibration["vae"].to(device), train["vae"].to(device))[0],
        "vae_heldout": nearest(
            calibration["vae"].to(device), heldout["vae"].to(device), *groups
        )[0],
    }
    distances = {k: v.cpu().numpy() for k, v in distances.items()}
    for key in ("pixel", "vae"):
        if not np.isfinite(distances[f"{key}_heldout"]).all():
            raise ValueError("A calibration slice found no held-out neighbour")

    def binary(labels):
        return np.array([label == "NC" for label in labels], dtype=np.int8)

    def pick(split, key):
        return np.array(split[key], dtype=np.int64)

    np.savez(
        args.output / "private-real-arrays.npz",
        train_gray64=train["gray64"].numpy(),
        train_vae=train["vae"].numpy(),
        heldout_gray64=heldout["gray64"].numpy(),
        heldout_vae=heldout["vae"].numpy(),
        heldout_is_nc=binary(heldout["labels"]),
    )
    heldout_refs, floor_refs = pick(heldout, "heldout_refs"), pick(floor, "floor_refs")
    floor_rows, train_refs = pick(floor, "floor"), pick(train, "train_refs")
    np.savez(
        args.output / "private-real-features.npz",
        heldout_inception=torch.stack(
            [heldout["inception"][i] for i in range(len(heldout["rows"]))]
        ).numpy(),
        heldout_vae=heldout["vae"].numpy(),
        heldout_is_nc=binary(heldout["labels"]),
        floor_inception=torch.stack(
            [floor["inception"][i] for i in floor_rows]
        ).numpy(),
        floor_vae=floor["vae"][floor_rows].numpy(),
        floor_is_nc=binary(floor["labels"])[floor_rows],
        heldout_ref_images=np.stack([heldout["images"][i] for i in heldout_refs]),
        heldout_ref_vae=heldout["vae"][heldout_refs].numpy(),
        heldout_ref_is_nc=binary(heldout["labels"])[heldout_refs],
        floor_ref_images=np.stack([floor["images"][i] for i in floor_refs]),
        floor_ref_vae=floor["vae"][floor_refs].numpy(),
        floor_ref_is_nc=binary(floor["labels"])[floor_refs],
        train_ref_vae=train["vae"][train_refs].numpy(),
        train_ref_is_nc=binary(train["labels"])[train_refs],
        calibration_is_nc=binary(calibration["labels"]),
        heldout_to_train_pixel=heldout_to_train.cpu().numpy(),
        **{f"calibration_{k}": v for k, v in distances.items()},
    )
    script = Path(__file__).resolve()
    report = {
        "protocol": "docs/final-evaluation-protocol.md",
        "mode": args.mode,
        "roles": role,
        "test_split_read": role["heldout"] == "test",
        "frozen_commit": args.frozen_commit,
        "counts": {
            split: {
                label: sum(x == label for x in data["labels"]) for label in ("AD", "NC")
            }
            for split, data in splits.items()
        },
        "floor_slices": len(floor_rows),
        "reference_patients": {
            "heldout": len(heldout_refs),
            "floor": len(floor_refs),
            "train": len(train_refs),
        },
        "heldout_to_train_pixel_rmse": summary(heldout_to_train.cpu().numpy()),
        "calibration_log_ratio": {
            key: float(
                np.mean(np.log(distances[f"{key}_train"] / distances[f"{key}_heldout"]))
            )
            for key in ("pixel", "vae")
        },
        "exact_copy_control_passed": True,
        "provenance_sha256": {
            "vae": hashlib.sha256(
                (args.vae_dir / "provenance.json").read_bytes()
            ).hexdigest(),
            "inception_weights": hashlib.sha256(
                args.inception_weights.read_bytes()
            ).hexdigest(),
            "manifest": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            **{
                name: hashlib.sha256((script.parent / name).read_bytes()).hexdigest()
                for name in ("final_common.py", "final_real_features.py")
            },
        },
        "seconds": time.perf_counter() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
