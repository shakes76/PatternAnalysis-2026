"""Shared pieces of the final evaluation (docs/final-evaluation-protocol.md).

Roles of the real data:

- train: the 21,280 training slices (nearest-neighbour reference).
- heldout: the split the models are judged against. The test split in the
  final run; the validation split in the rehearsal, so the rehearsal never
  reads a test slice.
- calibration: real held-out slices measured exactly like generated ones, for
  the training-preference ratio (validation in both modes; in the rehearsal
  it coincides with heldout, so each slice's own subject is excluded).
- floor: real slices treated as a perfect generator (validation in the final
  run, a training sample in the rehearsal).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ADNI_ROOT = Path("/home/groups/comp3710/ADNI")
MANIFEST = Path("data/manifest-seed3710.csv")
LABELS = ("AD", "NC")
# One slice per patient for SSIM references, as in every earlier SSIM audit.
PATIENTS_PER_CLASS = {"AD": 33, "NC": 69}
FLOOR_PER_CLASS = 512
INCEPTION_SIZE = 299
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ROLES = {
    "test": {"heldout": "test", "calibration": "validation", "floor": "validation"},
    "rehearsal": {
        "heldout": "validation",
        "calibration": "validation",
        "floor": "train",
    },
}


def roles(mode: str) -> dict[str, str]:
    if mode not in ROLES:
        raise ValueError(f"mode must be one of {tuple(ROLES)}")
    return ROLES[mode]


def balanced_sample(labels: list[str], per_class: int, seed: int) -> list[int]:
    """`per_class` row indices of each label, drawn without replacement."""
    random = np.random.default_rng(seed)
    chosen = []
    for label in LABELS:
        candidates = np.array([i for i, x in enumerate(labels) if x == label])
        if len(candidates) < per_class:
            raise ValueError(f"Only {len(candidates)} {label} rows for {per_class}")
        chosen.extend(random.choice(candidates, per_class, replace=False).tolist())
    return sorted(chosen)


def subject_codes(rows: list[dict]) -> np.ndarray:
    """An integer per subject, so private arrays need no patient IDs."""
    order = {p: i for i, p in enumerate(sorted({r["patient_id"] for r in rows}))}
    return np.array([order[r["patient_id"]] for r in rows], dtype=np.int32)


def cohort_records(folder: Path) -> list[dict]:
    """The cohort's records, each PNG checked against its recorded SHA-256."""
    records = json.loads((folder / "report.json").read_text())["records"]
    for record in records:
        content = (folder / record["file"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != record["png_sha256"]:
            raise ValueError(f"Generated image hash mismatch: {record['file']}")
    return records


def generated_views(path: Path) -> dict[str, torch.Tensor]:
    """The three views every earlier audit used, from one generated PNG.

    gray64: grayscale 64 px in [0, 1] (pixel nearest neighbours);
    rgb256: RGB 256 px in [-1, 1] (VAE features);
    rgb: RGB at the stored size in [0, 1] (Inception features).
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        gray64 = np.asarray(
            rgb.convert("L").resize((64, 64), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        rgb256 = np.asarray(rgb.resize((256, 256), Image.Resampling.BILINEAR))
        native = np.asarray(rgb)
    return {
        "gray64": torch.from_numpy(gray64.copy()).flatten() / 255,
        "rgb256": torch.from_numpy(rgb256.copy()).permute(2, 0, 1).float() / 127.5 - 1,
        "rgb": torch.from_numpy(native.copy()).permute(2, 0, 1).float() / 255,
    }


def inception_input(rgb: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] in [0, 1] -> 299 px, ImageNet-normalised."""
    resized = F.interpolate(
        rgb,
        size=(INCEPTION_SIZE, INCEPTION_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor(IMAGENET_MEAN, device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb.device).view(1, 3, 1, 1)
    return (resized - mean) / std


def load_inception(weights: Path, device: torch.device) -> torch.nn.Module:
    """torchvision's Inception-v3 (ImageNet weights), returning the 2,048-d pool."""
    from torchvision.models import inception_v3

    model = inception_v3(
        weights=None, aux_logits=True, transform_input=True, init_weights=False
    )
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    return model.eval().requires_grad_(False).to(device)


@torch.no_grad()
def vae_features(vae, rgb256: torch.Tensor) -> torch.Tensor:
    """Posterior mean pooled to 4x8x8, the feature of every earlier audit."""
    latent = vae.encode(rgb256).latent_dist.mode()
    return F.adaptive_avg_pool2d(latent, (8, 8)).flatten(1)


@torch.no_grad()
def nearest(
    query: torch.Tensor,
    references: torch.Tensor,
    query_groups: torch.Tensor | None = None,
    reference_groups: torch.Tensor | None = None,
    chunk: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSE to the nearest reference and its index, per query row.

    References in the same group as the query (its own subject) are skipped
    when groups are given. Direct differences, not the matrix-product shortcut,
    so an exact copy gives exactly zero, as in audit_cohort.py.
    """
    distances, indices = [], []
    scale = query.shape[1] ** 0.5
    for start in range(0, len(query), chunk):
        block = torch.cdist(
            query[start : start + chunk],
            references,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        if query_groups is not None:
            same = query_groups[start : start + chunk, None] == reference_groups[None]
            block = block.masked_fill(same, float("inf"))
        value, index = block.min(dim=1)
        distances.append(value / scale)
        indices.append(index)
    return torch.cat(distances), torch.cat(indices)
