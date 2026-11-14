"""Post-hoc: does the GAN generator ignore its noise? (GPU job)

Tests the explanation of README Section 8, autopsy 3: B2 and B2b collapsed to
one template per class because the generator ignored its noise input. For
each generator and class it draws the noise of seeds 37200 onward exactly as
`baseline_gan.py generate` does and measures:

- input: the first layer is a linear map of [noise, class embedding], so its
  output splits exactly into a noise part W_z z and a class part W_y e; their
  norms say how much the noise still enters;
- every layer: the share of activation variance that is due to the noise
  (law of total variance over noise draws and units: mean over units of the
  variance across draws, divided by the total variance);
- output, on 64 px copies made as for every cohort: the same share, the mean
  pairwise RMSE within the class, the mean correlation of each sample with the
  class mean image, and how much of the image sits in the flat part of tanh.

Real training slices of each class, measured the same way at 64 px, are the
reference (training arrays of a rehearsal-mode final_real_features run; no
test data). With --check-cohort the first images are compared with the B2
cohort's PNGs to confirm the noise is drawn as in the cohort.

    python tools/gan_noise.py --generator b2=PATH --generator b2b-AD=PATH:AD \
        --real outputs/final-real-rehearsal-<job> --output DIR
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_gan import LABELS, load_generator  # noqa: E402
from dataset import ADNISliceDataset  # noqa: E402
from final_common import ADNI_ROOT, MANIFEST  # noqa: E402

SATURATED = 0.99  # |tanh| above this: the flat part, where gradients vanish


class Moments:
    """Running per-unit sums over noise draws, for the variance share."""

    def __init__(self):
        self.n, self.total, self.squares = 0, None, None

    def add(self, values: torch.Tensor) -> None:
        values = values.double().flatten(1)
        if self.total is None:
            self.total = torch.zeros(values.shape[1], dtype=torch.float64, device=values.device)
            self.squares = torch.zeros_like(self.total)
        self.n += len(values)
        self.total += values.sum(0)
        self.squares += (values**2).sum(0)

    def share(self) -> float:
        """Mean over units of the variance across draws / total variance."""
        mean = self.total / self.n
        within = (self.squares / self.n - mean**2).clamp_min(0).mean()
        overall = self.squares.sum() / (self.n * len(mean)) - (mean.mean()) ** 2
        return float(within / overall)


def image_measures(images: torch.Tensor) -> dict:
    """Share, pairwise RMSE and template correlation of [N, 4096] images in [0, 1]."""
    moments = Moments()
    moments.add(images)
    pairs = torch.pdist(images) / images.shape[1] ** 0.5
    centred = images - images.mean(1, keepdim=True)
    template = images.mean(0) - images.mean()
    correlation = (centred @ template) / (centred.norm(dim=1) * template.norm())
    return {
        "noise_variance_share": moments.share(),
        "within_class_pairwise_rmse": float(pairs.mean()),
        "correlation_with_class_mean": float(correlation.mean()),
    }


def as_cohort_gray64(image: torch.Tensor) -> torch.Tensor:
    """One [1, H, W] generator output -> 64 px grey in [0, 1], as for the cohorts."""
    pixels = ((image[0].clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
    small = Image.fromarray(pixels).convert("RGB").convert("L")
    small = small.resize((64, 64), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(small, dtype=np.float32).copy()).flatten() / 255


@torch.no_grad()
def measure(generator, state, label_index, seeds, device, check_folder=None):
    """All measures for one generator and class."""
    z_dim = state["z_dim"]
    weight = generator.project.weight  # [out, 2 * z_dim]: noise columns, then class
    embedding = generator.label_embedding.weight[label_index]
    class_part = weight[:, z_dim:] @ embedding
    layers = {"project_norm": generator.project_norm}
    layers.update({f"block{i + 1}": b for i, b in enumerate(generator.blocks)})
    layers["to_image"] = generator.to_image
    moments = {name: Moments() for name in layers}
    captured = {}
    hooks = [
        module.register_forward_hook(
            lambda _m, _i, out, name=name: captured.__setitem__(name, out)
        )
        for name, module in layers.items()
    ]
    noise_norms, gray, saturated, mismatches = [], [], [], []
    for seed in seeds:
        z = torch.randn(
            (1, z_dim),
            generator=torch.Generator(device=device).manual_seed(seed),
            device=device,
        )
        image = generator(z, torch.tensor([label_index], device=device))
        noise_norms.append(float((weight[:, :z_dim] @ z[0]).norm()))
        for name in layers:
            moments[name].add(captured[name])
        saturated.append(float((image.abs() > SATURATED).float().mean()))
        small = as_cohort_gray64(image[0])
        gray.append(small)
        if check_folder is not None and seed < seeds[0] + 16:
            name = f"gan-{LABELS[label_index]}-{seed}.png"
            pixels = ((image[0, 0].clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
            with Image.open(check_folder / name) as saved:
                stored = np.asarray(saved.convert("L"))
            mismatches.append(int(np.abs(stored.astype(int) - pixels.astype(int)).max()))
    for hook in hooks:
        hook.remove()
    result = {
        "first_layer": {
            "noise_part_norm_mean": float(np.mean(noise_norms)),
            "class_part_norm": float(class_part.norm()),
            "noise_weight_norm": float(weight[:, :z_dim].norm()),
            "class_weight_norm": float(weight[:, z_dim:].norm()),
        },
        "noise_variance_share_by_layer": {n: m.share() for n, m in moments.items()},
        "saturated_output_fraction": float(np.mean(saturated)),
        "output_64px": image_measures(torch.stack(gray).to(device)),
    }
    if mismatches:
        result["max_pixel_difference_to_cohort_png"] = max(mismatches)
    return result


def main() -> None:
    """Measure every generator and the real reference; write report.json."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--generator",
        action="append",
        required=True,
        metavar="NAME=PATH[:CLASS]",
        help="a per-class generator names the one class it was trained on",
    )
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--check-cohort", type=Path, help="B2 cohort folder")
    parser.add_argument("--count", type=int, default=512, help="noise draws per class")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    real_report = json.loads((args.real / "report.json").read_text())
    if real_report["test_split_read"]:
        raise ValueError("--real must be a rehearsal-mode run")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    seeds = list(range(37200, 37200 + args.count))

    generators = {}
    for item in args.generator:
        name, source = item.split("=", 1)
        path, _, only = source.partition(":")
        generator, state = load_generator(Path(path), device)
        classes = [only] if only else list(LABELS)
        generators[name] = {
            "checkpoint": path,
            "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "step": state["step"],
            "classes": {
                label: measure(
                    generator,
                    state,
                    LABELS.index(label),
                    seeds,
                    device,
                    args.check_cohort if name == "b2" else None,
                )
                for label in classes
            },
        }
        print(name, json.dumps(generators[name]), flush=True)

    # Real reference: training slices of each class at 64 px.
    labels = [r["class_name"] for r in ADNISliceDataset(ADNI_ROOT, MANIFEST, "train", 256).rows]
    train = np.load(args.real / "private-real-arrays.npz")["train_gray64"]
    if len(labels) != len(train):
        raise ValueError("Training arrays and manifest rows disagree")
    random = np.random.default_rng(3710)
    real = {}
    for label in LABELS:
        rows = np.flatnonzero(np.array(labels) == label)
        chosen = np.sort(random.choice(rows, args.count, replace=False))
        images = torch.from_numpy(train[chosen]).to(device).float() / 255
        real[label] = image_measures(images)

    script = Path(__file__).resolve()
    report = {
        "post_hoc": True,
        "noise_draws_per_class": args.count,
        "first_seed": seeds[0],
        "saturation_threshold": SATURATED,
        "generators": generators,
        "real_training_slices_64px": real,
        "test_split_read": False,
        "code_sha256": {
            n: hashlib.sha256((script.parent / n).read_bytes()).hexdigest()
            for n in ("gan_noise.py", "baseline_gan.py")
        },
        "seconds": time.perf_counter() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["real_training_slices_64px"], indent=2))


if __name__ == "__main__":
    main()
