"""B2: a conditional DCGAN-style GAN baseline at 128 px (docs/b2-gan-protocol.md).

Adapted from the student's own COMP3710 demonstration-2 GAN (OASIS), which
settled on this recipe after three collapses: an upsample-then-convolve
generator with BatchNorm, a discriminator without normalisation, the
non-saturating logistic loss with an R1 penalty (Mescheder et al., 2018), Adam
with beta1 = 0, an EMA of the generator, and a minibatch standard-deviation
feature (Karras et al., 2018). New here: AD/NC conditioning (a label embedding
in the generator, a projection term in the discriminator; Miyato & Koyama,
2018), ADNI data with E2's defective-scan exclusion, and output in the cohort
format the audit tools read.

    python tools/baseline_gan.py train --output outputs/b2-<job> --steps 30000
    python tools/baseline_gan.py generate --checkpoint outputs/b2-<job>/final.pt \
        --output outputs/b2-cohort-<job>
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import (  # noqa: E402
    ADNISliceDataset,
    defective_scans,
    foreground_share,
)

LABELS = ("AD", "NC")
MAX_CHANNELS = 512
DIVERSITY_BATCH, DIVERSITY_SIZE = 64, 64  # demonstration 2's measure


def channels_at(size: int, resolution: int, width: int) -> int:
    """Channels double each time the resolution halves, capped at MAX_CHANNELS."""
    return min(MAX_CHANNELS, width * resolution // size)


class UpBlock(nn.Module):
    """Nearest upsample, then two 3x3 convolutions: no transposed-conv checkerboards."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class ConditionalGenerator(nn.Module):
    """(noise, label) -> one grayscale image in [-1, 1]."""

    def __init__(self, resolution: int = 128, z_dim: int = 128, width: int = 32):
        super().__init__()
        self.resolution, self.z_dim = resolution, z_dim
        self.label_embedding = nn.Embedding(len(LABELS), z_dim)
        self.start = channels_at(4, resolution, width)
        self.project = nn.Linear(2 * z_dim, self.start * 16, bias=False)
        self.project_norm = nn.Sequential(
            nn.BatchNorm2d(self.start), nn.ReLU(inplace=True)
        )
        blocks, size = [], 4
        while size < resolution:
            blocks.append(
                UpBlock(
                    channels_at(size, resolution, width),
                    channels_at(size * 2, resolution, width),
                )
            )
            size *= 2
        self.blocks = nn.Sequential(*blocks)
        self.to_image = nn.Conv2d(
            channels_at(resolution, resolution, width), 1, 3, padding=1
        )

    def forward(self, z, labels):
        hidden = self.project(torch.cat([z, self.label_embedding(labels)], dim=1))
        hidden = self.project_norm(hidden.view(-1, self.start, 4, 4))
        return torch.tanh(self.to_image(self.blocks(hidden)))


class MinibatchStd(nn.Module):
    """Append the spread of a group of images as an extra channel, so the
    discriminator can see that a collapsed batch is too uniform."""

    def __init__(self, group: int = 4):
        super().__init__()
        self.group = group

    def forward(self, x):
        batch, channels, height, width = x.shape
        group = min(self.group, batch)
        if batch % group:
            raise ValueError(f"batch {batch} is not divisible by group {group}")
        grouped = x.view(group, -1, channels, height, width)
        spread = (grouped.var(dim=0, unbiased=False) + 1e-8).sqrt().mean(dim=(1, 2, 3))
        feature = spread.view(-1, 1, 1, 1).repeat(group, 1, height, width)
        return torch.cat([x, feature], dim=1)


class ConditionalDiscriminator(nn.Module):
    """(image, label) -> one unbounded realness score.

    The label enters as a projection: score = w . phi(x) + embed(y) . phi(x)
    (Miyato & Koyama, 2018), so the discriminator judges "a real AD slice"
    rather than only "a real slice". No normalisation layers: R1 constrains it.
    """

    def __init__(self, resolution: int = 128, width: int = 32):
        super().__init__()
        layers = [
            nn.Conv2d(1, channels_at(resolution, resolution, width), 3, padding=1)
        ]
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        size = resolution
        while size > 4:
            left, right = (
                channels_at(size, resolution, width),
                channels_at(size // 2, resolution, width),
            )
            layers += [
                nn.Conv2d(left, left, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(left, right, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            size //= 2
        self.features = nn.Sequential(*layers)
        end = channels_at(4, resolution, width)
        self.minibatch_std = MinibatchStd()
        self.head = nn.Sequential(
            nn.Conv2d(end + 1, end, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        self.score = nn.Linear(end * 16, 1)
        self.label_projection = nn.Embedding(len(LABELS), end * 16)

    def forward(self, images, labels):
        phi = self.head(self.minibatch_std(self.features(images)))
        projection = (self.label_projection(labels) * phi).sum(dim=1)
        return self.score(phi).squeeze(1) + projection


def discriminator_loss(real_scores, fake_scores):
    """-log sigmoid(real) - log(1 - sigmoid(fake)), written with softplus."""
    return F.softplus(-real_scores).mean() + F.softplus(fake_scores).mean()


def generator_loss(fake_scores):
    """Non-saturating: -log sigmoid(fake), strong when the generator is poor."""
    return F.softplus(-fake_scores).mean()


def r1_penalty(real_scores, real_images):
    """Squared gradient of the score on real images (Mescheder et al., 2018)."""
    (gradient,) = torch.autograd.grad(real_scores.sum(), real_images, create_graph=True)
    return gradient.pow(2).flatten(1).sum(1).mean()


@torch.no_grad()
def update_ema(shadow: nn.Module, live: nn.Module, decay: float) -> None:
    for averaged, current in zip(shadow.parameters(), live.parameters()):
        averaged.lerp_(current, 1.0 - decay)


@torch.no_grad()
def recalibrate_batchnorm(
    generator, device, seed: int, batches: int = 32, size: int = 64, label=None
):
    """A copy whose BatchNorm statistics match its own (averaged) weights.

    Seeded, so the same weights always give the same calibrated generator.
    `label` fixes the label for a per-class GAN: the other label's embedding
    was never trained, and mixing it in would distort the statistics.
    """
    calibrated = copy.deepcopy(generator).to(device)
    for module in calibrated.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            module.momentum = None  # cumulative average
    random = torch.Generator(device=device).manual_seed(seed)
    calibrated.train()
    for _ in range(batches):
        z = torch.randn(size, calibrated.z_dim, generator=random, device=device)
        if label is None:
            labels = torch.randint(
                0, len(LABELS), (size,), generator=random, device=device
            )
        else:
            labels = torch.full((size,), label, device=device)
        calibrated(z, labels)
    return calibrated.eval()


@torch.no_grad()
def mean_pairwise_distance(images: torch.Tensor) -> float:
    """Mean Euclidean distance between all pairs, on 64x64 copies in [0, 1]."""
    small = F.interpolate(
        (images.float() + 1) / 2, size=(DIVERSITY_SIZE,) * 2, mode="area"
    )
    flat = small.flatten(1)
    count = flat.shape[0]
    return (torch.cdist(flat, flat).sum() / (count * (count - 1))).item()


class CollapseRule:
    """The protocol's rule: from `after` on, keep a snapshot whenever the
    diversity ratio is at least `healthy`; stop after `patience` consecutive
    ratios below `collapse`."""

    def __init__(self, after=5000, healthy=0.8, collapse=0.6, patience=2):
        self.after, self.healthy, self.collapse, self.patience = (
            after,
            healthy,
            collapse,
            patience,
        )
        self.low, self.snapshot_step, self.stopped_at = 0, None, None

    def observe(self, step: int, ratio: float) -> tuple[bool, bool]:
        """Returns (save a snapshot now, stop training now)."""
        if step < self.after:
            return False, False
        save = ratio >= self.healthy
        if save:
            self.snapshot_step = step
        self.low = self.low + 1 if ratio < self.collapse else 0
        if self.low >= self.patience:
            self.stopped_at = step
        return save, self.stopped_at is not None


def load_training_images(args, device):
    """E2's training slices: hash-checked, defective scans excluded at 256 px,
    then area-downsampled to the GAN's resolution."""
    data = ADNISliceDataset(
        args.data_root, args.manifest, "train", 256, verify_hash=True
    )
    images, labels, shares = [], [], []
    for batch in DataLoader(data, batch_size=128, num_workers=args.num_workers):
        pixels = batch["pixel_values"][:, :1]
        shares.extend(foreground_share(batch["pixel_values"]).tolist())
        images.append(F.interpolate(pixels, size=(args.resolution,) * 2, mode="area"))
        labels.extend(LABELS.index(name) for name in batch["class_name"])
    flagged = defective_scans(data.rows, shares)
    keep = torch.tensor([row["scan_id"] not in flagged for row in data.rows])
    excluded = {"scans": len(flagged), "slices": int((~keep).sum())}
    if (
        args.expect_excluded
        and [excluded["scans"], excluded["slices"]] != args.expect_excluded
    ):
        raise SystemExit(
            f"Excluded {excluded}, protocol expects {args.expect_excluded}"
        )
    labels = torch.tensor(labels)
    if args.train_class:
        # B2b: one GAN per class. The exclusion above still runs on the whole
        # split, so its expected count is the same as for every other model.
        keep &= labels == LABELS.index(args.train_class)
    images = torch.cat(images)[keep].to(device)
    labels = labels[keep].to(device)
    return images, labels, excluded


def save_grid(images, labels, path, columns=8):
    rows = (len(images) + columns - 1) // columns
    size = images.shape[-1]
    grid = Image.new("L", (columns * size, rows * (size + 14)), 255)
    draw = ImageDraw.Draw(grid)
    pixels = ((images[:, 0].clamp(-1, 1) + 1) * 127.5).round().byte().cpu()
    for i, (tile, label) in enumerate(zip(pixels, labels.tolist())):
        x, y = (i % columns) * size, (i // columns) * (size + 14)
        grid.paste(
            Image.frombytes("L", (size, size), bytes(tile.flatten().tolist())), (x, y)
        )
        draw.text((x + 2, y + size + 1), LABELS[label], fill=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    images, labels, excluded = load_training_images(args, device)
    print(json.dumps({"excluded": excluded, "train_images": len(images)}), flush=True)
    generator = ConditionalGenerator(args.resolution, args.z_dim, args.width).to(device)
    discriminator = ConditionalDiscriminator(args.resolution, args.width).to(device)
    ema = copy.deepcopy(generator).eval().requires_grad_(False)
    optimiser_g = torch.optim.Adam(
        generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    optimiser_d = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    random = torch.Generator(device=device).manual_seed(args.seed)
    fixed_z = torch.randn(16, args.z_dim, generator=random, device=device)
    # A per-class GAN only ever sees its own label, so its label input is a
    # constant and every batch and every collapse check stays within the class.
    single = LABELS.index(args.train_class) if args.train_class else None

    def label_batch(size, alternating=False):
        if single is not None:
            return torch.full((size,), single, device=device)
        if alternating:
            return torch.arange(size, device=device) % 2
        return torch.randint(0, 2, (size,), generator=random, device=device)

    fixed_labels = (
        torch.arange(16, device=device) // 8 if single is None else label_batch(16)
    )
    rule = CollapseRule(after=args.collapse_after)
    history, diversity = [], []
    window = {"d_loss": 0.0, "g_loss": 0.0, "r1": 0.0, "count": 0}
    setup_seconds = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    step, stop_reason = 0, "completed"

    def checkpoint(model, step_number, path, ratio=None):
        calibrated = recalibrate_batchnorm(model, device, seed=args.seed, label=single)
        torch.save(
            {
                "generator": calibrated.state_dict(),
                "resolution": args.resolution,
                "z_dim": args.z_dim,
                "width": args.width,
                "step": step_number,
                "diversity_ratio": ratio,
            },
            path,
        )
        return calibrated

    while step < args.steps:
        index = torch.randint(
            0, len(images), (args.batch_size,), generator=random, device=device
        )
        real, real_labels = images[index].requires_grad_(True), labels[index]
        z = torch.randn(args.batch_size, args.z_dim, generator=random, device=device)
        fake_labels = label_batch(args.batch_size)
        with torch.no_grad():
            fake = generator(z, fake_labels)
        real_scores = discriminator(real, real_labels)
        fake_scores = discriminator(fake, fake_labels)
        r1 = r1_penalty(real_scores, real)
        d_loss = discriminator_loss(real_scores, fake_scores) + 0.5 * args.r1_gamma * r1
        optimiser_d.zero_grad(set_to_none=True)
        d_loss.backward()
        optimiser_d.step()

        z = torch.randn(args.batch_size, args.z_dim, generator=random, device=device)
        g_loss = generator_loss(discriminator(generator(z, fake_labels), fake_labels))
        optimiser_g.zero_grad(set_to_none=True)
        g_loss.backward()
        optimiser_g.step()
        update_ema(ema, generator, args.ema_decay)
        step += 1
        window["d_loss"] += d_loss.item()
        window["g_loss"] += g_loss.item()
        window["r1"] += r1.item()
        window["count"] += 1

        if step % args.log_every == 0:
            row = {k: window[k] / window["count"] for k in ("d_loss", "g_loss", "r1")}
            row["step"] = step
            row["seconds"] = time.perf_counter() - train_started
            history.append(row)
            print(json.dumps(row), flush=True)
            window = {"d_loss": 0.0, "g_loss": 0.0, "r1": 0.0, "count": 0}

        if step % args.sample_every == 0 or step == args.steps:
            sampler = recalibrate_batchnorm(ema, device, seed=args.seed, label=single)
            with torch.no_grad():
                save_grid(
                    sampler(fixed_z, fixed_labels),
                    fixed_labels,
                    args.output / "progress" / f"step-{step:06d}.png",
                )
                z = torch.randn(
                    DIVERSITY_BATCH, args.z_dim, generator=random, device=device
                )
                generated = sampler(z, label_batch(DIVERSITY_BATCH, alternating=True))
                real_index = torch.randperm(
                    len(images), generator=random, device=device
                )[:DIVERSITY_BATCH]
                ratio = mean_pairwise_distance(generated) / mean_pairwise_distance(
                    images[real_index]
                )
            save, stop = rule.observe(step, ratio)
            diversity.append({"step": step, "ratio": ratio, "snapshot": save})
            print(json.dumps(diversity[-1]), flush=True)
            if save:
                checkpoint(ema, step, args.output / "snapshot.pt", ratio)
            if stop:
                stop_reason = "collapse"
                break
            minutes = (time.perf_counter() - started) / 60
            if args.max_minutes and minutes > args.max_minutes:
                stop_reason = "time_budget"
                break

    train_seconds = time.perf_counter() - train_started
    if stop_reason == "collapse":
        if rule.snapshot_step is None:
            raise SystemExit("Collapsed before any healthy snapshot; no result to keep")
        (args.output / "snapshot.pt").replace(args.output / "final.pt")
        result_step = rule.snapshot_step
    else:
        checkpoint(
            ema,
            step,
            args.output / "final.pt",
            diversity[-1]["ratio"] if diversity else None,
        )
        result_step = step
    for name, rows in (("history.csv", history), ("diversity.csv", diversity)):
        if rows:
            with (args.output / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    plot_curves(history, diversity, args.output / "curves.png")
    metrics = {
        "protocol": "docs/b2-gan-protocol.md",
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "excluded": excluded,
        "train_images": len(images),
        "steps": step,
        "stop_reason": stop_reason,
        "result_step": result_step,
        "collapse_stopped_at": rule.stopped_at,
        "generator_parameters": sum(p.numel() for p in generator.parameters()),
        "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()),
        "setup_seconds": setup_seconds,
        "train_seconds": train_seconds,
        "images_per_second": step * args.batch_size / train_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
        if device.type == "cuda"
        else None,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch": torch.__version__,
        "test_split_read": False,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in ("stop_reason", "steps", "result_step", "images_per_second")
            }
        )
    )


def plot_curves(history, diversity, path):
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    steps = [r["step"] for r in history]
    left.plot(steps, [r["d_loss"] for r in history], lw=0.8, label="discriminator")
    left.plot(steps, [r["g_loss"] for r in history], lw=0.8, label="generator")
    left.set(xlabel="step", ylabel="loss", title="GAN losses")
    left.legend()
    if diversity:
        right.plot(
            [r["step"] for r in diversity], [r["ratio"] for r in diversity], marker="o"
        )
        right.axhline(0.8, ls="--", color="green", lw=0.8, label="healthy (0.8)")
        right.axhline(0.6, ls="--", color="red", lw=0.8, label="collapse (0.6)")
        right.legend()
    right.set(xlabel="step", ylabel="generated / real", title="Diversity ratio")
    for axis in (left, right):
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def load_generator(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    generator = ConditionalGenerator(
        state["resolution"], state["z_dim"], state["width"]
    )
    generator.load_state_dict(state["generator"])
    return generator.to(device).eval(), state


@torch.no_grad()
def generate(args):
    """Fixed seeds from 37200, `count` per class, as the generator's noise.

    Either one conditional checkpoint (B2), or one checkpoint per class (B2b,
    --class-checkpoint AD=PATH --class-checkpoint NC=PATH). With --allow-partial,
    a class without a result is left out, as the B2b protocol allows.
    """
    device = torch.device(args.device)
    if bool(args.checkpoint) == bool(args.class_checkpoint):
        raise SystemExit("Give either --checkpoint or one --class-checkpoint per class")
    if args.checkpoint:
        sources = {label: args.checkpoint for label in LABELS}
    else:
        sources = dict(item.split("=", 1) for item in args.class_checkpoint)
        if not set(sources) <= set(LABELS) or (
            set(sources) != set(LABELS) and not args.allow_partial
        ):
            raise SystemExit(f"Need a checkpoint for each of {LABELS}")
        sources = {label: Path(path) for label, path in sources.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    records, tiles, tile_labels, used = [], [], [], {}
    for label_index, label in enumerate(LABELS):
        if label not in sources:
            continue
        generator, state = load_generator(sources[label], device)
        used[label] = {
            "checkpoint": str(sources[label]),
            "checkpoint_sha256": hashlib.sha256(
                sources[label].read_bytes()
            ).hexdigest(),
            "checkpoint_step": state["step"],
        }
        for seed in range(args.first_seed, args.first_seed + args.count):
            z = torch.randn(
                (1, state["z_dim"]),
                generator=torch.Generator(device=device).manual_seed(seed),
                device=device,
            )
            image = generator(z, torch.tensor([label_index], device=device))
            pixels = ((image[0, 0].clamp(-1, 1) + 1) * 127.5).round().byte().cpu()
            name = f"gan-{label}-{seed}.png"
            size = pixels.shape[-1]
            Image.frombytes("L", (size, size), bytes(pixels.flatten().tolist())).save(
                args.output / name
            )
            records.append(
                {
                    "file": name,
                    "label": label,
                    "seed": seed,
                    "png_sha256": hashlib.sha256(
                        (args.output / name).read_bytes()
                    ).hexdigest(),
                }
            )
            if seed < args.first_seed + 32:  # the grid shows the first 32 per class
                tiles.append(image)
                tile_labels.append(label_index)
    save_grid(torch.cat(tiles), torch.tensor(tile_labels), args.output / "grid.png")
    report = {
        "checkpoints": used,
        "resolution": state["resolution"],
        "records": records,
        "scope": f"{len(records)} predeclared images; no test-split data read",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"images": len(records), "checkpoints": used}))


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--steps", type=int, default=30000)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--resolution", type=int, default=128)
    train_parser.add_argument("--z-dim", type=int, default=128)
    train_parser.add_argument("--width", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--r1-gamma", type=float, default=10.0)
    train_parser.add_argument("--ema-decay", type=float, default=0.999)
    train_parser.add_argument("--log-every", type=int, default=100)
    train_parser.add_argument("--sample-every", type=int, default=1000)
    train_parser.add_argument("--collapse-after", type=int, default=5000)
    train_parser.add_argument("--max-minutes", type=float)
    train_parser.add_argument(
        "--expect-excluded", type=int, nargs=2, metavar=("SCANS", "SLICES")
    )
    train_parser.add_argument(
        "--train-class", choices=LABELS, help="B2b: train on one class only"
    )
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=3710)
    train_parser.add_argument(
        "--data-root", type=Path, default=Path("/home/groups/comp3710/ADNI")
    )
    train_parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifest-seed3710.csv")
    )
    train_parser.add_argument("--device", default="cuda")
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--checkpoint", type=Path)
    generate_parser.add_argument(
        "--class-checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="B2b: one per class instead of --checkpoint",
    )
    generate_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="B2b: generate only the classes given a --class-checkpoint",
    )
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--first-seed", type=int, default=37200)
    generate_parser.add_argument("--count", type=int, default=32)
    generate_parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments) if arguments.command == "train" else generate(arguments)
