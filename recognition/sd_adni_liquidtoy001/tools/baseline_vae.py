"""Small conditional convolutional VAE, trained from scratch at 128 pixels.

Analytic Gaussian KL follows the standard VAE objective, as illustrated by
https://github.com/pytorch/examples/blob/main/vae/main.py .
Here Gaussian reconstruction uses summed squared error, not Bernoulli BCE.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from itertools import pairwise
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import ADNISliceDataset


class SmallVAE(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (1, 16, 32, 64, 128, 128)
        layers = []
        for left, right in pairwise(channels):
            layers.extend((nn.Conv2d(left, right, 4, 2, 1), nn.SiLU()))
        self.encoder = nn.Sequential(*layers, nn.Flatten())
        self.mean = nn.Linear(128 * 4 * 4, 64)
        self.log_variance = nn.Linear(128 * 4 * 4, 64)
        self.expand = nn.Linear(66, 128 * 4 * 4)
        channels = (128, 128, 64, 32, 16, 1)
        layers = []
        for index, (left, right) in enumerate(pairwise(channels)):
            layers.append(nn.ConvTranspose2d(left, right, 4, 2, 1))
            layers.append(nn.Sigmoid() if index == 4 else nn.SiLU())
        self.decoder = nn.Sequential(*layers)

    def decode(self, latent, labels):
        condition = torch.nn.functional.one_hot(labels, 2).to(latent.dtype)
        hidden = self.expand(torch.cat((latent, condition), dim=1))
        return self.decoder(hidden.reshape(-1, 128, 4, 4))

    def forward(self, pixels, labels):
        hidden = self.encoder(pixels)
        mean, log_variance = self.mean(hidden), self.log_variance(hidden)
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        return self.decode(latent, labels), mean, log_variance


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def generate(model, output, count=32, device="cuda"):
    """Seeds 37200 onwards, `count` per class; the grid shows the first 32."""
    model.eval()
    records = []
    grid = Image.new("L", (128 * 8, 152 * 8), 255)
    draw = ImageDraw.Draw(grid)
    for label_index, label in enumerate(("AD", "NC")):
        for index, seed in enumerate(range(37200, 37200 + count)):
            generator = torch.Generator(device=device).manual_seed(seed)
            latent = torch.randn((1, 64), generator=generator, device=device)
            pixels = model.decode(latent, torch.tensor([label_index], device=device))
            if not torch.isfinite(pixels).all():
                raise RuntimeError("Non-finite generation")
            image = Image.fromarray(pixels[0, 0].mul(255).round().byte().cpu().numpy())
            name = f"vae-{label}-{seed}.png"
            image.save(output / name)
            if index < 32:
                x, y = (index % 8) * 128, (label_index * 4 + index // 8) * 152
                grid.paste(image, (x, y))
                draw.text((x + 2, y + 130), f"{label} {seed}", fill=0)
            records.append(
                {
                    "file": name,
                    "label": label,
                    "seed": seed,
                    "png_sha256": digest(output / name),
                }
            )
    grid.save(output / "grid.png")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(3710)
    torch.cuda.manual_seed_all(3710)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    dataset = ADNISliceDataset(
        Path("/home/groups/comp3710/ADNI"),
        Path("data/manifest-seed3710.csv"),
        "train",
        128,
    )
    contract = {
        "model": "small-conditional-vae128-latent64",
        "manifest": digest("data/manifest-seed3710.csv"),
        "code": digest(__file__),
        "dataset_code": digest("dataset.py"),
        "batch": 32,
        "lr": 0.001,
        "seed": 3710,
        "epochs": 5,
        "loss": "sum squared pixels + sum KL per sample; batch mean",
        "gate": args.gate,
    }
    model = SmallVAE().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_epoch, traces = 0, []
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["contract"] != contract:
            raise ValueError("Resume contract differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        traces = state["traces"]
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    initial = next(model.parameters()).detach().clone()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(start_epoch, 1 if args.gate else 5):
        generator = torch.Generator().manual_seed(3710 + epoch)
        loader = DataLoader(
            dataset, batch_size=32, shuffle=True, generator=generator, num_workers=0
        )
        model.train()
        total, reconstruction_total, kl_total, seen = 0.0, 0.0, 0.0, 0
        for batch_index, batch in enumerate(loader):
            pixels = (batch["pixel_values"][:, :1].to("cuda") + 1) / 2
            labels = torch.tensor(
                [0 if x == "AD" else 1 for x in batch["class_name"]], device="cuda"
            )
            optimizer.zero_grad(set_to_none=True)
            reconstructed, mean, log_variance = model(pixels, labels)
            reconstruction = (reconstructed - pixels).square().flatten(1).sum(1).mean()
            kl = (
                -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(1)
            ).mean()
            loss = reconstruction + kl
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite VAE objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            if not norm > 0:
                raise RuntimeError("Zero gradient")
            optimizer.step()
            count = len(pixels)
            total += loss.item() * count
            reconstruction_total += reconstruction.item() * count
            kl_total += kl.item() * count
            seen += count
            if args.gate and batch_index == 1:
                break
        trace = {
            "epoch": epoch + 1,
            "samples": seen,
            "loss": total / seen,
            "reconstruction_sse": reconstruction_total / seen,
            "kl": kl_total / seen,
        }
        traces.append(trace)
        print(json.dumps(trace), flush=True)
        checkpoint = args.output / f"epoch-{epoch + 1}.pt"
        atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "epoch": epoch + 1,
                "traces": traces,
                "contract": contract,
            },
            checkpoint,
        )
    if torch.equal(initial, next(model.parameters()).detach()):
        raise AssertionError("No actual parameter update")
    if args.gate:
        restored = SmallVAE().to("cuda")
        restored.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        )
        with torch.no_grad():
            z = torch.zeros((2, 64), device="cuda")
            c = torch.tensor([0, 1], device="cuda")
            torch.testing.assert_close(
                model.decode(z, c), restored.decode(z, c), atol=0, rtol=0
            )
        records = []
    else:
        records = generate(model, args.output)
    report = {
        "gate": args.gate,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": digest(checkpoint),
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "seconds": time.perf_counter() - started,
        "traces": traces,
        "contract": contract,
        "records": records,
        "gate_save_reload_equal": args.gate,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
