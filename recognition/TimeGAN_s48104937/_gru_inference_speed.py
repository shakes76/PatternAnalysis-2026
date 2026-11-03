"""Inference speed of the saved AR-GRU baseline: time to generate one window of N steps.

Times the generator only (no rescaling or decoding into books), after a warm-up,
synchronising the GPU before each clock read. Reports a single window (latency)
and a large batch (throughput), on every available device.
"""
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

CHECKPOINT_PATH = Path("models/saved/baseline.pt")

WINDOW_LENGTH: int | None = None   # None = the length the baseline was trained on
BATCH_SIZES = (1, 1000)            # 1 = latency for a single window; large = throughput
WARMUP: int = 3
REPEATS: int = 20
SEED: int = 0


class Generator(nn.Module):
    """Same layers and sampling as the baseline's Generator, so its state dict loads as is."""

    def __init__(self, n_features: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, 2 * n_features)

    def step(self, x: torch.Tensor, h: torch.Tensor | None = None):
        y, h = self.gru(x, h)
        mean, logvar = self.out(y).chunk(2, dim=-1)
        return mean, logvar.clamp(-10, 5), h

    @torch.no_grad()
    def sample(self, n: int, window_length: int) -> torch.Tensor:
        device = next(self.parameters()).device
        x = torch.zeros(n, 1, self.out.out_features // 2, device=device)
        h, steps = None, []
        for _ in range(window_length):
            mean, logvar, h = self.step(x, h)
            x = (mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)).clamp(-1, 1)
            steps.append(x)
        return torch.cat(steps, dim=1)


def time_inference(gen: Generator, device: torch.device, window_length: int, batch_size: int) -> np.ndarray:
    """Seconds per call of gen.sample over REPEATS runs, after WARMUP untimed runs."""
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    for _ in range(WARMUP):   # first calls pay for CUDA/cuDNN set-up
        gen.sample(batch_size, window_length)
    sync()

    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        gen.sample(batch_size, window_length)
        sync()                # wait for the GPU to finish before stopping the clock
        times.append(time.perf_counter() - start)
    return np.array(times)


def main() -> None:
    torch.manual_seed(SEED)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    cfg = ckpt["config"]
    length = WINDOW_LENGTH or cfg["window_length"]

    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    print(f"AR-GRU baseline, {length}-step windows, {REPEATS} timed runs after {WARMUP} warm-up\n")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}\n")

    for device in devices:
        gen = Generator(cfg["n_features"], cfg["hidden_dim"], cfg["num_layers"])
        gen.load_state_dict(ckpt["generator"])
        gen.to(device).eval()

        for batch in BATCH_SIZES:
            t = time_inference(gen, device, length, batch) * 1e3   # ms
            print(f"{device.type.upper():<4} batch {batch:>5}: {t.mean():8.1f} ± {t.std():6.1f} ms per batch, "
                  f"{t.mean() / batch:8.3f} ms per window")


if __name__ == "__main__":
    main()