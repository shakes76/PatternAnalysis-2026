import json
import time
import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from common.constants import DEVICE
from common.losses import bce
from common.losses import distribution_losses
from common.losses import grouped_mse
from common.losses import moment_loss
from common.losses import reconstruction_loss
from common.losses import supervised_loss
from common.misc import Config
from common.visualizations import plot_losses
from components.sampler import BatchSampler
from components.scaler import LogMinMaxScaler
from dataset import FEATURE_COLS
from dataset import encode
from dataset import load_books
from modules import TimeGAN


# Load config.toml as dataclass
with open(Path("config.toml"), "rb") as f:
    cfg = Config(**tomllib.load(f))

# Final model path; the loss history is saved beside it with the same name
MODEL_PATH = Path("models/recent_primary.pt")
LOSSES_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.stem}_losses.json")


# Training Phase 1: Embedder
def train_embedding(model: TimeGAN, loader: DataLoader, val_x: torch.Tensor, val_c: torch.Tensor, epochs: int,
                    losses: dict, on_epoch: Callable[[], None]) -> None:
    """Phase 1: train embedder + recovery to reconstruct real windows. Appends to losses each epoch."""
    params = list(model.embedder.parameters()) + list(model.recovery.parameters())
    opt = torch.optim.Adam(params, lr=cfg.learning_rate)

    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, y in loader:
            x, c = x.to(DEVICE), model.one_hot(y.to(DEVICE))
            loss = reconstruction_loss(model, x, c)
            opt.zero_grad()
            (10 * torch.sqrt(loss)).backward()
            opt.step()
            total += loss.item()

        with torch.no_grad():
            val = reconstruction_loss(model, val_x, val_c).item()
        losses["train"].append(total / len(loader))
        losses["val"].append(val)
        print(f"[Embedding  {epoch:3d}/{epochs}] recon MSE train {losses['train'][-1]:.5f}  val {val:.5f}")
        on_epoch()

# Training Phase 2: Supervisor
def train_supervisor(model: TimeGAN, loader: DataLoader, val_x: torch.Tensor, val_c: torch.Tensor, epochs: int,
                     losses: dict, on_epoch: Callable[[], None]) -> None:
    """Phase 2: train the supervisor on real latents (embedder frozen). Appends to losses each epoch."""
    opt = torch.optim.Adam(model.supervisor.parameters(), lr=cfg.learning_rate)

    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, y in loader:
            x, c = x.to(DEVICE), model.one_hot(y.to(DEVICE))
            with torch.no_grad():
                h = model.embedder(x, c)
            loss = supervised_loss(model, h, c)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        with torch.no_grad():
            val = supervised_loss(model, model.embedder(val_x, val_c), val_c).item()
        losses["train"].append(total / len(loader))
        losses["val"].append(val)
        print(f"[Supervised {epoch:3d}/{epochs}] next-step MSE train {losses['train'][-1]:.5f}  val {val:.5f}")
        on_epoch()

# Training Phase 3: Joint Generator and Discriminator
def train_joint(model: TimeGAN, loader: DataLoader, scalers: list[LogMinMaxScaler], epochs: int,
                losses: dict, on_epoch: Callable[[], None]) -> None:
    """Phase 3: adversarial training with embedder fine-tuning. Appends to losses each epoch.

    Each batch performs two generator + embedder updates and then one
    discriminator update, skipped while the discriminator is already winning
    (loss below D_THRESHOLD) so it cannot overpower the generator.
    """
    opt_g = torch.optim.Adam(
        list(model.generator.parameters()) + list(model.supervisor.parameters()), lr=cfg.learning_rate)
    opt_e = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=cfg.learning_rate)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.learning_rate)

    for epoch in range(1, epochs + 1):
        totals = dict.fromkeys(losses, 0.0)
        for x, y in loader:
            x, c = x.to(DEVICE), model.one_hot(y.to(DEVICE))
            n, seq_len = x.shape[:2]
            scaler = scalers[int(y[0])]   # batches hold a single ticker

            for _ in range(2):
                # Generator + supervisor
                h = model.embedder(x, c)
                e_hat = model.generator(model.noise(n, seq_len), c)
                h_hat = model.supervisor(e_hat, c)
                x_hat = model.recovery(h_hat, c)

                g_adv = bce(model.discriminator(h_hat, c), 1.0)
                g_adv_e = bce(model.discriminator(e_hat, c), 1.0)
                g_sup = supervised_loss(model, h, c)
                g_moment = moment_loss(x_hat, x)
                g_spread_w1, g_mid_w1 = distribution_losses(x_hat, x, scaler)
                g_loss = g_adv + g_adv_e + 100 * torch.sqrt(g_sup) + 100 * g_moment + (g_spread_w1 + g_mid_w1)

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                # Embedder + recovery fine-tuning
                h = model.embedder(x, c)
                e_loss = (10 * torch.sqrt(grouped_mse(model.recovery(h, c), x)) + 0.1 * supervised_loss(model, h, c))
                opt_e.zero_grad()
                e_loss.backward()
                opt_e.step()

            # Discriminator update on fresh synthetic windows
            with torch.no_grad():
                h = model.embedder(x, c)
                e_hat = model.generator(model.noise(n, seq_len), c)
                h_hat = model.supervisor(e_hat, c)
            d_loss = (bce(model.discriminator(h, c), 1.0)
                      + bce(model.discriminator(h_hat, c), 0.0)
                      + bce(model.discriminator(e_hat, c), 0.0))
            if d_loss.item() > cfg.d_threshold:
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            totals["d_loss"] += d_loss.item()
            totals["g_adv"] += g_adv.item()
            totals["g_sup"] += g_sup.item()
            totals["g_moment"] += g_moment.item()
            totals["g_spread_w1"] += g_spread_w1.item()
            totals["g_mid_w1"] += g_mid_w1.item()
            totals["e_loss"] += e_loss.item()

        for name in losses:
            losses[name].append(totals[name] / len(loader))
        print(f"[Joint      {epoch:3d}/{epochs}] " + "  ".join(f"{k} {v[-1]:.4f}" for k, v in losses.items()))
        on_epoch()


def write_json(data: dict, path: Path) -> None:
    """Write via a temporary file, so a kill mid-write never leaves a corrupted JSON."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def main() -> None:
    # Initialize rng and required filepaths
    Path("models").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    torch.manual_seed(cfg.train_seed)

    # Data pipeline: one scaler per ticker, fitted on the training split only
    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes, cfg.train_frac, cfg.val_frac, cfg.test_frac)
    scalers = {ticker: LogMinMaxScaler() for ticker in cfg.tickers}
    train_windows, _, train_labels = encode(books["train"], scalers, cfg.window_length, fit=True)
    val_windows, _, val_labels = encode(books["val"], scalers, cfg.window_length, fit=False)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows).float(), torch.from_numpy(train_labels)),
        batch_sampler=BatchSampler(
            labels=train_labels,
            batch_size=cfg.batch_size,
            seed=cfg.train_seed,
        ),
    )

    # Validation subset
    val_idx = np.arange(0, len(val_windows), 8)
    val_x = torch.from_numpy(val_windows[val_idx]).float().to(DEVICE)
    val_y = torch.from_numpy(val_labels[val_idx]).to(DEVICE)

    # Model and training
    feature_dim = train_windows.shape[2]
    model = TimeGAN(feature_dim, cfg.hidden_dim, cfg.noise_dim, cfg.num_layers, len(cfg.tickers)).to(DEVICE)
    val_c = model.one_hot(val_y)

    # Metadata saved with the final model
    meta = {
        "config": cfg.to_dict(),
        "scaler_lo": {ticker: s.lo.tolist() for ticker, s in scalers.items()},
        "scaler_hi": {ticker: s.hi.tolist() for ticker, s in scalers.items()},
        "feature_cols": FEATURE_COLS,
        "feature_dim": feature_dim,
    }

    # Loss history, filled in place by the training functions and written after every epoch
    history = {
        "embedding": {"train": [], "val": []},
        "supervised": {"train": [], "val": []},
        "joint": {name: [] for name in ("d_loss", "g_adv", "g_sup", "g_moment", "g_spread_w1", "g_mid_w1", "e_loss")},
        "profile": {},
        "completed": False,
    }

    # Resource profiling: reset the GPU's peak-memory counter, then time each phase
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = {}
    running = {"phase": None, "start": 0.0}   # the phase in progress, so a partial phase is timed too

    def save_history() -> None:
        t = dict(times)
        if running["phase"] is not None:
            t[running["phase"]] = time.perf_counter() - running["start"]
        history["profile"] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_minutes": {phase: s / 60 for phase, s in t.items()},
            "total_train_minutes": sum(t.values()) / 60,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else None,
        }
        write_json(history, LOSSES_PATH)

    phases = [
        ("embedding", lambda: train_embedding(model, loader, val_x, val_c, cfg.embedding_epochs,
                                              history["embedding"], save_history)),
        ("supervised", lambda: train_supervisor(model, loader, val_x, val_c, cfg.supervised_epochs,
                                                history["supervised"], save_history)),
        ("joint", lambda: train_joint(model, loader, list(scalers.values()), cfg.joint_epochs,
                                      history["joint"], save_history)),
    ]

    try:
        for phase, run in phases:
            running.update(phase=phase, start=time.perf_counter())
            run()
            times[phase] = time.perf_counter() - running["start"]
            running["phase"] = None
        history["completed"] = True
    except KeyboardInterrupt:
        print(f"\nStopped early during the {running['phase']} phase; saving what was trained so far")
    finally:
        save_history()
        torch.save({**meta, "model": model.state_dict()}, MODEL_PATH)

        profile = history["profile"]
        print(f"\nParameters        {profile['parameters']:,}")
        for phase, minutes in profile["train_minutes"].items():
            print(f"Training time     {phase:<11}{minutes:6.1f} min")
        print(f"Total             {'':<11}{profile['total_train_minutes']:6.1f} min")
        if profile["peak_vram_mib"] is not None:
            print(f"Peak GPU memory   {profile['peak_vram_mib']:.0f} MiB")
        print(f"Saved model       {MODEL_PATH}")
        print(f"Saved losses      {LOSSES_PATH}")

        plot_losses(history["embedding"], history["supervised"], history["joint"],
                    Path("figures/timegan_losses.png"))


if __name__ == "__main__":
    main()