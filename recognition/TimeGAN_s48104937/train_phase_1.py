"""Train only the TimeGAN autoencoder (phase 1: embedder + recovery) and save it.

The checkpoint uses the same format as the full model, so load_model reads it and
model.embedder / model.recovery work as usual; the other networks stay untrained.
The loss history is written after every epoch, so stopping early still keeps it.
"""
import json
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from common.constants import DEVICE
from common.losses import reconstruction_loss
from common.misc import Config
from components.sampler import BatchSampler
from components.scaler import LogMinMaxScaler
from dataset import FEATURE_COLS
from dataset import encode
from dataset import load_books
from modules import TimeGAN

quit(1)

# Load config.toml as dataclass
with open(Path("config.toml"), "rb") as f:
    cfg = Config(**tomllib.load(f))

# Checkpoint path; the loss history is saved beside it with the same name
MODEL_PATH = Path("models/checkpoint_autoencoder.pt")
LOSSES_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.stem}_losses.json")


def write_json(data: dict, path: Path) -> None:
    """Write via a temporary file, so a kill mid-write never leaves a corrupted JSON."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def main() -> None:
    Path("models").mkdir(exist_ok=True)
    torch.manual_seed(cfg.train_seed)

    # Data pipeline: one scaler per ticker, fitted on the training split only
    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes, cfg.train_frac, cfg.val_frac, cfg.test_frac)
    scalers = {ticker: LogMinMaxScaler() for ticker in cfg.tickers}
    train_windows, _, train_labels = encode(books["train"], scalers, cfg.window_length, fit=True)
    val_windows, _, val_labels = encode(books["val"], scalers, cfg.window_length, fit=False)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows).float(), torch.from_numpy(train_labels)),
        batch_sampler=BatchSampler(labels=train_labels, batch_size=cfg.batch_size, seed=cfg.train_seed),
    )

    # Validation subset
    val_idx = np.arange(0, len(val_windows), 8)
    val_x = torch.from_numpy(val_windows[val_idx]).float().to(DEVICE)
    val_y = torch.from_numpy(val_labels[val_idx]).to(DEVICE)

    # Model
    feature_dim = train_windows.shape[2]
    model = TimeGAN(feature_dim, cfg.hidden_dim, cfg.noise_dim, cfg.num_layers, len(cfg.tickers)).to(DEVICE)
    val_c = model.one_hot(val_y)
    params = list(model.embedder.parameters()) + list(model.recovery.parameters())
    opt = torch.optim.Adam(params, lr=cfg.learning_rate)

    meta = {
        "config": cfg.to_dict(),
        "scaler_lo": {ticker: s.lo.tolist() for ticker, s in scalers.items()},
        "scaler_hi": {ticker: s.hi.tolist() for ticker, s in scalers.items()},
        "feature_cols": FEATURE_COLS,
        "feature_dim": feature_dim,
    }

    # Loss history and resource profile, written after every epoch
    history = {"embedding": {"train": [], "val": []}, "profile": {}, "completed": False}
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    def save_history() -> None:
        minutes = (time.perf_counter() - start) / 60
        history["profile"] = {
            "parameters": sum(p.numel() for p in params),   # embedder + recovery only
            "train_minutes": {"embedding": minutes},
            "total_train_minutes": minutes,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else None,
        }
        write_json(history, LOSSES_PATH)

    losses = history["embedding"]
    epochs = cfg.embedding_epochs
    try:
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
            save_history()
        history["completed"] = True
    except KeyboardInterrupt:
        print(f"\nStopped early after {len(losses['train'])} epochs; saving what was trained so far")
    finally:
        save_history()
        torch.save({**meta, "model": model.state_dict()}, MODEL_PATH)

        profile = history["profile"]
        print(f"\nParameters        {profile['parameters']:,} (embedder + recovery)")
        print(f"Training time     {profile['total_train_minutes']:6.1f} min")
        if profile["peak_vram_mib"] is not None:
            print(f"Peak GPU memory   {profile['peak_vram_mib']:.0f} MiB")
        print(f"Saved model       {MODEL_PATH}")
        print(f"Saved losses      {LOSSES_PATH}")


if __name__ == "__main__":
    main()