"""Train the TimeGAN supervisor (phase 2) on top of the saved autoencoder checkpoint.

Loads models/checkpoint_autoencoder.pt, keeps its embedder frozen, trains the
supervisor to predict the next latent step, and saves models/checkpoint_supervisor.pt
in the same format. Data and architecture settings come from the autoencoder
checkpoint, so the scalers and model shape match it exactly; only the number of
supervised epochs and the learning rate are read from config.toml, so they can be
changed between runs. The loss history (including the autoencoder's, if its JSON is
found) is written after every epoch, so stopping early still keeps it.
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
from common.losses import supervised_loss
from common.misc import Config
from common.misc import load_model
from components.sampler import BatchSampler
from dataset import encode
from dataset import load_books


IN_PATH = Path("models/checkpoint_autoencoder.pt")
OUT_PATH = Path("models/checkpoint_supervisor.pt")
IN_LOSSES = IN_PATH.with_name(f"{IN_PATH.stem}_losses.json")
OUT_LOSSES = OUT_PATH.with_name(f"{OUT_PATH.stem}_losses.json")

# Phase-2 hyperparameters from the current config.toml
with open(Path("config.toml"), "rb") as f:
    run_cfg = Config(**tomllib.load(f))


def write_json(data: dict, path: Path) -> None:
    """Write via a temporary file, so a kill mid-write never leaves a corrupted JSON."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def main() -> None:
    # Autoencoder checkpoint: trained embedder/recovery, fitted scalers and the config they were built with
    model, scalers, cfg = load_model(IN_PATH, device=DEVICE)
    model.train()
    meta = {k: v for k, v in torch.load(IN_PATH, map_location="cpu").items() if k != "model"}
    torch.manual_seed(cfg.train_seed)

    # Same data pipeline as phase 1, reusing the fitted scalers
    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes, cfg.train_frac, cfg.val_frac, cfg.test_frac)
    train_windows, _, train_labels = encode(books["train"], scalers, cfg.window_length, fit=False)
    val_windows, _, val_labels = encode(books["val"], scalers, cfg.window_length, fit=False)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows).float(), torch.from_numpy(train_labels)),
        batch_sampler=BatchSampler(labels=train_labels, batch_size=cfg.batch_size, seed=cfg.train_seed),
    )

    # Validation subset, latents from the frozen embedder
    val_idx = np.arange(0, len(val_windows), 8)
    val_x = torch.from_numpy(val_windows[val_idx]).float().to(DEVICE)
    val_c = model.one_hot(torch.from_numpy(val_labels[val_idx]).to(DEVICE))

    opt = torch.optim.Adam(model.supervisor.parameters(), lr=run_cfg.learning_rate)
    epochs = run_cfg.supervised_epochs

    # Loss history: carry over the autoencoder's, so this file covers phases 1 and 2
    previous = json.loads(IN_LOSSES.read_text()) if IN_LOSSES.exists() else {}
    history = {
        "embedding": previous.get("embedding", {"train": [], "val": []}),
        "supervised": {"train": [], "val": []},
        "profile": {},
        "completed": False,
    }
    previous_minutes = previous.get("profile", {}).get("train_minutes", {})
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    def save_history() -> None:
        minutes = {**previous_minutes, "supervised": (time.perf_counter() - start) / 60}
        history["profile"] = {
            "parameters": sum(p.numel() for p in model.supervisor.parameters()),   # supervisor only
            "train_minutes": minutes,
            "total_train_minutes": sum(minutes.values()),
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else None,
        }
        write_json(history, OUT_LOSSES)

    losses = history["supervised"]
    try:
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
            save_history()
        history["completed"] = True
    except KeyboardInterrupt:
        print(f"\nStopped early after {len(losses['train'])} epochs; saving what was trained so far")
    finally:
        save_history()
        torch.save({**meta, "model": model.state_dict()}, OUT_PATH)

        profile = history["profile"]
        print(f"\nParameters        {profile['parameters']:,} (supervisor)")
        print(f"Training time     {profile['train_minutes']['supervised']:6.1f} min (this phase)")
        if profile["peak_vram_mib"] is not None:
            print(f"Peak GPU memory   {profile['peak_vram_mib']:.0f} MiB (this phase)")
        print(f"Saved model       {OUT_PATH}")
        print(f"Saved losses      {OUT_LOSSES}")


if __name__ == "__main__":
    main()