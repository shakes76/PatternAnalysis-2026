"""Train TimeGAN phase 3 (joint adversarial training) on top of the saved supervisor checkpoint.

Loads models/checkpoint_supervisor.pt, runs the joint generator / embedder /
discriminator training from train.py, and saves models/checkpoint_joint.pt in the
same format. Data and architecture settings come from the checkpoint, so the scalers
and model shape match it exactly; the number of joint epochs, the learning rate and
the discriminator threshold are read from config.toml, so they can be changed
between runs. The loss history (including phases 1 and 2, if their JSON is found)
is written after every epoch, so stopping early still keeps it.

Every SNAPSHOT_EVERY epochs (and before the first epoch and after the last) a real
window and a synthetic one are saved side by side with plot_book_pair_v2. The real
window and the generator noise are fixed, so the snapshots show how the same sample
changes as training goes on.
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
from common.losses import bce
from common.losses import distribution_losses
from common.losses import grouped_mse
from common.losses import moment_loss
from common.losses import supervised_loss
from common.misc import Config
from common.misc import load_model
from common.visualizations import plot_book_pair_v2
from components.sampler import BatchSampler
from dataset import decode
from dataset import encode
from dataset import load_books


IN_PATH = Path("models/checkpoint_supervisor.pt")
OUT_PATH = Path("models/checkpoint_joint.pt")
IN_LOSSES = IN_PATH.with_name(f"{IN_PATH.stem}_losses.json")
OUT_LOSSES = OUT_PATH.with_name(f"{OUT_PATH.stem}_losses.json")

SNAPSHOT_DIR = Path("figures/phase3_snapshots")
SNAPSHOT_EVERY: int = 10
SNAPSHOT_TICKER: str = "AMZN"
SNAPSHOT_SEED: int = 0          # fixes the real window and the generator noise across snapshots

# Phase-3 hyperparameters from the current config.toml
with open(Path("config.toml"), "rb") as f:
    run_cfg = Config(**tomllib.load(f))

JOINT_KEYS = ("d_loss", "g_adv", "g_sup", "g_moment", "g_spread_w1", "g_mid_w1", "e_loss")


def write_json(data: dict, path: Path) -> None:
    """Write via a temporary file, so a kill mid-write never leaves a corrupted JSON."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def make_snapshotter(model, scalers, cfg, books):
    """Return snapshot(epoch): saves a fixed real window beside a synthetic one, without touching training RNG."""
    k = cfg.tickers.index(SNAPSHOT_TICKER)
    real_x, refs, labels = encode(books, scalers, cfg.sample_length, fit=False)
    pool = np.flatnonzero(labels == k)
    i = np.random.default_rng(SNAPSHOT_SEED).choice(pool, size=1)
    real = decode(real_x[i], refs[i], labels[i], scalers, np.random.default_rng(SNAPSHOT_SEED))
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    def snapshot(epoch: int) -> None:
        model.eval()
        # Fork the RNG so the fixed snapshot noise does not change the training run
        with torch.random.fork_rng(devices=[DEVICE] if DEVICE.type == "cuda" else []):
            torch.manual_seed(SNAPSHOT_SEED)
            with torch.no_grad():
                x = model.sample(n=1, window_length=cfg.sample_length + cfg.burn_in, ticker=k).cpu().numpy()
        model.train()
        synth = decode(x[:, cfg.burn_in:], refs[i], labels[i], scalers, np.random.default_rng(SNAPSHOT_SEED))

        path = SNAPSHOT_DIR / f"epoch_{epoch:03d}.png"
        plot_book_pair_v2(real, synth, path, title=None, col_titles=("Real Sample", f"TimeGAN, Epoch {epoch}"))
        print(f"  snapshot -> {path}")

    return snapshot


def main() -> None:
    # Supervisor checkpoint: trained embedder/recovery/supervisor, fitted scalers and their config
    model, scalers, cfg = load_model(IN_PATH, device=DEVICE)
    model.train()
    meta = {k: v for k, v in torch.load(IN_PATH, map_location="cpu").items() if k != "model"}
    torch.manual_seed(cfg.train_seed)
    scaler_list = list(scalers.values()) if isinstance(scalers, dict) else list(scalers)   # indexed by label

    # Same data pipeline as phases 1 and 2, reusing the fitted scalers
    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes, cfg.train_frac, cfg.val_frac, cfg.test_frac)
    train_windows, _, train_labels = encode(books["train"], scalers, cfg.window_length, fit=False)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows).float(), torch.from_numpy(train_labels)),
        batch_sampler=BatchSampler(labels=train_labels, batch_size=cfg.batch_size, seed=cfg.train_seed),
    )

    opt_g = torch.optim.Adam(
        list(model.generator.parameters()) + list(model.supervisor.parameters()), lr=run_cfg.learning_rate)
    opt_e = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=run_cfg.learning_rate)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=run_cfg.learning_rate)
    epochs = run_cfg.joint_epochs

    # Loss history: carry over phases 1 and 2, so this file covers the whole run
    previous = json.loads(IN_LOSSES.read_text()) if IN_LOSSES.exists() else {}
    history = {
        "embedding": previous.get("embedding", {"train": [], "val": []}),
        "supervised": previous.get("supervised", {"train": [], "val": []}),
        "joint": {name: [] for name in JOINT_KEYS},
        "profile": {},
        "completed": False,
    }
    previous_minutes = previous.get("profile", {}).get("train_minutes", {})
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    def save_history() -> None:
        minutes = {**previous_minutes, "joint": (time.perf_counter() - start) / 60}
        history["profile"] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_minutes": minutes,
            "total_train_minutes": sum(minutes.values()),
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else None,
        }
        write_json(history, OUT_LOSSES)

    snapshot = make_snapshotter(model, scalers, cfg, books["val"])
    snapshot(0)   # the starting point, before any adversarial training

    losses = history["joint"]
    try:
        for epoch in range(1, epochs + 1):
            totals = dict.fromkeys(losses, 0.0)
            for x, y in loader:
                x, c = x.to(DEVICE), model.one_hot(y.to(DEVICE))
                n, seq_len = x.shape[:2]
                scaler = scaler_list[int(y[0])]   # batches hold a single ticker

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
                    e_loss = (10 * torch.sqrt(grouped_mse(model.recovery(h, c), x))
                              + 0.1 * supervised_loss(model, h, c))
                    opt_e.zero_grad()
                    e_loss.backward()
                    opt_e.step()

                # Discriminator update on fresh synthetic windows, skipped while it is already winning
                with torch.no_grad():
                    h = model.embedder(x, c)
                    e_hat = model.generator(model.noise(n, seq_len), c)
                    h_hat = model.supervisor(e_hat, c)
                d_loss = (bce(model.discriminator(h, c), 1.0)
                          + bce(model.discriminator(h_hat, c), 0.0)
                          + bce(model.discriminator(e_hat, c), 0.0))
                if d_loss.item() > run_cfg.d_threshold:
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
            save_history()
            if epoch % SNAPSHOT_EVERY == 0:
                snapshot(epoch)
        history["completed"] = True
    except KeyboardInterrupt:
        print(f"\nStopped early after {len(losses['d_loss'])} epochs; saving what was trained so far")
    finally:
        save_history()
        torch.save({**meta, "model": model.state_dict()}, OUT_PATH)
        done = len(losses["d_loss"])
        if done and done % SNAPSHOT_EVERY:   # the last epoch, if it was not already snapshotted
            snapshot(done)

        profile = history["profile"]
        print(f"\nParameters        {profile['parameters']:,}")
        print(f"Training time     {profile['train_minutes']['joint']:6.1f} min (this phase)")
        if profile["peak_vram_mib"] is not None:
            print(f"Peak GPU memory   {profile['peak_vram_mib']:.0f} MiB (this phase)")
        print(f"Saved model       {OUT_PATH}")
        print(f"Saved losses      {OUT_LOSSES}")
        print(f"Saved snapshots   {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()