from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common.constants import DEVICE
from common.misc import load_model
from dataset import FEATURE_COLS
from dataset import decode
from dataset import encode
from dataset import load_books


MODELS = {                          # column title -> checkpoint, left to right
    "Default MSE Autoencoder": Path("models/saved/autoencoder_default.pt"),
    "Grouped MSE Autoencoder": Path("models/saved/primary.pt"),
}
SAMPLE_FROM: str = "Grouped MSE Autoencoder"   # model whose encoding picks the real window (as in the earlier figure)
WINDOW_LENGTH: int | None = None           # None = that model's training window length
TICKER: str = "AMZN"                       # ticker to evaluate (must be in every model's cfg.tickers)
SPLIT: str = "val"                         # split to take the real windows from
SEED: int = 48
EVAL_WINDOWS: int = 1000                   # random windows per model for the printed metrics
SHOW = {"mid_rel": "Mid (relative)", "spread": "Spread"}
OUT_DIR = Path("dummy/figures")

# Feature groups in the scaled model space (the change flags follow the features)
N_FEAT = len(FEATURE_COLS)
GROUPS = {
    "mid": [FEATURE_COLS.index("mid_rel")],
    "spread": [FEATURE_COLS.index("spread")],
    "gaps": [i for i, c in enumerate(FEATURE_COLS) if "_gap_" in c],
    "sizes": [i for i, c in enumerate(FEATURE_COLS) if "_sz_" in c],
}


def load_split(path: Path) -> pd.DataFrame:
    """The evaluation split, built with this model's data settings."""
    _, _, cfg = load_model(path, device=DEVICE)
    return load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes,
                      cfg.train_frac, cfg.val_frac, cfg.test_frac)[SPLIT]


def pick_window(path: Path, books: pd.DataFrame) -> pd.DataFrame:
    """Choose the real window exactly as before, then return its raw book rows."""
    rng = np.random.default_rng(SEED)
    _, scalers, cfg = load_model(path, device=DEVICE)
    k = cfg.tickers.index(TICKER)
    length = WINDOW_LENGTH or cfg.window_length

    _, _, labels = encode(books=books, scalers=scalers, window_length=length)
    i = rng.choice(np.flatnonzero(labels == k))

    # Windows are stride-1 and grouped by ticker, so the offset within the ticker is the start row
    start = i - np.flatnonzero(labels == k)[0]
    ticker_books = books[books["ticker"] == TICKER]
    return ticker_books.iloc[start:start + length]


def reconstruct(model, x: np.ndarray, labels: np.ndarray, batch: int = 256) -> np.ndarray:
    """Real windows through the autoencoder only: embedder -> latent -> recovery."""
    with torch.no_grad():
        xt = torch.from_numpy(x).float().to(DEVICE)
        c = model.one_hot(torch.from_numpy(labels).long().to(DEVICE))
        out = [model.recovery(model.embedder(xb, cb), cb) for xb, cb in zip(xt.split(batch), c.split(batch))]
    return torch.cat(out).cpu().numpy()


def metrics(model, scalers, cfg, books: pd.DataFrame) -> dict[str, float]:
    """Reconstruction quality over EVAL_WINDOWS random windows of TICKER."""
    rng = np.random.default_rng(SEED)
    k = cfg.tickers.index(TICKER)
    x, refs, labels = encode(books=books, scalers=scalers, window_length=WINDOW_LENGTH or cfg.window_length)
    idx = np.flatnonzero(labels == k)
    idx = rng.choice(idx, size=min(EVAL_WINDOWS, len(idx)), replace=False)
    x, refs, labels = x[idx], refs[idx], labels[idx]
    recon = reconstruct(model, x, labels)

    out = {"MSE, all inputs (scaled)": ((recon - x) ** 2).mean()}

    # R^2 per feature, averaged within each group: 1 = perfect, 0 = no better than the mean
    for name, cols in GROUPS.items():
        mse = ((recon[..., cols] - x[..., cols]) ** 2).mean(axis=(0, 1))
        var = x[..., cols].var(axis=(0, 1)) + 1e-8
        out[f"R² {name}"] = (1 - mse / var).mean()

    # Change flags: are change / no-change steps recovered?
    real_f, recon_f = x[..., N_FEAT:] > 0.5, recon[..., N_FEAT:] > 0.5
    out["Flag accuracy"] = (real_f == recon_f).mean()
    out["Flag recall (changed steps)"] = recon_f[real_f].mean()

    # Decoded books in real units
    real_b = decode(x, refs, labels, scalers, rng)
    recon_b = decode(recon, refs, labels, scalers, rng)
    out["|Error| best ask price (ticks)"] = np.abs(recon_b["ask_px"][..., 0] - real_b["ask_px"][..., 0]).mean()
    out["|Error| best ask size (shares)"] = np.abs(recon_b["ask_sz"][..., 0] - real_b["ask_sz"][..., 0]).mean()
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    books = load_split(MODELS[SAMPLE_FROM])
    window_books = pick_window(MODELS[SAMPLE_FROM], books)

    results, scores = {}, {}
    for name, path in MODELS.items():
        model, scalers, cfg = load_model(path, device=DEVICE)
        model.eval()
        scores[name] = metrics(model, scalers, cfg, books)

        # The shown window, encoded with this model's own scalers -> one window
        real_x, _, labels = encode(books=window_books, scalers=scalers, window_length=len(window_books))
        results[name] = (real_x[0], reconstruct(model, real_x, labels)[0])

    # Plot: one column per model, one row per feature
    fig, axes = plt.subplots(len(SHOW), len(MODELS), figsize=(14, 2.6 * len(SHOW)),
                             sharex=True, squeeze=False)
    for col, (name, (real, recon)) in enumerate(results.items()):
        axes[0, col].text(0.5, 1.18, name, transform=axes[0, col].transAxes,
                          ha="center", va="bottom", fontsize=12, weight="bold")
        for row, (feat, label) in enumerate(SHOW.items()):
            ax = axes[row, col]
            j = FEATURE_COLS.index(feat)
            ax.plot(real[:, j], color="black", lw=1.3, alpha=0.8, label="Real", zorder=1)
            ax.plot(recon[:, j], color="C1", lw=1.3, alpha=0.9, label="Reconstruction", zorder=2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=11)
        axes[-1, col].set_xlabel("Time (s)", fontsize=11)

    for ax in axes[0]:
        ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=10,
                  frameon=False, borderaxespad=0.2, handlelength=1.8)
    fig.tight_layout()
    path = OUT_DIR / f"autoencoder_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()