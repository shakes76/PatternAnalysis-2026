from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.constants import DEVICE
from common.misc import load_model
from common.visualizations import plot_book_pair, plot_book_pair_v2
from dataset import decode
from dataset import encode
from dataset import load_books

TICKER = "AMZN"
SEED = 28686773
# SEED = 286867760

# (file-name slug, column title, checkpoint)
MODELS = [
    ("w10", "Wasserstein Weight ×10", Path("models_w1x10/timegan_w1x10.pt")),
    ("w1", "Wasserstein Weight ×1", Path("models/saved/primary.pt")),
]


def generate(model, scalers, cfg, ref: np.ndarray, seed: int) -> dict:
    """One synthetic window from this model, decoded at the given reference."""
    k = cfg.tickers.index(TICKER)
    torch.manual_seed(seed)   # same noise draw for every model
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        x = model.sample(n=1, window_length=cfg.sample_length + cfg.burn_in, ticker=k).cpu().numpy()
        x = x[:, cfg.burn_in:]
    return decode(x, ref, np.array([k]), scalers, rng)


def main() -> None:
    Path("figures").mkdir(exist_ok=True)

    # Load both models from their checkpoints
    loaded = [(slug, label, *load_model(path, device=DEVICE)) for slug, label, path in MODELS]
    slugs = "_vs_".join(slug for slug, *_ in loaded)
    col_titles = tuple(label for _, label, *_ in loaded)

    # Real train windows only supply the shared decode anchors (refs)
    _, _, _, scalers, cfg = loaded[0]
    books = load_books(cfg.tickers, cfg.resample_freq, "", cfg.train_frac, cfg.val_frac, cfg.test_frac)
    _, refs, labels = encode(books=books["train"], scalers=scalers, window_length=cfg.sample_length)
    pool = np.flatnonzero(labels == cfg.tickers.index(TICKER))

    title = f"Wasserstein Loss - 128s Windows (AMZN)"

    # The seed drives the anchor choice, each model's generator noise and decode
    idx = np.random.default_rng(SEED).choice(pool, size=1)

    # Both models decoded at the same reference, so they start from the same book
    books_out = [generate(model, scalers, cfg, refs[idx], SEED) for *_, model, scalers, cfg in loaded]

    path = Path(f"figures/timegan_{TICKER}_{slugs}_seed{SEED}.png")
    plot_book_pair_v2(*books_out, path, title=None, col_titles=col_titles)
    print(f"Seed {SEED}  -> {path}")


if __name__ == "__main__":
    main()