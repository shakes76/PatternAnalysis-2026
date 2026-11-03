from pathlib import Path

import numpy as np
import torch

from common.constants import DEVICE
from common.misc import load_model
from common.visualizations import plot_book_pair, plot_book_pair_v2
from dataset import FEATURE_COLS
from dataset import decode
from dataset import encode
from dataset import load_books

TICKER = "AMZN"
SEED = 53466788
# SEED = 286867760
CHECKPOINT = Path("models/saved/primary.pt")
COL_TITLES = ("Without Flags", "With Flags")


def main() -> None:
    Path("figures").mkdir(exist_ok=True)
    model, scalers, cfg = load_model(CHECKPOINT, device=DEVICE)
    k = cfg.tickers.index(TICKER)

    # Real train windows only supply the decode anchor (ref)
    books = load_books(cfg.tickers, cfg.resample_freq, "", cfg.train_frac, cfg.val_frac, cfg.test_frac)
    _, refs, labels = encode(books=books["train"], scalers=scalers, window_length=cfg.sample_length)
    pool = np.flatnonzero(labels == k)

    # The seed drives the anchor choice, the generator noise and the flag sampling
    idx = np.random.default_rng(SEED).choice(pool, size=1)
    torch.manual_seed(SEED)
    with torch.no_grad():
        x = model.sample(n=1, window_length=cfg.sample_length + cfg.burn_in, ticker=k).cpu().numpy()
        x = x[:, cfg.burn_in:]

    # The same generated window decoded twice: normally, and with every flag
    # probability forced to 1 so every feature updates every step (nothing held)
    no_flags = x.copy()
    no_flags[..., len(FEATURE_COLS):] = 1.0
    books_out = [decode(w, refs[idx], np.array([k]), scalers, np.random.default_rng(SEED)) for w in (no_flags, x)]

    path = Path(f"figures/timegan_{TICKER}_flags_vs_no_flags_seed{SEED}.png")
    plot_book_pair_v2(*books_out, path, title=None, col_titles=COL_TITLES)
    print(f"Seed {SEED}  -> {path}")


if __name__ == "__main__":
    main()
