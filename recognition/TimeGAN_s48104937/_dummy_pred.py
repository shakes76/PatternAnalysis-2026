from pathlib import Path

import numpy as np
import torch

from common.constants import DEVICE
from common.misc import load_model
from common.visualizations import plot_sample_pair
from dataset import decode
from dataset import encode
from dataset import load_books


WINDOW_LENGTH: int = 512       # steps to generate and compare
BURN_IN: int = 10              # number of steps to let the GRU warm-up
TICKER: str = "AMZN"          # index into the checkpoint's tickers
SEED: int = 478               # seed for np and torch rng


def main() -> None:
    Path("dummy/figures").mkdir(exist_ok=True)
    rng: np.random.Generator = np.random.default_rng()
    # torch.manual_seed()

    model, scalers, cfg = load_model(Path("models/timegan.pt"), device=DEVICE)

    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes, cfg.train_frac, cfg.val_frac, cfg.test_frac)
    real_x, refs, labels = encode(books=books["val"], scalers=scalers, window_length=WINDOW_LENGTH)

    # One random real window and one synthetic window of the same ticker
    print(real_x.shape, refs.shape, real_x.dtype)
    k = cfg.tickers.index(TICKER)
    idx = rng.choice(np.flatnonzero(labels == k), size=1)
    with torch.no_grad():
        synth_x = model.sample(n=1, window_length=WINDOW_LENGTH + BURN_IN, ticker=k).cpu().numpy()[:, BURN_IN:]

    real = decode(real_x[idx], refs[idx], labels[idx], scalers, rng)
    print(f"real: {real}")
    synth = decode(synth_x, refs[idx], labels[idx], scalers, rng)
    plot_sample_pair(real, synth, 0, Path(f"figures/timegan_{TICKER}_long_pair.png"))


if __name__ == "__main__":
    main()