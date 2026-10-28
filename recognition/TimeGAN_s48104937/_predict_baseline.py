"""Generate one long (512-step) synthetic window with the baseline GRU and plot it
beside a real window of the same length.

Put this file next to the baseline training script (assumed to be `baseline.py`);
it reuses that script's data pipeline, model and decoding.
"""
import numpy as np
import torch

from tests.smoke_test_model import COLS
from tests.smoke_test_model import CHECKPOINT_PATH
from tests.smoke_test_model import DEVICE
from tests.smoke_test_model import FIGURE_DIR
from tests.smoke_test_model import Generator
from tests.smoke_test_model import MinMax
from tests.smoke_test_model import load_book
from tests.smoke_test_model import to_books
from tests.smoke_test_model import to_features
from common.visualizations import plot_book_pair


WINDOW_LENGTH: int = 512       # steps to generate and compare
BURN_IN: int = 10              # steps to let the GRU warm up before keeping samples
TICKER: str = "AMZN"           # must be one of the checkpoint's tickers
SEED: int = 0                  # seed for np and torch rng


def load_baseline(path=CHECKPOINT_PATH) -> tuple[Generator, MinMax, dict]:
    """Rebuild the generator and scaler saved by baseline.py."""
    ckpt = torch.load(path, map_location=DEVICE)
    cfg = ckpt["config"]
    gen = Generator(cfg["n_features"], cfg["hidden_dim"], cfg["num_layers"]).to(DEVICE)
    gen.load_state_dict(ckpt["generator"])
    gen.eval()
    scaler = MinMax(np.array(ckpt["scaler_lo"]), np.array(ckpt["scaler_hi"]))
    return gen, scaler, cfg


def real_window(ticker: str, train_frac: float, window_length: int,
                rng: np.random.Generator) -> np.ndarray:
    """One random window from the held-out part of the day (after the train split),
    shape (1, window_length, 40)."""
    book = load_book(ticker)
    values = book[COLS].to_numpy(np.float32)[int(len(book) * train_frac):]
    if len(values) < window_length:
        raise ValueError(f"Held-out split has {len(values)} steps, need {window_length}")
    start = rng.integers(0, len(values) - window_length + 1)
    return values[None, start:start + window_length]


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    gen, scaler, cfg = load_baseline()
    if TICKER not in cfg["tickers"]:
        raise ValueError(f"{TICKER} not in checkpoint tickers {cfg['tickers']}")
    print(f"Loaded baseline trained on {cfg['window_length']}-step windows; "
          f"generating {WINDOW_LENGTH} steps (+{BURN_IN} burn-in)")

    # Real window and its anchor (first mid price)
    raw = real_window(TICKER, cfg["train_frac"], WINDOW_LENGTH, rng)
    real_feats, refs = to_features(raw)
    print(raw.shape, refs.shape, raw.dtype)

    # Synthetic window: sample with burn-in, drop the warm-up steps
    scaled = gen.sample(1, WINDOW_LENGTH + BURN_IN).cpu().numpy()[:, BURN_IN:]
    synth_feats = scaler.inverse_transform(scaled)

    real = to_books(real_feats, refs)
    synth = to_books(synth_feats, refs)     # anchored at the same mid as the real window

    path = FIGURE_DIR / f"baseline_{TICKER}_long_pair.png"
    plot_book_pair(real, synth, path, 0, 0,
                   title=f"Real vs Baseline - {WINDOW_LENGTH} Second Windows ({TICKER})",
                   col_titles=("Real Sample", "Baseline Model"))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()