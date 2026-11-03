"""KL divergence of spread, mid-price returns and VWAP returns between real and TimeGAN windows, per ticker.

Synthetic windows are decoded at their real partner's reference, as predict.py does.
KL is real -> synthetic, so lower is better.
"""
from pathlib import Path

import numpy as np
import torch

from common.constants import DEVICE
from common.metrics import kl_divergence
from common.metrics import mid_returns
from common.metrics import spread
from common.metrics import vwap_returns
from common.misc import load_model
from dataset import decode
from dataset import encode
from dataset import load_books


MODEL_PATH = Path("models/saved/primary.pt")
WINDOW_LENGTH: int | None = None   # None = the model's sample length
NUM_SAMPLES: int = 500             # windows per draw, per ticker
NUM_DRAWS: int = 5                 # independent random draws, to show the spread
SPLIT: str = "train"               # same split predict.py compares against
SEED: int = 48

# Column title -> KL between real and synthetic books (same settings as predict.py)
METRICS = {
    "KL Spread": lambda r, s: kl_divergence(spread(r), spread(s), bins=50),
    "KL Mid Returns": lambda r, s: kl_divergence(mid_returns(r), mid_returns(s)),
    "KL VWAP Returns": lambda r, s: kl_divergence(vwap_returns(r), vwap_returns(s)),
}


def draw(model, scalers, cfg, real_x, refs, labels, k: int, length: int, rng) -> list[float]:
    """One draw: KL for every metric between real and synthetic windows."""
    pool = np.flatnonzero(labels == k)
    idx = rng.choice(pool, size=min(NUM_SAMPLES, len(pool)), replace=False)

    real = decode(real_x[idx], refs[idx], labels[idx], scalers, rng)
    with torch.no_grad():
        synth_x = model.sample(n=len(idx), window_length=length + cfg.burn_in, ticker=k).cpu().numpy()
    synth = decode(synth_x[:, cfg.burn_in:], refs[idx], labels[idx], scalers, rng)

    return [kl(real, synth) for kl in METRICS.values()]


def row(label: str, scores: np.ndarray, bold: bool = False) -> str:
    """Markdown row of mean ± std over draws, one cell per metric."""
    cells = [f"{c.mean():.3f} ± {c.std():.3f}" for c in scores.T]
    if bold:
        label, cells = f"**{label}**", [f"**{c}**" for c in cells]
    return f"| {label} | " + " | ".join(cells) + " |"


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    model, scalers, cfg = load_model(MODEL_PATH, device=DEVICE)
    model.eval()
    length = WINDOW_LENGTH or cfg.sample_length

    books = load_books(cfg.tickers, resample_freq=cfg.resample_freq, trim_regimes=cfg.trim_regimes,
                       train_frac=cfg.train_frac, val_frac=cfg.val_frac, test_frac=cfg.test_frac)[SPLIT]
    real_x, refs, labels = encode(books=books, scalers=scalers, window_length=length)

    print(f"KL divergence, real vs synthetic, {SPLIT} split, {NUM_DRAWS} draws of {NUM_SAMPLES} windows, "
          f"{length}-step windows\n")
    print("| Ticker | " + " | ".join(METRICS) + " |")
    print("| :----- |" + " ---: |" * len(METRICS))

    per_ticker = []   # (draws, metrics) per ticker
    for k, ticker in enumerate(cfg.tickers):
        scores = np.array([draw(model, scalers, cfg, real_x, refs, labels, k, length, rng)
                           for _ in range(NUM_DRAWS)])
        per_ticker.append(scores)
        print(row(ticker, scores))

    # Total: mean over tickers within each draw, then mean ± std over draws
    print(row("Total", np.mean(per_ticker, axis=0), bold=True))


if __name__ == "__main__":
    main()