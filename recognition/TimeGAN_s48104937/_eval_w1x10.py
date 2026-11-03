"""Real vs two TimeGAN checkpoints on one ticker, printed as a markdown table.

Prices are in cents.
"""
from pathlib import Path

import numpy as np
import torch

from common.constants import DEVICE
from common.metrics import abs_return_autocorr
from common.metrics import kl_divergence
from common.metrics import mid_returns
from common.metrics import spread
from common.metrics import vwap_returns
from common.misc import load_model
from dataset import decode
from dataset import encode
from dataset import load_books

TICKER = "AMZN"
N_WINDOWS = 200
SEED = 286867760

# (label, checkpoint); the first model's scalers/config encode the real windows
MODELS = [
    ("Weight ×1", Path("models/saved/primary.pt")),
    ("Weight ×10", Path("models_w1x10/timegan_w1x10.pt")),
]


def mid_range(book: dict) -> float:
    """Highest minus lowest mid price within each window (cents), averaged over windows."""
    mid = (book["ask_px"][..., 0] + book["bid_px"][..., 0]) / 2
    return float(np.ptp(mid, axis=1).mean())


def level_spacing_variability(book: dict) -> float:
    """Std over time of the gap between adjacent price levels (cents), averaged over levels, sides and windows."""
    return float(np.mean([np.abs(np.diff(book[f"{s}_px"], axis=2)).std(axis=1).mean() for s in ("ask", "bid")]))


def best_quote_comovement(book: dict) -> float:
    """Correlation of step changes in best ask and best bid, over steps where either moves."""
    da = np.diff(book["ask_px"][..., 0], axis=1)
    db = np.diff(book["bid_px"][..., 0], axis=1)
    moving = (da != 0) | (db != 0)
    a, b = da[moving], db[moving]
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def summarise(book: dict, real: dict) -> dict:
    """Metric name -> value. KL rows are against real, so they are blank for the real column."""
    is_real = book is real
    return {
        "Mid-price range per window (¢)": mid_range(book),
        "Lag-1 volatility clustering": abs_return_autocorr(mid_returns(book))[0],
        "Price level spacing variability (¢)": level_spacing_variability(book),
        "Best quote co-movement": best_quote_comovement(book),
        "KL divergence, spread": np.nan if is_real else kl_divergence(spread(real), spread(book), bins=50),
        "KL divergence, VWAP returns": np.nan if is_real else kl_divergence(vwap_returns(real), vwap_returns(book)),
    }


def print_markdown(summaries: dict) -> None:
    names = list(summaries)
    print("| Metric | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    for row in next(iter(summaries.values())):
        cells = ["–" if np.isnan(summaries[n][row]) else f"{summaries[n][row]:.3f}".replace("-", "−")
                 for n in names]
        print(f"| {row} | " + " | ".join(cells) + " |")


def main() -> None:
    rng = np.random.default_rng(SEED)

    loaded = [(name, *load_model(path, device=DEVICE)) for name, path in MODELS]
    lengths = {cfg.sample_length for *_, cfg in loaded}
    assert len(lengths) == 1, f"models use different window lengths: {lengths}"

    # Real windows, encoded with the first model's scalers
    _, _, scalers, cfg = loaded[0]
    books = load_books(cfg.tickers, cfg.resample_freq, "", cfg.train_frac, cfg.val_frac, cfg.test_frac)
    real_x, refs, labels = encode(books=books["train"], scalers=scalers, window_length=cfg.sample_length)
    pool = np.flatnonzero(labels == cfg.tickers.index(TICKER))
    idx = rng.choice(pool, size=min(N_WINDOWS, len(pool)), replace=False)

    out = {"Real": decode(real_x[idx], refs[idx], labels[idx], scalers, rng)}

    # Synthetic windows from each model, anchored at the same references as the real ones
    for name, model, m_scalers, m_cfg in loaded:
        k = m_cfg.tickers.index(TICKER)
        torch.manual_seed(SEED)
        with torch.no_grad():
            x = model.sample(n=len(idx), window_length=m_cfg.sample_length + m_cfg.burn_in, ticker=k).cpu().numpy()
        out[name] = decode(x[:, m_cfg.burn_in:], refs[idx], np.full(len(idx), k), m_scalers, rng)

    print(f"{TICKER}: {len(idx)} windows of {cfg.sample_length} steps per source, seed {SEED}\n")
    print_markdown({name: summarise(book, out["Real"]) for name, book in out.items()})


if __name__ == "__main__":
    main()