from pathlib import Path

import numpy as np
import torch

from common.constants import DEVICE
from common.metrics import abs_return_autocorr
from common.metrics import garch_fit
from common.metrics import kl_divergence
from common.metrics import mean_ssim
from common.metrics import mid_returns
from common.metrics import spread
from common.metrics import unchanged_rates
from common.metrics import violation_rates
from common.metrics import vwap_returns
from common.misc import load_model
from common.visualizations import plot_distributions
from common.visualizations import plot_sample_pair
from common.visualizations import plot_spread_by_step
from dataset import BOOK_COLS
from dataset import decode
from dataset import encode
from dataset import load_books


def report(
        ticker: str,
        real: dict,
        synth: dict,
        real_x: np.ndarray,
        synth_x: np.ndarray,
        ticker_books,
        train_len: int,
) -> None:
    """Print one ticker's real-vs-synthetic metrics and save its plots."""
    print(f"\n{'=' * 24} {ticker} {'=' * 24}")
    print(f"Comparing {len(real_x)} windows against the train split\n")
    print(f"{'Invariant violations':<24}{'Real':>10}{'Synthetic':>12}")
    real_v, synth_v = violation_rates(real), violation_rates(synth)
    for key in real_v:
        print(f"  {key:<22}{real_v[key]:>10.2%}{synth_v[key]:>12.2%}")

    print(f"\n{'Unchanged step-to-step':<24}{'Real':>10}{'Synthetic':>12}")
    real_u, synth_u = unchanged_rates(real), unchanged_rates(synth)
    for key in real_u:
        print(f"  {key:<22}{real_u[key]:>10.2%}{synth_u[key]:>12.2%}")

    print(f"\nKL spread       {kl_divergence(spread(real), spread(synth), bins=50):.4f}   (target <= 0.1)")
    print(f"KL mid returns  {kl_divergence(mid_returns(real), mid_returns(synth)):.4f}   (target <= 0.1)")
    print(f"KL VWAP returns {kl_divergence(vwap_returns(real), vwap_returns(synth)):.4f}")
    print(f"Depth SSIM      {mean_ssim(real, synth):.4f}   (target > 0.6)")

    print("\n|return| autocorrelation, lags 1-5:")
    print(f"  Real       {np.round(abs_return_autocorr(mid_returns(real))[:5], 3)}")
    print(f"  Synthetic  {np.round(abs_return_autocorr(mid_returns(synth))[:5], 3)}")

    # GARCH(1,1) on VWAP returns: how strongly volatility reacts to shocks (alpha)
    # and how long it persists (alpha + beta). The full-day fit on the unwindowed
    # train split is a reference: the pooled real-window fit should land near it
    day = {key: ticker_books[cols].to_numpy(float)[None] for key, cols in
           [
               ("ask_px", [c for c in BOOK_COLS if "ask_px" in c]),
               ("ask_sz", [c for c in BOOK_COLS if "ask_sz" in c]),
               ("bid_px", [c for c in BOOK_COLS if "bid_px" in c]),
               ("bid_sz", [c for c in BOOK_COLS if "bid_sz" in c]),
           ]}
    fits = {"Real": garch_fit(vwap_returns(real)), "Synthetic": garch_fit(vwap_returns(synth)),
            "Real full day": garch_fit(vwap_returns(day))}
    print(f"\n{'GARCH(1,1), VWAP ret (bp)':<26}{'alpha':>8}{'beta':>8}{'a+b':>8}{'half-life':>11}{'LR var':>10}")
    for name, f in fits.items():
        flag = "" if f["converged"] else "  (not converged)"
        print(f"  {name:<24}{f['alpha']:>8.3f}{f['beta']:>8.3f}{f['persistence']:>8.3f}"
              f"{f['half_life']:>11.1f}{f['long_run_var']:>10.3f}{flag}")

    # Collapse check on the scaled model channels: near-zero synthetic std means
    # the generator ignores its noise
    print("\nMean std of model channels (synthetic should be near real):")
    print(f"  across samples  real {real_x.std(axis=0).mean():.3f}  "
          f"synthetic {synth_x.std(axis=0).mean():.3f}")
    print(f"  across time     real {real_x.std(axis=1).mean():.3f}  "
          f"synthetic {synth_x.std(axis=1).mean():.3f}")

    # Plots
    plot_spread_by_step(real, synth, Path(f"figures/timegan_{ticker}_spread_by_step.png"), train_len)
    plot_distributions(real, synth, Path(f"figures/timegan_{ticker}_distributions.png"))
    for i in range(min(3, len(real_x))):
        plot_sample_pair(real, synth, i, Path(f"figures/timegan_{ticker}_sample_pair_{i}.png"))


def main() -> None:
    # Initialize rng and required filepaths
    Path("figures").mkdir(exist_ok=True)
    rng = np.random.default_rng(48)
    torch.manual_seed(48)

    # Load model from checkpoint; keep the training length for the extrapolation marker
    model, scalers, cfg = load_model(Path("models/timegan.pt"), device=DEVICE)

    # Load real books and encode every window of the chosen split
    books = load_books(cfg.tickers, cfg.resample_freq, "", cfg.train_frac, cfg.val_frac, cfg.test_frac)
    real_x, refs, labels = encode(books=books["train"], scalers=scalers, window_length=cfg.sample_length)
    train_by_ticker = dict(tuple(books["train"].groupby("ticker", sort=False)))

    for k, ticker in enumerate(cfg.tickers):
        # Randomly sample this ticker's real windows (no repeats) and generate as many synthetic ones
        pool = np.flatnonzero(labels == k)
        idx: np.ndarray = rng.choice(pool, size=min(100, len(pool)), replace=False)
        with torch.no_grad():
            synth_x = model.sample(n=len(idx), window_length=cfg.sample_length + cfg.burn_in, ticker=k).cpu().numpy()
            synth_x = synth_x[:, cfg.burn_in:]

        # Decode real and synthetic features back to books, anchored at the same references
        real = decode(real_x[idx], refs[idx], labels[idx], scalers, rng)
        synth = decode(synth_x, refs[idx], labels[idx], scalers, rng)

        report(ticker, real, synth, real_x[idx], synth_x, train_by_ticker[ticker], cfg.window_length)


if __name__ == "__main__":
    main()
