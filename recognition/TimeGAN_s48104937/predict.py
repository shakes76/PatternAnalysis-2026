from pathlib import Path
import numpy as np
import torch

from common.constants import DEVICE
from common.metrics import abs_return_autocorr
from common.metrics import kl_divergence
from common.metrics import mean_ssim
from common.metrics import spread
from common.metrics import violation_rates
from common.metrics import vwap_returns
from common.misc import load_model
from common.visualizations import plot_book_pair
from dataset import decode
from dataset import encode
from dataset import load_books
from baseline.smoke_test_model import BLOCKS
from baseline.smoke_test_model import DEPTH
from baseline.smoke_test_model import Generator
from baseline.smoke_test_model import PX_IDX
from baseline.smoke_test_model import SZ_IDX


# Table layouts (name, target, scale, decimals, unit)
VWAP_TABLE = (
    ("Real (train)", None, 1, 2, ""),
    ("Real (val)",   None, 1, 2, ""),
    ("Real (test)",  None, 1, 2, ""),
    ("TimeGAN",      None, 1, 2, ""),
)
METRIC_TABLE = (
    ("LOB Violation Rate", "≤ 0.1%", 100, 1, "%"),
    ("Spread KL", "≤ 0.1", 1, 3, ""),
    ("∆VWAP KL", "≤ 0.1", 1, 3, ""),
    ("∆VWAP Autocorr. Ratio", "0.6–1.4", 1, 2, ""),
    ("Depth SSIM Ratio", "0.6–1.4", 1, 2, ""),
)

# Calculate all evaluation metrics and return
def score(reference: dict, other: dict, book: dict) -> list[float]:
    """Metrics for `book` against `reference`; `other` is a second real sample for the SSIM reference."""
    return [
        max(violation_rates(book).values()),
        kl_divergence(spread(reference), spread(book), bins=50),
        kl_divergence(vwap_returns(reference), vwap_returns(book)),
        abs_return_autocorr(vwap_returns(book))[0] / abs_return_autocorr(vwap_returns(reference))[0],
        mean_ssim(reference, book) / mean_ssim(reference, other),
    ]

# Display evaluation table and print to output
def print_table(header: list[str], columns: list, specs: tuple) -> None:
    """Print one row per (index, layout) spec and one column per (draws, rows) array, as mean ± std."""
    columns = [np.asarray(c) for c in columns]
    rows = []
    for j, (name, target, scale, decimals, unit) in specs:
        row = [name]
        if target:
            row.append(target)
        for column in columns:
            values = column[:, j] * scale
            row.append(f"{values.mean():.{decimals}f}{unit} ± {values.std():.{decimals}f}")
        rows.append(row)

    table = [header, *rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for row in table:
        cells = [f"{row[0]:<{widths[0]}}"] + [f"{cell:>{width}}" for cell, width in zip(row[1:], widths[1:])]
        print("  ".join(cells))
    print()


def main() -> None:
    # Initialize required filepaths
    Path("figures").mkdir(exist_ok=True)

    # Load TimeGAN from checkpoint
    model, scalers, cfg = load_model(Path("models/primary_ready.pt"), device=DEVICE)

    # Load the AR-GRU baseline and the min-max bounds of its scaler
    ckpt = torch.load(Path("models/saved/baseline.pt"), map_location="cpu", weights_only=False)
    gen = Generator(ckpt["config"]["n_features"], ckpt["config"]["hidden_dim"], ckpt["config"]["num_layers"])
    gen.load_state_dict(ckpt["generator"])
    gen.to(DEVICE).eval()
    lo = np.asarray(ckpt["scaler_lo"], np.float32)
    hi = np.asarray(ckpt["scaler_hi"], np.float32)

    # Load real books and encode every window of each split
    books = load_books(cfg.tickers, cfg.resample_freq, cfg.trim_regimes, cfg.train_frac, cfg.val_frac, cfg.test_frac)
    windows = {}
    for split in ("train", "val", "test"):
        windows[split] = encode(books=books[split], scalers=scalers, window_length=cfg.sample_length)
    train_x, train_refs, train_labels = windows["train"]

    # VWAP return volatility of each real split and of TimeGAN, per ticker
    stds = {ticker: [] for ticker in cfg.tickers}
    for draw in range(cfg.num_draws):
        rng = np.random.default_rng(cfg.eval_seed + draw)
        torch.manual_seed(cfg.eval_seed + draw)

        for k, ticker in enumerate(cfg.tickers):
            # Randomly sample this ticker's real windows from each split
            row = []
            for split in ("train", "val", "test"):
                real_x, refs, labels = windows[split]
                pool = np.flatnonzero(labels == k)
                idx = rng.permutation(pool)[:cfg.num_samples]
                real = decode(real_x[idx], refs[idx], labels[idx], scalers, rng)
                row.append((vwap_returns(real) * 1e4).std())

            # Generate as many synthetic windows, anchored at real training references
            pool = np.flatnonzero(train_labels == k)
            idx = rng.permutation(pool)[:cfg.num_samples]
            with torch.no_grad():
                synth_x = model.sample(n=len(idx), window_length=cfg.sample_length + cfg.burn_in, ticker=k).cpu().numpy()
                synth_x = synth_x[:, cfg.burn_in:]
            synth = decode(synth_x, train_refs[idx], train_labels[idx], scalers, rng)
            row.append((vwap_returns(synth) * 1e4).std())
            stds[ticker].append(row)

    print_table(["VWAP return std (bp)", *cfg.tickers], list(stds.values()), tuple(enumerate(VWAP_TABLE)))

    # Score every source on AMZN against test (the AR-GRU only knows AMZN), then every ticker against validation
    scores = {}
    for split, tickers in (("test", ["AMZN"]), ("val", cfg.tickers)):
        real_x, refs, labels = windows[split]
        scores[split] = {source: {ticker: [] for ticker in tickers} for source in ("AR-GRU", "TimeGAN", "Train")}

        for draw in range(cfg.num_draws):
            rng = np.random.default_rng(cfg.eval_seed + draw)
            torch.manual_seed(cfg.eval_seed + draw)

            for ticker in tickers:
                k = cfg.tickers.index(ticker)

                # Two disjoint real samples: the reference, and a second one for the SSIM ratio
                pool = np.flatnonzero(labels == k)
                idx = rng.permutation(pool)[:2 * cfg.num_samples]
                idx_a, idx_b = idx[:len(idx) // 2], idx[len(idx) // 2:]
                reference = decode(real_x[idx_a], refs[idx_a], labels[idx_a], scalers, rng)
                other = decode(real_x[idx_b], refs[idx_b], labels[idx_b], scalers, rng)

                # Generate as many synthetic windows, anchored at the same references
                with torch.no_grad():
                    synth_x = model.sample(n=len(idx_a), window_length=cfg.sample_length + cfg.burn_in, ticker=k).cpu().numpy()
                    synth_x = synth_x[:, cfg.burn_in:]
                synth = decode(synth_x, refs[idx_a], labels[idx_a], scalers, rng)

                # Randomly sample as many real training windows (no repeats)
                pool = np.flatnonzero(train_labels == k)
                idx_t = rng.choice(pool, size=len(idx_a), replace=False)
                train = decode(train_x[idx_t], train_refs[idx_t], train_labels[idx_t], scalers, rng)

                scores[split]["TimeGAN"][ticker].append(score(reference, other, synth))
                scores[split]["Train"][ticker].append(score(reference, other, train))

                if split == "test":
                    # Undo the AR-GRU's min-max scaling and relative prices, anchored at the first mids
                    scaled = gen.sample(len(idx_a), cfg.sample_length).cpu().numpy()
                    raw = ((scaled + 1) / 2 * (hi - lo + 1e-8) + lo).astype(np.float64)
                    mids = (reference["ask_px"][:, 0, 0] + reference["bid_px"][:, 0, 0]) / 2
                    raw[..., PX_IDX] += mids[:, None, None]
                    raw[..., SZ_IDX] = np.expm1(raw[..., SZ_IDX])
                    raw = np.rint(raw).astype("int64")
                    baseline = {f"{s}_{c}": raw[..., i * DEPTH:(i + 1) * DEPTH] for i, (s, c) in enumerate(BLOCKS)}
                    scores[split]["AR-GRU"][ticker].append(score(reference, other, baseline))

    print_table(["Metric", "Target", "AR-GRU vs Test", "TimeGAN vs Test", "Train vs Test"],
                [scores["test"][source]["AMZN"] for source in ("AR-GRU", "TimeGAN", "Train")],
                tuple(enumerate(METRIC_TABLE)))
    print_table(["TimeGAN vs Validation", "Target", *cfg.tickers],
                list(scores["val"]["TimeGAN"].values()), tuple(enumerate(METRIC_TABLE))[1:])

    # Sample figures come last so they can't shift the seeded draws behind the tables
    rng = np.random.default_rng(cfg.eval_seed)
    torch.manual_seed(cfg.eval_seed)
    samples = [(ticker, cfg.sample_length) for ticker in cfg.tickers] + [("AMZN", 4 * cfg.sample_length)]
    for ticker, length in samples:
        k = cfg.tickers.index(ticker)

        # Randomly pick one real test window and generate a synthetic one
        real_x, refs, labels = encode(books=books["test"], scalers=scalers, window_length=length)
        pool = np.flatnonzero(labels == k)
        idx = rng.choice(pool, size=1)
        real = decode(real_x[idx], refs[idx], labels[idx], scalers, rng)
        with torch.no_grad():
            synth_x = model.sample(n=1, window_length=length + cfg.burn_in, ticker=k).cpu().numpy()
            synth_x = synth_x[:, cfg.burn_in:]
        synth = decode(synth_x, refs[idx], labels[idx], scalers, rng)

        plot_book_pair(real, synth, Path(f"figures/timegan_{ticker}_{length}s.png"),
                       col_titles=(f"{ticker} Real", f"{ticker} TimeGAN"))


if __name__ == "__main__":
    main()
