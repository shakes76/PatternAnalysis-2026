"""Search for seeds whose sample_all_tickers row shows a clear mid-price move in both windows.

Uses the exact sampling from sample_all_tickers.make_pair, so a seed found here draws the
same real and synthetic windows when put in sample_all_tickers.SEEDS. Run it on the same
device as the figure script: CPU and GPU draw different noise from the same seed.
"""
import numpy as np

from _sample_all_tickers import load_windows
from _sample_all_tickers import make_pair


TICKER: str = "INTC"
MIN_TICKS: float = 1       # minimum |mid change| from first to last step, in ticks (cents)
START: int = 0             # first seed to try
MAX_TRIES: int = 20000     # give up after this many seeds
N_FOUND: int = 5           # stop after this many matches


def mid_change(book: dict) -> float:
    """Mid at the last step minus mid at the first, in ticks."""
    mid = (book["ask_px"][0, :, 0] + book["bid_px"][0, :, 0]) / 2
    return float(mid[-1] - mid[0])


def main() -> None:
    model, scalers, cfg, real_x, refs, labels = load_windows()
    k = cfg.tickers.index(TICKER)

    print(f"{TICKER}: seeds where |mid change| >= {MIN_TICKS} ticks in both real and synthetic\n")
    print(f"  {'seed':>8}{'real':>8}{'synth':>8}")
    found = 0
    for seed in range(START, START + MAX_TRIES):
        real, synth = make_pair(model, scalers, cfg, real_x, refs, labels, k, seed)
        r, s = mid_change(real), mid_change(synth)
        if abs(r) >= MIN_TICKS and abs(s) >= MIN_TICKS:
            print(f"  {seed:>8}{r:>+8.1f}{s:>+8.1f}")
            found += 1
            if found == N_FOUND:
                break

    if not found:
        print(f"  none in seeds {START}-{START + MAX_TRIES - 1}; lower MIN_TICKS or raise MAX_TRIES")


if __name__ == "__main__":
    main()