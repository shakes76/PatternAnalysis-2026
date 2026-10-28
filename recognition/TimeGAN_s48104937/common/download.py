"""Download and unzip the free LOBSTER sample files (NASDAQ, 21 June 2012)."""

import urllib.request
from pathlib import Path

def download_dataset(ticker: str, path: Path) -> None:
    """Download the LOBSTER level-10 sample CSVs for `ticker` into `path`. Raises on failure."""
    base_url = "https://huggingface.co/datasets/totalorganfailure/lobster-data/resolve/main/"
    folder = f"LOBSTER_SampleFile_{ticker}_2012-06-21_10/"
    path.mkdir(parents=True, exist_ok=True)

    for kind in ("message", "orderbook"):
        name = f"{ticker}_2012-06-21_34200000_57600000_{kind}_10.csv"
        dest = path / name
        if dest.is_file():
            print(f"skip '{name}' (already downloaded)")
            continue

        tmp = dest.with_suffix(".part")
        try:
            req = urllib.request.Request(base_url + folder + name, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
                f.write(resp.read())
            tmp.replace(dest)
            print(f"downloaded '{name}' -> {path}")
        except Exception as e:
            raise RuntimeError(f"failed to download '{name}' -> {e}") from e
        finally:
            tmp.unlink(missing_ok=True)
