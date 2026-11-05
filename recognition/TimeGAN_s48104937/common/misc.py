import shutil
import urllib.request
from pathlib import Path

import torch

from components.config import Config
from components.scaler import LogMinMaxScaler
from recognition.TimeGAN_s48104937.modules import TimeGAN


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
        req = urllib.request.Request(base_url + folder + name, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
                shutil.copyfileobj(resp, f)
            tmp.replace(dest)
        except OSError as e:
            raise RuntimeError(f"failed to download '{name}': {e}") from e
        finally:
            tmp.unlink(missing_ok=True)
        print(f"downloaded '{name}' -> {path}")

def load_model(path: Path, device: torch.device) -> tuple[TimeGAN, dict[str, LogMinMaxScaler], Config]:
    """Rebuild TimeGAN and its per-ticker scalers from a checkpoint."""
    ckpt = torch.load(path, map_location=device)
    cfg = Config(**ckpt["config"])

    model = TimeGAN(
        ckpt["feature_dim"], cfg.hidden_dim, cfg.noise_dim, cfg.num_layers, len(cfg.tickers)
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scalers = {t: LogMinMaxScaler(ckpt["scaler_lo"][t], ckpt["scaler_hi"][t]) for t in cfg.tickers}
    return model, scalers, cfg
