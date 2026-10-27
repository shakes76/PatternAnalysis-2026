from pathlib import Path

import torch

from common.scalers import LogMinMaxScaler, QuantileScaler, DualScaler
from recognition.TimeGAN_s48104937.modules import TimeGAN


# --- Model Helpers ---
def load_model(path: Path | str, device: torch.device):
    """
    Rebuild TimeGAN and its scaler from a checkpoint. Returns (model, scaler, config).
    """
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = TimeGAN(cfg["feat_dim"], cfg["hidden_dim"], cfg["noise_dim"], cfg["num_layers"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scaler = DualScaler(
        len(ckpt["feature_names"]),
        ckpt["scaler_cols"],
        QuantileScaler(quantiles=ckpt["scaler_quantiles"]),
        LogMinMaxScaler(ckpt["scaler_lo"], ckpt["scaler_hi"]),
    )
    return model, scaler, cfg
