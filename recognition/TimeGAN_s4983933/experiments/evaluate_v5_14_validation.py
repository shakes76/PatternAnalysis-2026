import numpy as np
import torch

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42
from train_v5_14_multifactor import (
    MultiFactorVolumeHead, stats
)

torch.set_num_threads(4)
N = 512
SEED = 371114
ALPHAS = [1.0, 0.7, 0.4, 0.2, 0.1]

data = prepare_dataset_v4("data")
norm = data["normalizer"]

raw_val = norm.inverse_transform(
    np.asarray(data["val"][:N], dtype=np.float64)
)
real = np.maximum(
    np.expm1(raw_val[..., 1:40:2]), 0
)

gan_ck = torch.load(
    "outputs_v5_10_latent_matching/timegan_finetuned.pt",
    map_location="cpu", weights_only=True
)

gan = TimeGANV42(
    feature_dim=gan_ck["feature_dim"],
    hidden_dim=gan_ck["hidden_dim"],
    noise_dim=gan_ck["noise_dim"],
    num_layers=gan_ck["num_layers"]
)
gan.load_state_dict(gan_ck["model_state_dict"])
gan.eval()

ck = torch.load(
    "outputs_v5_14_multifactor/best_multifactor_head.pt",
    map_location="cpu", weights_only=True
)

# These tensors only establish the layer shapes.
# The actual trained parameters are loaded below.
head = MultiFactorVolumeHead(
    hidden_dim=ck["hidden_dim"],
    mean_log=torch.ones(20),
    std_log=torch.ones(20)
)
head.load_state_dict(ck["head_state_dict"])
head.eval()

rng = torch.Generator().manual_seed(SEED)

with torch.no_grad():
    z = torch.rand(
        N, 24, gan_ck["noise_dim"], generator=rng
    )
    hidden = gan.supervisor(gan.generator(z))
    eps = torch.randn(N, 24, 20, generator=rng)
    log_volume = head(hidden, eps)
    base = torch.expm1(log_volume).numpy()

def smooth(x, alpha):
    result = np.empty_like(x)
    result[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        result[:, t] = (
            alpha * x[:, t]
            + (1 - alpha) * result[:, t - 1]
        )
    return result

def summary(x):
    log_x = torch.tensor(
        np.log1p(np.maximum(x, 0)),
        dtype=torch.float32
    )
    s = stats(log_x)
    return {
        "mean": round(s["mean"], 2),
        "std": round(s["std"], 2),
        "p99": round(s["p99"], 2),
        "change": round(s["temporal_change"], 2),
        "rank": round(s["effective_rank"], 3),
        "pc1": round(s["pc1"], 4),
        "lag1": round(s["lag1"], 4),
        "lag5": round(s["lag5"], 4),
    }

print("=== V5.14 VALIDATION SMOOTHING ===")
print("REAL:", summary(real))

for alpha in ALPHAS:
    print(
        f"ALPHA {alpha}:",
        summary(smooth(base, alpha))
    )

print("\nValidation smoothing check complete!")
