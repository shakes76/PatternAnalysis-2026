import torch
import torch.nn.functional as F

from components.scaler import LogMinMaxScaler
from dataset import FEATURE_COLS
from dataset import SZ_IDX
from dataset import FLAG_GROUPS
from modules import TimeGAN


N_FEAT = len(FEATURE_COLS)
LOSS_GROUPS = [
    [FEATURE_COLS.index("mid_rel")],
    [FEATURE_COLS.index("spread")],
    [i for i, c in enumerate(FEATURE_COLS) if "_gap_" in c],
    SZ_IDX,
    list(range(N_FEAT, N_FEAT + len(FLAG_GROUPS))),
]


# Loss Functions
def grouped_mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean of the per-group MSEs, so every channel group weighs the same."""
    return torch.stack([F.mse_loss(x_hat[..., g], x[..., g]) for g in LOSS_GROUPS]).mean()

def reconstruction_loss(model: TimeGAN, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Group-weighted MSE between real features and their embed -> recover round trip."""
    return grouped_mse(model.recovery(model.embedder(x, c), c), x)

def supervised_loss(model: TimeGAN, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """MSE between the supervisor's next-step prediction and the actual next step."""
    return F.mse_loss(model.supervisor(h, c)[:, :-1], h[:, 1:])

def moment_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Match per-(timestep, feature) mean and std across the batch.

    Uses sqrt(var + eps) rather than std directly, because the gradient of
    std is undefined when a generated feature has zero variance, which is
    exactly the collapsed state this loss should push the generator out of.
    """
    var_hat, mean_hat = torch.var_mean(x_hat, dim=0)
    var, mean = torch.var_mean(x, dim=0)
    std_gap = (torch.sqrt(var_hat + 1e-6) - torch.sqrt(var + 1e-6)).abs().mean()
    return std_gap + (mean_hat - mean).abs().mean()

def unscale(x: torch.Tensor, scaler: LogMinMaxScaler, col: int) -> torch.Tensor:
    """Torch version of scaler.inverse_transform for one feature, so gradients flow through."""
    logged = x[..., col] * float(scaler.span[col]) + float(scaler.lo[col])
    return torch.sign(logged) * torch.expm1(logged.abs())

def w1(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """1D Wasserstein distance between two equal-sized samples, in units of the real std.

    Sorting both samples pairs their quantiles, so the whole distribution is
    compared, tails included, which matching mean and std alone cannot do.
    """
    fake, real = torch.sort(fake.flatten())[0], torch.sort(real.flatten())[0]
    return (fake - real).abs().mean() / (real.std() + 1e-6)

def distribution_losses(x_hat: torch.Tensor, x: torch.Tensor, scaler: LogMinMaxScaler) -> tuple[torch.Tensor, torch.Tensor]:
    """W1 between generated and real batches for the spread and the step-to-step mid change, in ticks."""
    spread_col, mid_col = FEATURE_COLS.index("spread"), FEATURE_COLS.index("mid_rel")
    spread_w1 = w1(unscale(x_hat, scaler, spread_col), unscale(x, scaler, spread_col))
    mid_w1 = w1(torch.diff(unscale(x_hat, scaler, mid_col), dim=1),
                torch.diff(unscale(x, scaler, mid_col), dim=1))
    return spread_w1, mid_w1

def bce(logits: torch.Tensor, target: float) -> torch.Tensor:
    """Binary cross-entropy against a constant label."""
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target))
