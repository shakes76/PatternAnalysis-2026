import torch

from modules import TimeGAN
import torch.nn.functional as F


def reconstruction_loss(model: TimeGAN, x: torch.Tensor) -> torch.Tensor:
    """MSE between real features and their embed -> recover round trip."""
    return F.mse_loss(model.recovery(model.embedder(x)), x)


def supervised_loss(model: TimeGAN, h: torch.Tensor) -> torch.Tensor:
    """MSE between the supervisor's next-step prediction and the actual next step."""
    return F.mse_loss(model.supervisor(h)[:, :-1], h[:, 1:])


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


def bce(logits: torch.Tensor, target: float) -> torch.Tensor:
    """Binary cross-entropy against a constant label."""
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target))
