import torch
import torch.nn.functional as F
from torch import nn


class GRUNet(nn.Module):
    """GRU + linear head, optionally conditioned on a per-window vector.

    The condition (e.g. a one-hot ticker) is repeated at every step and
    concatenated to the input, so in_dim excludes it and cond_dim adds it.
    """
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_layers: int = 3,
            activation: nn.Module | None = None,
            cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(in_dim + cond_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if c is not None:
            x = torch.cat([x, c[:, None, :].expand(-1, x.shape[1], -1)], dim=-1)
        h, _ = self.gru(x)
        out = self.linear(h)
        return self.activation(out) if self.activation is not None else out


class TimeGAN(nn.Module):
    """TimeGAN with every network conditioned on the ticker.

    The discriminator sees the ticker too: otherwise the generator could
    ignore its label and still fool it, and conditioning would do nothing.
    """
    def __init__(
            self,
            feature_dim: int,
            hidden_dim: int = 64,
            noise_dim: int = 32,
            num_layers: int = 3,
            k: int = 1,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.k = k
        self.embedder = GRUNet(feature_dim, hidden_dim, hidden_dim, num_layers, nn.Sigmoid(), k)
        self.recovery = GRUNet(hidden_dim, hidden_dim, feature_dim, num_layers, nn.Sigmoid(), k)
        self.generator = GRUNet(noise_dim, hidden_dim, hidden_dim, num_layers, nn.Sigmoid(), k)
        self.supervisor = GRUNet(hidden_dim, hidden_dim, hidden_dim, max(num_layers - 1, 1), nn.Sigmoid(), k)
        self.discriminator = GRUNet(hidden_dim, hidden_dim, 1, num_layers, cond_dim=k)

    def one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        """Ticker indices (batch,) -> one-hot condition vectors (batch, n_tickers)."""
        return F.one_hot(labels.long(), self.k).float()

    def noise(self, n: int, window_length: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.randn(n, window_length, self.noise_dim, device=device)

    def sample(self, n: int, window_length: int, ticker: int) -> torch.Tensor:
        """Generate n windows for the ticker with index `ticker`."""
        device = next(self.parameters()).device
        c = self.one_hot(torch.full((n,), ticker, device=device))
        e_hat = self.generator(self.noise(n, window_length), c)
        h_hat = self.supervisor(e_hat, c)
        return self.recovery(h_hat, c)
