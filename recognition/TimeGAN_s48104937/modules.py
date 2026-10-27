import torch
from torch import nn


class GRUNet(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_layers: int = 3,
            activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        out = self.linear(h)
        return self.activation(out) if self.activation is not None else out


class TimeGAN(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            hidden_dim: int = 64,
            noise_dim: int = 32,
            num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.embedder = GRUNet(feature_dim, hidden_dim, hidden_dim, num_layers, nn.Sigmoid())
        self.recovery = GRUNet(hidden_dim, hidden_dim, feature_dim, num_layers, nn.Sigmoid())
        self.generator = GRUNet(noise_dim, hidden_dim, hidden_dim, num_layers, nn.Sigmoid())
        self.supervisor = GRUNet(hidden_dim, hidden_dim, hidden_dim, max(num_layers - 1, 1), nn.Sigmoid())
        self.discriminator = GRUNet(hidden_dim, hidden_dim, 1, num_layers)

    def noise(self, n: int, window_length: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.randn(n, window_length, self.noise_dim, device=device)

    def sample(self, n: int, window_length: int) -> torch.Tensor:
        e_hat = self.generator(self.noise(n, window_length))
        h_hat = self.supervisor(e_hat)
        return self.recovery(h_hat)
