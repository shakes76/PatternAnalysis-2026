from modules import TimeGAN
from torch import nn


class TimeGANV32(TimeGAN):
    """TimeGAN with identity activation in Generator."""

    def __init__(
        self,
        feature_dim=40,
        hidden_dim=24,
        noise_dim=40,
        num_layers=2,
    ):
        super().__init__(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            noise_dim=noise_dim,
            num_layers=num_layers,
        )

        self.generator.activation = nn.Identity()