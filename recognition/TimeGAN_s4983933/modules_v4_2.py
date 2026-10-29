"""TimeGAN V4.2: Identity activation for generator and recovery."""

from torch import nn
from modules_v3_2 import TimeGANV32


class TimeGANV42(TimeGANV32):
    """Allow real-valued recovery outputs for asinh + Z-score returns."""

    def __init__(self, feature_dim=41, hidden_dim=24, noise_dim=41, num_layers=2):
        super().__init__(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            noise_dim=noise_dim,
            num_layers=num_layers,
        )
        self.recovery.activation = nn.Identity()
