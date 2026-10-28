"""
Model components for ADNI AD vs NC classification.

Two Models:
    - BaselineCNN: standard CNN used as the comparison baseline.
    - ConvNeXt: Modernized convolutional backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    4-layer basline CNN

    Input:  [B, 1, 240, 256] greyscale MRI slices
    Output: [B, 2] logits for NC / AD
    """

    def __init__(self, n_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class LayerNorm2d(nn.Module):
    """LayerNorm for [B, C, H, W] tensors because  pytorch and numpy hate me"""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DropPath(nn.Module):
    """Stochastic depth: during training, randomly skip a block's output for some images."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # one keep/drop decision per image in the batch: shape [B, 1, 1, 1]
        mask = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep_prob).float()
        return x * mask / keep_prob  # divide so the average output size stays the same


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block (Liu et al. 2022)
    """

    def __init__(self, dim, layer_scale_init=1e-6, drop_path=0.0):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        residual = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.gamma * x

        x = x.permute(0, 3, 1, 2)

        return residual + self.drop_path(x)


def make_stage(in_dim, out_dim, depth, drop_rates, downsample=True):
    """downsampling layer followed by depth ConvNeXt blocks with one drop rate per block"""
    layers = []

    if downsample:
        layers.append(LayerNorm2d(in_dim))
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2))

    for i in range(depth):
        layers.append(ConvNeXtBlock(out_dim, drop_path=drop_rates[i]))

    return nn.Sequential(*layers)


class ConvNeXt(nn.Module):
    """
    ConvNeXt for greyscale MRI slices

    Input:  [B, 1, 240, 256]
    Output: [B, 2] logits for NC / AD
    """

    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depths=(2, 2, 6, 2),
        dims=(48, 96, 192, 384),
        drop_path=0.0,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        # Drop rate rises linearly from the first block to the last block
        total_blocks = sum(depths)
        rates = [drop_path * i / (total_blocks - 1) for i in range(total_blocks)]
        end1 = depths[0]
        end2 = end1 + depths[1]
        end3 = end2 + depths[2]

        self.stage1 = make_stage(
            dims[0], dims[0], depths[0], rates[0:end1], downsample=False
        )
        self.stage2 = make_stage(dims[0], dims[1], depths[1], rates[end1:end2])
        self.stage3 = make_stage(dims[1], dims[2], depths[2], rates[end2:end3])
        self.stage4 = make_stage(dims[2], dims[3], depths[3], rates[end3:])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(dims[3], eps=1e-6)
        self.head = nn.Linear(dims[3], n_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.norm(x)
        x = self.head(x)

        return x
