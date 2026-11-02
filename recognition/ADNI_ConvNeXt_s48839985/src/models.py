"""
CNN and ConvNeXt model implementations.
Used to assess Alzheimer's presence in ADNI MRI images.

ConvNeXt references:
1. "ConvNet for the 2020s" (Liu et al, 2022)
2. https://www.geeksforgeeks.org/computer-vision/convnext/
3. PyTorch Image Models (timm) module
"""

import torch
import torch.nn as nn

from dataset import NUM_CLASSES

#################
# ADNI CNN Model
#################


class ADNICNN(nn.Module):
    """
    Basic CNN implementation to classify AD / NC patients in ADNI Dataset
    """

    def __init__(self):
        super(ADNICNN, self).__init__()
        self.features = nn.Sequential(
            self._cnn_block(1, 32),
            self._cnn_block(32, 64),
            self._cnn_block(64, 128),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, NUM_CLASSES),
        )

    @staticmethod
    def _cnn_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


#################
# ConvNeXt Model
#################


class DropPath2d(nn.Module):
    """
    Stochastically drop branches during training based on drop prob

    Args:
        drop_prob: probability of any given branch being 'dropped'
        scale_remaining: should remaining branches be scaled to
                         compensate for dropped branches?
    """

    def __init__(self, drop_prob: float = 0.0, scale_remaining: bool = True):
        super().__init__()

        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in the range [0, 1)")

        self.drop_prob = drop_prob
        self.scale_remaining = scale_remaining

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        # Getting random 0 (toss) or 1 (keep) value per image
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], 1, 1, 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        if self.scale_remaining:
            # Scaling based on keep probability
            random_tensor.div_(keep_prob)

        return x * random_tensor


class LayerNorm2d(nn.Module):
    """
    Implementation of a ConvNeXt Layer normalisation layer
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()

        if data_format not in ("channels_first", "channels_last"):
            raise ValueError(f"Data format {data_format} is BAD!!!")

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )

        # Channels-first: (B, C, H, W)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)

        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]

        return x


class ConvNeXtBlock(nn.Module):
    """
    Implementation of a ConvNeXt Block
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()

        # Depthwise convolution
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)

        # 1x1 convolution; implemented with linear layers
        self.pw_conv1 = nn.Linear(dim, 4 * dim)
        self.gelu = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * dim, dim)

        # Learnable scale values for channel outputs
        self.gamma = None
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

        self.drop_path = DropPath2d(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        orig_input = x
        x = self.dw_conv(x)

        # LayerNorm requires tensor channels_list format
        # i.e. (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        x = self.pw_conv1(x)
        x = self.gelu(x)
        x = self.pw_conv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Residual output = x + F(x)
        return orig_input + self.drop_path(x)


class ADNI_ConvNeXt(nn.Module):
    """
    Implementation of the ConvNeXt model
    """

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        super().__init__()

        # Stem and 3 downsampling convolution layers
        self.downsample_layers = nn.ModuleList()

        # ConvNeXt stem: Conv2d -> Layer Normalisation
        stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        # 3x Downsampling layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages
        self.stages = nn.ModuleList()

        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Last normalisation layer
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], NUM_CLASSES)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(
            x.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
