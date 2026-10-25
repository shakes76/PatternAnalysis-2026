"""
ResNet-18 and ConvNeXt-Tiny implementations with PyTorch

Both models start with random weights and return raw classification logits.
Input: (batch_size, 1, 224, 224), for single-channel grayscale images.
Output: (batch_size, 2), for AD/NC classification.

Architecture references:
- ResNet: https://arxiv.org/abs/1512.03385
- ConvNeXt: https://arxiv.org/abs/2201.03545
"""

import torch
from torch import nn

class ResNetBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection"""

    def __init__(self, in_channels, out_channels, stride=1):
        """Construct a two-convolution residual block with an optional projected shortcut.

        Args:
            in_channels: number of input feature channels
            out_channels: number of output feature channels
            stride: spatial stride of the first convolution and projected shortcut
        """
        super().__init__()

        # first convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # match the skip connection to the main branch when its shape changes
        self.downsample = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            # project the shortcut so it can be added to the main branch
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """Apply the residual branch, add the shortcut, and activate the result.

        Args:
            x: input features of shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: output of forward pass
        """
        # save the residual path before applying the main convolutions
        shortcut = self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        # combine both paths and apply the final activation
        return self.relu(x + shortcut)

class ResNet18(nn.Module):
    """ResNet-18 implementation of four stages with [2, 2, 2, 2] residual blocks"""

    def __init__(self, num_classes=2):
        """Construct a randomly initialized ResNet-18 for single-channel images.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()

        # the stem extracts low-level features and reduces 224x224 inputs to 56x56
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7,
            stride=2, padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_stage(64, 64, stride=1)    # (C, H, W) = (64, 56, 56) for 224x224 input
        self.layer2 = self._make_stage(64, 128, stride=2)   # (128, 28, 28)
        self.layer3 = self._make_stage(128, 256, stride=2)  # (256, 14, 14)
        self.layer4 = self._make_stage(256, 512, stride=2)  # (512, 7, 7)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _make_stage(in_channels, out_channels, stride):
        """Build a stage of two residual blocks with downsampling in the first block.

        Args:
            in_channels: number of channels entering the stage
            out_channels: number of channels produced by both blocks
            stride: spatial stride applied by the first block

        Returns:
            nn.Sequential: two ResNetBlock modules forming one residual stage
        """
        # only the first block changes the spatial size or channel count
        return nn.Sequential(
            ResNetBlock(in_channels, out_channels, stride=stride),
            ResNetBlock(out_channels, out_channels),
        )

    @staticmethod
    def _init_weights(layer):
        """Initialize convolution and batch-normalization parameters in place.

        Args:
            layer: layer to be initalised
        """
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(
                layer.weight, mode="fan_out", nonlinearity="relu",
            )
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Extract residual features and classify their spatial averages.

        Args:
            x: single-channel image batch, typically (N, 1, 224, 224)

        Returns:
            torch.Tensor: output of forward pass
        """
        # the first stage keeps the resolution, later stages downsample it
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)  # pool H/W, retaining batch and channel axes
        return self.fc(x)


class LayerNorm2d(nn.Module):
    """Normalize channels at each spatial position of a tensor"""

    def __init__(self, channels):
        """Create layer normalization over the channel dimension of image features.

        Args:
            channels: number of channels in each input feature map
        """
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        """Normalize channels independently at each spatial position.

        Args:
            x: features in (N, C, H, W) order, with C equal to channels

        Returns:
            torch.Tensor: mormalized features with the same shape and dimension order
        """
        # layerNorm operates on the last dimension, so temporarily move C last
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # NHWC -> NCHW

class DropPath(nn.Module):
    """Randomly drop a samples residual branch during training"""

    def __init__(self, probability=0.0):
        """Configure per-sample stochastic depth for a residual branch.

        Args:
            probability: Probability of dropping a branch

        Raises:
            ValueError: If probability is outside [0, 1).
        """
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("Drop-path probability must be in [0, 1).")
        self.probability = probability

    def forward(self, x):
        """Randomly zero whole sample branches during training and rescale survivors.

        Args:
            x: residual features with the batch dimension first

        Returns:
            torch.Tensor: output of forward pass
        """
        if not self.training or self.probability == 0.0:
            return x

        keep_probability = 1.0 - self.probability
        # one random decision per image
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(mask_shape).bernoulli_(keep_probability)
        # Rescale surviving branches to preserve their expected contribution.
        return x * mask / keep_probability

class ConvNeXtBlock(nn.Module):
    """Depthwise convolution, channel expansion, and a residual connection"""

    def __init__(self, channels, drop_path=0.0):
        """Construct a ConvNeXt residual block with layer scaling and stochastic depth.

        Args:
            channel: number of input and output feature channels
            drop_path: per-sample residual drop probability in [0, 1)
        """
        super().__init__()

        # groups=channels applies a separate spatial filter to each channel
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels,
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)

        # applied to NHWC tensors these linear layers act like 1x1 convolutions
        self.expand = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.project = nn.Linear(4 * channels, channels)

        # one learnable scale per channel, initialized close to zero
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-6))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        """Filter spatial features, mix channels, and add the scaled residual branch.

        Args:
            x: features of shape (N, channels, H, W)

        Returns:
            torch.Tensor: updated features of the same shape as x
        """
        shortcut = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.project(self.activation(self.expand(x)))
        x = x * self.layer_scale
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return shortcut + self.drop_path(x)

class ConvNeXt(nn.Module):
    """ConvNeXt with depths [3, 3, 9, 3] and widths [96, 192, 384, 768].
    drop_path_rate is the maximum stochastic-depth probability, it increases linearly across the 18 blocks.
    """

    def __init__(self, num_classes=2, drop_path_rate=0.1):
        """Construct a randomly initialized ConvNeXt-Tiny for single-channel images.

        Args:
            num_classes: number of output class logits
            drop_path_rate: maximum residual drop probability in [0, 1), increasing linearly from zero across the 18 blocks
        """
        super().__init__()

        # tiny
        depths = (3, 3, 9, 3)
        widths = (96, 192, 384, 768)

        # pico
        # depths = (2, 2, 6, 2)
        # widths = (64, 128, 256, 512)

        # the stem reduces spatial size by four 224x224 -> 56x56.
        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], kernel_size=4, stride=4),
            LayerNorm2d(widths[0]),
        )

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        block_index = 0
        total_blocks = sum(depths)

        for stage_index in range(4):
            channels = widths[stage_index]
            blocks = []

            for _ in range(depths[stage_index]):
                # Deeper blocks receive progressively stronger stochastic depth.
                probability = drop_path_rate * block_index / (total_blocks - 1)
                blocks.append(ConvNeXtBlock(channels, drop_path=probability))
                block_index += 1

            self.stages.append(nn.Sequential(*blocks))

            if stage_index < 3:
                # normalize, halve H/W, and double the channel count
                self.downsamples.append(nn.Sequential(
                    LayerNorm2d(channels),
                    nn.Conv2d(
                        channels, widths[stage_index + 1],
                        kernel_size=2, stride=2,
                    ),
                ))

        self.norm = nn.LayerNorm(widths[-1], eps=1e-6)
        self.head = nn.Linear(widths[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer):
        """Initialize convolution and linear parameters in place.

        Args:
            layer: layer to be initalised
        """
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Extract hierarchical features and classify their normalized spatial averages.

        Args:
            x: single-channel image batch, typically (N, 1, 224, 224)

        Returns:
            torch.Tensor: raw class logits of shape (N, num_classes), before softmax
        """
        x = self.stem(x)
        for stage_index in range(4):
            x = self.stages[stage_index](x)
            if stage_index < 3:
                x = self.downsamples[stage_index](x)

        x = x.mean(dim=(2, 3))  # global average pooling: (N, C, H, W) -> (N, C)
        x = self.norm(x)
        return self.head(x)

def build_model(
    model_name, num_classes=2,
):
    """Build a supported model with randomly initialized weights.

    Args:
        model_name: architecture name, either "resnet18" or "convnext"
        num_classes: number of output class logits

    Returns:
        nn.Module: a ResNet18 or ConvNeXt model

    Raises:
        ValueError: If model_name is not a supported architecture.
    """
    
    if model_name == "resnet18":
        return ResNet18(num_classes=num_classes)

    if model_name == "convnext":
        return ConvNeXt(
            num_classes=num_classes,
            drop_path_rate=0.1,
        )

    raise ValueError("model_name must be 'resnet18' or'convnext'.")


if __name__ == "__main__":
    for model_name in ("resnet18", "convnext"):
        model = build_model(model_name)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"{model_name}: {parameter_count} parameters")

