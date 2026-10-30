"""Model components for the COMP3710 ADNI project (PyTorch only).

A compact ConvNeXt (Liu et al., "A ConvNet for the 2020s", CVPR 2022) written
from scratch: a 4x4 patchify stem, four stages of ConvNeXt blocks separated by
2x2 down-sampling layers, global average pooling and a linear classifier.
No pretrained weights are used.
"""
import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Layer normalization for image feature maps."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """Randomly drop residual paths during training."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        mask = (
            torch.rand(shape, device=x.device) < keep_prob
        ).to(x.dtype)

        return x * mask / keep_prob


class ConvNeXtBlock(nn.Module):
    """A ConvNeXt residual block."""

    def __init__(self, channels, drop_path=0.0):
        super().__init__()

        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels
        )

        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, channels * 4)
        self.activation = nn.GELU()
        self.project = nn.Linear(channels * 4, channels)

        self.gamma = nn.Parameter(
            1e-6 * torch.ones(channels)
        )

        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        residual = x

        x = self.depthwise(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.project(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        return residual + self.drop_path(x)


class ConvNeXt(nn.Module):
    """Compact ConvNeXt for binary MRI classification."""

    def __init__(
        self,
        num_classes=2,
        in_channels=3,
        depths=(2, 2, 4, 2),
        dims=(32, 64, 128, 256),
        drop_path_rate=0.1
    ):
        super().__init__()

        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("ConvNeXt expects four stages")

        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    dims[0],
                    kernel_size=4,
                    stride=4
                ),
                LayerNorm2d(dims[0])
            )
        ])

        for i in range(1, 4):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i - 1]),
                    nn.Conv2d(
                        dims[i - 1],
                        dims[i],
                        kernel_size=2,
                        stride=2
                    )
                )
            )

        probabilities = torch.linspace(
            0, drop_path_rate, sum(depths)
        ).tolist()

        self.stages = nn.ModuleList()
        block_index = 0

        for stage_index in range(4):
            blocks = []

            for _ in range(depths[stage_index]):
                blocks.append(
                    ConvNeXtBlock(
                        dims[stage_index],
                        probabilities[block_index]
                    )
                )
                block_index += 1

            self.stages.append(nn.Sequential(*blocks))

        self.head_norm = nn.LayerNorm(
            dims[-1], eps=1e-6
        )
        self.classifier = nn.Linear(
            dims[-1], num_classes
        )

    def forward(self, x):
        for downsample, stage in zip(
            self.downsample_layers, self.stages
        ):
            x = downsample(x)
            x = stage(x)

        x = x.mean(dim=(2, 3))
        x = self.head_norm(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = ConvNeXt()
    model.eval()

    parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:", parameters)

    with torch.no_grad():
        images = torch.randn(2, 3, 224, 224)
        outputs = model(images)

    print("Input shape:", tuple(images.shape))
    print("Output shape:", tuple(outputs.shape))

    assert outputs.shape == (2, 2)

    print("ConvNeXt forward test passed.")
