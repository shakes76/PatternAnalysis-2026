import torch
import torch.nn as nn


CONVNEXT_CONFIGS = {
    "tiny": ((3, 1, 3, 1), (96, 192, 384, 768)),
    "compact": ((2, 2, 6, 2), (40, 80, 160, 320)),
    "paper_tiny": ((3, 3, 9, 3), (96, 192, 384, 768)),
}


class DropPath(nn.Module):
    def __init__(self, drop_probability=0.0):
        super().__init__()
        if not 0.0 <= drop_probability < 1.0:
            raise ValueError("drop_probability must be in [0, 1)")
        self.drop_probability = drop_probability

    def forward(self, x):
        if self.drop_probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.drop_probability
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(mask_shape).bernoulli_(keep_probability)
        return x * mask / keep_probability


class BaselineCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Preserve the earlier class name for any existing checkpoints or imports.
BaselineNN = BaselineCNN


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path_rate)
        
    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.drop_path(x) + input_x
        return x


class ConvNeXtTiny(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        depths=None,
        dims=None,
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init=1e-6,
        variant="tiny",
    ):
        super().__init__()
        if variant not in CONVNEXT_CONFIGS:
            raise ValueError(
                f"Unsupported ConvNeXt variant {variant!r}; "
                f"choose from {', '.join(CONVNEXT_CONFIGS)}"
            )
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must be in [0, 1)")
        if layer_scale_init < 0.0:
            raise ValueError("layer_scale_init must be non-negative")
        default_depths, default_dims = CONVNEXT_CONFIGS[variant]
        depths = default_depths if depths is None else depths
        dims = default_dims if dims is None else dims
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("ConvNeXt depths and dims must each contain four stages")
        if any(depth <= 0 for depth in depths) or any(dim <= 0 for dim in dims):
            raise ValueError("ConvNeXt depths and dims must contain positive integers")
        total_blocks = sum(depths)
        block_drop_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        next_block = 0
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                    LayerNorm2d(dims[0]),
                )
            ]
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        for stage_index, depth in enumerate(depths):
            stage_blocks = [
                ConvNeXtBlock(
                    dim=dims[stage_index],
                    drop_path_rate=block_drop_rates[next_block + block_index],
                )
                for block_index in range(depth)
            ]
            for block in stage_blocks:
                nn.init.constant_(block.gamma, layer_scale_init)
            self.stages.append(nn.Sequential(*stage_blocks))
            next_block += depth
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(dims[-1], num_classes)
        if variant == "paper_tiny":
            # Liu et al. (CVPR 2022): scratch initialization for the paper-sized variant.
            self.apply(self._initialize_paper_weights)

    @staticmethod
    def _initialize_paper_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = downsample(x)
            x = stage(x)

        x = x.mean([-2, -1])
        x = self.norm(x)
        x = self.head_dropout(x)
        return self.head(x)