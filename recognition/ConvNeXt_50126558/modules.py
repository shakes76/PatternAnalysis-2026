# constructs the structure of the model
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module): # defines layer normalization
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

# stochastic depth: randomly drops the residual branch per sample during training
def drop_path(x, p, training):
    if p == 0. or not training:
        return x
    return x * x.new_empty(x.shape[0], 1, 1, 1).bernoulli_(1 - p) / (1 - p)

class Block(nn.Module): # ConvNeXt block: 7x7 depthwise conv, then a 2-layer MLP, plus a residual connection
    def __init__(self, dim, dp=0.):
        super().__init__()
        self.dp = dp
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        # 1x1 convs implemented as linear layers (inverted bottleneck, 4x expansion)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # learnable layer scale, starts near zero so each block begins close to identity
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + drop_path(x, self.dp, self.training)


class ConvNeXt(nn.Module): # stem, 4 stages of stacked blocks, then pooling and a linear head
    def __init__(self, in_chans=1, num_classes=2,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)))

        self.stages = nn.ModuleList() # 4 stages of stacked blocks (3, 3, 9, 3)
        dp = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[Block(dims[i], dp[cur + j]) for j in range(depths[i])]))
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        return self.head(self.forward_features(x))


def convnext(**kwargs):
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)