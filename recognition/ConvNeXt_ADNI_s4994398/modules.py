"""
ConvNeXt (Liu et al., "A ConvNet for the 2020s", CVPR 2022) implemented from scratch.

The network is a 4-stage hierarchy:

    stem (4x4 conv, stride 4) -> [stage, downsample] x 3 -> stage -> GAP -> LN -> FC

Each ConvNeXt block is

    x -> 7x7 depthwise conv -> LayerNorm -> 1x1 conv (4C) -> GELU -> 1x1 conv (C)
      -> layer scale (gamma) -> stochastic depth -> + x

The 1x1 convs are implemented as ``nn.Linear`` on channels-last tensors, as in the
reference implementation. ``load_imagenet_weights`` maps torchvision's ImageNet-1k
ConvNeXt checkpoints onto this implementation so the model can be fine-tuned rather
than trained from scratch on ~20k slices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIGS = {
    "tiny": dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768)),
    "small": dict(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768)),
    "base": dict(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024)),
}


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dimension of an NCHW tensor."""

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """Stochastic depth: randomly drops the whole residual branch per sample."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = (self.gamma * x).permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=1, num_classes=2, depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768), drop_path_rate=0.0, head_dropout=0.0):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        # stochastic depth rate increases linearly with block index
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        k = 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[Block(dims[i], rates[k + j]) for j in range(depths[i])]))
            k += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        for down, stage in zip(self.downsample_layers, self.stages):
            x = stage(down(x))
        return self.norm(x.mean(dim=(-2, -1)))  # global average pool

    def forward(self, x):
        return self.head(self.head_drop(self.forward_features(x)))


def _torchvision_to_ours(sd):
    """Rename torchvision ``convnext_*`` state-dict keys to this module's names."""
    out = {}
    for k, v in sd.items():
        parts = k.split(".")
        if parts[0] == "classifier":
            if parts[1] == "0":  # final LayerNorm2d
                out[f"norm.{parts[2]}"] = v
            continue  # ImageNet FC head is discarded
        idx = int(parts[1])
        if idx % 2 == 0:  # stem (0) or downsample layer (2, 4, 6)
            out[f"downsample_layers.{idx // 2}.{parts[2]}.{parts[3]}"] = v
        else:  # stage
            stage, block = idx // 2, parts[2]
            prefix = f"stages.{stage}.{block}"
            if parts[3] == "layer_scale":
                out[f"{prefix}.gamma"] = v.flatten()
            else:
                name = {"0": "dwconv", "2": "norm", "3": "pwconv1", "5": "pwconv2"}[parts[4]]
                out[f"{prefix}.{name}.{parts[5]}"] = v
    return out


def load_imagenet_weights(model, variant="tiny"):
    """Initialise ``model`` from torchvision's ImageNet-1k ConvNeXt weights.

    The RGB stem filters are summed over the colour channels so the network
    accepts single-channel MRI slices while giving the same response to a
    greyscale image replicated across RGB.
    """
    import torchvision

    builder = getattr(torchvision.models, f"convnext_{variant}")
    sd = builder(weights="IMAGENET1K_V1").state_dict()
    sd = _torchvision_to_ours(sd)
    stem = sd["downsample_layers.0.0.weight"]
    in_chans = model.downsample_layers[0][0].in_channels
    if in_chans != stem.shape[1]:
        sd["downsample_layers.0.0.weight"] = stem.sum(dim=1, keepdim=True).repeat(1, in_chans, 1, 1) / in_chans
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) == {"head.weight", "head.bias"}, missing
    assert not unexpected, unexpected
    return model


def build_model(variant="tiny", pretrained=True, num_classes=2, in_chans=1,
                drop_path_rate=0.1, head_dropout=0.0):
    model = ConvNeXt(in_chans=in_chans, num_classes=num_classes,
                     drop_path_rate=drop_path_rate, head_dropout=head_dropout,
                     **CONFIGS[variant])
    if pretrained:
        load_imagenet_weights(model, variant)
    return model


if __name__ == "__main__":
    # Check the weight mapping reproduces torchvision's features exactly.
    import torchvision

    ref = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1").eval()
    ours = ConvNeXt(in_chans=3, num_classes=1000, **CONFIGS["tiny"])
    load_imagenet_weights(ours, "tiny")
    ours.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        a = ref.classifier[1](ref.classifier[0](ref.avgpool(ref.features(x))))
        b = ours.forward_features(x)
    print("max feature diff vs torchvision:", (a - b).abs().max().item())
    m = build_model()
    print("ConvNeXt-T params: %.1fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
    print("output", tuple(m(torch.randn(2, 1, 224, 224)).shape))
