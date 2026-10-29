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

    # nn.LayerNorm normalises over the *last* dimension, so move channels last,
    # normalise each pixel's C-dim feature vector, then move channels back.
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)  # NHWC -> NCHW


class DropPath(nn.Module):
    """Stochastic depth: randomly drops the whole residual branch per sample."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        # one Bernoulli draw per sample, broadcast over C,H,W: a sample either keeps
        # the whole branch or loses it (the residual shortcut still passes through)
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        # divide by keep so the expected output equals the input (like inverted dropout),
        # which lets the full network be used unchanged at test time
        return x * mask / keep


class Block(nn.Module):
    """One ConvNeXt block. Input and output shape: [B, dim, H, W]."""

    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        # Spatial mixing: groups=dim makes it depthwise (one 7x7 filter per channel),
        # so it is cheap even with a large kernel. padding=3 keeps H and W unchanged.
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Channel mixing: an MLP applied independently at every pixel. A 1x1 conv is
        # the same operation as a Linear layer on channels-last data (and faster).
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # expand 4x (inverted bottleneck)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # project back
        # Layer scale: per-channel multiplier starting near 0, so each block begins
        # close to the identity function, which stabilises training of deep stacks.
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)                                       # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)                                # [B, H, W, C] for LayerNorm/Linear
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))   # [B, H, W, C] (via 4C)
        x = (self.gamma * x).permute(0, 3, 1, 2)                 # back to [B, C, H, W]
        return shortcut + self.drop_path(x)                      # residual connection


class ConvNeXt(nn.Module):
    """Full network. ``depths`` = blocks per stage, ``dims`` = channels per stage.

    For a 224x224 input the feature maps are 56x56 -> 28x28 -> 14x14 -> 7x7.
    """

    def __init__(self, in_chans=1, num_classes=2, depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768), drop_path_rate=0.0, head_dropout=0.0):
        super().__init__()
        # downsample_layers[0] is the "patchify" stem: a 4x4 conv with stride 4 splits
        # the image into non-overlapping 4x4 patches (224 -> 56), like a ViT patch embedding.
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        ))
        # Between stages: LayerNorm then a 2x2 stride-2 conv halves H, W and widens channels.
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        # Stochastic depth rate increases linearly with block index: early blocks
        # (generic low-level features) are almost never dropped, deep ones most often.
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        k = 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[Block(dims[i], rates[k + j]) for j in range(depths[i])]))
            k += depths[i]

        # Classification head: global average pool -> LayerNorm -> (dropout) -> Linear.
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Truncated-normal init from the ConvNeXt paper; only matters when training
        # from scratch, since pretrained weights overwrite everything except the head.
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """[B, in_chans, H, W] -> pooled feature vector [B, dims[-1]]."""
        for down, stage in zip(self.downsample_layers, self.stages):
            x = stage(down(x))
        return self.norm(x.mean(dim=(-2, -1)))  # global average pool over H, W

    def forward(self, x):
        """Returns raw logits [B, num_classes]; apply softmax for probabilities."""
        return self.head(self.head_drop(self.forward_features(x)))


def _torchvision_to_ours(sd):
    """Rename torchvision ``convnext_*`` state-dict keys to this module's names.

    torchvision stores stem/downsample layers and stages interleaved in one
    ``features`` Sequential: features.0 = stem, features.1 = stage 0,
    features.2 = downsample 1, features.3 = stage 1, ... so even indices are
    down-sampling layers and odd indices are stages. Inside a block, its
    ``block`` Sequential is [dwconv, permute, norm, linear1, GELU, linear2, permute],
    so sub-indices 0/2/3/5 hold the weights.
    Example: features.5.2.block.3.weight -> stages.2.2.pwconv1.weight
    """
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
                out[f"{prefix}.gamma"] = v.flatten()  # torchvision shape (C,1,1) -> ours (C,)
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
    # Stem weight is [96, 3, 4, 4] for RGB. A greyscale image copied into R, G and B
    # gives sum_c(w_c * x) = (sum_c w_c) * x, so summing over the colour dim reproduces
    # the pretrained response exactly with a single input channel.
    stem = sd["downsample_layers.0.0.weight"]
    in_chans = model.downsample_layers[0][0].in_channels
    if in_chans != stem.shape[1]:
        sd["downsample_layers.0.0.weight"] = stem.sum(dim=1, keepdim=True).repeat(1, in_chans, 1, 1) / in_chans
    # strict=False because the ImageNet head (1000 classes) is dropped; the asserts make
    # sure *only* our new 2-class head is left uninitialised and nothing else mismatched.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) == {"head.weight", "head.bias"}, missing
    assert not unexpected, unexpected
    return model


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class BaselineCNN(nn.Module):
    """Plain VGG-style CNN used as the comparison baseline.

    Four stages of two 3x3 conv-BN-ReLU layers followed by 2x2 max-pooling,
    then global average pooling, dropout and a linear classifier. Trained from
    scratch, so it isolates what ConvNeXt's architecture + ImageNet pretraining add.
    """

    def __init__(self, in_chans=1, num_classes=2, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        layers, cin = [], in_chans
        for w in widths:
            layers += [ConvBNReLU(cin, w), ConvBNReLU(w, w), nn.MaxPool2d(2)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.features(x).mean(dim=(-2, -1))  # 4 max-pools: 224 -> 14, then global average pool
        return self.head(self.drop(x))


def build_model(arch="convnext", variant="tiny", pretrained=True, num_classes=2, in_chans=1,
                drop_path_rate=0.1, head_dropout=0.0):
    """Factory used by train.py / predict.py. ``arch`` is "convnext" or "baseline"."""
    if arch == "baseline":
        return BaselineCNN(in_chans=in_chans, num_classes=num_classes)
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
