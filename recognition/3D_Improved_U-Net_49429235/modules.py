import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormAct(nn.Module): 
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
 
    def forward(self, x):
        return self.block(x)

class ContextModule(nn.Module):
    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.Dropout3d(p=dropout),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
        )
 
    def forward(self, x):
        return x + self.block(x)

class UpsampleModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = ConvNormAct(in_ch, out_ch, kernel_size=3)

    def forward(self, x):
        return self.conv(self.up(x))


class LocalisationModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3 = ConvNormAct(in_ch, out_ch, kernel_size=3)
        self.conv1 = ConvNormAct(out_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv1(self.conv3(x))



class Improved3DUNet(nn.Module):

    def __init__(self, in_channels=1, num_classes=6, base_channels=16,
                 dropout=0.3, deep_supervision=True):
        super().__init__()
        b = base_channels
        ch = [b, 2 * b, 4 * b, 8 * b, 16 * b]
        self.deep_supervision = deep_supervision

        # Encoder
        self.enc0 = nn.Sequential(ConvNormAct(in_channels, ch[0]), ContextModule(ch[0], dropout))
        self.down = nn.ModuleList([ConvNormAct(ch[i - 1], ch[i], stride=2) for i in range(1, 5)])
        self.ctx = nn.ModuleList([ContextModule(ch[i], dropout) for i in range(1, 5)])

        # Decoder
        self.up4 = UpsampleModule(ch[4], ch[3])
        self.loc3 = LocalisationModule(ch[3] * 2, ch[3])
        self.up3 = UpsampleModule(ch[3], ch[2])
        self.loc2 = LocalisationModule(ch[2] * 2, ch[2])
        self.up2 = UpsampleModule(ch[2], ch[1])
        self.loc1 = LocalisationModule(ch[1] * 2, ch[1])
        self.up1 = UpsampleModule(ch[1], ch[0])
        self.final_conv = ConvNormAct(ch[0] * 2, ch[1], kernel_size=3)

        # Segmentation heads (1x1x1 convs)
        self.seg0 = nn.Conv3d(ch[1], num_classes, kernel_size=1)  # full resolution
        if deep_supervision:
            self.seg1 = nn.Conv3d(ch[1], num_classes, kernel_size=1)  # 1/2 resolution
            self.seg2 = nn.Conv3d(ch[2], num_classes, kernel_size=1)  # 1/4 resolution

    @staticmethod
    def _check_shape(x):
        if any(s % 16 != 0 for s in x.shape[2:]):
            raise ValueError(
                f"Spatial dims must be divisible by 16, got {tuple(x.shape[2:])}")

    def forward(self, x):
        self._check_shape(x)

        # Encoder
        x0 = self.enc0(x)
        feats = [x0]
        h = x0
        for down, ctx in zip(self.down, self.ctx):
            h = down(h)
            h = ctx(h)
            feats.append(h)
        x0, x1, x2, x3, x4 = feats

        # Decoder
        d3 = self.loc3(torch.cat([self.up4(x4), x3], dim=1))   # 1/8
        d2 = self.loc2(torch.cat([self.up3(d3), x2], dim=1))   # 1/4
        d1 = self.loc1(torch.cat([self.up2(d2), x1], dim=1))   # 1/2
        d0 = self.final_conv(torch.cat([self.up1(d1), x0], dim=1))  # full

        o0 = self.seg0(d0)
        if not self.deep_supervision:
            return o0

        o1 = self.seg1(d1)
        o2 = self.seg2(d2)
        up = lambda t, **kw: F.interpolate(t, mode="trilinear", align_corners=False, **kw)
        comb1 = up(o2, scale_factor=2) + o1          # 1/2 resolution
        final = up(comb1, scale_factor=2) + o0       # full resolution

        if self.training:
            full = x.shape[2:]
            return [final, up(comb1, size=full), up(o2, size=full)]
        return final


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):

    def __init__(self, in_channels=1, num_classes=6, base_channels=16):
        super().__init__()
        b = base_channels
        ch = [b, 2 * b, 4 * b, 8 * b, 16 * b]
        self.encoders = nn.ModuleList(
            [DoubleConv(in_channels, ch[0])] + [DoubleConv(ch[i - 1], ch[i]) for i in range(1, 5)])
        self.pool = nn.MaxPool3d(2)
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose3d(ch[i], ch[i - 1], kernel_size=2, stride=2) for i in range(4, 0, -1)])
        self.decoders = nn.ModuleList([DoubleConv(ch[i - 1] * 2, ch[i - 1]) for i in range(4, 0, -1)])
        self.head = nn.Conv3d(ch[0], num_classes, kernel_size=1)

    def forward(self, x):
        if any(s % 16 != 0 for s in x.shape[2:]):
            raise ValueError(f"Spatial dims must be divisible by 16, got {tuple(x.shape[2:])}")
        skips = []
        h = x
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            if i < len(self.encoders) - 1:
                skips.append(h)
                h = self.pool(h)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = dec(torch.cat([up(h), skip], dim=1))
        return self.head(h)

def build_model(name, in_channels=1, num_classes=6, base_channels=16,
                dropout=0.3, deep_supervision=True):
    """Factory: name is 'improved' or 'baseline'."""
    if name == "improved":
        return Improved3DUNet(in_channels, num_classes, base_channels, dropout, deep_supervision)
    if name == "baseline":
        return UNet3D(in_channels, num_classes, base_channels)
    raise ValueError(f"Unknown model '{name}' (use 'improved' or 'baseline')")
 
 
if __name__ == "__main__":
    # Quick shape / parameter smoke test (no data needed).
    for name in ("improved", "baseline"):
        net = build_model(name, base_channels=8)
        x = torch.randn(1, 1, 32, 32, 16)
        net.train()
        out = net(x)
        shapes = [tuple(o.shape) for o in out] if isinstance(out, list) else tuple(out.shape)
        net.eval()
        with torch.no_grad():
            ev = net(x)
        print(name, "train out:", shapes, "| eval out:", tuple(ev.shape),
              "| params:", sum(p.numel() for p in net.parameters()))


