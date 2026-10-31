# Basic 2D UNet baseline. It takes whole volumes like the 3D model, but each slice goes through 2D convs on its own,
# so it trains with train.py's main (same split, augmentation, dice loss, optimiser, scheduler, filters) and only differs by being 2D

import torch
import torch.nn as nn
import torch.nn.functional as F

import train

def double_conv(in_channels, out_channels):
    # two 3x3 convs, padding 1 keeps the resolution the same so the skip connections line up
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=6, base_filters=16):
        super().__init__()
        f = base_filters
        # encoder, each stage halves the resolution and doubles the filters
        self.enc1 = double_conv(in_channels, f)
        self.enc2 = double_conv(f, f * 2)
        self.enc3 = double_conv(f * 2, f * 4)
        self.enc4 = double_conv(f * 4, f * 8)
        self.enc5 = double_conv(f * 8, f * 16) # bottleneck
        # decoder, upsample then join with the encoder output of the same size (*2 filters because of the skip)
        self.up4, self.dec4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2), double_conv(f * 16, f * 8)
        self.up3, self.dec3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2), double_conv(f * 8, f * 4)
        self.up2, self.dec2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2), double_conv(f * 4, f * 2)
        self.up1, self.dec1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2), double_conv(f * 2, f)
        self.out = nn.Conv2d(f, num_classes, kernel_size=1) # one score per class per pixel

    def forward(self, volumes):
        B, D = volumes.shape[0], volumes.shape[-1]
        x = volumes.permute(0, 4, 1, 2, 3).flatten(0, 1) # (B, 1, H, W, D) -> (B*D, 1, H, W), every slice is its own image

        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        x4 = self.enc4(F.max_pool2d(x3, 2))
        x5 = self.enc5(F.max_pool2d(x4, 2))
        x = self.dec4(torch.cat([x4, self.up4(x5)], dim=1))
        x = self.dec3(torch.cat([x3, self.up3(x)], dim=1))
        x = self.dec2(torch.cat([x2, self.up2(x)], dim=1))
        x = self.dec1(torch.cat([x1, self.up1(x)], dim=1))

        x = self.out(x) # (B*D, C, H, W)
        return x.unflatten(0, (B, D)).permute(0, 2, 3, 4, 1) # back to (B, C, H, W, D), the same as the 3D model

if __name__ == "__main__":
    train.main(UNet2D(1, train.num_classes, train.base_filters), name="unet_2d")
