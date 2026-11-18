# Containing the source code of the components of your model. Each component must be implemented as a class or a funciton in PyTorch or TensorFlow.
# Your implementation of the model should not depend on NumPy (or other python libraries not part of Pytorch / tensorFlow) in any way unless otherwise
# approved by teaching staff.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    # Encoder block: two 3x3x3 convs with dropout in between

    def __init__(self, channels, dropout=0.3):
        # instanceNorm -> leakyrelu -> 3x3x3 conv -> dropout -> instancenorm -> leakyrelu -> 3x3x3 conv (channel count stays same)
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout3d(dropout),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # run x through layers and return result
        return self.block(x)

class UpsampleBlock(nn.Module):
    # Decoder step (doubles spatial size)

    def __init__(self, in_channels, out_channels):
        # upsample scale 2-> 3x3x3 conv (in -> out channels) -> instance norm -> leaky relu
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
        )

    def forward(self, x):
        # upsample x by 2 and convolve to out_channels
        return self.block(x)

class LocalisationBlock(nn.Module):
    # decoder run after concatting with skip connection

    def __init__(self, in_channels, out_channels):
        # 3x3x3 convs (keeps channels) -> instance norm -> leakyrelu -> 1x1x1 conv (in->out chans) -> instance norm -> leakyrelu
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
        )

    def forward(self, x):
        # combine concat features and reduce chan count
        return self.block(x)

class Improved3DUNet(nn.Module):
    # the full improved 3d unet

    def __init__(self, in_channels=1, num_classes=6, base_filters=16):
        # encoder (5 levels, filters 16, 32, 64, 128, 256): each lvl = 3x3x3 conv (stride 2 except lvl 1) + res block
        # decoder (lvls 4 -> 1): upsamp -> concat skip -> localisation block (lvl 1 ends with a 3x3x3 conv instead)
        # segmentation layers: 1x1x1 convs to num_classes at decoder lvls 3, 2, 1

        super().__init__()
        f1 = base_filters
        f2 = base_filters * 2
        f3 = base_filters * 4
        f4 = base_filters * 8
        f5 = base_filters * 16

        # encoder
        self.conv1 = nn.Conv3d(in_channels, f1, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(f1)
        self.conv2 = nn.Conv3d(f1, f2, kernel_size=3, stride=2, padding=1)
        self.res2 = ResidualBlock(f2)
        self.conv3 = nn.Conv3d(f2, f3, kernel_size=3, stride=2, padding=1)
        self.res3 = ResidualBlock(f3)
        self.conv4 = nn.Conv3d(f3, f4, kernel_size=3, stride=2, padding=1)
        self.res4 = ResidualBlock(f4)
        self.conv5 = nn.Conv3d(f4, f5, kernel_size=3, stride=2, padding=1)
        self.res5 = ResidualBlock(f5)

        # decoder (concat with skip doubles channels, localisation halves them again)
        self.up4 = UpsampleBlock(f5, f4)
        self.loc4 = LocalisationBlock(f4 * 2, f4)
        self.up3 = UpsampleBlock(f4, f3)
        self.loc3 = LocalisationBlock(f3 * 2, f3)
        self.up2 = UpsampleBlock(f3, f2)
        self.loc2 = LocalisationBlock(f2 * 2, f2)
        self.up1 = UpsampleBlock(f2, f1)
        # lvl 1 ends with 3x3x3 conv, instead of localisation (keeps 32 chans, in paper) ####### TODO
        self.final_conv = nn.Sequential(
            nn.Conv3d(f1 * 2, f1 * 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(f1 * 2, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
        )

        # segmentation layers at decoder lvls 3,2,1 (deep supervision)
        self.seg3 = nn.Conv3d(f3, num_classes, kernel_size=1)
        self.seg2 = nn.Conv3d(f2, num_classes, kernel_size=1)
        self.seg1 = nn.Conv3d(f1 * 2, num_classes, kernel_size=1)

    def forward(self, x):
        # encoder: at each lvl, out = conv(x); out = out + res(out); save out as a skip connection

        out = self.conv1(x)
        skip1 = out + self.res1(out)
        out = self.conv2(skip1)
        skip2 = out + self.res2(out)
        out = self.conv3(skip2)
        skip3 = out + self.res3(out)
        out = self.conv4(skip3)
        skip4 = out + self.res4(out)
        out = self.conv5(skip4)
        out = out + self.res5(out)

        # decoder: upsamp, concat with skip, localise
        out = self.loc4(torch.cat([self.up4(out), skip4], dim=1))
        lvl3 = self.loc3(torch.cat([self.up3(out), skip3], dim=1))
        lvl2 = self.loc2(torch.cat([self.up2(lvl3), skip2], dim=1))
        lvl1 = self.final_conv(torch.cat([self.up1(lvl2), skip1], dim=1))
        
        # deep supervision: upscale seg out from lvl3, add to lvl2, upscale, add to lvl1.
        seg = self.seg3(lvl3)
        seg = self.seg2(lvl2) + F.interpolate(seg, scale_factor=2, mode="trilinear", align_corners=False)
        seg = self.seg1(lvl1) + F.interpolate(seg, scale_factor=2, mode="trilinear", align_corners=False)

        # (N, num_classes, H, W, D), apply softmax or use cross entripy + dice
        return seg


