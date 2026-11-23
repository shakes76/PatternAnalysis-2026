import torch
import torch.nn as nn


class ContextModule(nn.Module):
    """
    2D adaptation of the pre-activation residual context module
    described by Isensee et al. (2018).

    Input and output feature channels remain the same.
    """

    def __init__(self, channels, dropout_p=0.3):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )
        self.norm1 = nn.InstanceNorm2d(channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        #Using normal dropout - randomly zeros feature units rather than entire feature layers
        self.dropout = nn.Dropout(p=dropout_p) # Changed the dropout layer during verification
        
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )
        self.norm2 = nn.InstanceNorm2d(channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(self.act1(self.norm1(x)))
        out = self.dropout(out)
        out = self.conv2(self.act2(self.norm2(out)))

        return out + residual

#Strided Convolution Downsampling

class Downsample(nn.Module):
    """
    Reduces spatial resolution by a factor of two
    using a strided 3x3 convolution.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)

#localisation module


class LocalizationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1
        )

        self.norm = nn.InstanceNorm2d(in_channels)
        self.act = nn.LeakyReLU(
            negative_slope=0.01,
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(self.norm(x))
        x = self.conv2(x)
        return x


#Upsampling

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

#Constructing the encoder

class Encoder(nn.Module):
    #Layers of the encoder
    def __init__(self, in_channels=1):
        super().__init__()

        self.initial_conv = nn.Conv2d(
            in_channels, 16, kernel_size=3, padding=1
        )

        self.context1 = ContextModule(16)

        self.down1 = Downsample(16, 32)
        self.context2 = ContextModule(32)

        self.down2 = Downsample(32, 64)
        self.context3 = ContextModule(64)

        self.down3 = Downsample(64, 128)
        self.context4 = ContextModule(128)

        self.down4 = Downsample(128, 256)
        self.context5 = ContextModule(256)

    #Forward pass - image passing through the layers
    def forward(self, x):
        x1 = self.context1(self.initial_conv(x))

        x2 = self.context2(self.down1(x1))

        x3 = self.context3(self.down2(x2))

        x4 = self.context4(self.down3(x3))

        x5 = self.context5(self.down4(x4))

        return x1, x2, x3, x4, x5


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.upsample = Upsample(in_channels, skip_channels)

        self.localization = LocalizationModule(
            skip_channels * 2,
            out_channels
        )

    def forward(self, x, skip):
        x = self.upsample(x)

        # Combine decoder and encoder features
        x = torch.cat([x, skip], dim=1)

        # Refine the combined features
        x = self.localization(x)

        return x

#Connecting all 4 decoder stages

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder4 = DecoderBlock(256, 128, 128)
        self.decoder3 = DecoderBlock(128, 64, 64)
        self.decoder2 = DecoderBlock(64, 32, 32)
        self.decoder1 = DecoderBlock(32, 16, 16)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features

        d4 = self.decoder4(x5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)

        return d1, d2, d3, d4

#6 class segmentation masks

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=6):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)

#Adding the Deep supervision layer- predictions at multiple resoultons are generated
# Upsamples the lower-resolution predictions and adds their logits to form the final six-class prediction.

class DeepSupervision(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.head_d3 = SegmentationHead(64, num_classes)
        self.head_d2 = SegmentationHead(32, num_classes)
        self.head_d1 = SegmentationHead(16, num_classes)

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )

    def forward(self, d1, d2, d3):
        pred_d3 = self.head_d3(d3)
        pred_d2 = self.head_d2(d2)
        pred_d1 = self.head_d1(d1)

        combined_d2 = self.upsample(pred_d3) + pred_d2
        combined_d1 = self.upsample(combined_d2) + pred_d1

        return combined_d1


#Assembling all the layers to a complete U-net architecture

class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder()
        self.deep_supervision = DeepSupervision(
            num_classes=num_classes
        )

    def forward(self, x):
        encoder_features = self.encoder(x)

        d1, d2, d3, d4 = self.decoder(encoder_features)

        logits = self.deep_supervision(d1, d2, d3)

        return logits


#Code sanity check
if __name__ == "__main__":
    model = ContextModule(channels=32)

    '''x = torch.randn(1, 32, 256, 144)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


    downsample = Downsample(32, 64)

    x = torch.randn(1, 32, 256, 144)
    output = downsample(x)

    print("Downsampling input:", x.shape)
    print("Downsampling output:", output.shape)


    localization = LocalizationModule(128, 64)

    x = torch.randn(1, 128, 128, 72)
    output = localization(x)

    print("Localization input:", x.shape)
    print("Localization output:", output.shape)


    upsample = Upsample(128, 64)

    x = torch.randn(1, 128, 64, 36)
    output = upsample(x)

    print("Upsampling input:", x.shape)
    print("Upsampling output:", output.shape)

    #Adding skip connections: transfer feature maps from encoder to decoder
    encoder_features = torch.randn(1, 64, 128, 72)
    decoder_features = torch.randn(1, 64, 128, 72)

    combined = torch.cat(
        [encoder_features, decoder_features],
        dim=1
    )

    print("Skip connection output:", combined.shape)


    encoder = Encoder(in_channels=1)
    #tensor with random numbers
    x = torch.randn(1, 1, 256, 144)
    features = encoder(x)

    for i, feature in enumerate(features, start=1):
        print(f"Encoder level {i}:", feature.shape)

    
    decoder_block = DecoderBlock(256, 128, 128)

    bottleneck = torch.randn(1, 256, 16, 9)
    skip = torch.randn(1, 128, 32, 18)

    output = decoder_block(bottleneck, skip)

    print("Decoder input:", bottleneck.shape)
    print("Skip input:", skip.shape)
    print("Decoder output:", output.shape)'''

    #Testing decoder block

    encoder = Encoder(in_channels=1)
    decoder = Decoder()

    x = torch.randn(1, 1, 256, 144)

    features = encoder(x)
    decoder_outputs = decoder(features)

    for i, output in enumerate(decoder_outputs, start=1):
        print(f"Decoder output {i}: {output.shape}")


    #6-class output
    segmentation_head = SegmentationHead(
        in_channels=16,
        num_classes=6
    )

    final_features = decoder_outputs[0]
    logits = segmentation_head(final_features)

    print("Segmentation output:", logits.shape)


    deep_supervision = DeepSupervision(num_classes=6)

    d1, d2, d3, d4 = decoder_outputs

    final_logits = deep_supervision(d1, d2, d3)

    print("Deep supervision output:", final_logits.shape)

    assert final_logits.shape == (1, 6, 256, 144)
    print("Deep supervision shape test passed!")

    #Testing the whole model after assembling all the layers including deep supervision and localisation
    model = ImprovedUNet(in_channels=1, num_classes=6)

    x = torch.randn(1, 1, 256, 144)
    logits = model(x)

    print("Full model output:", logits.shape)

    assert logits.shape == (1, 6, 256, 144)

    predicted_mask = torch.argmax(logits, dim=1)
    print("Predicted mask:", predicted_mask.shape)

    assert predicted_mask.shape == (1, 256, 144)

    print("Full model shape tests passed!")


