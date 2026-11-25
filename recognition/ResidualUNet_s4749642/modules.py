import torch
import torch.nn as nn

NUM_CLASSES = 6
BASE = 16

def conv_block(in_channels, out_channels, dropout=0):
    """Two consecutive 3x3 convolutions, followed by ReLU."""
    layers = [
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(),
    ]

    # Apply dropout only where requested by the architecture.
    if dropout:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)

class UNet(nn.Module):
    # Architecture adapted from GeeksforGeeks:
    # https://www.geeksforgeeks.org/deep-learning/image-segmentation-using-u-net/
    # Adaptations: PyTorch, single-channel MRI input, 16 starting filters, and raw output logits for CrossEntropyLoss.

    def __init__(self, num_classes=NUM_CLASSES, base=BASE):
        super().__init__()

        # Feature channels at each depth: [16, 32, 64, 128, 256].
        widths = [base * 2**i for i in range(5)]

        # ModuleList registers the blocks so PyTorch can train their weights.
        self.encoders = nn.ModuleList()
        in_channels = 1  # Grayscale MRI has one intensity channel.

        for i, width in enumerate(widths):
            self.encoders.append(
                conv_block(
                    in_channels,
                    width,
                    # Apply dropout in the two deepest blocks.
                    dropout=0.3 if i >= 3 else 0,
                )
            )

            # The next block receives this block's output channels.
            in_channels = width

        # Halve height and width by taking the maximum in each 2×2 region.
        self.pool = nn.MaxPool2d(2)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Build the decoder from deepest to shallowest level.
        # i takes the values 4, 3, 2, 1.
        for i in range(4, 0, -1):
            self.upsamples.append(
                nn.ConvTranspose2d(
                    widths[i],
                    widths[i - 1],
                    kernel_size=3,
                    # Double height and width.
                    stride=2,         
                    padding=1,
                    # Resolve output size to exactly double.
                    output_padding=1,
                )
            )

            # Upsampled features and matching encoder features are joined.
            # Each has widths[i - 1] channels, giving twice that many inputs.
            self.decoders.append(
                conv_block(2 * widths[i - 1], widths[i - 1])
            )

        # Convert the final 16 features into six class scores per pixel.
        # A 1×1 convolution combines channels without changing image size.
        self.output = nn.Conv2d(base, num_classes, kernel_size=1)

        # Initialise feature-extraction Conv2d weights for ReLU activations.
        # These are starting weights, not pretrained weights.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) and layer is not self.output:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Four halvings require dimensions divisible by 16
        # so the encoder and decoder feature maps match.
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("Input height and width must be divisible by 16")

        skips = []

        # Process the first four encoder blocks.
        # Save their outputs before pooling, preserving finer spatial detail.
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        # Process the deepest block without further downsampling.
        x = self.encoders[-1](x)

        # Match each decoder stage with the corresponding encoder output,
        # starting with the deepest saved features.
        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips)
        ):
            x = upsample(x)

            # Join feature maps along dimension 1: the channel dimension.
            # This concatenates features; it does not add their values.
            x = decoder(torch.cat([x, skip], dim=1))

        # Returns raw scores (logits)
        return self.output(x)

class ResidualBlock(nn.Module):
    """Baseline convolution block with an added residual shortcut."""
    # Quick residual test, AI generated.

    def __init__(self, in_channels, out_channels, dropout=0):
        super().__init__()

        # Keep the baseline's convolution and dropout arrangement.
        self.layers = conv_block(in_channels, out_channels, dropout)

        # Match channel counts before adding the two paths.
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

        # Initialise the new shortcut consistently with the feature layers.
        if isinstance(self.shortcut, nn.Conv2d):
            nn.init.kaiming_normal_(
                self.shortcut.weight, nonlinearity="relu"
            )
            nn.init.zeros_(self.shortcut.bias)

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)


class ResidualUNet(UNet):
    def __init__(self, num_classes=NUM_CLASSES, base=BASE):
        super().__init__(num_classes=num_classes, base=base)

        widths = [base * 2**i for i in range(5)]
        residual_encoders = nn.ModuleList()

        # Wrap the existing encoder blocks, preserving their initial weights.
        for i, width in enumerate(widths):
            in_channels = 1 if i == 0 else widths[i - 1]

            block = ResidualBlock(
                in_channels,
                width,
                dropout=0.3 if i >= 3 else 0,
            )
            block.layers = self.encoders[i]
            residual_encoders.append(block)

        self.encoders = residual_encoders

        # Inherit the baseline's forward method and unchanged decoder.

class ResidualContextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = (nn.Identity() if in_channels == out_channels
                         else nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)


def decoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(),
    )

class IsenseeUNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, base=BASE):
        super().__init__()
        # Prepare for the successive layers with increasing / doubling of image
        # resolution
        widths = [base * 2**i for i in range(5)]
        # Prepare the encoder layer list
        self.encoders = nn.ModuleList()
        in_channels = 1
        # Add the successive ResidualContextBlock with increasing image resolution to the encoder
        # layers
        for i, width in enumerate(widths):
            self.encoders.append(ResidualContextBlock(
                in_channels, width, dropout=0.3 if i >= 3 else 0))
            in_channels = width
        self.pool = nn.MaxPool2d(2)
        self.upsamples = nn.ModuleList()
        # Prepare the decoder layer list
        self.decoders = nn.ModuleList()
        # Add the successive ResidualContextBlock with increasing image resolution to the encoder
        # layers
        for i in range(4, 0, -1):
            self.upsamples.append(nn.ConvTranspose2d(
                widths[i], widths[i-1], 3, stride=2, padding=1, output_padding=1))
            self.decoders.append(decoder_block(2 * widths[i-1], widths[i-1]))
        self.output = nn.Conv2d(base, num_classes, 1)
        # Initialise encoder convolutions for leaky ReLU and decoder ones for ReLU.
        for block in self.encoders:
            for layer in block.modules():
                # fills an input tensor with values sampled from normal distribution
                # using the Kaiming (He) initialization method. This technique is 
                # specifically optimized to preserve variance across deep networks using 
                # rectifying activation functions like ReLU or Leaky ReLU.
                # Also, initialise the bias to zero
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, a=0.01, nonlinearity="leaky_relu")
                    nn.init.zeros_(layer.bias)
        for block in self.decoders:
            for layer in block.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("Input height and width must be divisible by 16")
        skips = []
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.encoders[-1](x)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = decoder(torch.cat([upsample(x), skip], dim=1))
        return self.output(x)  # Raw class scores; do not apply softmax before CE.

def get_model(model_name):
    if model_name == "baseline":
        return UNet(num_classes=NUM_CLASSES, base=BASE)
    elif model_name == "residual":
        return ResidualUNet(num_classes=NUM_CLASSES, base=BASE)
    elif model_name == "improved":
        return IsenseeUNet(num_classes=NUM_CLASSES, base=BASE)
    else:
        raise ValueError(f"Unknown model name: {model_name}")