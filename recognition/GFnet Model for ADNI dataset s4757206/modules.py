"""CNN baseline and GFNet-inspired models for ADNI slice classification.

Inputs:
    Grayscale tensors with shape [batch, 1, height, width].

Outputs:
    Two logits per image: NC = 0, AD = 1.

Architecture reference for global filtering:
    https://github.com/raoyongming/GFNet/blob/master/gfnet.py

The GFNet-inspired implementation uses separate residual connections
around the global filter and MLP branches.
"""

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    Three-stage CNN baseline for binary MRI classification (NC vs AD).

    Each convolutional stage extracts image features using a 3x3 convolution,
    applies ReLU activation, and reduces spatial resolution using max pooling.

    Global average pooling converts the final feature maps into a
    128-dimensional feature vector, which is passed to a linear classifier.
    """

    def __init__(self):
        super().__init__()

        # Feature extraction: three convolutional stages.
        # Input shape: (batch_size, 1, 256, 256)
        self.features = nn.Sequential(

            # Stage 1: 1 -> 32 channels; spatial size 256 -> 128.
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Stage 2: 32 -> 64 channels; spatial size 128 -> 64.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Stage 3: 64 -> 128 channels; spatial size 64 -> 32.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Reduce each 32x32 feature map to a single average value.
        # Output shape: (batch_size, 128, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Map the 128 extracted features to two class logits:
        # class 0 = NC (Normal Control), class 1 = AD (Alzheimer's Disease).
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        # Extract hierarchical spatial features from the input MRI slices.
        # Shape: (batch_size, 1, 256, 256) -> (batch_size, 128, 32, 32)
        x = self.features(x)

        # Apply global average pooling.
        # Shape: (batch_size, 128, 32, 32) -> (batch_size, 128, 1, 1)
        x = self.pool(x)

        # Flatten the pooled feature maps into feature vectors.
        # Shape: (batch_size, 128, 1, 1) -> (batch_size, 128)
        x = torch.flatten(x, 1)

        # Return raw logits for NC and AD.
        # Shape: (batch_size, 128) -> (batch_size, 2)
        # Softmax is not applied because CrossEntropyLoss expects raw logits.
        return self.classifier(x)


class GlobalFilter(nn.Module):
    """
    Learnable frequency-domain filter for channels-first feature maps.

    The input features are transformed into the frequency domain using
    a 2D real FFT. Learnable complex-valued weights are then applied
    through element-wise multiplication.

    An inverse FFT transforms the filtered features back into the
    spatial domain while preserving the original spatial dimensions.
    """

    def __init__(self, height, width, channels):
        super().__init__()

        # Store the expected spatial dimensions and number of channels.
        self.height = height
        self.width = width
        self.channels = channels

        # Define learnable frequency-domain weights for each channel.
        # The frequency width is width // 2 + 1 because rfft2 stores
        # only the non-redundant frequencies along the last dimension.
        #
        # Weight shape:
        # (channels, height, width // 2 + 1, 2)
        #
        # The final dimension stores the real and imaginary components.
        # Small random values are used to initialise the weights.
        self.complex_weight = nn.Parameter(
            torch.randn(
                channels,
                height,
                width // 2 + 1,
                2,
            ) * 0.02
        )

    def forward(self, x):
        # Input shape: (batch_size, channels, height, width)

        # Apply a 2D real FFT along the spatial dimensions.
        # This converts spatial features into frequency representations.
        #
        # Output shape:
        # (batch_size, channels, height, width // 2 + 1)
        x = torch.fft.rfft2(
            x,
            dim=(-2, -1),
            norm="ortho",
        )

        # Convert the stored real and imaginary components into
        # complex-valued frequency-domain weights.
        #
        # Weight shape:
        # (channels, height, width // 2 + 1)
        weight = torch.view_as_complex(self.complex_weight)

        # Apply learnable global filtering through element-wise
        # multiplication in the frequency domain.
        #
        # The weights are broadcast across the batch dimension.
        x = x * weight

        # Apply the inverse real FFT to return to the spatial domain.
        # The original height and width are explicitly specified.
        #
        # Output shape:
        # (batch_size, channels, height, width)
        return torch.fft.irfft2(
            x,
            s=(self.height, self.width),
            dim=(-2, -1),
            norm="ortho",
        )


class GFBlock(nn.Module):
    """
    Global filter block with an MLP and two residual connections.

    The block first applies layer normalization and frequency-domain
    filtering, followed by a residual connection.

    A second layer normalization and MLP are then applied, followed
    by another residual connection.

    Input and output shapes:
        (batch_size, height, width, channels)
    """

    def __init__(self, dim, grid_size, mlp_ratio=4, dropout=0.1):
        super().__init__()

        # Normalize the channel features before global filtering.
        self.norm1 = nn.LayerNorm(dim)

        # Apply learnable frequency-domain filtering.
        # The filter operates on a grid_size x grid_size feature map.
        self.filter = GlobalFilter(grid_size, grid_size, dim)

        # Normalize the channel features before the MLP.
        self.norm2 = nn.LayerNorm(dim)

        # Set the hidden dimension of the MLP.
        # By default, the hidden dimension is four times the input dimension.
        hidden_dim = dim * mlp_ratio

        # Channel-wise MLP for nonlinear feature transformation.
        # Linear layers expand and then restore the channel dimension.
        # GELU introduces nonlinearity, while dropout provides regularization.
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Input shape: (batch_size, height, width, channels)

        # Save the original input for the first residual connection.
        residual = x

        # Apply layer normalization before global filtering.
        x = self.norm1(x)

        # GlobalFilter expects channels-first tensors.
        # Convert from (batch, height, width, channels)
        # to (batch, channels, height, width).
        x = x.permute(0, 3, 1, 2).contiguous()

        # Apply learnable global filtering in the frequency domain.
        x = self.filter(x)

        # Convert the filtered features back to channels-last format.
        # Shape: (batch, channels, height, width)
        #     -> (batch, height, width, channels)
        x = x.permute(0, 2, 3, 1).contiguous()

        # First residual connection:
        # Add the original input to the globally filtered features.
        x = residual + x

        # Second residual connection:
        # Apply layer normalization and the MLP, then add its output
        # to the features from the first residual connection.
        x = x + self.mlp(self.norm2(x))

        # Return features with the same shape as the input.
        return x


class GFNet(nn.Module):
    """
    GFNet-inspired classifier for binary MRI classification (NC vs AD).

    The model divides each input image into non-overlapping patches,
    projects them into feature embeddings, and processes the features
    through a sequence of global filter blocks.

    Global average pooling and a linear classifier are used to
    produce the final class logits.

    Default hyperparameters match the final experiment:
        image_size = 256
        patch_size = 16
        dim = 128
        depth = 8
        dropout = 0.1

    Input shape:
        (batch_size, 1, 256, 256)

    Output shape:
        (batch_size, 2)
    """

    def __init__(
        self,
        image_size=256,
        patch_size=16,
        dim=128,
        depth=8,
        dropout=0.1,
    ):
        super().__init__()

        # Ensure the image can be divided into non-overlapping patches
        # without leaving any unused pixels.
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        # Calculate the number of patches along each spatial dimension.
        # For a 256x256 image with 16x16 patches:
        # grid_size = 256 // 16 = 16.
        grid_size = image_size // patch_size

        # Patch embedding:
        # Split the grayscale MRI image into non-overlapping patches
        # and project each patch into a dim-dimensional feature vector.
        #
        # Input shape:  (batch_size, 1, 256, 256)
        # Output shape: (batch_size, 128, 16, 16)
        self.patch_embed = nn.Conv2d(
            1,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Learnable positional embeddings for the patch grid.
        # These embeddings provide spatial position information.
        #
        # Shape: (1, grid_size, grid_size, dim)
        # Default: (1, 16, 16, 128)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, grid_size, grid_size, dim)
        )

        # Initialize positional embeddings with a truncated
        # normal distribution using a standard deviation of 0.02.
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Apply dropout after adding positional embeddings
        # to reduce overfitting during training.
        self.pos_drop = nn.Dropout(dropout)

        # Stack multiple global filter blocks.
        # Each block contains frequency-domain filtering,
        # an MLP, and two residual connections.
        #
        # The default depth is 8 blocks.
        # Feature shape remains (batch_size, 16, 16, 128).
        self.blocks = nn.Sequential(*[
            GFBlock(dim, grid_size, dropout=dropout)
            for _ in range(depth)
        ])

        # Normalize the output features after all global filter blocks.
        self.norm = nn.LayerNorm(dim)

        # Final linear classifier for binary classification.
        # Class 0 = NC (Normal Control)
        # Class 1 = AD (Alzheimer's Disease)
        self.classifier = nn.Linear(dim, 2)

    def forward(self, x):
        # Convert the input MRI images into patch embeddings.
        #
        # Shape: (batch_size, 1, 256, 256)
        #     -> (batch_size, 128, 16, 16)
        x = self.patch_embed(x)

        # Convert from channels-first to channels-last format
        # because GFBlock expects (batch, height, width, channels).
        #
        # Shape: (batch_size, 128, 16, 16)
        #     -> (batch_size, 16, 16, 128)
        x = x.permute(0, 2, 3, 1).contiguous()

        # Add learnable positional embeddings and apply dropout.
        # The positional embeddings are broadcast across the batch.
        x = self.pos_drop(x + self.pos_embed)

        # Process the patch features through the global filter blocks.
        # Each block preserves the spatial grid and channel dimensions.
        x = self.blocks(x)

        # Apply layer normalization to the final feature representations.
        x = self.norm(x)

        # Apply global average pooling across the spatial dimensions.
        # Average the 16x16 patch grid into one feature vector per image.
        #
        # Shape: (batch_size, 16, 16, 128)
        #     -> (batch_size, 128)
        x = x.mean(dim=(1, 2))

        # Produce two raw class logits for NC and AD.
        # Shape: (batch_size, 128) -> (batch_size, 2)
        #
        # Softmax is not applied because CrossEntropyLoss
        # expects raw logits during training.
        return self.classifier(x)