"""
Author: Lawrence Tiang
Student ID: s4892210

Description:
    Defines the neural network architectures used for binary
    classification of ADNI brain MRI images. Implements a baseline
    convolutional neural network (CNN) and a ConvNeXt-Tiny
    model for comparison. Also provides reusable components
    for convolutional blocks, channel-wise Layer Normalisation,
    residual connections, stochastic depth and model construction.

Usage:
    The model functions are imported into train.py and predict.py
    to initialise the selected model for training and evaluation.
"""

import torch
import torch.nn as nn
import torchvision

class LayerNorm2d(nn.Module):
    """
    Apply Layer Normalisation to a four-dimensional image tensor.

    Args:
        num_channels (int): Number of input channels.
        eps (float): Small value added for numerical stability.

    Returns:
        torch.Tensor: Normalised tensor with the original shape (N, C, H, W).
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)   # (N, C, H, W) -> (N, H, W, C)
        x = self.layer_norm(x)      # Apply LayerNorm across channel dimension
        x = x.permute(0, 3, 1, 2)   # (N, H, W, C) -> (N, C, H, W)

        return x


class ConvBNReLU(nn.Module):
    """
    Define a convolutional block containing convolution,
    batch normalisation and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Step size of the convolution.
        padding (int): Padding added around the input.

    Returns:
        torch.Tensor: Feature maps produced by the convolutional block.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # Combine convolution, normalisation and activation into one block
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Pass the input through the complete convolutional block
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    Define a baseline CNN for binary classification of ADNI MRI images.

    Args:
        num_classes (int): Number of output classes. Defaults to 2.
        dropout (float): Dropout probability used for regularisation.

    Returns:
        torch.Tensor: Classification logits for the AD and NC classes.
    """

    def __init__(self, num_classes=2, dropout=0.35):
        super().__init__()

        # Extract increasingly complex features of input image.
        self.features = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),      # 224 -> 112 spatial dimensions
            ConvBNReLU(32, 64, stride=2),     # 112 -> 56  spatial dimensions
            ConvBNReLU(64, 128, stride=2),    # 56 -> 28   spatial dimensions
            ConvBNReLU(128, 256, stride=2),   # 28 -> 14   spatial dimensions
            ConvBNReLU(256, 256, stride=1)    # Further feature extraction
        )

        # Reduce each feature map to a single spatial value
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Flatten the features, apply dropout and predict class logits
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Initialise the model parameters before training
        self._initialize_weights()

    def _initialize_weights(self):
        # Apply an appropriate initialisation method to each layer type
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming initialisation for ReLU activations
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                # Start batch normalisation with neutral scaling and offset
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                # Initialise the classifier with small weights
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Extract image features through the convolutional layers
        x = self.features(x)
        # Aggregate spatial features into a single vector per image
        x = self.pool(x)
        # Produce the final classification logits
        x = self.classifier(x)

        return x


class ConvNeXtBlock(nn.Module):
    """
    Define a ConvNeXt-Tiny residual block for feature extraction.
    The block uses depthwise convolution, Layer Normalisation,
    pointwise convolutions, GELU activation, optional layer scaling
    and optional stochastic depth.

    Args:
        dim (int): Number of input and output channels.
        drop_path (float): Probability used by stochastic depth.
        layer_scale_initial_value (float): Initial value of the
            learnable layer-scaling parameter.

    Returns:
        torch.Tensor: Feature maps after residual processing.
    """

    def __init__(self, dim, drop_path, layer_scale_initial_value):
        super().__init__()

        # Apply a separate spatial convolution to each channel
        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            padding=3,
            groups=dim
        )

        # Normalise the feature channels
        self.norm = LayerNorm2d(num_channels=dim, eps=1e-6)

        # Expand the channel dimension to learn richer representations
        self.pwconv1 = nn.Conv2d(
            in_channels=dim,
            out_channels=4 * dim,
            kernel_size=1
        )

        # Introduce non-linearity after channel expansion
        self.activation = nn.GELU()

        # Project the expanded features back to the original dimension
        self.pwconv2 = nn.Conv2d(
            in_channels=4 * dim,
            out_channels=dim,
            kernel_size=1
        )

        # Optionally scale the block output using a learnable parameter
        if layer_scale_initial_value > 0:
            self.gamma = nn.Parameter(
                data=layer_scale_initial_value * torch.ones(
                    (dim),
                    requires_grad=True
                )
            )
        else:
            self.gamma = None

        # Optionally apply stochastic depth during training
        if drop_path > 0:
            self.stochastic_depth = torchvision.ops.StochasticDepth(
                p=drop_path,
                mode="batch"
            )
        else:
            self.stochastic_depth = nn.Identity()

    def forward(self, x):
        # Preserve the input for the residual connection
        residual = x
        # Extract spatial features and normalise the channels
        x = self.dwconv(x)
        x = self.norm(x)
        # Expand, activate, and project the channel features
        x = self.pwconv1(x)
        x = self.activation(x)
        x = self.pwconv2(x)

        # Apply learnable channel-wise scaling when enabled
        if self.gamma is not None:
            x = self.gamma[:, None, None] * x

        # Apply stochastic depth and add the original input
        x = residual + self.stochastic_depth(x)
        return x

class ConvNextStage(nn.Module):
    """
    Define a stage consisting of multiple ConvNeXt blocks.

    Args:
        depth (int): Number of blocks in the stage.
        dim (int): Number of channels processed by each block.
        drop_path_rates (list[float]): Stochastic depth probability
            assigned to each block.
        layer_scale_initial_value (float): Initial layer-scaling value.

    Returns:
        torch.Tensor: Feature maps after all blocks in the stage.
    """
    def __init__(self, depth, dim, drop_path_rates, layer_scale_initial_value):
        super().__init__()

        # Create the specified number of ConvNeXt blocks
        blocks = []
        for i in range(depth):
            block = ConvNeXtBlock(
                dim=dim,
                drop_path=drop_path_rates[i],
                layer_scale_initial_value=layer_scale_initial_value
            )
            blocks.append(block)

        # Combine the blocks into a sequential processing stage
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ConvNeXtClassifier(nn.Module):
    """
    Define a ConvNeXt-Tiny hierarchical image classifier.
    The model uses multiple ConvNeXt stages, global average
    pooling and a linear classification head.

    Args:
        input_channels (int): Number of input image channels.
        num_classes (int): Number of output classes.
        depths (list[int]): Number of blocks in each stage.
        dims (list[int]): Channel dimensions for the stages.
        drop_path_rate (float): Maximum stochastic depth probability.
        layer_scale_initial_value (float): Initial layer-scaling value.

    Returns:
        torch.Tensor: Classification logits for each input image.
    """
    def __init__(
        self,
        input_channels=3,
        num_classes=2,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1,
        layer_scale_initial_value=1e-6
    ):
        super().__init__()

        self.num_stages = len(depths)
        self.downsampling_layers = nn.ModuleList()

        # Patchify stem: convert the input into non-overlapping 4x4 patches
        # (224 x 224 -> 56 x 56 spatial dimensions)
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=dims[0],
                kernel_size=4,
                stride=4,
            ),
            LayerNorm2d(num_channels=dims[0], eps=1e-6)
        )
        self.downsampling_layers.append(stem)

        # Create downsampling layers between consecutive stages
        # (56 x 56 -> 28 x 28 -> 14 x 14 -> 7 x 7 spatial dimensions)
        for i in range(len(dims) - 1):
            downsample = nn.Sequential(
                LayerNorm2d(num_channels=dims[i], eps=1e-6),
                nn.Conv2d(
                    in_channels=dims[i],
                    out_channels=dims[i + 1],
                    kernel_size=2,
                    stride=2
                )
            )

            self.downsampling_layers.append(downsample)

        # Gradually increases stochastic depth probability across all blocks
        drop_path_rates = [
            tensor.item() for tensor in torch.linspace(
                start=0.0,
                end=drop_path_rate,
                steps=sum(depths)
            )
        ]

        # Construct each stage using its configured depth and channel size
        self.stages = nn.ModuleList()
        rate_index = 0
        for i in range(self.num_stages):
            stage_rates = drop_path_rates[rate_index: rate_index + depths[i]]
            self.stages.append(
                ConvNextStage(
                    depth=depths[i],
                    dim=dims[i],
                    drop_path_rates=stage_rates,
                    layer_scale_initial_value=layer_scale_initial_value
                )
            )

            # Move to the next stage's section of drop path rates
            rate_index += depths[i]

        # Aggregate spatial features and classify each image
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=dims[-1], out_features=num_classes)
        )

    def forward(self, x):
        # Apply each downsampling layer followed by its corresponding stage
        for i in range(self.num_stages):
            x = self.downsampling_layers[i](x)
            x = self.stages[i](x)

        # Reduce each feature map to a single value per channel
        x = self.avgpool(x)
        # Convert the extracted features into classification logits
        x = self.classifier(x)
        return x


def build_model(model_name, num_classes=2):
    """
    Create a model using the requested architecture name.

    Args:
        model_name (str): Model architecture to create.
            Supported values are "baseline" and "convnext".
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: An initialised instance of the selected model.

    Raises:
        ValueError: If model_name is not a supported architecture.
    """
    # Construct the baseline CNN when requested
    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)

    # Construct the ConvNeXt-Tiny classifier when requested
    if model_name == "convnext":
        return ConvNeXtClassifier(input_channels=3, num_classes=num_classes)

    # Reject unsupported model names
    raise ValueError(f"Unknown model: {model_name}")