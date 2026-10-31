"""
Contains the source code for the components of the model/s
"""
from typing import Literal

import torch
from torch import nn
from torchvision.ops import StochasticDepth
from dataset import CLASS_TO_IDX
DEFAULT_CONFIDENCE_THRESHOLD = 0.8

###### General prediction function with confidence thresholding ######
def predict(
    x: torch.Tensor,
    model: torch.nn.Module,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    mode: Literal["flag", "mask"] = "mask"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Predicts classes from the tensor x, returning a tensor of shape (batch, 2) with the predicted 
    class and confidence. For each sample, if the confidence is below the threshold, the predicted class 
    is set to -1, indicating the desire for human input

    Args:
        x (torch.Tensor): input tensor of shape (batch, 1, 256,240)
        model (torch.nn.Module): The trained model to use for prediction
        confidence_threshold (float, optional): Threshold for determining class predictions. 
        Defaults to DEFAULT_CONFIDENCE_THRESHOLD.
        mode (Literal["flag", "mask"], optional): The mode for handling uncertain predictions. mask will
        default low confidence predicitions to AD and flag will default low confidence predictions 
        to -1.
        Defaults to "mask".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Predicted classes and confidences.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        logits = model.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)

        if mode == "flag":
            # Apply confidence threshold
            # If the confidence is below the threshold, set the prediction to -1, indicating uncertainty
            # in the prediction and that human input may be required for this sample
            predictions[confidences < confidence_threshold] = -1
            return predictions, confidences

        # Low confidence predictions will be defaulted to AD
        predictions[confidences < confidence_threshold] = CLASS_TO_IDX["AD"]
        return predictions, confidences

###### BASELINE CNN ######
class ConvBlock(nn.Module):
    """Two convolution layers with batch normalisation and relu activation function, followed by
    a final max pool"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """A CNN baseline for AD vs NC classification.
    Uses four ConvBlocks, then average pooling and linear layers,
    somewhat mimicking the ConvNeXt architecture.

    Input: (batch, 1, H, W)
    Output: (batch, num_classes) in raw logits
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def predict(self,
        x: torch.Tensor,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        mode: Literal["flag", "mask"] = "mask"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calls predict() with the baseline CNN model
        """
        return predict(x, self, confidence_threshold, mode)

###### ConvNeXt ######
class ConvNeXtLayerNorm(nn.Module):
    """
    Customised layer norm implementation. This is necessary since nn.LayerNorm()
    uses BHWC, but the convolutions expect BCHW. We permute the tensor, compute
    the layer norm and then permute the tensor back
    """
    def __init__(self, in_channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor):
        # permute to have channels last (B, C, H, W) -> (B, H, W, C)
        x_perm = x.permute(0, 2, 3, 1)
        x_normed = self.layer_norm(x_perm) # norms last dimension

        # (B, H, W, C) -> (B, C, H, W)
        x_out = x_normed.permute(0, 3, 1, 2)
        return x_out


class ConvNeXtBlock(nn.Module):
    """
    A single block of the ConvNeXt algorithm
    references: 
        - https://www.geeksforgeeks.org/computer-vision/ConvNeXt/
        - https://www.researchgate.net/figure/Structure-chart-of-ConvNeXt-Block_fig4_363389821
    """
    def __init__(self,
        in_channels: int,
        layer_scale_init: float = 1e-6,
        p_drop: float = 0.0
    ):
        super().__init__()
        self.layer_scale_init = layer_scale_init

        if p_drop < 0.0 or p_drop >= 1.0:
            raise ValueError("p_drop must be at least 0 and less than 1")

        self.block = nn.Sequential(
            nn.Conv2d( # depthwise convolution
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=in_channels # each channel computed independently
            ),
            ConvNeXtLayerNorm(in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=4*in_channels,
                kernel_size=1
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=4*in_channels,
                out_channels=in_channels,
                kernel_size=1
            )
        )

        # layer scale (one learnable scaling parameter per channel)
        self.gamma = nn.Parameter(
                self.layer_scale_init * 
                torch.ones(in_channels)
        )

        # drop path
        if p_drop > 0:
            self.drop_path = StochasticDepth(p_drop, mode="row")
        else:
            self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.block(x)
        x = x * self.gamma.view(1, -1, 1, 1)
        x = self.drop_path(x)
        return residual + x


class ConvNeXtDownsample(nn.Module):
    """
    ConvNeXt Downsample block
    references: 
        - https://www.geeksforgeeks.org/computer-vision/ConvNeXt/
    """
    def __init__(self, in_channels, channel_scale):
        super().__init__()
        self.downsample = nn.Sequential(
            ConvNeXtLayerNorm(in_channels),
            nn.Conv2d( # halves height and width
                in_channels=in_channels,
                out_channels=channel_scale*in_channels,
                kernel_size=2,
                stride=2
            )
        )

    def forward(self, x):
        return self.downsample(x)


class ConvNeXt(nn.Module):
    """
    Modified ConvNeXt algorithm with a (batch x 1 x 240 x 256) input size.
    The implementation attempts to mirror the downsampling scales used by the 
    original ConvNeXt model.
    Consists of:
        Stem:
            (1,240,256) -> (96,60,64) 4x4 Convolution stride 4 then LayerNorm
        1 ConvNeXt Block:
            (96,60,64) -> (96,60,64)
        Downsample:
            (96,60,64) -> (192,30,32) LayerNorm + 2x2 Convolution stride 2
        1 ConvNeXt Block:
            (192,30,32) -> (192,30,32)
        Downsample:
            (192,30,32) -> (384,15,16)
        3 ConvNeXt Blocks:
            (384,15,16) -> (384,15,16)
        Downsample:
            (384,15,16) -> (768,7,8)
        1 ConvNeXt Block:
            (768,7,8) -> (768,7,8)
        Global Average Pooling:
            (768,7,8) -> (768,1,1)
        Flatten + Linear:
            (768) -> (2)
    
    ConvNeXt blocks perform depthwise convolutions (convolutions independent 
    of channels) and do not change the input size.
    Downsampling reduces the input sizes by ~1/2 and increases the channel size
    by 2.
    Global average pooling averages per channel and reduces each to size 1.
    Final linear layer reduces the channel size to the number of classes (2).
    Each ConvNeXt block increases the drop path rate linearly from the next up to the maximum p_drop_max.
    References:
        https://www.geeksforgeeks.org/computer-vision/ConvNeXt/
    """
    in_channels = 1
    stem_out_channels = 96
    channel_scale = 2
    block_nums = (1, 1, 3, 1)


    def __init__(self,
        num_classes: int = 2,
        initial_layer_scale: float = 1e-6,
        p_drop_max: float = 0.0
    ):
        super().__init__()

        drop_path_rates = torch.linspace(
            0.0, p_drop_max, steps=len(self.block_nums)
        ).tolist()

        self.stem = nn.Sequential(
            nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.stem_out_channels,
            kernel_size=4,
            stride=4
            ),
            ConvNeXtLayerNorm(self.stem_out_channels)
        )

        # 3 ConvNeXt blocks + downsample
        self.stage1 = nn.Sequential(
            *[
                ConvNeXtBlock(
                    in_channels=self.stem_out_channels, 
                    p_drop=drop_path_rates[0], 
                    layer_scale_init=initial_layer_scale
                ) for _ in range(self.block_nums[0])
            ],
            ConvNeXtDownsample(
                in_channels=self.stem_out_channels,
                channel_scale=self.channel_scale
            )
        )

        # 3 ConvNeXt blocks (w/ 2x channels) + downsample
        stage2_channels = self.channel_scale * self.stem_out_channels
        self.stage2 = nn.Sequential(
            *[
                ConvNeXtBlock(
                    in_channels=stage2_channels,
                    p_drop=drop_path_rates[1], 
                    layer_scale_init=initial_layer_scale
                ) for _ in range(self.block_nums[1])
            ],
            ConvNeXtDownsample(
                in_channels=stage2_channels,
                channel_scale=self.channel_scale
            )
        )

        # 9 ConvNeXt blocks (w/ 2x channels) + downsample
        stage3_channels = self.channel_scale**2 * self.stem_out_channels
        self.stage3 = nn.Sequential(
            *[
                ConvNeXtBlock(
                    in_channels=stage3_channels,
                    p_drop=drop_path_rates[2], 
                    layer_scale_init=initial_layer_scale
                ) for _ in range(self.block_nums[2])
            ],
            ConvNeXtDownsample(
                in_channels=stage3_channels,
                channel_scale=self.channel_scale
            )
        )

        # 3 ConvNeXt blocks but no downsample
        stage4_channels = self.channel_scale**3 * self.stem_out_channels
        self.stage4 = nn.Sequential(*[
            ConvNeXtBlock(
                in_channels=stage4_channels,
                p_drop=drop_path_rates[3], 
                layer_scale_init=initial_layer_scale
            ) for _ in range(self.block_nums[3])
        ])

        # global average pooling
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        # final linear layer
        self.linear = nn.Linear(
            in_features=stage4_channels,
            out_features=num_classes # two class output (matches baseline)
        )


    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1) # flatten C,H,W
        x = self.linear(x)
        return x

    def predict(self,
        x: torch.Tensor,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        mode: Literal["flag", "mask"] = "mask"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calls predict() with the ConvNeXt model
        """
        return predict(x, self, confidence_threshold, mode)
