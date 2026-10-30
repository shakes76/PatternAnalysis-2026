# This code was taken from Francesco Saverio Zuppichini, as the building blocks for the convnext architecture
# reference to this can be found in the readme
# 
import torch
from torch import nn
from torch import Tensor
from typing import List
from torchvision.ops import StochasticDepth

class LayerScaler(nn.Module):
    """
    Channel-wise scaling applied to the output of a block
    """
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)), 
                                    requires_grad=True)
        
    def forward(self, x):
        return self.gamma[None,...,None,None] * x

class ConvNormAct(nn.Sequential):
    """
    A utility layer composed by (convvolution) -> (normalisation) -> (activation)
    This is a reusable building block for convolutional operations
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm = nn.BatchNorm2d,
        act = nn.ReLU,
        **kwargs
    ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs
            ),
            norm(out_features),
            act(),
        )

class BottleNeckBlock(nn.Module):
    """
    Inverted bottle neck - main ConvNeXt building block
    Here expansion occurs before depthwise convolution
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 4,
        drop_p: float = .0,
        layer_scaler_init_value: float = 1e-6,
    ):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv2d(
                in_features, in_features, kernel_size=7, padding=3, bias=False, groups=in_features
            ),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            # wide -> wide 
            nn.Conv2d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv2d(expanded_features, out_features, kernel_size=1),
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
        self.drop_path = StochasticDepth(drop_p, mode="batch")

        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x = self.layer_scaler(x)
        x = self.drop_path(x)
        x += res
        return x




class ConvNexStage(nn.Sequential):
    """
    Represents one of the 4 stages that builds up a ConvNeXt model
    Consists of downsampling followed by multiple bottle neck blocks
    Each stage helps to process at progresively lower resolutions
    """
    def __init__(
        self, in_features: int, out_features: int, depth: int, **kwargs
    ):
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_features),
                nn.Conv2d(in_features, out_features, kernel_size=2, stride=2)
            ),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth)
            ],
        )

class ConvNextStem(nn.Sequential):
    """
    Input layer to the model - here we apply a 4x4 convolution with a stride of 4 to mimick the patch embeddings of ViTs
    This is the entry point to ConvNeXt
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=4),
            nn.BatchNorm2d(out_features)
        )

class ConvNextEncoder(nn.Module):
    """
    Encapsulates 1 stem followed by 4 stages
    """
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
        drop_p: float = .0,
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)

        in_out_widths = list(zip(widths, widths[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))] 
        
        self.stages = nn.ModuleList(
            [
                ConvNexStage(stem_features, widths[0], depths[0], drop_p=drop_probs[0]),
                *[
                    ConvNexStage(in_features, out_features, depth, drop_p=drop_p)
                    for (in_features, out_features), depth, drop_p in zip(
                        in_out_widths, depths[1:], drop_probs[1:]
                    )
                ],
            ]
        )
        

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

class ClassificationHead(nn.Sequential):
    """
    Final layer of the model - converts the feature map into a classifier based off the number of channels provided
    Flattens and applies a linear layer to perform classification
    """
    def __init__(self, num_channels: int, num_classes: int = 1000):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_classes)
        )
    
    
class ConvNextForImageClassification(nn.Sequential):
    """
    Overarching class to encapsulate the entire ConvNeXt model
    Contains an encoder followed by a classification head
    """
    def __init__(self,  
                 in_channels: int,
                 stem_features: int,
                 depths: List[int],
                 widths: List[int],
                 drop_p: float = .0,
                 num_classes: int = 1000):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels, stem_features, depths, widths, drop_p)
        self.head = ClassificationHead(widths[-1], num_classes)

# Initialisation of our ConvNeXt model
# widths indicated the dimensions of each stage, depths indicates how many convolutional layers and in each stage
classifier = ConvNextForImageClassification(in_channels=1, stem_features=24, depths=[1,1,2,1], widths=[24, 48, 96, 192], num_classes=2)
