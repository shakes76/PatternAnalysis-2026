"""Small from-scratch grayscale CNN baseline; ConvNeXt is a later stage."""
from torch import nn


class SimpleCNN(nn.Module):
    """Produce two raw logits (NC=0, AD=1) from [B,1,H,W] MRI slices.

    Three convolution/ReLU/pooling blocks reduce spatial size; global average
    pooling avoids a large flattened classifier. No pretrained weights.
    """
    def __init__(self):
        super().__init__()
        blocks=[]
        for inputs, outputs in [(1,16),(16,32),(32,64)]:
            blocks.extend([nn.Conv2d(inputs,outputs,3,padding=1),
                           nn.ReLU(),nn.MaxPool2d(2)])
        self.features=nn.Sequential(*blocks,nn.AdaptiveAvgPool2d(1))
        self.classifier=nn.Linear(64,2)

    def forward(self, images):
        return self.classifier(self.features(images).flatten(1))
