# This code was taken from the 3710 blackboard page where example code for a CNN MNIST classifier was made
# It has been tweaked slightly to be used for binary classification
import time
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data_loader import train_loader, validation_loader


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_vram = torch.cuda.get_device_properties(device).total_memory
print(f"GPU: {torch.cuda.get_device_name(device)}")
print(f"Total VRAM: {total_vram / (1024**3):.2f} GiB")

if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU.")


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        dim,
        kernel_size,
        stride=1,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        dropout_p=0.3
    ):
        super().__init__()
        self.features = nn.Conv2d(
            in_dim, dim, kernel_size=kernel_size,
            stride=stride, bias=True
        )
        self.norm1 = norm_layer(dim)
        self.act_layer = act_layer()
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        x = self.features(x)
        x = self.norm1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        return x


class ConvNetwork(nn.Module):
    def __init__(self, dim, in_channels, num_classes=2, dropout_p=0.3):
        super().__init__()

        self.block1 = BasicBlock(
            in_channels, dim, 3, stride=2, dropout_p=dropout_p
        )
        self.block2 = BasicBlock(
            dim, dim * 2, 3, stride=2, dropout_p=dropout_p
        )

        self.ave_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.ave_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


channels = 1 # channels 1 since the ADNI brain scans are black and white
depth = 64
model = ConvNetwork(depth, channels, num_classes=2).to(device) # this is what our train.py file will import when training basic cnn
