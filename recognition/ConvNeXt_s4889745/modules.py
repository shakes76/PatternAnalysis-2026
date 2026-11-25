import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Convnext bottlneck block
'''
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # Add the original input (passed through the shortcut path) to the output
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial stem layers: typical for standard ImageNet ResNet
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The 4 ResNet Stages (Layers)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def ResNet50(num_classes=2):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

class LayerNorm(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super().__init__()

        self.norm = nn.LayerNorm(
            channels,
            eps=eps
        )

    def forward(self, x):

        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)

        # [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

        return x

'''
Stochastic depth

Used to regularise during training
'''
class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):

        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = ( keep_prob + torch.rand( shape, dtype=x.dtype, device=x.device))

        random_tensor.floor_()

        return (x / keep_prob) * random_tensor

class ConvNeXtBlock(nn.Module):

    def __init__(self, dim, drop_path=0.0):
        super().__init__()

        self.drop_path = DropPath(drop_path)

        # 7x7 depthwise convolution
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # One LayerNorm per block
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Inverted bottleneck: C -> 4C
        self.expand = nn.Linear(dim, 4 * dim)

        # Single activation used in the block
        self.gelu = nn.GELU()

        # 4C -> C
        self.reduce = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):

        residual = x

        # Spatial mixing
        x = self.depthwise_conv(x)

        # [B,C,H,W] -> [B,H,W,C]
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)

        # Channel mixing
        x = self.expand(x)

        x = self.gelu(x)

        x = self.reduce(x)

        x = self.gamma * x

        # [B,H,W,C] -> [B,C,H,W]
        x = x.permute(0, 3, 1, 2)

        # Residual connection
        x = self.drop_path(x) + residual

        return x

class ConvNeXt(nn.Module):

    def __init__(self, in_channels=1, num_classes=2, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), drop_path_rate=0.1):
        super().__init__()
        # Patchify stem:
        # 4x4 non-overlapping convolution, stride 4
        self.stem = nn.Sequential(nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4), LayerNorm(dims[0]))

        total_blocks = sum(depths)

        drop_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        # Four ConvNeXt stages
        self.stages = nn.ModuleList()

        block_index = 0

        for stage_index in range(4):

            blocks = []

            for _ in range(depths[stage_index]):
                blocks.append(ConvNeXtBlock(dims[stage_index], drop_path=drop_rates[block_index]))

                block_index += 1

            self.stages.append(nn.Sequential(*blocks))

        # Separate downsampling layers between stages
        self.downsample_layers = nn.ModuleList()

        for i in range(3):

            self.downsample_layers.append(
                nn.Sequential(LayerNorm(dims[i]), nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)))

        # Final LayerNorm after global average pooling
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        # AD vs NC
        self.dropout = nn.Dropout(0.4)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):

        # Stem
        x = self.stem(x)

        # Stage 1
        x = self.stages[0](x)

        # Stage 2
        x = self.downsample_layers[0](x)
        x = self.stages[1](x)

        # Stage 3
        x = self.downsample_layers[1](x)
        x = self.stages[2](x)

        # Stage 4
        x = self.downsample_layers[2](x)
        x = self.stages[3](x)

        # Global average pooling
        x = x.mean(dim=[2, 3])

        # Final normalisation
        x = self.norm(x)

        # Classification head
        x = self.head(self.dropout(x))

        return x

'''
forward pass and output shape
model = ResNet50()

x = torch.randn(4, 3, 224, 224)

output = model(x)

print("Input:", x.shape)
print("Output:", output.shape)

total_params = sum(
    p.numel() for p in model.parameters()
)

print("Total parameters:", total_params)

model = ConvNeXt()

model.eval()

x = torch.randn(4, 3, 224, 224)

output = model(x)

print("Input:", x.shape)
print("Output:", output.shape)

total_params = sum(p.numel() for p in model.parameters())

print("Total parameters:", total_params)
'''
