import torch
import torch.nn as nn

def drop_path(x, p, training):
    if p == 0.0 or not training:
        return x
    keep = 1 - p
    mask = x.new_empty(x.shape[0], 1, 1, 1).bernoulli_(keep)
    return x * mask / keep

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, downsample: nn.Module=None, drop_path_p=0.0):
        super().__init__()
        expansion_factor = 4

        self.conv1 = nn.Conv2d(inplanes, outplanes, 1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(outplanes, outplanes, 3, stride, padding=1)
        self.bn2 = nn.BatchNorm2d(outplanes)

        self.conv3 = nn.Conv2d(outplanes, outplanes*expansion_factor, 1)
        self.bn3 = nn.BatchNorm2d(outplanes*expansion_factor)
        self.downsample = downsample
        self.drop_path_p = drop_path_p
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        # Drop path for stochastic depth regularisation before skip connection
        out = drop_path(out, self.drop_path_p, self.training)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, drop_path_rate=0.0):
        super().__init__()
        self.expansion_factor = 4
        self.inplanes = 64

        self._dp = iter(torch.linspace(0, drop_path_rate, 16).tolist())
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self.make_layer(64, 3, 1)
        self.layer2 = self.make_layer(128, 4, 2)
        self.layer3 = self.make_layer(256, 6, 2)
        self.layer4 = self.make_layer(512, 3, 2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(512 * self.expansion_factor, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    def make_layer(self, planes, num_blocks, stride):
        
        outplanes = planes*self.expansion_factor
     
        downsample = None

        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, outplanes, 1, stride), nn.BatchNorm2d(outplanes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample, drop_path_p=next(self._dp))]
        self.inplanes = outplanes
        for i in range(1, num_blocks):
            layers.append(BasicBlock(self.inplanes, planes, drop_path_p=next(self._dp)))
        return nn.Sequential(*layers)

"""
-----------------------------------------------------------------------------------------------------------------
Below is the adjusted model for ConvNeXt, built off of the above model
-----------------------------------------------------------------------------------------------------------------
"""
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels for (B, C, H, W) tensors."""
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class BasicBlockConv(nn.Module):
    def __init__(self, planes, drop_path_p=0.0, layer_scale=1e-6):
        super().__init__()
        self.dw = nn.Conv2d(planes, planes, 7, padding=3, groups=planes)
        self.norm = LayerNorm2d(planes, eps=1e-6)
        
        self.pw1 = nn.Conv2d(planes, 4 * planes, 1)
       
        
        self.pw2 = nn.Conv2d(4 * planes, planes, 1)
        self.gamma = nn.Parameter(layer_scale * torch.ones(planes))
        self.drop_path_p = drop_path_p

        self.gelu = nn.GELU()
    
    def forward(self, x):
        identity = x

        out = self.dw(x)
        out = self.norm(out)

        out = self.pw1(out)
        out = self.gelu(out)

        out = self.pw2(out)
        out = out * self.gamma[:, None, None]
        # Drop path for stochastic depth regularisation before skip connection
        out = drop_path(out, self.drop_path_p, self.training)
        out += identity
        return out

class ConvNeXt(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, drop_path_rate=0.0):
        super().__init__()
        self.inplanes = 96
        self._dp = iter(torch.linspace(0, drop_path_rate, 18).tolist())

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 96, 4, stride=4),
            LayerNorm2d(96, eps=1e-6),
        )

        self.layer1 = self.make_layer(96, 96, 3, downsample=False)
        self.layer2 = self.make_layer(96, 192, 3)
        self.layer3 = self.make_layer(192, 384, 9)
        self.layer4 = self.make_layer(384, 768, 3)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.fc = nn.Linear(768, num_classes)

        self.apply(self.init_weights)

    def forward(self, x):
        out = self.stem(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.pool(out)
        out = self.flatten(out)
        out = self.norm(out)
        out = self.fc(out)
        return out

    def make_layer(self, inplanes, outplanes, num_blocks, downsample=True):
        layers = []
        if downsample:
            layers += [LayerNorm2d(inplanes, eps=1e-6),
                       nn.Conv2d(inplanes, outplanes, kernel_size=2, stride=2)]
        layers += [BasicBlockConv(outplanes, next(self._dp)) for _ in range(num_blocks)]
        return nn.Sequential(*layers)

    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)