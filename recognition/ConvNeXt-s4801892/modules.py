# Contains source code of the components of your model. Each component must be implemented as a class or a function in PyTorch 
# CNN
import torch.nn as nn
import torch

# baseline model
class BasicCNN(nn.Module):
    """
    A standard baseline Convolutional Neural Network (CNN) consisting of 
    three sequential convolutional blocks followed by global pooling and a linear classifier.
    """
    def __init__(self, num_classes=2):
        """
        Initialises the BasicCNN architecture.
        
        Args:
            num_classes (int): Number of target output classes for classification.
        """
        super().__init__()

        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ConvNeXt components and architecture
class DropPath(nn.Module):
    """
    Implements Drop Paths (Stochastic Depth) per sample. 
    Applied randomly to the residual branch during training for regularization.
    """
    def __init__(self, drop_prob=0.0):
        """
        Initialises DropPath.
        
        Args:
            drop_prob (float): Probability of zeroing out paths.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        Applies stochastic depth drop path during training.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Processed tensor with dropped paths if training, else original.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # shape for broadcasting across batch dimension: (B, 1, 1, 1) for 4D tensors
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor.div_(keep_prob)
        return x * random_tensor


class ConvNeXtBlock(nn.Module):
    """
    A single ConvNeXt residual block featuring an inverted bottleneck design, 
    large kernel depthwise convolutions, LayerNorm equivalents, and GELU activations.
    """
    def __init__(self, channels, expansion=4, drop_path=0.0):
        """
        Initialises the ConvNeXt block.
        
        Args:
            channels (int): Number of input/output feature channels.
            expansion (int): Factor by which hidden channel dimensions expand in the bottleneck.
            drop_path (float): Stochastic depth drop probability.
        """
        super().__init__()

        self.block = nn.Sequential(
            # depthwise convolution
            nn.Conv2d(
                channels,
                channels,
                kernel_size=7,
                padding=3,
                groups=channels,
            ),

            # normalise channels
            nn.GroupNorm(1, channels),

            # channel expansion and projection
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels * expansion,
                channels,
                kernel_size=1,
            ),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Forward pass combining input with residual block transformation.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor with residual connection added.
        """
        return x + self.drop_path(self.block(x))


# ConvNeXt model
class ConvNeXt(nn.Module):
    """
    A ConvNeXt architecture tailored for 2D medical image classification 
    (such as MRI slices), utilizing hierarchical stages and inverted bottlenecks.
    """
    def __init__(self, num_classes=2, dropout_rate=0.5, drop_path_rate=0.1):
        """
        Initialises the ConvNeXt model.
        
        Args:
            num_classes (int): Number of target output classes.
            dropout_rate (float): Dropout probability for the classifier head.
            drop_path_rate (float): Base stochastic depth drop path rate across blocks.
        """
        super().__init__()

        self.stem = nn.Sequential(
            # input MRI image: 1 channel
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, 32),
        )

        self.features = nn.Sequential(
            # stage 1: 32 channels, 2 blocks
            ConvNeXtBlock(32, expansion=4, drop_path=drop_path_rate),
            ConvNeXtBlock(32, expansion=4, drop_path=drop_path_rate),

            # downsample: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.GroupNorm(1, 64),

            # stage 2: 64 channels, 2 blocks
            ConvNeXtBlock(64, expansion=4, drop_path=drop_path_rate),
            ConvNeXtBlock(64, expansion=4, drop_path=drop_path_rate),

            # downsample: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.GroupNorm(1, 128),

            # stage 3: 128 channels, 3 blocks
            ConvNeXtBlock(128, expansion=4, drop_path=drop_path_rate),
            ConvNeXtBlock(128, expansion=4, drop_path=drop_path_rate),
            ConvNeXtBlock(128, expansion=4, drop_path=drop_path_rate),

            # downsample: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.GroupNorm(1, 256),

            # stage 4: 256 channels, 2 blocks
            ConvNeXtBlock(256, expansion=4, drop_path=drop_path_rate),
            ConvNeXtBlock(256, expansion=4, drop_path=drop_path_rate),

            # global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass for ConvNeXt.
        
        Args:
            x (torch.Tensor): Input batch of images of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Raw logits of shape (B, num_classes).
        """
        x = self.stem(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)