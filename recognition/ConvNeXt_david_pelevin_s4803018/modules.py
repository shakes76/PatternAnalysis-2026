import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A simple convolutional neural network for 256x240px images

    Code developed using the following websites
    https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    def __init__(self):
        """
        Initialises the components of the CNN with the relevant dimensions
        """
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(5824, 128)
        
        self.fc2 = nn.Linear(128, 2)

    def forward(self, input):
        """
        Classifies a batch of images using the CNN and returns logits
        representing the confidence of the model in each class.

        Parameters:
            input (Tensor N, C, H, W / C, H, W): The batch of images to be categorised.
        
        Returns:
            Tensor (int, int): The logits for the confidence in each class.
        """
        input = self.pool(F.relu(self.conv1(input)))
        input = self.pool(F.relu(self.conv2(input)))
        input = self.pool(F.relu(self.conv3(input)))
        input = self.pool(F.relu(self.conv4(input)))

        input = torch.flatten(input, 1)
        input = F.relu(self.fc1(input))
        input = self.fc2(input)
        return input


class ConvNeXtBlock(nn.Module):
    """
    A single ConvNeXt block based on the architecture
    shown in https://www.geeksforgeeks.org/computer-vision/convnext/

    Performs depthwise convolution, GELU activation, layer scale, and contains a drop path.
    """
    def __init__(self, dimension, layer_scale=1e-6):
        """
        Initialises a ConvNeXt block with the relevant dimensions.

        Setups up the depthwise convolution, GELU activation, and layer scale.

        Parameters:
            dimension (int): Number of channels in the image inputs
            layer_scale (float): The hyperparameter in charge of layer scale within the block
        """
        super().__init__()

        ## Depthwise convolution requires groups=dimension. This setting was found in
        ## https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html
        self.depthConv2d = nn.Conv2d(dimension, dimension, 7, padding=3, groups=dimension)

        self.l1 = nn.LayerNorm(dimension)
        self.conv1 = nn.Conv2d(dimension, dimension * 4, 1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(dimension * 4, dimension, 1)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dimension))

    def forward(self, input):
        """
        Runs a batch of images through the ConvNeXt block.

        Parameters:
            input (Tensor N, C, H, W / C, H, W): The batch of images to be categorised.
        
        Returns:
            Tensor (N, C, H, W): The convolved batch of images.
        """
        carryThrough = input

        input = self.depthConv2d(input)
        ## Layer norm requires channels to be last (N, H, W, C)
        input = self.l1(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        input = self.conv1(input)
        input = self.gelu(input)
        input = self.conv2(input)
        ## Layer scaling requires channels to be last (N, H, W, C)
        input = (self.gamma * input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        ## Stochastic Depth A.K.A Drop Path
        input = torchvision.ops.stochastic_depth(input, 0.1, "row", self.training)

        return input + carryThrough


class downSample(nn.Module):
    """
    Implements downsampling of the image.
    Each downsample doubles the number of channels and halves the height and width of the image.

    Based on the architecture shown in
    https://www.geeksforgeeks.org/computer-vision/convnext/
    """
    def __init__(self, dimension):
        """
        Initalises the layer norm and convolution in charge of downsampling.

        Parameters:
            dimension (int): The number of channels in the inputted image.
        """
        super().__init__()

        self.l1 = nn.LayerNorm(dimension)
        self.conv1 = nn.Conv2d(dimension, dimension * 2, 2, 2)

    def forward(self, input):
        """
        Downsamples an image by doubling the number of channels and halving the height and width.

        Parameters:
            input (Tensor N, C, H, W / C, H, W): The batch of images to be downsampled.

        Returns:
            Tensor N, C * 2, H / 2, W / 2: The downsampled image tensors
        """
        ## Layer norm requires channels to be last (N, H, W, C)
        input = self.l1(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        input = self.conv1(input)
        return input


class ConvNeXt(nn.Module):
    """
    A Tiny ConvNeXt model for 256x240px images based on the architecture
    shown in https://www.geeksforgeeks.org/computer-vision/convnext/

    Implements a ConvNeXt model with dimensions 96 and batch sizes (3, 3, 9, 3).
    """
    def __init__(self):
        """
        Initialises the components of the ConvNeXt model.
        """
        super().__init__()

        convnext_size = 96

        self.conv1 = nn.Conv2d(1, convnext_size, 4, 4)
        self.l1 = nn.LayerNorm(convnext_size)

        self.convnext1 = nn.Sequential(
            ConvNeXtBlock(convnext_size),
            ConvNeXtBlock(convnext_size),
            ConvNeXtBlock(convnext_size)
        )

        self.downSample1 = downSample(convnext_size)

        self.convnext2 = nn.Sequential(
            ConvNeXtBlock(convnext_size * 2),
            ConvNeXtBlock(convnext_size * 2),
            ConvNeXtBlock(convnext_size * 2)
        )

        self.downSample2 = downSample(convnext_size * 2)

        self.convnext3 = nn.Sequential(
            ConvNeXtBlock(convnext_size * 4),
            ConvNeXtBlock(convnext_size * 4),
            ConvNeXtBlock(convnext_size * 4),
            ConvNeXtBlock(convnext_size * 4),
            ConvNeXtBlock(convnext_size * 4),
            ConvNeXtBlock(convnext_size * 4),
            ConvNeXtBlock(convnext_size * 4),
            ConvNeXtBlock(convnext_size * 4),
            ConvNeXtBlock(convnext_size * 4)
        )

        self.downSample3 = downSample(convnext_size * 4)

        self.convnext4 = nn.Sequential(
            ConvNeXtBlock(convnext_size * 8),
            ConvNeXtBlock(convnext_size * 8),
            ConvNeXtBlock(convnext_size * 8)
        )

        self.averagePool = nn.AdaptiveAvgPool2d(1)

        self.l2 = nn.LayerNorm(convnext_size * 8)

        self.linear = nn.Linear(convnext_size * 8, 2)

    def forward(self, input):
        """
        Classifies a batch of images using the ConvNeXt model and returns logits
        representing the confidence of the model in each class.

        Parameters:
            input (Tensor N, C, H, W / C, H, W): The batch of images to be categorised.

        Returns:
            Tensor (int, int): The logits for the confidence in each class.
        """
        input = self.conv1(input)
        ## Layer norm requires channels to be last (N, H, W, C)
        input = self.l1(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        input = self.convnext1(input)
        input = self.downSample1(input)
        input = self.convnext2(input)
        input = self.downSample2(input)
        input = self.convnext3(input)
        input = self.downSample3(input)
        input = self.convnext4(input)
        input = self.averagePool(input)
        ## Layer norm requires channels to be last (N, H, W, C)
        input = self.l2(input.permute(0, 2, 3, 1))
        input = self.linear(input).permute(0, 3, 1, 2)
        input = torch.flatten(input, 1)
        return input