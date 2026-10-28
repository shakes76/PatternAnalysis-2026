import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveInstanceNorm(nn.Module):
    """Weighted combination of identity and instance norm function."""
    def __init__(self, channels):
        super().__init__()
        self.instancenorm = nn.InstanceNorm3d(channels)
        self.weights = nn.Parameter(torch.rand(2))

    def forward(self, x):
        return self.weights[0] * x + self.weights[1] * self.instancenorm(x)

class CAN3D(nn.Module):
    """Progressive dilation network

    Implementation is based on descriptions in:
        W. Dai, et al. “CAN3D: Fast 3D medical image segmentation via compact 
        context aggregation”, 2021, https://doi.org/10.1016/j.media.2022.102562.

    I have added an additional layer in the CAM for greater context awareness.
    """
    def __init__(self, depth=256, height=256, width=128, outputs = 6):
        super().__init__()
        self.depth = depth
        self.height = height
        self.width = width

        self.inlayer = self.create_layer(1, 3, 1)
        self.cammodule = nn.Sequential(
            self.create_layer(3, 6, 1, stride=2),
            self.create_layer(6, 36, 1, stride=2),

            self.create_layer(36, 36, 2, initialiser='identity', padding=2),
            self.create_layer(36, 36, 4, initialiser='identity', padding=4),
            self.create_layer(36, 36, 8, initialiser='identity', padding=8),
            self.create_layer(36, 36, 16, initialiser='identity', padding=16),
            self.create_layer(36, 48, 1, initialiser='identity'),

            nn.Upsample(scale_factor=2),
            self.create_layer(48, 24, 1, initialiser='identity'),
            nn.Upsample(scale_factor=2),

        )
        self.skip = self.create_layer(3, 24, 1, kernel_size=1, padding=0)
        self.segmentationlayer = nn.Conv3d(24, outputs, 1)

    def create_layer(self, in_channels, out_channels, dilation, stride=1, padding=1, kernel_size=3, initialiser='xavier'):
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        if initialiser == 'identity':
            with torch.no_grad():
                conv.weight.fill_(0.0)
                conv.weight.data += (torch.randn_like(conv.weight.data) / out_channels) * 0.25
                channel_dilation_factor = in_channels / out_channels
                for i in range(out_channels):
                    j = math.floor(channel_dilation_factor * i)
                    conv.weight.data[i, j, kernel_size//2, kernel_size//2] = channel_dilation_factor
                
        return nn.Sequential(
            conv,
            AdaptiveInstanceNorm(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        # Take no channels, and add 1
        x = x.view(x.shape[0], 1, self.depth, self.height, self.width)
        
        x = self.inlayer(x)
        u = self.skip(x)
        x = self.cammodule(x)
        x = x + u
        x = self.segmentationlayer(x)
        return x

class CAN2D(nn.Module):
    """Progressive dilation network

    Adapted architecture from Fisher Yu, by adding AdaIn as Wai Dai suggests, and leaky relu.
    """
    def __init__(self, height=256, width=128, outputs=6):
        super().__init__()
        self.height = height
        self.width = width

        self.cammodule = nn.Sequential(
            nn.MaxPool2d(2, 2),
            self.create_layer(1, 8, 1),
            self.create_layer(8, 16, 1),
            self.create_layer(16, 32, 2, 1, 3, 'identity', padding=2),
            self.create_layer(32, 64, 4, 1, 3, 'identity', padding=4),
            self.create_layer(64, 64, 8, 1, 3, 'identity', padding=8),
            self.create_layer(64, 64, 16, 1, 3, 'identity', padding=16),
            self.create_layer(64, 64, 1, 1),
            nn.Conv2d(64, outputs, 1),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        x = x.view(x.shape[0], 1, self.height, self.width)
        x = self.cammodule(x)
        return x

    def create_layer(self, in_channels, out_channels, dilation, stride=1, kernel_size=3, initialiser='xavier', padding=1):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        if initialiser == 'identity':
            with torch.no_grad():
                conv.weight.fill_(0.0)
                conv.weight.data += (torch.randn_like(conv.weight.data) / out_channels) * 0.5
                channel_dilation_factor = in_channels / out_channels
                for i in range(out_channels):
                    j = math.floor(channel_dilation_factor * i)
                    conv.weight.data[i, j, kernel_size//2, kernel_size//2] = channel_dilation_factor
        return nn.Sequential(
            conv,
            AdaptiveInstanceNorm(out_channels),
            nn.LeakyReLU(0.1)
        )


def compute_dsfl(y_probs, y_true, weights=None, alphas=None, smooth = 0.01, gamma=2):
    """Dice squared focal loss, as described in the (W. Dai, 2021) paper."""
    if weights is None or alphas is None:
        raise ValueError("Weights and alphas must be provided as tensors on the device.")

    voxel_count = y_probs.shape[2] * y_probs.shape[3] * y_probs.shape[4]
    y_probs = y_probs.reshape(y_probs.shape[0], y_probs.shape[1], -1)
    y_true = y_true.view(y_true.shape[0], y_true.shape[1], -1)

    difference = y_probs - y_true
    phi = torch.sum(difference * difference, dim=-1)
    denominator = torch.sum(y_probs * y_probs + y_true * y_true + smooth, dim=-1)

    dsl_per_class = phi / denominator + phi / voxel_count
    dsl_per_class = torch.mean(dsl_per_class, dim=0)
    fl_per_class = -torch.sum(y_true * torch.pow(1.0 - y_probs, gamma) * torch.log(y_probs + smooth), dim=-1) / voxel_count
    fl_per_class = torch.mean(fl_per_class, dim=0)
    dsfl = torch.dot(dsl_per_class, weights) + torch.dot(fl_per_class, alphas)
    return dsfl

def compute_loss(predictions, labels, weights, alphas, gamma = 2):
    """Takes logits predictions as input
    Note: gamma < 1 can cause instability."""
    predictions = nn.functional.softmax(predictions, dim=1)
    labels = nn.functional.one_hot(labels, predictions.shape[1]).permute(0, 4, 1, 2, 3)
    return compute_dsfl(predictions, labels, weights, alphas, gamma=gamma)

def calculate_descrete_dsc(prediction_logits, label, smooth = 1e-6):
    """Calculates the descrete Dice-Sorenson Coefficient for each class"""
    batch_size, num_classes = prediction_logits.shape[0], prediction_logits.shape[1]
    y_preds = torch.argmax(prediction_logits, dim=1)
    y_preds = nn.functional.one_hot(y_preds, num_classes).permute(0, 4, 1, 2, 3).view(batch_size, num_classes, -1)
    y_true = nn.functional.one_hot(label, num_classes).permute(0, 4, 1, 2, 3).view(batch_size, num_classes, -1)

    intersection = torch.sum(y_preds * y_true, dim=-1)
    cardinality = torch.sum(y_preds + y_true, dim=-1)

    dice_per_class = (2. * intersection + smooth) / (cardinality + smooth)
    sum_dice_per_class = torch.sum(dice_per_class, dim=0)
    return sum_dice_per_class, batch_size

def calculate_intersection_over_union(prediction_logits, label, smooth = 1e-6):
    """Calculates the intersection over union statistic for each class"""
    batch_size, num_classes = prediction_logits.shape[0], prediction_logits.shape[1]
    y_preds = torch.argmax(prediction_logits, dim=1)
    y_preds = nn.functional.one_hot(y_preds, num_classes).permute(0, 4, 1, 2, 3).view(batch_size, num_classes, -1)
    y_true = nn.functional.one_hot(label, num_classes).permute(0, 4, 1, 2, 3).view(batch_size, num_classes, -1)

    intersection = torch.sum(y_preds * y_true, dim=-1)
    union = torch.sum(y_preds + y_true, dim=-1) - intersection
    iou_per_class = torch.sum(intersection / (union + smooth), dim=0)
    return iou_per_class, batch_size


def compute_dsfl_2d(y_probs, y_true, weights=None, alphas=None, smooth = 0.01, gamma=2):
    """Dice squared focal loss, as described in the (W. Dai, 2021) paper."""
    if weights is None or alphas is None:
        raise ValueError("Weights and alphas must be provided as tensors on the device.")

    pixel_count = y_probs.shape[2] * y_probs.shape[3]
    y_probs = y_probs.reshape(y_probs.shape[0], y_probs.shape[1], -1)
    y_true = y_true.view(y_true.shape[0], y_true.shape[1], -1)

    difference = y_probs - y_true
    phi = torch.sum(difference * difference, dim=-1)
    denominator = torch.sum(y_probs * y_probs + y_true * y_true + smooth, dim=-1)

    dsl_per_class = phi / denominator + phi / pixel_count
    dsl_per_class = torch.mean(dsl_per_class, dim=0)
    fl_per_class = -torch.mean(y_true * torch.pow(1.0 - y_probs, gamma) * torch.log(y_probs + smooth), dim=-1)
    fl_per_class = torch.mean(fl_per_class, dim=0)
    dsfl = torch.dot(dsl_per_class, weights) + torch.dot(fl_per_class, alphas)

    return dsfl

def compute_loss_2d(predictions, labels, weights, alphas, gamma = 2):
    """Takes logits predictions as input.
    Note: gamma < 1 can cause instability."""
    predictions = nn.functional.softmax(predictions, dim=1)
    labels = nn.functional.one_hot(labels, predictions.shape[1]).permute(0, 3, 1, 2)
    dsfl = compute_dsfl_2d(predictions, labels, weights, alphas, gamma = gamma)
    return dsfl