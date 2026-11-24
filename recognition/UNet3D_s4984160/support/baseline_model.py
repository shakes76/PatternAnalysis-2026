"""Standard U-Net factory with normalization recorded in each checkpoint."""
from torch import nn
from modules import UNet3D


def make_model(base_channels=8, normalization='instance'):
    model = UNet3D(in_channels=1, num_classes=6, base_channels=base_channels)
    if normalization == 'instance':
        def replace(parent):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.BatchNorm3d):
                    setattr(parent, name, nn.InstanceNorm3d(
                        child.num_features, eps=child.eps, affine=True,
                        track_running_stats=False))
                else:
                    replace(child)
        replace(model)
    elif normalization != 'batch':
        raise ValueError(normalization)
    return model
