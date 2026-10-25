"""
Helpers shared by train.py and predict.py: building models, whole-volume
inference for both the 3D model and the slice-wise 2D baseline, and
overlap metrics.
"""

import torch

from modules import ImprovedUNet2D, ImprovedUNet3D, UNet2D

MODELS = {"unet3d": ImprovedUNet3D, "improved2d": ImprovedUNet2D, "unet2d": UNet2D}


def build_model(name, num_classes):
    """Create a model by name (``unet3d``, ``improved2d`` or ``unet2d``)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {list(MODELS)}")
    return MODELS[name](in_channels=1, num_classes=num_classes)


def load_checkpoint(path, device):
    """Rebuild a model from a checkpoint written by train.py."""
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["model"], checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_volume(model, image, model_name, slice_batch=32):
    """
    Class probabilities ``(C, D, H, W)`` for one volume ``image (1, D, H, W)``.

    The 3D model sees the whole volume at once. The 2D models are run on
    every axial slice independently and the slices are stacked back, so all
    models are scored on exactly the same voxels.
    """
    if model_name == "unet3d":
        return model(image[None]).softmax(dim=1)[0]

    slices = image.permute(1, 0, 2, 3)  # (D, 1, H, W)
    probs = [model(slices[i:i + slice_batch]).softmax(dim=1)
             for i in range(0, slices.shape[0], slice_batch)]
    return torch.cat(probs).permute(1, 0, 2, 3)


def dice_iou(pred, target, num_classes):
    """
    Per-class Dice and IoU between two hard label maps of equal shape.

    Returns two float tensors of length ``num_classes``. A class absent from
    both prediction and ground truth gets NaN so it can be skipped when
    averaging.
    """
    dice = torch.full((num_classes,), float("nan"))
    iou = torch.full((num_classes,), float("nan"))
    for c in range(num_classes):
        p, t = pred == c, target == c
        inter = (p & t).sum().item()
        total = p.sum().item() + t.sum().item()
        if total > 0:
            dice[c] = 2 * inter / total
            iou[c] = inter / (total - inter)
    return dice, iou
