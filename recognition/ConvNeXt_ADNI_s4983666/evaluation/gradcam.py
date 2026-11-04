"""Grad-CAM saliency maps for the one-logit AD classifiers.

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017, https://arxiv.org/abs/1610.02391 .
The feature maps of a chosen layer are weighted by the spatial mean of the
gradient of the score and summed; negative evidence is removed by ReLU. The
score is the AD logit for slices predicted AD and its negative for slices
predicted NC, so the map always explains the predicted class.
"""

import torch
from torch import nn
import torch.nn.functional as F


def target_layer(model: nn.Module, depth: str = "last") -> nn.Module:
    """Last (or penultimate) feature stage of a ConvNeXt or the last CNN block."""
    if hasattr(model, "stages"):
        return model.stages[-1] if depth == "last" else model.stages[-2]
    if hasattr(model, "features"):  # Four blocks of (conv, norm, ReLU, pool); -5 ends the third block.
        return model.features if depth == "last" else model.features[-5]
    raise ValueError("Grad-CAM supports the ConvNeXt and SmallCNN models only.")


def grad_cam(model: nn.Module, images: torch.Tensor, layer: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (maps in [0, 1] with the input's height and width, AD logits) for a batch."""
    captured = {}

    def keep(_module, _inputs, output):
        output.retain_grad()
        captured["features"] = output

    handle = layer.register_forward_hook(keep)
    model.eval()
    try:
        with torch.enable_grad():
            images = images.detach().requires_grad_(True)  # Ensures a graph even if the first layer is frozen.
            logits = model(images)
            scores = torch.where(logits >= 0, logits, -logits)  # Explain the predicted class.
            model.zero_grad(set_to_none=True)
            scores.sum().backward()
    finally:
        handle.remove()
    features, gradients = captured["features"], captured["features"].grad
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    maps = F.relu((weights * features).sum(dim=1, keepdim=True))
    maps = F.interpolate(maps, size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    peak = maps.flatten(1).amax(dim=1).clamp_min(1e-12).view(-1, 1, 1)
    return (maps / peak).detach(), logits.detach()


def outside_share(maps: torch.Tensor, head_masks: torch.Tensor) -> torch.Tensor:
    """Fraction of each map's mass that falls outside the head (on the black background)."""
    total = maps.flatten(1).sum(dim=1).clamp_min(1e-12)
    return (maps * (~head_masks)).flatten(1).sum(dim=1) / total
