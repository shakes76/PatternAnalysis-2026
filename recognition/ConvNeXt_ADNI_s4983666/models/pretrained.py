"""Load the ImageNet-1k ConvNeXt-Tiny weights of Liu et al. into ``ConvNeXtTinyImageNet``.

Source: the authors' release, https://github.com/facebookresearch/ConvNeXt
(file ``convnext_tiny_1k_224_ema.pth``, ImageNet-1k, 224x224, EMA weights;
Liu et al., "A ConvNet for the 2020s", CVPR 2022). The file is downloaded
once by hand (see README) and passed to train.py; nothing is downloaded here.

The official parameter names differ from this implementation, so they are
mapped explicitly. Every backbone tensor must be found with the same shape;
only the 1000-class head is discarded (the AD classifier stays random).
"""

import hashlib
from pathlib import Path

import torch
from torch import nn

IMAGENET_SOURCE = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"


def official_name(name: str) -> str | None:
    """Official checkpoint key of one of our parameter names (None for the new classifier)."""
    parts = name.split(".")
    if parts[0] == "stem":                       # stem.{0,1} -> downsample_layers.0.{0,1}
        return "downsample_layers.0." + ".".join(parts[1:])
    if parts[0] == "downsample_layers":          # ours 0-2 -> official 1-3
        return f"downsample_layers.{int(parts[1]) + 1}." + ".".join(parts[2:])
    if parts[0] == "stages":
        renamed = {"depthwise": "dwconv", "norm": "norm", "expand": "pwconv1",
                   "project": "pwconv2", "layer_scale": "gamma"}[parts[3]]
        return ".".join(parts[:3] + [renamed] + parts[4:])
    if parts[0] == "final_norm":
        return "norm." + parts[1]
    return None                                  # classifier and input-normalization buffers


def file_sha256(path: Path) -> str:
    """SHA-256 of the weight file, recorded in the run config."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_imagenet_weights(model: nn.Module, path: Path) -> dict:
    """Copy every backbone tensor from the official checkpoint into ``model``; return provenance."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    official = checkpoint.get("model", checkpoint)
    state = model.state_dict()
    loaded = {}
    for name, tensor in state.items():
        source = official_name(name)
        if source is None:
            continue
        if source not in official or official[source].shape != tensor.shape:
            raise ValueError(f"Pretrained weights do not match {name} (expected {source}, {tuple(tensor.shape)}).")
        loaded[name] = official[source]
    expected = {official_name(name) for name in loaded}
    unused = sorted(key for key in official if key not in expected and not key.startswith("head."))
    if unused:
        raise ValueError(f"Unused backbone tensors in the pretrained file: {unused[:5]}")
    model.load_state_dict({**state, **loaded})
    return {"source": IMAGENET_SOURCE, "file": Path(path).name, "sha256": file_sha256(path),
            "loaded_tensors": len(loaded), "random_tensors": sorted(set(state) - set(loaded))}
