"""Training controls shared by train.py, predict.py and evaluation.

BCE definition: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
Autocast (mixed precision): https://docs.pytorch.org/docs/stable/amp.html
Mixup: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018,
https://arxiv.org/abs/1710.09412
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ExecutionControls:
    """Settings that change the model input, objective or numerics; saved in checkpoints."""

    context_slices: int = 1           # 1 = single slice, 3 = slice stacked with its neighbours.
    loss: str = "weighted_bce"        # weighted_bce uses the train-only NC/AD slice ratio.
    precision: str = "fp32"           # amp_bf16 runs the forward pass in bfloat16 on CUDA.
    drop_path: float | None = None    # None keeps the architecture default (ConvNeXt only).
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.context_slices not in (1, 3):
            raise ValueError("context_slices must be 1 or 3.")
        if self.loss not in ("bce", "weighted_bce"):
            raise ValueError("loss must be bce or weighted_bce.")
        if self.precision not in ("fp32", "amp_bf16"):
            raise ValueError("precision must be fp32 or amp_bf16.")
        if self.drop_path is not None and not 0 <= self.drop_path < 1:
            raise ValueError("drop_path must be in [0, 1).")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1).")
        if not 0 <= self.mixup_alpha <= 1:
            raise ValueError("mixup_alpha must be in [0, 1]; zero disables Mixup.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExecutionControls":
        return cls(**value)


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the ExecutionControls fields on a command line."""
    parser.add_argument("--context-slices", type=int, choices=(1, 3), default=3,
                        help="1: single slice; 3: stack each slice with its two neighbours (default: 3).")
    parser.add_argument("--loss", choices=("bce", "weighted_bce"), default="weighted_bce")
    parser.add_argument("--precision", choices=("fp32", "amp_bf16"), default="fp32")
    parser.add_argument("--drop-path", type=float, default=None,
                        help="Maximum ConvNeXt DropPath; omitted keeps 0.1.")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Beta(alpha, alpha) Mixup on training batches; 0 disables it.")


def controls_from_args(args: argparse.Namespace) -> ExecutionControls:
    """Collect the parsed ExecutionControls fields."""
    return ExecutionControls(**{name: getattr(args, name) for name in ExecutionControls.__dataclass_fields__})


def validate_device(controls: ExecutionControls, device: torch.device) -> None:
    """bfloat16 autocast is only used on CUDA GPUs that support it."""
    if controls.precision == "amp_bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("amp_bf16 requires a CUDA GPU with bfloat16 support; use --precision fp32.")


def autocast_context(controls: ExecutionControls, device: torch.device) -> Any:
    """Return the autocast context for one forward pass."""
    if controls.precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def make_criterion(controls: ExecutionControls, pos_weight: float, device: torch.device) -> nn.Module:
    """BCE on the AD logit, optionally class-weighted and label-smoothed."""
    weight = pos_weight if controls.loss == "weighted_bce" else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    return SmoothedBinaryLoss(criterion, controls.label_smoothing) if controls.label_smoothing else criterion


class SmoothedBinaryLoss(nn.Module):
    """Move binary targets toward 0.5: y -> (1 - epsilon) * y + epsilon / 2.

    This is the two-class version of the smoothing used by PyTorch's
    CrossEntropyLoss(label_smoothing=...).
    """

    def __init__(self, criterion: nn.Module, epsilon: float) -> None:
        super().__init__()
        self.criterion, self.epsilon = criterion, epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets * (1 - self.epsilon) + self.epsilon / 2)


def mixup_batch(images: torch.Tensor, labels: torch.Tensor,
                alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mix each training image with another image from the same minibatch.

    Returns the mixed images, both label sets and the mixing coefficient. The
    loss is ``lam * loss(labels_a) + (1 - lam) * loss(labels_b)``.
    """
    if alpha == 0 or labels.numel() < 2:
        return images, labels, labels, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    permutation = torch.randperm(labels.numel(), device=images.device)
    return lam * images + (1 - lam) * images[permutation], labels, labels[permutation], lam
