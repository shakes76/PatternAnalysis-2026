"""Pure PyTorch neural network modules for ADNI Brain MRI classification.

Author: Lingxiao Li (s49852413)
Course: COMP3710 / COMP7710 Pattern Recognition and Analysis (2026)

Strict Compliance Note (Section 4.1):
    This file depends EXCLUSIVELY on PyTorch (`torch`, `torch.nn`, `torch.nn.functional`)
    and Python standard library typing annotations. It contains zero dependency on NumPy
    or any external third-party libraries.

Implemented Components:
    1. `LayerNorm2d`: Channel-first 2D Layer Normalization for `[B, C, H, W]` feature maps.
    2. `DropPath`: Stochastic Depth per-sample residual branch regularization.
    3. `ConvNeXtBlock`: Modernized ConvNet block with 7x7 depthwise convolution, inverted
       4x bottleneck, GELU activation, and learnable LayerScale (Liu et al., CVPR 2022).
    4. `ConvNeXt`: 4-stage hierarchical ConvNeXt backbone (`[Hard Difficulty]`).
    5. `SimpleCNNBaseline`: Standard 3-stage 3x3 CNN benchmark (`Baseline Comparison`).
    6. `TemperatureScaledClassifier`: Post-hoc temperature scaling wrapper (`T > 0`) and
       predictive uncertainty / Shannon entropy quantifier for clinical triage routing.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-first 2D Layer Normalization for `[B, C, H, W]` feature maps.

    Mathematical formulation:
        y_{b,c,h,w} = gamma_c * (x_{b,c,h,w} - mu_{b,h,w}) / sqrt(sigma^2_{b,h,w} + eps) + beta_c
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class DropPath(nn.Module):
    """Stochastic Depth (DropPath) regularization per sample in residual branches."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class ConvNeXtBlock(nn.Module):
    """Modernized ConvNeXt Block (Liu et al., 'A ConvNet for the 2020s', CVPR 2022).

    Architecture:
        x -> 7x7 Depthwise Conv (groups=dim) -> Permute [B, H, W, C] -> LayerNorm(dim)
          -> Linear(dim, 4 * dim) -> GELU -> Linear(4 * dim, dim) -> LayerScale(gamma)
          -> Permute [B, C, H, W] -> DropPath -> + x
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(mlp_ratio * dim)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        if layer_scale_init_value > 0.0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones((dim,)), requires_grad=True
            )
        else:
            self.gamma = None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        return shortcut + self.drop_path(x)


class ConvNeXt(nn.Module):
    """4-Stage Hierarchical ConvNeXt Backbone for Neuroimaging Classification (`[Hard Difficulty]`).

    Adapts the ConvNeXt architecture (Liu et al., CVPR 2022) to 1-channel T1-weighted
    structural brain MRI scans. Uses a patchify stem followed by four spatial stages
    with 2x2 stride-2 downsampling layers between stages, global average pooling,
    final LayerNorm, and a linear classification head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        depths: Sequence[int] = (2, 2, 6, 2),
        dims: Sequence[int] = (32, 64, 128, 256),
        drop_path_rate: float = 0.10,
        layer_scale_init_value: float = 0.1,
        head_dropout: float = 0.2,
        stem_stride: int = 4,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("ConvNeXt requires 4 stages for `depths` and `dims`.")

        self.depths = tuple(depths)
        self.dims = tuple(dims)

        # Downsampling layers: 4x4 non-overlapping patchify stem + 3 inter-stage transition layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dims[0],
                kernel_size=4,
                stride=stem_stride,
                padding=1 if stem_stride == 2 else 0,
            ),
            LayerNorm2d(dims[0], eps=1e-6),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Stochastic depth decay schedule across all blocks
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0.0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage_blocks = [
                ConvNeXtBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                )
                for j in range(depths[i])
            ]
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head_drop = nn.Dropout(p=head_dropout)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def get_last_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return final spatial feature map `[B, C_4, H_4, W_4]` prior to global pooling."""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalized global pooled representation `[B, C_4]`."""
        feat_map = self.get_last_feature_map(x)
        pooled = feat_map.mean(dim=(-2, -1))  # Global average pooling [B, C]
        return self.norm(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized class logits `[B, num_classes]`."""
        feats = self.forward_features(x)
        feats = self.head_drop(feats)
        return self.head(feats)


class SimpleCNNBaseline(nn.Module):
    """Standard 2-stage Convolutional Neural Network Baseline for ADNI AD vs. NC screening.

    Serves as the mandatory Section 2.1 baseline benchmark using conventional 3x3
    convolutions, ReLU activations, and MaxPool2d without residual connections,
    large-kernel depthwise mixing, or LayerNorm.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 16,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2

        self.features = nn.Sequential(
            # Stage 1: local 3x3 receptive field
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            # Stage 2: local 3x3 receptive field
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c2, c2),
            nn.ReLU(inplace=True),
            nn.Linear(c2, num_classes),
        )

    def get_last_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return final spatial feature map `[B, C, H', W']` for spatial attribution."""
        return self.features(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pre-logit pooled feature vector `[B, C]`."""
        feat_map = self.features(x)
        return self.pool(feat_map).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized class logits `[B, num_classes]`."""
        feat_map = self.features(x)
        pooled = self.pool(feat_map)
        return self.classifier(pooled)


class TemperatureScaledClassifier(nn.Module):
    """Post-hoc Temperature Scaling Calibration & Uncertainty Quantifier (Guo et al., ICML 2017).

    Addresses the Section 2.1 Open Research Question ('The Confidence and Ambiguity
    Dilemma in Automated Triage') by calibrating raw network logits `z` via a single
    scalar temperature `T = exp(log_temperature) > 0` optimized on held-out validation
    patients without altering class ranking:
        p_hat(y = c | x) = softmax(z / T)_c

    Also computes normalized Shannon predictive entropy H(p_hat) in [0, 1] and
    maximum predicted confidence max_c p_hat_c for reject-option clinical routing.
    """

    def __init__(self, base_model: nn.Module, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.base_model = base_model
        init_t = max(float(init_temperature), 1e-3)
        self.log_temperature = nn.Parameter(torch.tensor(init_t).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.05, max=20.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return temperature-calibrated logits `z / T`."""
        logits = self.base_model(x)
        return logits / self.temperature

    def forward_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute raw logits, calibrated probabilities, confidence, and normalized entropy."""
        raw_logits = self.base_model(x)
        cal_logits = raw_logits / self.temperature
        raw_probs = F.softmax(raw_logits, dim=-1)
        cal_probs = F.softmax(cal_logits, dim=-1)

        confidence, preds = cal_probs.max(dim=-1)
        raw_confidence, _ = raw_probs.max(dim=-1)

        # Normalized binary Shannon entropy H(p) / log(K) in [0, 1]
        num_classes = cal_probs.shape[-1]
        log_k = torch.tensor(float(num_classes), device=cal_probs.device).log()
        entropy = -(cal_probs * (cal_probs.clamp(min=1e-8)).log()).sum(dim=-1) / log_k

        return {
            "raw_logits": raw_logits,
            "calibrated_logits": cal_logits,
            "raw_probs": raw_probs,
            "calibrated_probs": cal_probs,
            "predictions": preds,
            "confidence": confidence,
            "raw_confidence": raw_confidence,
            "entropy": entropy,
        }

    def fit_temperature(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 200,
        lr: float = 0.05,
    ) -> float:
        """Optimize scalar temperature `T` by minimizing NLL on validation logits."""
        device = val_logits.device
        self.to(device)
        logits_detached = val_logits.detach()
        labels_detached = val_labels.detach().long()

        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def _eval_closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = logits_detached / self.temperature
            loss = F.cross_entropy(scaled, labels_detached)
            loss.backward()
            return loss

        optimizer.step(_eval_closure)
        return float(self.temperature.detach().cpu().item())


def compute_spatial_attribution(model: nn.Module, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
    """Compute Grad-CAM spatial attribution heatmap `[B, H, W]` in pure PyTorch.

    Highlights which anatomical regions (e.g., lateral ventricles, hippocampal / medial
    temporal cortex, or peripheral skull boundaries) drove the network's prediction.
    """
    base = model.base_model if isinstance(model, TemperatureScaledClassifier) else model
    base.eval()

    x_in = x.detach().clone().requires_grad_(True)
    saved_acts: list[torch.Tensor] = []
    saved_grads: list[torch.Tensor] = []

    # Hook onto the final convolutional stage
    target_layer = base.stages[-1] if isinstance(base, ConvNeXt) else base.features

    def _fwd_hook(_mod: nn.Module, _inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        saved_acts.append(out)
        out.register_hook(lambda g: saved_grads.append(g))

    handle = target_layer.register_forward_hook(_fwd_hook)
    try:
        logits = base(x_in)
        if target_class is None:
            target = logits.argmax(dim=-1)
        else:
            target = torch.full((logits.shape[0],), int(target_class), device=logits.device, dtype=torch.long)
        selected = logits.gather(1, target.unsqueeze(1)).sum()
        base.zero_grad()
        selected.backward()

        acts = saved_acts[0].detach()   # [B, C, H', W']
        grads = saved_grads[0].detach()  # [B, C, H', W']
        weights = grads.mean(dim=(-2, -1), keepdim=True)  # [B, C, 1, 1]
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # [B, 1, H', W']
        cam_up = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

        # Normalize each heatmap to [0, 1]
        b = cam_up.shape[0]
        cam_flat = cam_up.view(b, -1)
        c_min = cam_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1)
        c_max = cam_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1)
        return (cam_up - c_min) / (c_max - c_min + 1e-8)
    finally:
        handle.remove()


def build_model(
    model_name: str = "convnext",
    in_channels: int = 1,
    num_classes: int = 2,
    drop_path_rate: float = 0.15,
) -> nn.Module:
    """Factory function to instantiate `ConvNeXt` variants or `SimpleCNNBaseline`."""
    name = model_name.lower().strip()
    if name in ("convnext", "convnext_med"):
        return ConvNeXt(
            in_channels=in_channels,
            num_classes=num_classes,
            depths=(2, 2, 6, 2),
            dims=(32, 64, 128, 256),
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=0.1,
            stem_stride=4,
        )
    if name == "convnext_tiny":
        return ConvNeXt(
            in_channels=in_channels,
            num_classes=num_classes,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            drop_path_rate=drop_path_rate,
            stem_stride=4,
        )
    if name in ("baseline", "simple_cnn", "cnn"):
        return SimpleCNNBaseline(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model_name '{model_name}'. Choose from: convnext, convnext_tiny, baseline.")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return `(total_params, trainable_params)` for a PyTorch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    x_dummy = torch.randn(4, 1, 128, 128)
    for m_name in ("baseline", "convnext"):
        net = build_model(m_name)
        tot_p, tr_p = count_parameters(net)
        out = net(x_dummy)
        cam_map = compute_spatial_attribution(net, x_dummy)
        print(
            f"[{m_name}] params={tot_p:,} (trainable={tr_p:,}) | "
            f"logits={tuple(out.shape)} | gradcam={tuple(cam_map.shape)}"
        )
