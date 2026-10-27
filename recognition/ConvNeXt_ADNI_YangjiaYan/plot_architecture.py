"""Draw model architecture summaries for the report."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch.nn as nn

from modules import build_model, count_parameters

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

cnn = build_model("cnn", pretrained=False)
convnext = build_model("convnext", pretrained=False)

# Verify the dimensions shown against the model definitions.
cnn_convs = [
    m for m in cnn.features if isinstance(m, nn.Conv2d)
]
assert [m.out_channels for m in cnn_convs] == [32, 64, 128, 256]
assert count_parameters(cnn)[0] == 422530

backbone = convnext.backbone
stage_depths = [len(backbone.features[i]) for i in (1, 3, 5, 7)]
assert stage_depths == [3, 3, 9, 3]
assert backbone.classifier[2].in_features == 768
assert backbone.classifier[2].out_features == 2
assert count_parameters(convnext)[0] == 27821666

cnn_boxes = [
    "Input MRI slice\nRGB: 3 × 224 × 224",
    "Conv 3×3: 3 → 32 | BN | ReLU\nMaxPool 2×2 → 32 × 112 × 112",
    "Conv 3×3: 32 → 64 | BN | ReLU\nMaxPool 2×2 → 64 × 56 × 56",
    "Conv 3×3: 64 → 128 | BN | ReLU\nMaxPool 2×2 → 128 × 28 × 28",
    "Conv 3×3: 128 → 256 | BN | ReLU\nAdaptive average pooling → 256 × 1 × 1",
    "Flatten | Dropout (p = 0.3)\nLinear: 256 → 128 | ReLU",
    "Dropout (p = 0.3)\nLinear: 128 → 2",
    "Output: 2 logits\nEvaluation: Softmax → P(NC), P(AD)",
]

convnext_boxes = [
    "Input MRI slice\nRGB: 3 × 224 × 224",
    "Stem: Conv 4×4, stride 4 | LayerNorm\n96 × 56 × 56",
    f"Stage 1: {stage_depths[0]} ConvNeXt blocks\n96 × 56 × 56",
    f"Downsample: LayerNorm | Conv 2×2, stride 2\n"
    f"Stage 2: {stage_depths[1]} blocks → 192 × 28 × 28",
    f"Downsample: LayerNorm | Conv 2×2, stride 2\n"
    f"Stage 3: {stage_depths[2]} blocks → 384 × 14 × 14",
    f"Downsample: LayerNorm | Conv 2×2, stride 2\n"
    f"Stage 4: {stage_depths[3]} blocks → 768 × 7 × 7",
    "Adaptive average pooling → 768 × 1 × 1\n"
    "LayerNorm | Flatten | Linear: 768 → 2",
    "Output: 2 logits\nEvaluation: Softmax → P(NC), P(AD)",
]

fig, axes = plt.subplots(1, 2, figsize=(13, 12))

for ax, boxes, title, colour, note in [
    (
        axes[0], cnn_boxes, "SimpleCNN baseline", "#EAF2FA",
        "Random initialisation\n"
        "422,530 parameters; all trainable",
    ),
    (
        axes[1], convnext_boxes, "ConvNeXt-Tiny", "#FFF0E5",
        "ImageNet-pretrained backbone; new 2-class output layer\n"
        "27,821,666 parameters; all fine-tuned",
    ),
]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=17, pad=18)

    centres = [0.91 - i * 0.115 for i in range(len(boxes))]
    height = 0.085

    for i, (text, y) in enumerate(zip(boxes, centres)):
        patch = FancyBboxPatch(
            (0.04, y - height / 2), 0.92, height,
            boxstyle="round,pad=0.008",
            facecolor=colour, edgecolor="#455A64",
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(
            0.5, y, text,
            ha="center", va="center", fontsize=10.5,
        )
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(0.5, centres[i + 1] + height / 2 + 0.008),
                xytext=(0.5, y - height / 2 - 0.008),
                arrowprops=dict(arrowstyle="->", color="#455A64"),
            )

    ax.text(0.5, 0.025, note, ha="center", va="center", fontsize=10)

fig.suptitle("ADNI binary classification model architectures", fontsize=19)
fig.text(
    0.5, 0.018,
    "Feature dimensions: channels × height × width; batch dimension omitted. "
    "BN = BatchNorm. Training loss uses logits directly.",
    ha="center", fontsize=10,
)
fig.subplots_adjust(top=0.92, bottom=0.06, wspace=0.18)

for extension in ("png", "pdf"):
    path = OUT / f"model_architectures.{extension}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)

plt.close(fig)
print("Architecture checks: PASSED")
