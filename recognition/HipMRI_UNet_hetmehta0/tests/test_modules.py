"""Architecture tests for both 2D U-Net variants (synthetic tensors only)."""

from __future__ import annotations

import ast

import pytest
import torch
from conftest import PROJECT_DIR

from baselines.unet2d import unet2d_256, unet2d_512
from losses import DiceCrossEntropyLoss
from modules import (
    UNET2D_VARIANTS,
    DecoderStage,
    EncoderStage,
    UNet2D,
    build_model,
    count_parameters,
)

VARIANTS = sorted(UNET2D_VARIANTS)


def expected_parameters(bottleneck: int, encoder=(32, 64, 128, 256), in_ch=1, classes=6) -> int:
    """Independent hand count. Conv has no bias (BatchNorm follows); BN has 2 params/channel."""

    def double_conv(i, o):
        return 9 * i * o + 2 * o + 9 * o * o + 2 * o

    total, prev = 0, in_ch
    for width in encoder:
        total += double_conv(prev, width)
        prev = width
    total += double_conv(prev, bottleneck)
    prev = bottleneck
    for width in reversed(encoder):
        total += 4 * prev * width + width  # ConvTranspose2d 2x2 with bias
        total += double_conv(2 * width, width)
        prev = width
    total += prev * classes + classes  # 1x1 head with bias
    return total


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("size", [(256, 256), (64, 96)])
def test_output_shape_and_six_channels(variant, size):
    model = build_model(variant).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, *size))
    assert out.shape == (2, 6, *size)


@pytest.mark.parametrize("variant", VARIANTS)
def test_stage_shapes(variant):
    """Encoder skips, bottleneck and decoder outputs have the agreed widths and resolutions."""
    model = build_model(variant).eval()
    b = UNET2D_VARIANTS[variant]
    shapes = {}

    def record(name):
        def hook(_module, _inputs, output):
            shapes[name] = (
                tuple(output[1].shape) if isinstance(output, tuple) else tuple(output.shape)
            )

        return hook

    for i, stage in enumerate(model.encoder.stages):
        stage.register_forward_hook(record(f"enc{i}_skip"))
    model.bottleneck.register_forward_hook(record("bottleneck"))
    for i, stage in enumerate(model.decoder.stages):
        stage.register_forward_hook(record(f"dec{i}"))
    with torch.no_grad():
        model(torch.randn(1, 1, 256, 256))

    assert shapes["enc0_skip"] == (1, 32, 256, 256)
    assert shapes["enc1_skip"] == (1, 64, 128, 128)
    assert shapes["enc2_skip"] == (1, 128, 64, 64)
    assert shapes["enc3_skip"] == (1, 256, 32, 32)
    assert shapes["bottleneck"] == (1, b, 16, 16)
    assert shapes["dec0"] == (1, 256, 32, 32)
    assert shapes["dec1"] == (1, 128, 64, 64)
    assert shapes["dec2"] == (1, 64, 128, 128)
    assert shapes["dec3"] == (1, 32, 256, 256)


@pytest.mark.parametrize("variant", VARIANTS)
def test_decoder_input_is_skip_then_upsampled(variant):
    """Each decoder DoubleConv receives cat(encoder skip, upsampled), in that order."""
    model = build_model(variant).eval()
    skips, ups, conv_inputs = [], [], []
    for stage in model.encoder.stages:
        stage.register_forward_hook(lambda m, i, o: skips.append(o[1]))
    for stage in model.decoder.stages:
        stage.up.register_forward_hook(lambda m, i, o: ups.append(o))
        stage.conv.register_forward_hook(lambda m, i, o: conv_inputs.append(i[0]))
    with torch.no_grad():
        model(torch.randn(1, 1, 64, 64))

    for skip, up, conv_in in zip(reversed(skips), ups, conv_inputs, strict=True):
        c = skip.shape[1]
        assert conv_in.shape[1] == 2 * c
        assert torch.equal(conv_in[:, :c], skip)
        assert torch.equal(conv_in[:, c:], up)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("level", range(4))
def test_each_skip_connection_affects_output(variant, level):
    model = build_model(variant).eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        reference = model(x)
        handle = model.encoder.stages[level].register_forward_hook(
            lambda m, i, o: (o[0], torch.zeros_like(o[1]))
        )
        ablated = model(x)
        handle.remove()
    assert not torch.allclose(reference, ablated)


def test_variants_differ_only_in_bottleneck():
    """Only the bottleneck and the first up-convolution's input width may differ."""
    a = build_model("unet2d_256").state_dict()
    b = build_model("unet2d_512").state_dict()
    assert a.keys() == b.keys()
    differing = {k for k in a if a[k].shape != b[k].shape}
    allowed = {k for k in a if k.startswith("bottleneck.")} | {"decoder.stages.0.up.weight"}
    assert differing and differing <= allowed
    assert a["decoder.stages.0.up.weight"].shape[0] == 256
    assert b["decoder.stages.0.up.weight"].shape[0] == 512
    # Everything after the first up-convolution is shape-identical.
    assert a["decoder.stages.0.up.bias"].shape == b["decoder.stages.0.up.bias"].shape


@pytest.mark.parametrize("variant", VARIANTS)
def test_parameter_count_matches_hand_count(variant):
    model = build_model(variant)
    assert count_parameters(model) == expected_parameters(UNET2D_VARIANTS[variant])


def test_parameter_counts_values():
    counts = {v: count_parameters(build_model(v)) for v in VARIANTS}
    print("parameter counts:", counts)
    assert counts["unet2d_512"] > counts["unet2d_256"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_backward_finite(variant):
    torch.manual_seed(0)
    model = build_model(variant).train()
    x = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 6, (2, 64, 64))
    terms = DiceCrossEntropyLoss()(model(x), target)
    assert torch.isfinite(terms.total)
    terms.total.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())


def test_rejects_size_not_divisible_by_16():
    with pytest.raises(ValueError, match="divisible by 16"):
        build_model("unet2d_256")(torch.randn(1, 1, 250, 256))


def test_rejects_wrong_rank():
    with pytest.raises(ValueError, match="Expected input"):
        build_model("unet2d_256")(torch.randn(1, 256, 256))


def test_unknown_variant():
    with pytest.raises(ValueError, match="Unknown variant"):
        build_model("unet2d_1024")


def test_thin_entry_points():
    a, b = unet2d_256.build_model(), unet2d_512.build_model()
    assert isinstance(a, UNet2D) and isinstance(b, UNet2D)
    assert a.bottleneck_channels == 256 and b.bottleneck_channels == 512
    assert isinstance(a.encoder.stages[0], EncoderStage)
    assert isinstance(a.decoder.stages[0], DecoderStage)


def test_modules_does_not_import_numpy():
    tree = ast.parse((PROJECT_DIR / "modules.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "numpy" not in imported
    assert imported <= {"__future__", "collections", "torch"}


def test_no_softmax_in_model_output():
    """Logits are unnormalised: per-pixel class scores do not sum to 1."""
    torch.manual_seed(0)
    with torch.no_grad():
        out = build_model("unet2d_256").eval()(torch.randn(1, 1, 32, 32))
    assert not torch.allclose(out.sum(dim=1), torch.ones(1, 32, 32))
