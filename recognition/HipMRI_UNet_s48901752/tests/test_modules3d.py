"""Architecture tests for both standard 3D U-Net variants (synthetic tensors, CPU only)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from baselines.unet3d import unet3d_256, unet3d_512
from dataset import PatchDataset, PatchSampler, VolumeSet
from losses import DiceCrossEntropyLoss
from modules import (
    ENCODER_CHANNELS,
    UNET2D_VARIANTS,
    UNET3D_VARIANTS,
    DecoderStage3D,
    EncoderStage3D,
    UNet3D,
    build_model,
    build_model3d,
    count_parameters,
)
from sliding_window import sliding_window_logits

VARIANTS = sorted(UNET3D_VARIANTS)


def expected_parameters(bottleneck: int, encoder=(32, 64, 128, 256), in_ch=1, classes=6) -> int:
    """Independent hand count. Conv3d has no bias (BatchNorm3d follows); BN has 2 params/channel."""

    def double_conv(i, o):
        return 27 * i * o + 2 * o + 27 * o * o + 2 * o

    total, prev = 0, in_ch
    for width in encoder:
        total += double_conv(prev, width)
        prev = width
    total += double_conv(prev, bottleneck)
    prev = bottleneck
    for width in reversed(encoder):
        total += 8 * prev * width + width  # ConvTranspose3d 2x2x2 with bias
        total += double_conv(2 * width, width)
        prev = width
    total += prev * classes + classes  # 1x1x1 head with bias
    return total


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("size", [64, 96])
def test_output_shape_and_six_channels(variant, size):
    model = build_model3d(variant).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, size, size, size))
    assert out.shape == (1, 6, size, size, size) and out.dtype == torch.float32


def test_non_cubic_input():
    with torch.no_grad():
        out = build_model3d("unet3d_256").eval()(torch.randn(2, 1, 16, 32, 48))
    assert out.shape == (2, 6, 16, 32, 48)


@pytest.mark.parametrize("variant", VARIANTS)
def test_stage_shapes(variant):
    """Encoder skips, bottleneck and decoder outputs have the agreed widths and resolutions."""
    model = build_model3d(variant).eval()
    b = UNET3D_VARIANTS[variant]
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
        model(torch.randn(1, 1, 64, 64, 64))

    assert shapes["enc0_skip"] == (1, 32, 64, 64, 64)
    assert shapes["enc1_skip"] == (1, 64, 32, 32, 32)
    assert shapes["enc2_skip"] == (1, 128, 16, 16, 16)
    assert shapes["enc3_skip"] == (1, 256, 8, 8, 8)
    assert shapes["bottleneck"] == (1, b, 4, 4, 4)
    assert shapes["dec0"] == (1, 256, 8, 8, 8)
    assert shapes["dec1"] == (1, 128, 16, 16, 16)
    assert shapes["dec2"] == (1, 64, 32, 32, 32)
    assert shapes["dec3"] == (1, 32, 64, 64, 64)


@pytest.mark.parametrize("variant", VARIANTS)
def test_decoder_upsamples_and_concatenates_skip_then_upsampled(variant):
    """Each up-conv doubles D, H and W; its DoubleConv3D receives cat(skip, upsampled)."""
    model = build_model3d(variant).eval()
    skips, up_inputs, ups, conv_inputs = [], [], [], []
    for stage in model.encoder.stages:
        stage.register_forward_hook(lambda m, i, o: skips.append(o[1]))

    def record_up(_module, inputs, output):  # returns None, so the output is not replaced
        up_inputs.append(inputs[0])
        ups.append(output)

    for stage in model.decoder.stages:
        stage.up.register_forward_hook(record_up)
        stage.conv.register_forward_hook(lambda m, i, o: conv_inputs.append(i[0]))
    with torch.no_grad():
        model(torch.randn(1, 1, 32, 32, 32))

    for x, up in zip(up_inputs, ups, strict=True):
        assert up.shape[2:] == tuple(2 * n for n in x.shape[2:])
    for skip, up, conv_in in zip(reversed(skips), ups, conv_inputs, strict=True):
        c = skip.shape[1]
        assert up.shape[1] == c and conv_in.shape[1] == 2 * c
        assert torch.equal(conv_in[:, :c], skip)
        assert torch.equal(conv_in[:, c:], up)


@pytest.mark.parametrize("level", range(4))
def test_each_skip_connection_affects_output(level):
    # Train mode: BatchNorm uses batch statistics. In eval mode an untrained model's default
    # running statistics shrink activations with depth, hiding the deepest skip (~1e-7 change).
    model = build_model3d("unet3d_256").train()
    x = torch.randn(1, 1, 32, 32, 32)
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
    a = build_model3d("unet3d_256").state_dict()
    b = build_model3d("unet3d_512").state_dict()
    assert a.keys() == b.keys()
    differing = {k for k in a if a[k].shape != b[k].shape}
    allowed = {k for k in a if k.startswith("bottleneck.")} | {"decoder.stages.0.up.weight"}
    assert differing and differing <= allowed
    assert a["decoder.stages.0.up.weight"].shape == (256, 256, 2, 2, 2)
    assert b["decoder.stages.0.up.weight"].shape == (512, 256, 2, 2, 2)


def test_same_state_dict_layout_as_2d():
    """3D and 2D models share module names, so the two architectures read the same way."""
    assert (
        build_model3d("unet3d_256").state_dict().keys()
        == build_model("unet2d_256").state_dict().keys()
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_parameter_count_matches_hand_count(variant):
    model = build_model3d(variant)
    assert count_parameters(model) == expected_parameters(UNET3D_VARIANTS[variant])


def test_parameter_counts_values():
    counts = {v: count_parameters(build_model3d(v)) for v in VARIANTS}
    print("parameter counts:", counts)
    assert counts == {"unet3d_256": 14_975_238, "unet3d_512": 22_578_438}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("size", [32, 64])
def test_forward_backward_finite(variant, size):
    torch.manual_seed(0)
    model = build_model3d(variant).train()
    x = torch.randn(1, 1, size, size, size)
    target = torch.randint(0, 6, (1, size, size, size))
    terms = DiceCrossEntropyLoss()(model(x), target)
    assert torch.isfinite(terms.total)
    terms.total.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())


def test_rejects_size_not_divisible_by_16():
    with pytest.raises(ValueError, match="divisible by 16"):
        build_model3d("unet3d_256")(torch.randn(1, 1, 40, 64, 64))


def test_rejects_wrong_rank():
    with pytest.raises(ValueError, match="Expected input"):
        build_model3d("unet3d_256")(torch.randn(1, 64, 64, 64))


def test_unknown_variant_and_registries_are_separate():
    with pytest.raises(ValueError, match="Unknown variant"):
        build_model3d("unet2d_256")
    with pytest.raises(ValueError, match="Unknown variant"):
        build_model("unet3d_256")  # 2D builder (used by train.py / predict.py) is unchanged
    assert not set(UNET2D_VARIANTS) & set(UNET3D_VARIANTS)
    assert UNET2D_VARIANTS == {"unet2d_256": 256, "unet2d_512": 512}


def test_thin_entry_points():
    a, b = unet3d_256.build_model(), unet3d_512.build_model()
    assert isinstance(a, UNet3D) and isinstance(b, UNet3D)
    assert a.bottleneck_channels == 256 and b.bottleneck_channels == 512
    assert isinstance(a.encoder.stages[0], EncoderStage3D)
    assert isinstance(a.decoder.stages[0], DecoderStage3D)
    assert [s.conv.block[0].out_channels for s in a.encoder.stages] == list(ENCODER_CHANNELS)


def test_no_softmax_in_model_output():
    torch.manual_seed(0)
    with torch.no_grad():
        out = build_model3d("unet3d_256").eval()(torch.randn(1, 1, 16, 16, 16))
    assert not torch.allclose(out.sum(dim=1), torch.ones(1, 16, 16, 16))


# -----------------------------------------------------------------------------
# Fit with the existing 3D pipeline (no training)
# -----------------------------------------------------------------------------
def test_patch_batches_feed_the_model_and_loss():
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 6, size=(80, 72, 70), dtype=np.uint8)  # [H, W, D]
    image = rng.standard_normal(mask.shape).astype(np.float32)
    volumes = VolumeSet(["S0"], ["P0"], [image], [mask])
    plan = PatchSampler(volumes, 64, "class_balanced", seed=3710, fg_prob=0.5).plan(1, 2)
    images, masks = next(iter(DataLoader(PatchDataset(volumes, plan, 64), batch_size=2)))
    model = build_model3d("unet3d_256").train()
    logits = model(images)
    assert logits.shape[2:] == masks.shape[1:] and logits.shape[:2] == (2, 6)
    assert torch.isfinite(DiceCrossEntropyLoss()(logits, masks).total)


def test_sliding_window_inference_with_unet3d():
    """Volume with depth < patch (padded) and sides that are not multiples of 16."""
    torch.manual_seed(0)
    model = build_model3d("unet3d_256").eval()
    image = torch.randn(1, 40, 80, 70)
    logits = sliding_window_logits(model, image, (64, 64, 64), num_classes=6, batch_size=2)
    assert logits.shape == (6, 40, 80, 70) and torch.isfinite(logits).all()
