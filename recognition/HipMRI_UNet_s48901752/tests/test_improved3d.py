"""Tests of the improved residual 3D U-Net and its train3d.py integration (CPU only)."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

import dataset
import train3d
from improved.unet3d import unet3d_256 as res_256
from improved.unet3d import unet3d_512 as res_512
from improved.unet3d.residual_blocks import (
    RESUNET3D_VARIANTS,
    ResidualBlock3D,
    ResUNet3D,
    build_resunet3d,
)
from losses import DiceCrossEntropyLoss
from modules import UNET3D_VARIANTS, DoubleConv3D, UNet3D, build_model3d, count_parameters

VARIANTS = sorted(RESUNET3D_VARIANTS)
SYNTHETIC_COUNTS = {"train": 3, "val": 3, "test": 3}


def conv_blocks(model):
    """The nine convolution blocks: four encoder stages, bottleneck, four decoder stages."""
    return [s.conv for s in (*model.encoder.stages, model.bottleneck, *model.decoder.stages)]


def standard_parameters(bottleneck: int, encoder=(32, 64, 128, 256), in_ch=1, classes=6) -> int:
    """Hand count for the standard UNet3D (same formula as test_modules3d)."""

    def double_conv(i, o):
        return 27 * i * o + 2 * o + 27 * o * o + 2 * o

    total, prev = 0, in_ch
    for width in encoder:
        total += double_conv(prev, width)
        prev = width
    total += double_conv(prev, bottleneck)
    prev = bottleneck
    for width in reversed(encoder):
        total += 8 * prev * width + width + double_conv(2 * width, width)
        prev = width
    return total + prev * classes + classes


def projection_parameters(bottleneck: int, encoder=(32, 64, 128, 256), in_ch=1) -> int:
    """1x1x1 projection conv (no bias) + BatchNorm for every block whose width changes."""
    widths = [in_ch, *encoder]
    pairs = list(zip(widths[:-1], widths[1:], strict=True))
    pairs.append((encoder[-1], bottleneck))
    pairs += [(2 * w, w) for w in reversed(encoder)]
    return sum(i * o + 2 * o for i, o in pairs if i != o)


# -----------------------------------------------------------------------------
# Residual block
# -----------------------------------------------------------------------------
def test_identity_shortcut_when_widths_match():
    block = ResidualBlock3D(8, 8)
    assert isinstance(block.shortcut, nn.Identity)
    assert not list(block.shortcut.parameters())


def test_projection_shortcut_when_widths_differ():
    block = ResidualBlock3D(4, 8)
    conv, norm = block.shortcut
    assert isinstance(conv, nn.Conv3d) and conv.kernel_size == (1, 1, 1) and conv.bias is None
    assert isinstance(norm, nn.BatchNorm3d) and norm.num_features == 8


@pytest.mark.parametrize("widths", [(8, 8), (4, 8)])
def test_output_is_relu_of_residual_plus_shortcut(widths):
    torch.manual_seed(0)
    block = ResidualBlock3D(*widths).eval()
    x = torch.randn(2, widths[0], 6, 8, 8)
    with torch.no_grad():
        expected = torch.relu(block.residual(x) + block.shortcut(x))
        assert torch.allclose(block(x), expected)


def test_zero_residual_branch_passes_input_through_identity():
    """With F(x) = 0 the identity block reduces to ReLU(x): the shortcut really is added."""
    block = ResidualBlock3D(8, 8).eval()
    nn.init.zeros_(block.residual[-1].weight)
    nn.init.zeros_(block.residual[-1].bias)
    x = torch.randn(1, 8, 4, 4, 4)
    with torch.no_grad():
        assert torch.equal(block(x), torch.relu(x))


def test_unet_uses_identity_only_where_widths_match():
    kinds = {
        v: [type(b.shortcut).__name__ for b in conv_blocks(build_resunet3d(v))] for v in VARIANTS
    }
    # Encoder 1->32->64->128->256, bottleneck 256->B, decoder 2w->w: only 256->256 is identity.
    assert kinds["resunet3d_256"][4] == "Identity"
    assert kinds["resunet3d_256"].count("Identity") == 1
    assert kinds["resunet3d_512"].count("Identity") == 0


# -----------------------------------------------------------------------------
# Residual 3D U-Net
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("size", [64, 96])
def test_output_shape_and_six_channels(variant, size):
    model = build_resunet3d(variant).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, size, size, size))
    assert out.shape == (1, 6, size, size, size) and out.dtype == torch.float32


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_block_is_residual_with_standard_widths(variant):
    """Same (in, out) widths as the standard model; no DoubleConv3D left."""
    model = build_resunet3d(variant)
    standard = build_model3d(variant.removeprefix("res"))
    assert isinstance(model, ResUNet3D) and isinstance(model, UNet3D)
    assert not any(isinstance(m, DoubleConv3D) for m in model.modules())
    for block, reference in zip(conv_blocks(model), conv_blocks(standard), strict=True):
        assert isinstance(block, ResidualBlock3D)
        assert block.residual[0].in_channels == reference.block[0].in_channels
        assert block.residual[3].out_channels == reference.block[3].out_channels


@pytest.mark.parametrize("variant", VARIANTS)
def test_everything_outside_the_blocks_matches_the_standard_model(variant):
    """Up-convolutions and head have identical names and shapes to the standard UNet3D."""
    improved = build_resunet3d(variant).state_dict()
    standard = build_model3d(variant.removeprefix("res")).state_dict()

    def outside(state):
        return {k: v.shape for k, v in state.items() if ".conv." not in k}

    assert outside(improved) == outside(standard)
    assert set(outside(improved)) >= {"head.weight", "decoder.stages.0.up.weight"}


@pytest.mark.parametrize("variant", VARIANTS)
def test_stage_shapes(variant):
    model = build_resunet3d(variant).eval()
    b = RESUNET3D_VARIANTS[variant]
    shapes = {}
    for i, stage in enumerate(model.encoder.stages):
        stage.register_forward_hook(lambda m, i_, o, k=i: shapes.__setitem__(f"enc{k}", o[1].shape))
    model.bottleneck.register_forward_hook(lambda m, i_, o: shapes.__setitem__("bott", o.shape))
    for i, stage in enumerate(model.decoder.stages):
        stage.register_forward_hook(lambda m, i_, o, k=i: shapes.__setitem__(f"dec{k}", o.shape))
    with torch.no_grad():
        model(torch.randn(1, 1, 32, 32, 32))
    assert [tuple(shapes[f"enc{k}"]) for k in range(4)] == [
        (1, 32, 32, 32, 32),
        (1, 64, 16, 16, 16),
        (1, 128, 8, 8, 8),
        (1, 256, 4, 4, 4),
    ]
    assert tuple(shapes["bott"]) == (1, b, 2, 2, 2)
    assert [tuple(shapes[f"dec{k}"]) for k in range(4)] == [
        (1, 256, 4, 4, 4),
        (1, 128, 8, 8, 8),
        (1, 64, 16, 16, 16),
        (1, 32, 32, 32, 32),
    ]


def test_long_skip_connections_concatenate_skip_then_upsampled():
    model = build_resunet3d("resunet3d_256").eval()
    skips, ups, conv_inputs = [], [], []

    def record_skip(_m, _i, output):
        skips.append(output[1])

    def record_up(_m, _i, output):
        ups.append(output)

    def record_conv_input(_m, inputs, _o):
        conv_inputs.append(inputs[0])

    for stage in model.encoder.stages:
        stage.register_forward_hook(record_skip)
    for stage in model.decoder.stages:
        stage.up.register_forward_hook(record_up)
        stage.conv.register_forward_hook(record_conv_input)
    with torch.no_grad():
        model(torch.randn(1, 1, 32, 32, 32))
    for skip, up, conv_in in zip(reversed(skips), ups, conv_inputs, strict=True):
        c = skip.shape[1]
        assert torch.equal(conv_in[:, :c], skip) and torch.equal(conv_in[:, c:], up)


def test_variants_differ_only_in_bottleneck():
    a = build_resunet3d("resunet3d_256").state_dict()
    b = build_resunet3d("resunet3d_512").state_dict()
    # The 512 bottleneck has a projection shortcut; the 256 one is an identity.
    extra = set(b) - set(a)
    assert extra and all(k.startswith("bottleneck.conv.shortcut.") for k in extra)
    assert set(a) <= set(b)
    differing = {k for k in a if a[k].shape != b[k].shape}
    allowed = {k for k in a if k.startswith("bottleneck.")} | {"decoder.stages.0.up.weight"}
    assert differing and differing <= allowed


@pytest.mark.parametrize("variant", VARIANTS)
def test_parameter_count_matches_hand_count(variant):
    b = RESUNET3D_VARIANTS[variant]
    expected = standard_parameters(b) + projection_parameters(b)
    assert count_parameters(build_resunet3d(variant)) == expected


def test_parameter_counts_values():
    counts = {v: count_parameters(build_resunet3d(v)) for v in VARIANTS}
    standard = {v: count_parameters(build_model3d(v)) for v in sorted(UNET3D_VARIANTS)}
    print("residual:", counts, "standard:", standard)
    assert counts == {"resunet3d_256": 15_194_278, "resunet3d_512": 22_929_574}
    assert standard == {"unet3d_256": 14_975_238, "unet3d_512": 22_578_438}


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_backward_finite(variant):
    torch.manual_seed(0)
    model = build_resunet3d(variant).train()
    x = torch.randn(2, 1, 32, 32, 32)
    target = torch.randint(0, 6, (2, 32, 32, 32))
    terms = DiceCrossEntropyLoss()(model(x), target)
    assert torch.isfinite(terms.total)
    terms.total.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())


def test_inherited_input_checks_and_raw_logits():
    model = build_resunet3d("resunet3d_256").eval()
    with pytest.raises(ValueError, match="divisible by 16"):
        model(torch.randn(1, 1, 40, 64, 64))
    with pytest.raises(ValueError, match="Expected input"):
        model(torch.randn(1, 32, 32, 32))
    with torch.no_grad():
        out = model(torch.randn(1, 1, 16, 16, 16))
    assert not torch.allclose(out.sum(dim=1), torch.ones(1, 16, 16, 16))


def test_unknown_variant_and_thin_entry_points():
    with pytest.raises(ValueError, match="Unknown variant"):
        build_resunet3d("unet3d_256")
    a, b = res_256.build_model(), res_512.build_model()
    assert isinstance(a, ResUNet3D) and isinstance(b, ResUNet3D)
    assert a.bottleneck_channels == 256 and b.bottleneck_channels == 512


# -----------------------------------------------------------------------------
# train3d.py integration and the controlled comparison
# -----------------------------------------------------------------------------
def test_standard_variants_build_exactly_as_before():
    """build_variant gives the same standard UNet3D (type and initial weights) as build_model3d."""
    for variant in sorted(UNET3D_VARIANTS):
        torch.manual_seed(3710)
        via_train3d = train3d.build_variant(variant)
        torch.manual_seed(3710)
        direct = build_model3d(variant)
        assert type(via_train3d) is UNet3D
        a, b = via_train3d.state_dict(), direct.state_dict()
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert type(train3d.build_variant("resunet3d_512")) is ResUNet3D
    assert train3d.MODEL_VARIANTS == {
        "unet3d_256": 256,
        "unet3d_512": 512,
        "resunet3d_256": 256,
        "resunet3d_512": 512,
    }


def test_same_cli_settings_apart_from_model_and_out_dir():
    common = ["--metadata", "m.csv"]
    standard = vars(train3d.parse_args(["--model", "unet3d_256", "--out-dir", "a", *common]))
    improved = vars(train3d.parse_args(["--model", "resunet3d_256", "--out-dir", "b", *common]))
    differing = {k for k in standard if standard[k] != improved[k]}
    assert differing == {"model", "out_dir"}
    assert improved["patch_size"] == 96 and improved["sampling"] == "class_balanced"
    assert improved["fg_prob"] == 0.33 and improved["seed"] == 3710


def _train(manifest, out_dir, model):
    args = train3d.parse_args(
        [
            "--model",
            model,
            "--metadata",
            str(manifest),
            "--out-dir",
            str(out_dir),
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--patch-size",
            "16",
            "--patches-per-epoch",
            "4",
            "--device",
            "cpu",
        ]
    )
    return train3d.run_training(args, SYNTHETIC_COUNTS, expected_sha256=None)


def test_residual_training_smoke_sees_the_same_patches_as_standard(
    synthetic_manifest, tmp_path, monkeypatch
):
    """Both models train on identical patch plans: only the architecture differs."""
    plans = {"unet3d_256": [], "resunet3d_256": []}
    current = []
    original_plan = dataset.PatchSampler.plan

    def recording_plan(self, epoch, count):
        result = original_plan(self, epoch, count)
        plans[current[0]].append(result)
        return result

    monkeypatch.setattr(dataset.PatchSampler, "plan", recording_plan)
    summaries = {}
    for model in plans:
        current[:] = [model]
        summaries[model] = _train(synthetic_manifest, tmp_path / model, model)
    assert plans["unet3d_256"] == plans["resunet3d_256"] and len(plans["unet3d_256"]) == 2

    out_dir = tmp_path / "resunet3d_256"
    saved = json.loads((out_dir / "summary.json").read_text())
    assert saved["variant"] == "resunet3d_256" and saved["trainable_parameters"] == 15_194_278
    assert saved["best_metrics"]["per_volume"]["volumes"] == 6
    checkpoint = torch.load(out_dir / "best.pt", weights_only=False)
    assert checkpoint["variant"] == "resunet3d_256" and checkpoint["bottleneck_channels"] == 256
    model = ResUNet3D(256)
    model.load_state_dict(checkpoint["model_state"])
    assert summaries["unet3d_256"]["trainable_parameters"] == 14_975_238
