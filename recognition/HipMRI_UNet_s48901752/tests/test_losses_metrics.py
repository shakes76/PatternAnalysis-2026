"""Loss (batch-level Dice + CE) and evaluation-metric tests with hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from losses import DiceCrossEntropyLoss, batch_dice_loss, batch_dice_per_class
from metrics import confusion_matrix, dice_from_confusion, global_metrics, per_volume_metrics

EPS = 1e-5


def uniform_probabilities(q, batch: int, h: int = 2, w: int = 2) -> torch.Tensor:
    """Same class-probability vector q at every pixel: [batch, 6, h, w]."""
    q = torch.tensor(q, dtype=torch.float32)
    return q.view(1, 6, 1, 1).expand(batch, 6, h, w).clone()


# Fixed example used by L1, L2 and L5:
#   target (1 x 2 x 2) = [[0, 1], [2, 0]];  classes 3, 4, 5 absent.
#   every pixel has p = Q.
Q = [0.4, 0.3, 0.2, 0.05, 0.04, 0.01]
TARGET = torch.tensor([[[0, 1], [2, 0]]])


def hand_dice(q_c: float, gt_pixels: int, pixels: int = 4) -> float:
    intersection = q_c * gt_pixels
    predicted = q_c * pixels
    return (2 * intersection + EPS) / (predicted + gt_pixels + EPS)


def test_L5_hand_computed_value():
    """D_1 = (2*0.3 + e)/(1.2 + 1 + e), D_2 = (2*0.2 + e)/(0.8 + 1 + e); loss = 1 - mean(D_1, D_2)."""
    probs = uniform_probabilities(Q, batch=1)
    dice, present = batch_dice_per_class(probs, TARGET, 6, EPS)
    assert present.tolist() == [True, True, False, False, False]
    assert dice[0].item() == pytest.approx(hand_dice(0.3, 1), abs=1e-6)
    assert dice[1].item() == pytest.approx(hand_dice(0.2, 1), abs=1e-6)
    expected = 1 - (hand_dice(0.3, 1) + hand_dice(0.2, 1)) / 2
    assert batch_dice_loss(probs, TARGET, 6, EPS).item() == pytest.approx(expected, abs=1e-6)


def test_L1_only_present_classes_are_averaged():
    """Absent classes 3-5 are excluded: result != the 5-class average."""
    probs = uniform_probabilities(Q, batch=1)
    loss = batch_dice_loss(probs, TARGET, 6, EPS).item()
    present_only = 1 - (hand_dice(0.3, 1) + hand_dice(0.2, 1)) / 2
    absent = [EPS / (4 * q + EPS) for q in Q[3:]]  # D_c when G_c = 0
    all_five = 1 - (hand_dice(0.3, 1) + hand_dice(0.2, 1) + sum(absent)) / 5
    assert loss == pytest.approx(present_only, abs=1e-6)
    assert abs(loss - all_five) > 0.01


def test_L2_absent_class_false_positives_change_ce_not_dice():
    """Move background probability to absent class 5: Dice term unchanged, CE larger."""
    q_fp = [0.2, 0.3, 0.2, 0.05, 0.04, 0.21]
    criterion = DiceCrossEntropyLoss()
    base = criterion(torch.log(uniform_probabilities(Q, 1)), TARGET)
    shifted = criterion(torch.log(uniform_probabilities(q_fp, 1)), TARGET)
    assert shifted.dice.item() == pytest.approx(base.dice.item(), abs=1e-6)
    assert shifted.ce.item() > base.ce.item() + 0.1
    # Sanity: CE on log-probabilities equals -mean log p_target.
    expected_ce = -(2 * np.log(0.4) + np.log(0.3) + np.log(0.2)) / 4
    assert base.ce.item() == pytest.approx(expected_ce, abs=1e-5)


def test_L3_presence_is_per_batch_not_per_sample():
    """Class 1 appears only in sample 0; sample 1 is background-only."""
    q1 = 0.3
    probs = uniform_probabilities(Q, batch=2)
    target = torch.zeros(2, 2, 2, dtype=torch.long)
    target[0, 0, 0] = 1
    loss = batch_dice_loss(probs, target, 6, EPS).item()

    batch_level = 1 - (2 * q1 + EPS) / (8 * q1 + 1 + EPS)  # sums over both samples
    per_sample = 1 - (2 * q1 + EPS) / (4 * q1 + 1 + EPS)  # sample 0 only
    assert loss == pytest.approx(batch_level, abs=1e-6)
    assert abs(loss - per_sample) > 0.05


def test_L4_all_background_batch():
    torch.manual_seed(0)
    logits = torch.randn(3, 6, 8, 8, requires_grad=True)
    target = torch.zeros(3, 8, 8, dtype=torch.long)
    terms = DiceCrossEntropyLoss()(logits, target)
    assert terms.dice.item() == 0.0
    assert terms.total.item() == pytest.approx(0.5 * terms.ce.item(), abs=1e-7)
    assert terms.dice.requires_grad
    terms.total.backward()
    assert torch.isfinite(logits.grad).all()


def test_L6_single_pixel_class_is_finite():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 64, 64, requires_grad=True)
    target = torch.zeros(2, 64, 64, dtype=torch.long)
    target[1, 10, 10] = 5
    terms = DiceCrossEntropyLoss()(logits, target)
    assert torch.isfinite(terms.total)
    terms.total.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, 5].abs().sum() > 0


def test_L7_loss_is_fp32_under_autocast():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(1, 6, 1)
    x = torch.randn(2, 1, 16, 16)
    target = torch.randint(0, 6, (2, 16, 16))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = conv(x)
        assert logits.dtype == torch.bfloat16
        terms = DiceCrossEntropyLoss()(logits, target)
    for t in terms:
        assert t.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fp16 autocast check needs a GPU")
def test_L7_loss_is_fp32_under_cuda_autocast():
    conv = torch.nn.Conv2d(1, 6, 1).cuda()
    x = torch.randn(2, 1, 16, 16, device="cuda")
    target = torch.randint(0, 6, (2, 16, 16), device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        terms = DiceCrossEntropyLoss()(conv(x), target)
    assert terms.total.dtype == torch.float32


def test_perfect_prediction_gives_zero_dice_loss():
    target = torch.tensor([[[0, 1, 2], [3, 4, 5]]])
    probs = F.one_hot(target, 6).movedim(-1, 1).float()
    assert batch_dice_loss(probs, target, 6, EPS).item() == pytest.approx(0.0, abs=1e-6)


def test_ce_matches_torch_reference():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 8, 8)
    target = torch.randint(0, 6, (2, 8, 8))
    terms = DiceCrossEntropyLoss()(logits, target)
    assert terms.ce.item() == pytest.approx(F.cross_entropy(logits, target).item(), abs=1e-6)
    assert terms.total.item() == pytest.approx(
        0.5 * terms.dice.item() + 0.5 * terms.ce.item(), abs=1e-6
    )


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def test_confusion_matrix_orientation():
    target = torch.tensor([0, 1, 1, 2])
    prediction = torch.tensor([0, 1, 2, 2])
    matrix = confusion_matrix(target, prediction, 3)
    assert matrix.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]


def test_confusion_matrix_rejects_out_of_range():
    with pytest.raises(ValueError):
        confusion_matrix(torch.tensor([0, 6]), torch.tensor([0, 0]), 6)


def test_M3_global_dice_hand_computed():
    # rows = ground truth, cols = prediction
    matrix = np.zeros((6, 6), dtype=np.int64)
    matrix[0, 0] = 100
    matrix[1, 1], matrix[1, 0], matrix[0, 1] = 8, 2, 2  # D1 = 16 / (16 + 2 + 2) = 0.8
    matrix[2, 2], matrix[2, 3] = 6, 4  # D2 = 12/(12+0+4) = 0.75 ; class 3 FP = 4
    matrix[3, 3] = 4  # D3 = 8/(8+4) = 0.6667
    matrix[4, 4] = 10  # D4 = 1
    matrix[5, 5], matrix[5, 0] = 1, 1  # D5 = 2/3
    metrics = global_metrics(matrix)
    d = metrics["dice_per_class"]
    assert d["bladder"] == pytest.approx(0.8)
    assert d["body"] == pytest.approx(0.75)
    assert d["bone"] == pytest.approx(8 / 12)
    assert d["rectum"] == pytest.approx(1.0)
    assert d["prostate"] == pytest.approx(2 / 3)
    assert metrics["mean_foreground_dice"] == pytest.approx((0.8 + 0.75 + 8 / 12 + 1 + 2 / 3) / 5)
    assert metrics["pixel_accuracy"] == pytest.approx(129 / 138)


def test_M1_absent_in_truth_and_prediction_is_nan_and_counted():
    v1 = np.diag([50, 5, 5, 5, 5, 0])  # prostate absent in GT and prediction
    v2 = np.diag([50, 5, 5, 5, 5, 5])
    assert np.isnan(dice_from_confusion(v1)[5])
    summary = per_volume_metrics([v1, v2])
    assert summary["undefined_counts"]["prostate"] == 1
    assert summary["dice_per_class"]["prostate"]["n"] == 1
    # v1 foreground mean uses its 4 defined classes (all 1.0); both volumes = 1.0
    assert summary["mean_foreground_dice"]["mean"] == pytest.approx(1.0)
    assert summary["mean_foreground_dice"]["n"] == 2


def test_M2_absent_in_truth_but_predicted_is_zero():
    matrix = np.diag([50, 5, 5, 5, 5, 0])
    matrix[0, 5] = 3  # three background voxels predicted as prostate
    assert dice_from_confusion(matrix)[5] == 0.0
    assert global_metrics(matrix)["dice_per_class"]["prostate"] == 0.0


def test_M3_global_includes_false_positives_from_volume_without_class():
    with_class = np.diag([50, 5, 5, 5, 5, 4])
    without_class = np.diag([50, 5, 5, 5, 5, 0])
    without_class[0, 5] = 4
    total = global_metrics(with_class + without_class)
    assert total["dice_per_class"]["prostate"] == pytest.approx(8 / (8 + 4))


def test_per_volume_mean_and_sample_sd():
    def volume(tp1):  # class 1: tp1 correct, 10 - tp1 missed; others perfect
        m = np.diag([100, tp1, 10, 10, 10, 10])
        m[1, 0] = 10 - tp1
        return m

    volumes = [volume(10), volume(5), volume(0)]
    d1 = [1.0, 10 / 15, 0.0]
    summary = per_volume_metrics(volumes)
    assert summary["dice_per_class"]["bladder"]["mean"] == pytest.approx(np.mean(d1))
    assert summary["dice_per_class"]["bladder"]["sd"] == pytest.approx(np.std(d1, ddof=1))
    fg = [(d + 4) / 5 for d in d1]
    assert summary["mean_foreground_dice"]["mean"] == pytest.approx(np.mean(fg))
    assert summary["mean_foreground_dice"]["sd"] == pytest.approx(np.std(fg, ddof=1))
