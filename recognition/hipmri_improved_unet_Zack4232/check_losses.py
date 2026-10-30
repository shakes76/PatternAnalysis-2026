"""Verify combined-loss values, edge cases, and gradients."""

import math

import torch

from losses import CrossEntropyDiceLoss


def main():
    criterion = CrossEntropyDiceLoss(dice_weight=1.0)

    # Uniform probabilities: CE = log(6), soft Dice = 2/7.
    target = torch.full((1, 2, 2), 5, dtype=torch.int64)
    logits = torch.zeros((1, 6, 2, 2), requires_grad=True)
    ce, dice_loss = criterion.components(logits, target)

    assert math.isclose(ce.item(), math.log(6), abs_tol=1e-6)
    assert math.isclose(dice_loss.item(), 5 / 7, abs_tol=1e-6)

    loss = criterion(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert (logits.grad[:, 5] < 0).all()

    print("Uniform CE:", ce.item())
    print("Uniform soft Dice loss:", dice_loss.item())
    print("Gradient direction for the true class: PASS")

    perfect = torch.full((1, 6, 2, 2), -15.0)
    perfect[:, 5] = 15.0
    wrong = torch.full((1, 6, 2, 2), -15.0)
    wrong[:, 0] = 15.0

    perfect_loss = criterion(perfect, target).item()
    wrong_loss = criterion(wrong, target).item()
    assert perfect_loss < 1e-5
    assert wrong_loss > perfect_loss + 1.0
    print("Near-perfect loss:", perfect_loss)
    print("Wrong-class loss:", wrong_loss)

    background = torch.zeros((1, 2, 2), dtype=torch.int64)
    background_logits = torch.zeros(
        (1, 6, 2, 2), requires_grad=True
    )
    bg_ce, bg_dice = criterion.components(background_logits, background)
    assert bg_dice.item() == 0.0
    bg_loss = criterion(background_logits, background)
    assert torch.allclose(bg_loss, bg_ce)
    bg_loss.backward()
    assert torch.isfinite(background_logits.grad).all()
    assert background_logits.grad.abs().sum() > 0
    print("Background-only batch: PASS")

    # Adding a background-only slice must not dilute the positive-pair mean.
    mixed_target = torch.cat((target, background), dim=0)
    mixed_logits = torch.zeros((2, 6, 2, 2))
    _, mixed_dice = criterion.components(mixed_logits, mixed_target)
    assert torch.allclose(mixed_dice, dice_loss.detach())
    print("Absent-class exclusion: PASS")

    print("Combined-loss checks: PASS")


if __name__ == "__main__":
    main()
