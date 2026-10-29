"""Quick self-checks (no GPU needed; reads only results/split.csv).

Run from this folder:  python test_pipeline.py     (pytest would also work)
Checks: the patient-level split has no leakage; random_flip keeps image and label
aligned; every model returns logits of the right shape for 2D and 3D inputs; the
Dice loss is ~0 for a perfect prediction and ~1 for a wrong one.
"""
import torch
import torch.nn.functional as F

from dataset import check_no_leakage, load_split, random_flip
from modules import ImprovedUNet, UNet2D
from train import NUM_CLASSES, dice_loss


def test_split_has_no_patient_leakage():
    """The saved split puts every patient in exactly one of train, val and test."""
    # raises AssertionError if any patient appears in more than one split
    check_no_leakage(load_split())


def test_random_flip_keeps_image_and_label_aligned():
    """After random_flip an image that equals its label must still equal it."""
    torch.manual_seed(0)
    y = torch.randint(0, NUM_CLASSES, (32, 16, 16))
    x = y.float()[:, None]                      # an "image" that equals its label
    x2, y2 = random_flip(x, y)
    assert torch.equal(x2[:, 0].long(), y2)     # still identical after the flip
    assert not torch.equal(y2, y)               # and some samples were flipped


def test_model_output_shapes():
    """Every model returns one logit map per class at the input's spatial size."""
    # (model, input shape): a small base width and image size keep the check fast on a CPU
    cases = [
        (UNet2D(base=4), (2, 1, 32, 32)),
        (ImprovedUNet(dim=2, base=4), (2, 1, 32, 32)),
        (ImprovedUNet(dim=3, base=4), (1, 1, 16, 32, 32)),
    ]
    for model, shape in cases:
        with torch.no_grad():
            out = model(torch.randn(*shape))
        assert out.shape == (shape[0], NUM_CLASSES, *shape[2:]), out.shape


def test_dice_loss_small_for_perfect_and_large_for_wrong_prediction():
    """Dice loss is near 0 for a perfect prediction and near 1 for a completely wrong one."""
    target = torch.randint(0, NUM_CLASSES, (2, 16, 16))
    # logits of 100 on the right class (perfect) or on a shifted class (wrong everywhere)
    perfect = F.one_hot(target, NUM_CLASSES).movedim(-1, 1).float() * 100
    wrong = F.one_hot((target + 1) % NUM_CLASSES, NUM_CLASSES).movedim(-1, 1).float() * 100
    assert dice_loss(perfect, target) < 0.01
    assert dice_loss(wrong, target) > 0.9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  ", name)
