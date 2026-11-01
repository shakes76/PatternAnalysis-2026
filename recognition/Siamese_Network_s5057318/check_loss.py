"""Check ContrastiveLoss against hand-calculated values and gradient behaviour."""
import torch

from modules import ContrastiveLoss

failures = 0


def report(name, ok, detail=""):
    """Print PASS or FAIL for one check and count the failures."""
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}")


def loss_value(d, y, margin=1.0):
    """ContrastiveLoss value for distances `d` and labels `y` given as lists."""
    return ContrastiveLoss(margin).loss_from_distance(torch.tensor(d), torch.tensor(y)).item()


cases = [
    ("d=0.3, y=0", [0.3], [0.0], 0.045),
    ("d=0.3, y=1, margin=1", [0.3], [1.0], 0.245),
    ("d=0.8, y=1, margin=1", [0.8], [1.0], 0.020),
    ("d=1.5, y=1, margin=1", [1.5], [1.0], 0.0),
    ("d=0.0, y=0", [0.0], [0.0], 0.0),
    ("batch of d=0.3 y=0 and d=0.3 y=1", [0.3, 0.3], [0.0, 1.0], 0.145),
]
for name, d, y, expected in cases:
    got = loss_value(d, y)
    report(name, abs(got - expected) < 1e-6, f"-> {got:.6f} (expected {expected})")


def grad_of(d, y):
    """Gradient of the loss with respect to one distance `d` with label `y`."""
    d = torch.tensor([d], requires_grad=True)
    ContrastiveLoss(1.0).loss_from_distance(d, torch.tensor([y])).backward()
    return d.grad.item()


g = grad_of(0.3, 0.0)
report("gradient non-zero: similar pair, d>0", g != 0, f"(grad={g:.4f})")
g = grad_of(0.3, 1.0)
report("gradient non-zero: dissimilar pair, d<margin", g != 0, f"(grad={g:.4f})")
g = grad_of(1.5, 1.0)
report("gradient zero: dissimilar pair, d>margin", g == 0, f"(grad={g:.4f})")

# End-to-end through forward() with embeddings: distance is |a-b| = 0.3 here.
a, b = torch.zeros(1, 4, requires_grad=True), torch.tensor([[0.3, 0.0, 0.0, 0.0]])
val = ContrastiveLoss()(a, b, torch.tensor([0.0])).item()
report("forward(emb1, emb2, y) matches d=0.3, y=0", abs(val - 0.045) < 1e-5, f"-> {val:.6f}")

print("\nALL PASSED" if not failures else f"\n{failures} FAILED")
raise SystemExit(1 if failures else 0)
