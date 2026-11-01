"""
test_quantizer.py - Checks for VectorQuantizer in modules.py.

Run from the project folder:
    python test_quantizer.py
Every check uses assert, so the first failure stops with a message saying what broke.
"""
import torch
import torch.nn.functional as F

from modules import VectorQuantizer

torch.manual_seed(0)
B, D, H, W, K = 2, 64, 8, 4, 512

vq = VectorQuantizer(num_codes=K, dim=D, beta=0.25)
z = torch.randn(B, D, H, W, requires_grad=True)
q, loss, ids = vq(z)
codebook = vq.embedding.weight

# 1. Shapes
assert q.shape == z.shape, f"q shape {tuple(q.shape)} != z shape {tuple(z.shape)}"
assert ids.shape == (B, H, W), f"ids shape {tuple(ids.shape)} != {(B, H, W)}"
print(f"1. Shapes ok: q {tuple(q.shape)}, ids {tuple(ids.shape)}")

# 2. Each position really got its nearest code (slow brute-force loop as the reference)
for b, h, w in [(0, 0, 0), (0, 7, 3), (1, 3, 2), (1, 5, 0)]:
    vec = z[b, :, h, w].detach()
    dists = [torch.sum((vec - codebook[k].detach()) ** 2).item() for k in range(K)]
    nearest = min(range(K), key=lambda k: dists[k])
    assert ids[b, h, w].item() == nearest, \
        f"position {(b, h, w)}: ids says {ids[b, h, w].item()}, brute force says {nearest}"
print("2. Nearest-code lookup matches brute force at 4 positions")

# 3. The output holds the chosen code vectors (straight-through keeps z_q's value)
expected = codebook[ids].permute(0, 3, 1, 2).detach()  # (B,H,W,D) -> (B,D,H,W)
assert torch.allclose(q, expected, atol=1e-5), "q does not equal the looked-up code vectors"
print("3. Output values equal the chosen code vectors")

# 4. Loss value: both terms are the same MSE in value, so loss = (1 + beta) * MSE
mse = F.mse_loss(expected, z.detach())
assert torch.allclose(loss, (1 + vq.beta) * mse, rtol=1e-4), \
    f"loss {loss.item():.6f} != (1 + beta) * mse = {(1 + vq.beta) * mse.item():.6f}"
print(f"4. Loss value ok: {loss.item():.6f}")

# 5. Gradients reach both the encoder side (z) and the codebook
(q.sum() + loss).backward()
assert z.grad is not None and z.grad.abs().sum() > 0, \
    "no gradient reached z: check the straight-through line"
assert codebook.grad is not None and codebook.grad.abs().sum() > 0, \
    "no gradient reached the codebook: check the codebook loss"
print("5. Gradients reach z and the codebook")

# 6. Beta is on the commitment term (paper eq. 3): from the loss alone,
#    d(loss)/dz must equal beta * d(MSE)/dz = beta * 2 * (z - z_q) / numel
z2 = z.detach().clone().requires_grad_(True)
vq.zero_grad()
_, loss2, _ = vq(z2)
loss2.backward()
expected_grad = vq.beta * 2 * (z2.detach() - expected) / z2.numel()
assert torch.allclose(z2.grad, expected_grad, atol=1e-7), \
    "gradient on z from the loss is not beta-scaled: beta may be on the wrong term"
print("6. Beta is on the commitment term, as in the paper")

# 7. embed_code turns ids back into the same vectors the forward pass chose
assert torch.allclose(vq.embed_code(ids), q.detach(), atol=1e-5), "embed_code does not match forward"
print("7. embed_code reproduces the quantised output")

# 8. Dead-code reset moves unused codes onto encoder outputs
vq3 = VectorQuantizer(num_codes=K, dim=D, beta=0.25, reset_every=1)
z3 = torch.randn(B, D, H, W)
_, _, ids3 = vq3(z3)                       # one training step triggers the reset
dead = torch.ones(K, dtype=torch.bool)
dead[ids3.unique()] = False
rows = z3.permute(0, 2, 3, 1).reshape(-1, D)
moved = torch.cdist(vq3.embedding.weight[dead].detach(), rows, compute_mode="donot_use_mm_for_euclid_dist").min(dim=1).values
assert torch.allclose(moved, torch.zeros_like(moved), atol=1e-5), "dead codes were not moved onto encoder outputs"
print(f"8. Dead-code reset moved {int(dead.sum())} unused codes onto encoder outputs")

print("All VQ tests passed.")