"""
test_pixelcnn.py - Checks for the PixelCNN prior in modules.py.

Run from the project folder:
    python test_pixelcnn.py
The key property: the prediction at a position must NOT depend on that position
or any later position (raster order), or the prior "cheats" during training and
fails at sampling time.
"""
import torch
import torch.nn.functional as F

from modules import MaskedConv2d, PixelCNNPrior

torch.manual_seed(0)
K, B, H, W = 16, 2, 8, 6  # small sizes so the test is fast

# 1. Mask patterns (5x5): rows below the centre are hidden, as is everything right of it;
#    type A also hides the centre itself, type B keeps it
mA = MaskedConv2d("A", 1, 1, 5).mask[0, 0]
mB = MaskedConv2d("B", 1, 1, 5).mask[0, 0]
assert mA[2, 2] == 0 and mB[2, 2] == 1, "centre: A must hide it, B must keep it"
assert mA[3:].sum() == 0 and mB[3:].sum() == 0, "rows below the centre must be hidden"
assert mA[2, 3:].sum() == 0 and mB[2, 3:].sum() == 0, "right of the centre must be hidden"
assert mA[:2].sum() == 10 and mA[2, :2].sum() == 2, "rows above and left of the centre must be visible"
print("1. Mask patterns correct:\n   type A:", mA.int().tolist(), "\n   type B:", mB.int().tolist())

# 2. Causality: changing the current and all later positions must not change the prediction
model = PixelCNNPrior(num_codes=K, channels=32, n_layers=4, kernel_size=5)
ids = torch.randint(0, K, (B, H, W))
logits = model(ids)
assert logits.shape == (B, K, H, W), f"logits shape {tuple(logits.shape)}"
for (i, j) in [(0, 0), (2, 3), (4, 0), (7, 5)]:
    p = i * W + j
    ids_future = ids.clone().view(B, -1)
    ids_future[:, p:] = (ids_future[:, p:] + 1) % K   # change this and every later position
    out = model(ids_future.view(B, H, W))
    assert torch.allclose(out[:, :, i, j], logits[:, :, i, j], atol=1e-6), \
        f"position {(i, j)} sees itself or the future: masks are leaking"
print("2. Causality holds: predictions ignore the current and later positions")

# 3. ...and the test is not vacuous: changing the position just LEFT of (i, j) does matter
i, j = 3, 3
ids_past = ids.clone()
ids_past[:, i, j - 1] = (ids_past[:, i, j - 1] + 1) % K
diff = (model(ids_past)[:, :, i, j] - logits[:, :, i, j]).abs().max().item()
assert diff > 1e-6, "prediction ignores the previous position: the model sees nothing"
print(f"3. Past positions do influence predictions (max change {diff:.2e})")

# 4. Conditional prior: bottom grid (H, W) conditioned on a top grid (H/2, W/2)
cmodel = PixelCNNPrior(num_codes=K, channels=32, n_layers=4, kernel_size=5, cond_num_codes=K)
top = torch.randint(0, K, (B, H // 2, W // 2))
clogits = cmodel(ids, top)
assert clogits.shape == (B, K, H, W), f"conditional logits shape {tuple(clogits.shape)}"
other_top = (top + 1) % K
assert not torch.allclose(cmodel(ids, other_top), clogits), "condition is ignored"
print("4. Conditional prior: correct shape, and the top grid changes the prediction")

# 5. Loss and gradients
loss = F.cross_entropy(clogits, ids)
loss.backward()
assert cmodel.embed.weight.grad is not None and cmodel.cond_embed.weight.grad is not None
print(f"5. Cross-entropy loss {loss.item():.3f} (about ln K = {torch.log(torch.tensor(float(K))):.3f}"
      f" when untrained); gradients reach both embeddings")

# 6. Sampling returns valid ids of the right shape
s = cmodel.sample(2, (H, W), torch.device("cpu"), cond_ids=top[:2])
assert s.shape == (2, H, W) and s.min() >= 0 and s.max() < K, "bad sample shape or range"
print(f"6. Sampling ok: shape {tuple(s.shape)}, ids in [0, {K})")

print("All PixelCNN tests passed.")