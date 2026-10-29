import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataset_v4 import prepare_dataset_v4

SEED = 371027
EPOCHS = 12
BATCH = 2048
LR = 1e-3
THRESHOLD = 1e-5

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

out = Path("outputs_v5_27_state_depth")
out.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

def extract(x):
    return np.maximum(
        np.expm1(x[..., 1:40:2]), 0
    )

train = extract(norm.inverse_transform(
    np.asarray(data["train"], dtype=np.float64)
))

val = extract(norm.inverse_transform(
    np.asarray(data["val"][:512], dtype=np.float64)
))

# Log-volume state representation.
train_log = np.log1p(train)
val_log = np.log1p(val)

def transitions(x):
    current = x[:, :-1, :]
    future = x[:, 1:, :]

    delta = np.abs(
        np.expm1(future) - np.expm1(current)
    )
    mask = (delta >= THRESHOLD).astype(np.float32)

    return (
        current.reshape(-1, 20).astype(np.float32),
        future.reshape(-1, 20).astype(np.float32),
        mask.reshape(-1, 20),
    )

xtr, ytr, mtr = transitions(train_log)
xva, yva, mva = transitions(val_log)

# Standardise using TRAIN only.
mu = xtr.mean(axis=0)
sd = xtr.std(axis=0) + 1e-6

xtr_n = (xtr - mu) / sd
xva_n = (xva - mu) / sd

class StateDepthNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.mask_head = nn.Linear(128, 20)
        self.value_head = nn.Linear(128, 20)

    def forward(self, x):
        hidden = self.encoder(x)
        return (
            self.mask_head(hidden),
            self.value_head(hidden)
        )

model = StateDepthNet()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LR
)

X = torch.from_numpy(xtr_n)
Y = torch.from_numpy(ytr)
M = torch.from_numpy(mtr)

XV = torch.from_numpy(xva_n)
YV = torch.from_numpy(yva)
MV = torch.from_numpy(mva)

def objective(logits, predicted, target, mask):
    # Training-only level frequencies.
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, mask
    )

    # Learn post-update values only at updated levels.
    regression = (
        nn.functional.smooth_l1_loss(
            predicted, target,
            reduction="none"
        ) * mask
    ).sum() / mask.sum().clamp_min(1)

    return bce + 0.5 * regression, bce, regression

best_val = float("inf")
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()

    order = torch.randperm(len(X))
    losses = []

    for ids in order.split(BATCH):
        logits, predicted = model(X[ids])

        loss, _, _ = objective(
            logits, predicted,
            Y[ids], M[ids]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        logits, predicted = model(XV)
        val_loss, val_bce, val_reg = objective(
            logits, predicted,
            YV, MV
        )

    record = {
        "epoch": epoch,
        "train_loss": float(np.mean(losses)),
        "val_loss": float(val_loss),
        "val_bce": float(val_bce),
        "val_regression": float(val_reg),
    }

    history.append(record)

    print(
        f"Epoch {epoch:02d}"
        f" | train={record['train_loss']:.4f}"
        f" | val={record['val_loss']:.4f}"
        f" | mask={record['val_bce']:.4f}"
        f" | value={record['val_regression']:.4f}"
    )

    if record["val_loss"] < best_val:
        best_val = record["val_loss"]

        torch.save({
            "model_state_dict": model.state_dict(),
            "mu": mu,
            "sd": sd,
            "epoch": epoch,
        }, out / "best_depth_state.pt")

with open(out / "history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n=== V5.27 TRAINING COMPLETE ===")
print("Best validation loss:", best_val)
print("Saved:", out / "best_depth_state.pt")
