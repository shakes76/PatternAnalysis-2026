import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
)
from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42

torch.manual_seed(371017)
np.random.seed(371017)
torch.set_num_threads(4)

data = prepare_dataset_v4("data")
norm = data["normalizer"]

train_x = np.asarray(data["train"], dtype=np.float32)
val_x = np.asarray(data["val"], dtype=np.float32)

train_raw = norm.inverse_transform(
    train_x.astype(np.float64)
)
val_raw = norm.inverse_transform(
    val_x.astype(np.float64)
)

ck = torch.load(
    "outputs_v5_10_latent_matching/timegan_finetuned.pt",
    map_location="cpu",
    weights_only=True,
)

model = TimeGANV42(
    feature_dim=ck["feature_dim"],
    hidden_dim=ck["hidden_dim"],
    noise_dim=ck["noise_dim"],
    num_layers=ck["num_layers"],
)
model.load_state_dict(ck["model_state_dict"])
model.eval()

def extract_hidden(x, raw, batch_size=128):
    all_hidden = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(
                x[start:start+batch_size],
                dtype=torch.float32
            )
            h = model.embedder(batch)
            all_hidden.append(h.cpu().numpy())

    hidden = np.concatenate(all_hidden, axis=0)

    # Predict the NEXT event's nonzero return.
    features = hidden[:, :-1, :].reshape(
        -1, hidden.shape[-1]
    )

    target = (
        np.abs(raw[:, 1:, 40]) > 1e-7
    ).astype(np.float32).reshape(-1)

    return features, target


X_train, y_train = extract_hidden(train_x, train_raw)
X_val, y_val = extract_hidden(val_x, val_raw)

# Chronological train and validation splits already
# come from the data preparation pipeline.
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0) + 1e-6

X_train = (X_train - mu) / sd
X_val = (X_val - mu) / sd

head = nn.Linear(X_train.shape[1], 1)
optimizer = torch.optim.Adam(
    head.parameters(), lr=1e-3
)

xtr = torch.tensor(X_train, dtype=torch.float32)
ytr = torch.tensor(y_train, dtype=torch.float32)
xva = torch.tensor(X_val, dtype=torch.float32)

positive = float(y_train.mean())

print("=== V5.17 HIDDEN-STATE PREDICTABILITY ===")
print("Train jump rate:", positive)
print("Validation jump rate:", float(y_val.mean()))

for epoch in range(1, 101):
    head.train()
    perm = torch.randperm(len(xtr))

    for ids in perm.split(2048):
        logits = head(xtr[ids]).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, ytr[ids]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch in [1, 5, 10, 20, 30, 50, 75, 100]:
        head.eval()
        with torch.no_grad():
            prob = torch.sigmoid(
                head(xva).squeeze(-1)
            ).numpy()

        baseline = np.full_like(prob, positive)

        print(
            f"Epoch {epoch:02d}"
            f" | AUC={roc_auc_score(y_val, prob):.4f}"
            f" | AP={average_precision_score(y_val, prob):.4f}"
            f" | LogLoss={log_loss(y_val, prob):.4f}"
            f" | Baseline LogLoss={log_loss(y_val, baseline):.4f}"
        )

print("V5.17 diagnostic complete!")
