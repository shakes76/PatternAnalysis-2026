import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataset import load_graph
data, num_classes = load_graph()

# ---------- 1. Data audit ----------
print("Nodes:", data.num_nodes, "| Features:", data.num_node_features,
      "| Classes:", num_classes)
print("Edge entries (PyG stores both directions):", data.edge_index.size(1))

counts = torch.bincount(data.y)
print("Class counts:", counts.tolist())
print("Class proportions:", [round(p, 3) for p in (counts / counts.sum()).tolist()])

src, dst = data.edge_index
deg = torch.bincount(src, minlength=data.num_nodes)
print("Isolated nodes:", (deg == 0).sum().item(), "| Mean degree:", round(deg.float().mean().item(), 2))
print("Edge homophily:", round((data.y[src] == data.y[dst]).float().mean().item(), 3))

# ---------- 2. Quick split (final version goes in dataset.py) ----------
n = data.num_nodes
perm = torch.randperm(n)
train_idx = perm[: int(0.6 * n)]
val_idx = perm[int(0.6 * n): int(0.8 * n)]
test_idx = perm[int(0.8 * n):]

# ---------- 3. MLP baseline smoke test ----------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.p = p

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), self.p, self.training)
        return self.fc2(x)

model = MLP(data.num_node_features, 64, num_classes).to(device)
x, y = data.x.to(device), data.y.to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

best_val, best_state, epochs = 0, None, 100
start = time.time()
for epoch in range(epochs):
    model.train()
    opt.zero_grad()
    loss = F.cross_entropy(model(x)[train_idx], y[train_idx])
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        val_acc = (model(x)[val_idx].argmax(1) == y[val_idx]).float().mean().item()
    if val_acc > best_val:
        best_val = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
train_time = time.time() - start

# ---------- 4. Evaluate on test ----------
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    probs = F.softmax(model(x)[test_idx], dim=1)
conf, pred = probs.max(1)
correct = pred == y[test_idx]

print("\n--- MLP smoke test ---")
print("Best val acc:", round(best_val, 3))
print("Test acc:", round(correct.float().mean().item(), 3))
print("Test macro-F1:", round(f1_score(y[test_idx].cpu(), pred.cpu(), average="macro"), 3))
print("Mean confidence (correct):", round(conf[correct].mean().item(), 3))
print("Mean confidence (wrong):", round(conf[~correct].mean().item(), 3))
print("Parameters:", sum(p.numel() for p in model.parameters()))
print(f"Train time for {epochs} epochs: {train_time:.1f}s ({train_time / epochs:.3f}s/epoch)")
if device.type == "cuda":
    print("Peak GPU memory (MB):", round(torch.cuda.max_memory_allocated() / 1e6, 1))

from sklearn.metrics import roc_auc_score

with torch.no_grad():
    val_probs = F.softmax(model(x)[val_idx], dim=1)
val_conf, val_pred = val_probs.max(1)

# Can confidence detect errors? (errors are the positive class)
auroc = roc_auc_score((~correct).cpu().numpy(), (1 - conf).cpu().numpy())
print("Error-detection AUROC (test):", round(auroc, 3))

# Reject rule: threshold chosen on VALIDATION, reported on TEST
val_correct = val_pred == y[val_idx]
for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
    vm, tm = val_conf >= t, conf >= t
    if vm.sum() == 0 or tm.sum() == 0:
        continue
    print(f"thr {t}: val acc {val_correct[vm].float().mean():.3f} (cov {vm.float().mean():.2f}) | "
          f"test acc {correct[tm].float().mean():.3f} (cov {tm.float().mean():.2f})")