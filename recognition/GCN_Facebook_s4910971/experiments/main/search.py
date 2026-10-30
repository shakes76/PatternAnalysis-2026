"""Search GCN hyperparameters."""

import csv
import sys
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


# Import project modules.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import load_graph
from modules import GCN



# Local paths.
DATA = ROOT / "data" / "facebook.npz"
OUT = ROOT / "experiments" / "main" / "search_results"


# Search settings.
SEED = 0
EPOCHS = 300
DEPTHS = [2, 3, 4, 6, 8]
WIDTHS = [16, 32, 64]
LRS = [0.005, 0.01, 0.02]
DROPS = [0.2, 0.5, 0.7]
DECAYS = [0.0, 0.0005, 0.005]



def train_one(data, device, depth, width, lr, drop, wd):
    """Train one combination with a fixed seed."""
    torch.manual_seed(SEED)
    model = GCN(data.num_node_features, width, depth, drop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


    # Train and check validation performance.
    best_f1 = -1
    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()


        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)[data.val_mask]
            y = data.y[data.val_mask]
            pred = out.argmax(dim=1)
            loss = F.cross_entropy(out, y).item()
            acc = (pred == y).float().mean().item()
            f1 = f1_score(y.cpu(), pred.cpu(), labels=[0, 1, 2, 3],
                          average="macro", zero_division=0)

        if f1 > best_f1 or (f1 == best_f1 and loss < best_loss):
            best_f1 = f1
            best_loss = loss
            best_acc = acc
            best_epoch = epoch

    return {
        "depth": depth,
        "hidden_dim": width,
        "learning_rate": lr,
        "dropout": drop,
        "weight_decay": wd,
        "val_accuracy": best_acc,
        "val_macro_f1": best_f1,
        "val_loss": best_loss,
        "best_epoch": best_epoch,
    }



def main():
    """Search combinations and save selected parameters."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_graph(DATA, seed=SEED).to(device)
    OUT.mkdir(parents=True, exist_ok=True)

    columns = [
        "depth", "hidden_dim", "learning_rate", "dropout", "weight_decay",
        "val_accuracy", "val_macro_f1", "val_loss", "best_epoch",
    ]
    best = None


    # Save one row per combination.
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for depth, width, lr, drop, wd in product(DEPTHS, WIDTHS, LRS, DROPS, DECAYS):
            row = train_one(data, device, depth, width, lr, drop, wd)
            writer.writerow(row)
            f.flush()

            # Select by macro F1, then loss.
            if best is None:
                best = row
            elif row["val_macro_f1"] > best["val_macro_f1"]:
                best = row
            elif row["val_macro_f1"] == best["val_macro_f1"]:
                if row["val_loss"] < best["val_loss"]:
                    best = row


    # Save selected parameters.
    with (OUT / "best_params.txt").open("w", encoding="utf-8") as f:
        f.write("Selection: highest validation macro F1 across epochs; ties use lower validation loss.\n")
        f.write(f"Epochs per combination: {EPOCHS}\nSeed: {SEED}\n")
        f.write("Split: 70% training / 10% validation / 20% test\nOptimizer: Adam\n")
        f.write(f"Depths searched: {DEPTHS}\nHidden widths searched: {WIDTHS}\n")
        f.write(f"Learning rates searched: {LRS}\nDropouts searched: {DROPS}\n")
        f.write(f"Weight decays searched: {DECAYS}\n\n")
        for name, value in best.items():
            f.write(f"{name} = {value}\n")

    print(f"Best: depth={best['depth']}, width={best['hidden_dim']}, "
          f"lr={best['learning_rate']}, dropout={best['dropout']}, wd={best['weight_decay']}")
    print(f"Saved: {OUT}")



if __name__ == "__main__":
    main()
