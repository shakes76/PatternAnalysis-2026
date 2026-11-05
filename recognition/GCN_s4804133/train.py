import argparse, time, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from dataset import load_graph, make_masks
from modules import MLP, GCN
import os

# Training settings: argument names and defaults (seed, lr, weight_decay, dropout) adapted from Kipf's pygcn train.py, https://github.com/tkipf/pygcn/blob/master/pygcn/train.py; --model, --patience, --num_layers and the other values are my own
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["mlp", "gcn"], default="gcn")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=300) 
parser.add_argument("--patience", type=int, default=50) # stop early if val macro-F1 hasn't improved for this many epochs in a row; prevents overfitting
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--hidden", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--tag", default="") # label for a tuning run so output files don't overwrite each other
args = parser.parse_args()
run_name = f"{args.model}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")

torch.manual_seed(args.seed)

# Device config for diff systems
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("Warning: CUDA and MPS not found. Using CPU")
print("Using device:", device)

# load the graph and create the train, validation and test masks
data = make_masks(load_graph()).to(device) 

# build the chosen model; 4 output classes
in_dim, out_dim = data.x.size(1), 4
if args.model == "gcn":
    model = GCN(in_dim, args.hidden, out_dim, num_layers=args.num_layers, dropout=args.dropout)
else:
    model = MLP(in_dim, args.hidden, out_dim, dropout=args.dropout)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# TRAIN 
def train_step():
    model.train()
    optimizer.zero_grad() # clear previous gradients
    out = model(data.x, data.edge_index) # run forward pass on the whole graph to get predictions for all nodes
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask]) # compute the cross entropy loss only for the training nodes
    loss.backward() # compute gradients for all model parameters
    optimizer.step() # update model weights
    return loss.item()

# EVALUATE
@torch.no_grad() # disable gradient calculation for evaluation; saves mem and time
def evaluate(model, data, mask):
    model.eval() # dropout off
    out = model(data.x, data.edge_index) 
    loss = F.cross_entropy(out[mask], data.y[mask]).item() 
    pred = out[mask].argmax(dim=1) 
    acc = (pred == data.y[mask]).float().mean().item() # compute the accuracy only for the nodes specified by the mask
    f1 = f1_score(data.y[mask].cpu(), pred.cpu(), average="macro") # macro-F1 weights all 4 classes equally regardless of class imbalance
    return loss, acc, f1

# TRAINING LOOP 
# best-validation epoch and stop early, since 10% labels overfit easily
best_val, best_state, wait = 0, None, 0
history = {"train_loss": [], "val_loss": [], "val_f1": []}
start = time.time()
for epoch in range(args.epochs):
    train_loss = train_step()
    val_loss, val_acc, val_f1 = evaluate(model, data, data.val_mask)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_f1"].append(val_f1)
    if val_f1 > best_val:
        best_val, wait = val_f1, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= args.patience: break
train_time = time.time() - start
model.load_state_dict(best_state) # restore best epoch, not the last one

# TEST: evaluated once, only after training and model selection are finished
test_loss, test_acc, test_f1 = evaluate(model, data, data.test_mask)
print(f"{args.model} seed={args.seed} | epochs run: {epoch + 1} | train time: {train_time:.1f}s")
print(f"best val macro-F1: {best_val:.4f}")
print(f"test loss: {test_loss:.4f} | test acc: {test_acc:.4f} | test macro-F1: {test_f1:.4f}")

# SAVE weights and hyperparameters
os.makedirs("outputs", exist_ok=True)
os.makedirs("figures", exist_ok=True)
torch.save({"state_dict": best_state, "args": vars(args)}, f"outputs/{run_name}.pt")

# LOG one row per run to results/tuning.csv; choose configs by best_val_f1, never test_f1
os.makedirs("results", exist_ok=True)
log_path = "results/tuning.csv"
header = "tag,model,seed,hidden,dropout,lr,weight_decay,num_layers,epochs_run,best_val_f1,test_f1,test_acc,train_time_s\n"
new_file = not os.path.exists(log_path)
with open(log_path, "a") as f:
    if new_file:
        f.write(header)
    f.write(f"{args.tag},{args.model},{args.seed},{args.hidden},{args.dropout},{args.lr},{args.weight_decay},"
            f"{args.num_layers},{epoch + 1},{best_val:.4f},{test_f1:.4f},{test_acc:.4f},{train_time:.1f}\n")

# PLOT loss and validation macro-F1 curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(history["train_loss"], label="train loss")
ax1.plot(history["val_loss"], label="val loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("cross-entropy"); ax1.legend()
ax2.plot(history["val_f1"], label="val macro-F1")
ax2.set_xlabel("epoch"); ax2.set_ylabel("macro-F1"); ax2.legend()
fig.suptitle(f"{args.model.upper()} training (seed {args.seed})")
fig.tight_layout()
fig.savefig(f"figures/{run_name}_curves.png")
plt.close(fig)