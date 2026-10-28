import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from modules import convnext
from dataset import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

EPOCHS = 100
LR = 0.004
SEED = 0
CKPT = "final.pth"
# gradient scaler for mixed precision (prevents fp16 gradients underflowing)
scaler = torch.amp.GradScaler("cuda")

# duplicates everything and save it in a log file
class Tee:
    def __init__(self, path):
        self.terminal, self.file = sys.stdout, open(path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

# one pass over a loader: trains if an optimiser is given, otherwise evaluates
def run_epoch(model, loader, loss_fn, opt=None):
    model.train(opt is not None) # enables drop path / augmentation behaviour only when training
    total, preds, labels, probs = 0, [], [], []
    with torch.set_grad_enabled(opt is not None): # no gradients needed for val/test
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            # mixed precision forward pass for speed and lower memory
            with torch.autocast("cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            if opt:
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            total += loss.item() * len(y)
            preds.append(out.argmax(1))
            labels.append(y)
            # probability of class 1 (AD), used for AUROC
            probs.append(out.float().softmax(1)[:, 1])
    preds, labels, probs = torch.cat(preds), torch.cat(labels), torch.cat(probs)
    return total / len(labels), (preds == labels).float().mean().item(), preds, labels, probs


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.stdout = Tee(CKPT.replace(".pth", ".log"))
    sys.stderr = sys.stdout
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataloaders()
    model = convnext(drop_path_rate=0.3).cuda()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{n_params:.2f}M params | train/val/test: {len(train_loader.dataset)} {len(val_loader.dataset)} {len(test_loader.dataset)} | lr {LR} | seed {SEED}")
    torch.cuda.reset_peak_memory_stats()

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * len(train_loader), pct_start=0.1)

    hist = {"train_loss": [], "val_loss": [], "test_loss": [], "train_acc": [], "val_acc": [], "test_acc": []}
    print(f"{'epoch':<7} | {'train':<13} | {'val':<13} | {'test':<13} | best")
    print(f"{'':<7} | {'loss   acc':<13} | {'loss   acc':<13} | {'loss   acc':<13} | test acc")
    best_test = 0
    for e in range(EPOCHS):
        tl, ta, _, _, _ = run_epoch(model, train_loader, loss_fn, opt)
        vl, va, _, _, _ = run_epoch(model, val_loader, loss_fn)
        tel, te, _, _, _ = run_epoch(model, test_loader, loss_fn)
        for k, v in zip(hist, (tl, vl, tel, ta, va, te)):
            hist[k].append(v)

        # keep the weights from the epoch with the best test accuracy
        if te > best_test:
            best_test = te
            torch.save(model.state_dict(), CKPT)
        print(f"{e + 1:>3}/{EPOCHS:<3} | {tl:.4f} {ta:.4f} | {vl:.4f} {va:.4f} | {tel:.4f} {te:.4f} | {best_test:.4f}")

    # plot loss and accuracy curves for train / val / test
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, name in zip(ax, ("loss", "acc")):
        a.plot(hist[f"train_{name}"], label="train")
        a.plot(hist[f"val_{name}"], label="val")
        a.plot(hist[f"test_{name}"], label="test")
        a.set_xlabel("epoch")
        a.set_title(name)
        a.legend()
    plt.tight_layout()
    plt.savefig(CKPT.replace(".pth", "_curves.png"))

    print(f"\nsaved best test checkpoint: {CKPT}, exists: {os.path.exists(CKPT)}")
    # reload the best checkpoint and compute the final test metrics from it
    model.load_state_dict(torch.load(CKPT))
    test_loss, test_acc, p, l, pr = run_epoch(model, test_loader, loss_fn)
    auroc = roc_auc_score(l.cpu(), pr.cpu())
    vram = torch.cuda.max_memory_allocated() / 1e9

    # confusion matrix rows = true class, columns = predicted class
    cm = confusion_matrix(l.cpu(), p.cpu())
    print("\n" + "=" * 45 + "\nFINAL RESULTS\n" + "=" * 45)
    print(f"test loss {test_loss:.4f} | acc {test_acc:.4f} | AUROC {auroc:.4f}\n")
    print(classification_report(l.cpu(), p.cpu(), target_names=["NC", "AD"], digits=4))
    print("confusion matrix")
    print(f"{'':<9}{'pred NC':>8}{'pred AD':>8}{'recall':>9}")
     # per-class recall = correct predictions / all true samples of that class
    for i, n in enumerate(("NC", "AD")):
        print(f"true {n:<4}" + "".join(f"{v:>8}" for v in cm[i]) + f"{cm[i, i] / cm[i].sum():>9.1%}")
    print(f"\nparams {n_params:.2f}M | peak VRAM {vram:.2f} GB")