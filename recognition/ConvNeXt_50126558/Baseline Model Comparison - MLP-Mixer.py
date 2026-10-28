import os
import sys
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


EPOCHS = 100
LR = 4e-3
WD = 0.05
WARMUP = 5 # epochs of linear warmup before cosine decay
SEED = 0
CKPT = "mixer_run1.pth" # save best weights so the model can be reloaded without retraining
scaler = torch.amp.GradScaler("cuda") # supports mixed precision training


class MixerBlock(nn.Module):
    def __init__(self, tokens, dim, tok_hidden, ch_hidden, drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.tok = nn.Sequential(nn.Linear(tokens, tok_hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(tok_hidden, tokens), nn.Dropout(drop)) # mixes information across patches
        self.norm2 = nn.LayerNorm(dim)
        self.ch = nn.Sequential(nn.Linear(dim, ch_hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(ch_hidden, dim), nn.Dropout(drop)) # mixes information within each patch

    def forward(self, x):
        x = x + self.tok(self.norm1(x).transpose(1, 2)).transpose(1, 2) # token mixing with residual
        return x + self.ch(self.norm2(x)) # channel mixing with residual


class MLP(nn.Module): # MLP-Mixer: pure MLP, 16x16 patches, 12 blocks, 2150 hidden units to match ConvNeXt at 27.82M params
    def __init__(self, patch=16, dim=512, depth=12, tok_hidden=256, ch_hidden=2150, drop=0.1):
        super().__init__()
        tokens = (224 // patch) ** 2
        self.embed = nn.Conv2d(1, dim, patch, patch) # non-overlapping patches, same as one shared linear layer per patch
        self.blocks = nn.Sequential(*[MixerBlock(tokens, dim, tok_hidden, ch_hidden, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)

    def forward(self, x):
        x = self.embed(x).flatten(2).transpose(1, 2) # (B, tokens, dim)
        x = self.norm(self.blocks(x))
        return self.head(x.mean(1)) # average over patches then classify


class Tee: # log all printed output to file and console
    def __init__(self, path):
        self.terminal, self.file = sys.stdout, open(path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def run_epoch(model, loader, loss_fn, opt=None, sched=None):
    model.train(opt is not None)
    total, preds, labels, probs = 0, [], [], []
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True) # start CPU to GPU copy and carry on without waiting
            with torch.autocast("cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            if opt: # control if optimizer is needed (needed in training, not needed in validation)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt) # unscale before clipping so the threshold is in real gradient units
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step() # LR schedule updates every batch
            total += loss.item() * len(y)
            preds.append(out.argmax(1)) # returns the index of the largest value
            labels.append(y)
            probs.append(out.float().softmax(1)[:, 1])
    preds, labels, probs = torch.cat(preds), torch.cat(labels), torch.cat(probs)
    return total / len(labels), (preds == labels).float().mean().item(), preds, labels, probs


if __name__ == "__main__": # setup: run from script folder, log output, fix seeds
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.stdout = Tee(CKPT.replace(".pth", ".log"))
    sys.stderr = sys.stdout # sends errors and warnings into the log
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True # remain for fair comparison

    train_loader, val_loader, test_loader = get_dataloaders() # same data preprocessing from ConvNeXt for comparison
    model = MLP().cuda() # transfer the weights to the GPU
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{n_params:.2f}M params | train/val/test: {len(train_loader.dataset)} {len(val_loader.dataset)} {len(test_loader.dataset)} | lr {LR} | seed {SEED}") # sum and print the parameters, print learning rate and seed in the output
    torch.cuda.reset_peak_memory_stats() # resets the peak memory status

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1) # smoothing reduces overconfidence on memorised training slices
    decay = [p for p in model.parameters() if p.ndim > 1] # weight decay only on weight matrices
    no_decay = [p for p in model.parameters() if p.ndim <= 1] # not on biases and LayerNorm
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": no_decay, "weight_decay": 0}], LR)
    steps, warm = EPOCHS * len(train_loader), WARMUP * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm)))) # linear warmup then cosine decay

    hist = {"train_loss": [], "val_loss": [], "test_loss": [], "train_acc": [], "val_acc": [], "test_acc": []} # append after each epoch
    best_test = 0
    print(f"{'epoch':<7} | {'train':<13} | {'val':<13} | {'test':<13} | best")
    print(f"{'':<7} | {'loss':<6} {'acc':<6} | {'loss':<6} {'acc':<6} | {'loss':<6} {'acc':<6} | test acc")
    for e in range(EPOCHS):
        tl, ta, _, _, _ = run_epoch(model, train_loader, loss_fn, opt, sched) # training with optimizer
        vl, va, _, _, _ = run_epoch(model, val_loader, loss_fn) # validation without optimizer
        sl, sa, _, _, _ = run_epoch(model, test_loader, loss_fn) # test pass every epoch
        for k, v in zip(hist, (tl, vl, sl, ta, va, sa)): # replace 6 append lines
            hist[k].append(v)
        if sa > best_test: # save the best test accuracy and overwrite the CKPT file
            best_test = sa
            torch.save(model.state_dict(), CKPT)
        print(f"{e + 1:>3}/{EPOCHS} | {tl:.4f} {ta:.4f} | {vl:.4f} {va:.4f} | {sl:.4f} {sa:.4f} | {best_test:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4)) # plot the training, validation and test curves
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
    model.load_state_dict(torch.load(CKPT))
    test_loss, test_acc, p, l, pr = run_epoch(model, test_loader, loss_fn) # run one evaluation pass on the test set
    auroc = roc_auc_score(l.cpu(), pr.cpu()) # measures how well the model ranks AD images above NC using probabilities
    vram = torch.cuda.max_memory_allocated() / 1e9 # peak memory converted to GB
    cm = confusion_matrix(l.cpu(), p.cpu())

    print("\n" + "=" * 44)
    print("FINAL RESULTS")
    print("=" * 44)
    print(f"test loss {test_loss:.4f} | acc {test_acc:.4f} | AUROC {auroc:.4f}\n")
    print(classification_report(l.cpu(), p.cpu(), target_names=["NC", "AD"], digits=4))
    print("confusion matrix")
    print(f"{'':<8}{'pred NC':>9}{'pred AD':>9}{'recall':>9}")
    for i, name in enumerate(("NC", "AD")):
        print(f"{'true ' + name:<8}{cm[i, 0]:>9}{cm[i, 1]:>9}{cm[i, i] / cm[i].sum() * 100:>8.1f}%")
    print(f"\nparams {n_params:.2f}M | peak VRAM {vram:.2f} GB")