import os
import time
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataset import get_dataloaders, HIPMRI_ROOT, NUM_CLASSES
from modules import ImprovedUNet2D, UNet2D


LABELS = ["background", "body", "bone", "bladder", "rectum", "prostate"]
BATCH_SIZE = 16
EPOCHS = 50
LR = 5e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_ce_loss(logits, masks):
    """Cross-entropy + soft Dice."""
    ce = F.cross_entropy(logits, masks)
    probs = logits.softmax(dim=1)
    one_hot = F.one_hot(masks, NUM_CLASSES).permute(0, 3, 1, 2).float()   # (B, 6, H, W)
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    total = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return ce + 1 - ((2 * inter + 1) / (total + 1)).mean()

def dice_iou(preds, truth):
    """Per-class Dice and IoU, with overlaps summed over the whole split."""
    dice, iou = [], []
    for c in range(NUM_CLASSES):
        p, t = preds == c, truth == c
        inter, p_sum, t_sum = (p & t).sum().item(), p.sum().item(), t.sum().item()
        dice.append(2 * inter / max(p_sum + t_sum, 1))
        iou.append(inter / max(p_sum + t_sum - inter, 1))
    return dice, iou

def edges(mask):
    m = mask.float().unsqueeze(1)
    eroded = -F.max_pool2d(-m, 3, stride=1, padding=1)
    return (m - eroded).squeeze(1) > 0


def boundary_f1(preds, truth, label=5, tol=2):
    """Boundary precision for one label."""
    pred_edge, true_edge = edges(preds == label), edges(truth == label)
    k = 2 * tol + 1
    near_true = F.max_pool2d(true_edge.float().unsqueeze(1), k, 1, tol).squeeze(1) > 0
    near_pred = F.max_pool2d(pred_edge.float().unsqueeze(1), k, 1, tol).squeeze(1) > 0
    precision = (pred_edge & near_true).sum().item() / max(pred_edge.sum().item(), 1)
    recall = (true_edge & near_pred).sum().item() / max(true_edge.sum().item(), 1)
    return 2 * precision * recall / max(precision + recall, 1e-8)


@torch.no_grad()
def predict_all(model, loader):
    """mean loss and the ms per slice."""
    model.eval()
    preds, total_loss, total_time = [], 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start
        total_loss += dice_ce_loss(logits, masks).item() * len(images)
        preds.append(logits.argmax(dim=1).cpu())
    n = len(loader.dataset)
    return torch.cat(preds), total_loss / n, 1000 * total_time / n
 

def train(name, model, train_loader, val_loader):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    val_truth = torch.from_numpy(val_loader.dataset.masks).long()
    path = os.path.join("outputs", f"{name}_best.pt")
    history = {"train_loss": [], "val_loss": [], "val_prostate_dice": []}
    best = -1.0
 
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
 
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = dice_ce_loss(model(images), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
 
        val_preds, val_loss, _ = predict_all(model, val_loader)
        val_dice = dice_iou(val_preds, val_truth)[0][5]
        history["train_loss"].append(total_loss / len(train_loader.dataset))
        history["val_loss"].append(val_loss)
        history["val_prostate_dice"].append(val_dice)
 
        if val_dice > best:   
            best = val_dice
            torch.save(model.state_dict(), path)
        print(f"[{name}] epoch {epoch:3d}/{EPOCHS}  train_loss={history['train_loss'][-1]:.4f}  "
              f"val_loss={val_loss:.4f}  val_prostate_dice={val_dice:.4f}")
 
    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
    minutes = (time.time() - start) / 60
    model.load_state_dict(torch.load(path, map_location=device))
    return history, peak_vram, minutes

def plot_failures(name, images, truth, preds, n=4):
    """Worst test slices by prostate Dice: input | truth | prediction | errors."""
    p, t = preds == 5, truth == 5
    inter = (p & t).flatten(1).sum(1).float()
    slice_dice = 2 * inter / (p.flatten(1).sum(1) + t.flatten(1).sum(1)).clamp(min=1)
    relevant = (p | t).flatten(1).any(1)       
    slice_dice[~relevant] = 2.0             
    n = min(n, int(relevant.sum()))
    if n == 0:
        return
    worst = slice_dice.argsort()[:n]
 
    fig, axes = plt.subplots(n, 4, figsize=(10, 4.5 * n), squeeze=False)
    for row, i in enumerate(worst.tolist()):
        fp, fn = p[i] & ~t[i], t[i] & ~p[i]
        errors = torch.zeros(*fp.shape, 4)
        errors[fp] = torch.tensor([1.0, 0.0, 0.0, 0.8])   # false positive: red
        errors[fn] = torch.tensor([0.0, 0.4, 1.0, 0.8])   # false negative: blue
 
        ax = axes[row]
        ax[0].imshow(images[i], cmap="gray")
        ax[1].imshow(truth[i], cmap="viridis", vmin=0, vmax=5, interpolation="nearest")
        ax[2].imshow(preds[i], cmap="viridis", vmin=0, vmax=5, interpolation="nearest")
        ax[3].imshow(images[i], cmap="gray")
        ax[3].imshow(errors.numpy(), interpolation="nearest")
        ax[0].set_title(f"Input (test slice {i})")
        ax[1].set_title("Ground truth")
        ax[2].set_title("Prediction")
        ax[3].set_title(f"Prostate errors, Dice {slice_dice[i]:.2f}\nred = FP, blue = FN")
        for a in ax:
            a.axis("off")
    fig.suptitle(f"{name}: worst test slices for prostate")
    fig.tight_layout()
    fig.savefig(os.path.join("outputs", f"{name}_failures.png"), dpi=150)
    plt.close(fig)
 

 
def plot_curves(histories):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in histories.items():
        epochs = range(1, len(h["train_loss"]) + 1)
        ax1.plot(epochs, h["train_loss"], label=f"{name} train")
        ax1.plot(epochs, h["val_loss"], "--", label=f"{name} validation")
        ax2.plot(epochs, h["val_prostate_dice"], label=name)
    ax2.axhline(0.75, color="grey", linestyle=":", label="target 0.75")
    ax1.set(xlabel="Epoch", ylabel="Dice + CE loss", title="Loss")
    ax2.set(xlabel="Epoch", ylabel="Dice", title="Validation prostate Dice", ylim=(0, 1))
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join("outputs", "training_curves.png"), dpi=150)
    plt.close(fig)

def print_results(results):
    """Markdown tables, ready to paste into the README."""
    names = list(results)
    print("\n| Label | " + " | ".join(f"{n} Dice | {n} IoU" for n in names) + " |")
    print("|---|" + "---|---|" * len(names))
    for c in range(NUM_CLASSES):
        cells = " | ".join(f"{results[n]['dice'][c]:.4f} | {results[n]['iou'][c]:.4f}" for n in names)
        print(f"| {LABELS[c]} | {cells} |")
 
    print("\n| Model | Prostate boundary F1 (2 px) | Parameters | Peak VRAM (GB) | Train time (min) | Inference (ms/slice) |")
    print("|---|---|---|---|---|---|")
    for n in names:
        r = results[n]
        print(f"| {n} | {r['boundary_f1']:.4f} | {r['params']:,} | {r['vram']:.2f} | "
              f"{r['minutes']:.1f} | {r['ms']:.2f} |")
        
def main():
    torch.manual_seed(35)
    os.makedirs("outputs", exist_ok=True)
    print("Using device:", device)
    train_loader, val_loader, test_loader = get_dataloaders(HIPMRI_ROOT, BATCH_SIZE, early_stop=False)
    test_images = test_loader.dataset.images
    test_truth = torch.from_numpy(test_loader.dataset.masks).long()
    models = {
        "baseline": UNet2D(in_channels=1, num_classes=6),
        "improved": ImprovedUNet2D(in_channels=1, num_classes=6),
    }
    histories, results = {}, {}
    for name, model in models.items():
        histories[name], vram, minutes = train(name, model, train_loader, val_loader)
        preds, _, ms = predict_all(model, test_loader)
        dice, iou = dice_iou(preds, test_truth)
        results[name] = {
            "dice": dice, "iou": iou, "boundary_f1": boundary_f1(preds, test_truth),
            "params": sum(p.numel() for p in model.parameters()),
            "vram": vram, "minutes": minutes, "ms": ms,
        }
        plot_failures(name, test_images, test_truth, preds)
    plot_curves(histories)
    print_results(results)

if __name__ == "__main__":
    main()