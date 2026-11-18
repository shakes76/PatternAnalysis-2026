import argparse,json,random,time
from pathlib import Path

import numpy as np
import torch

from dataset import get_dataloader
from modules import ImprovedUNet,DiceCeLoss,dice_per_class,count_parameters

CLASS_NAMES = ["background","body","bone","bladder","rectum","prostate"]

def parse_args():
    p = argparse.ArgumentParser(description="Train Improved U-Net on HipMRI")
    p.add_argument("--root", default=None,help="data folder (default: the one set in dataset.py)")
    p.add_argument("--dims", type=int, choices=[2, 3], default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device,non_blocking = True)
        labels = batch["label"].to(device,non_blocking = True).long()

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item() * images.size(0)

    return total_loss/len(loader.dataset)

@torch.no_grad
def validate(model,loader,criterion,device,num_classes = 6):
    model.eval()
    total_loss = 0.0
    inter = torch.zeros(num_classes,device=device)
    pred_sum = torch.zeros(num_classes,device=device)
    true_sum = torch.zeros(num_classes,device=device)

    for batch in loader:
        images = batch["image"].to(device,non_blocking = True)
        labels = batch["label"].to(device,non_blocking = True).long() 

        logits = model(images)
        total_loss += criterion(logits,labels).item()*images.size(0)

        preds = logits.argmax(dim=1)
        for c in range(num_classes):
            p = preds == c
            t = labels == c
            inter[c] += (p&t).sum()
            pred_sum[c] += p.sum()
            true_sum[c] += t.sum()
    dice = 2 * inter / (pred_sum + true_sum).clamp(min=1)
    return total_loss/ len(loader.dataset),dice.tolist()

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    loader_kwargs = dict(dims=args.dims, batch_size=args.batch_size,
                     num_workers=args.num_workers, subset=args.subset, seed=args.seed,
                     crop_size=(128, 128, 64) if args.dims == 3 else None)

    if args.root is not None:
        loader_kwargs["root"]= args.root

    loaders,split = get_dataloader(**loader_kwargs)
    model = ImprovedUNet(in_channels=1, num_classes=6, dims=args.dims).to(device)
    
    print(f"Parameters: {count_parameters(model):,}")

    criterion = DiceCeLoss(num_classes=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)

    history = {"train_loss" : [],"val_loss" : [], "val_dice" : [],"lr":[],"epoch_time":[]}
    best_dice = -1.0
    prostate_idx = CLASS_NAMES.index("prostate")
    check_point_path = out_dir/f"best_model_{args.dims}d.pt"
    history_path = out_dir/f"history_{args.dims}d.json"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model,loaders["train"],criterion,optimizer,device)
        val_loss ,val_dice = validate(model,loaders["val"],criterion,device)

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_time = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["lr"].append(lr)
        history["epoch_time"].append(epoch_time)

        saved = ""
        if val_dice[prostate_idx]>best_dice:
            best_dice = val_dice[prostate_idx]
            torch.save({
                "model":model.state_dict(),
                "epoch":epoch,
                "dims": args.dims,
                "val_dice":val_dice,
                "args":vars(args),

            },check_point_path)
            saved = "  <- best, saved"

        dice_str = " ".join(f"{n}={d:.3f}" for n,d in zip(CLASS_NAMES,val_dice) )
        print(f"Epoch {epoch+1}/{args.epochs}: train loss {train_loss:.4f} | val {val_loss:.4f} | "
              f"{dice_str} | lr {lr:.2e} | {epoch_time:.1f}s{saved}")

        with open(history_path,"w") as f:
            json.dump(history,f,indent=2)
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    summary = {
    "dims": args.dims,
    "params": count_parameters(model),
    "best_prostate_dice": best_dice,
    "peak_vram_gb": peak_vram_gb,
    "mean_epoch_time_s": sum(history["epoch_time"]) / len(history["epoch_time"]),
    "split": split,
    }
    with open(out_dir/f"summary_{args.dims}d.json","w") as f:
        json.dump(summary,f,indent=2)
    print(f"Done.Best prostate Dice {best_dice:.4f} | peak Vram {peak_vram_gb:.2f}GB")


if __name__ == "__main__":
    main()