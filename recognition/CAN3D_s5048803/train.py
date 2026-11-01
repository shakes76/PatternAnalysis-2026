"""
train.py: Containing the source code for training, validating, testing, and saving your model.
The model should be imported from modules.py and the data loader should be imported from
dataset.py. Make sure to plot the losses and metrics during training.

Usage:
	python train.py --epochs 15 --scale 0.5 --model can2d
"""

import argparse, os, time
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from modules import CAN
from dataset import (get_pairs, split_pairs, HipMRI3D, get_slices, HipMRI2D,
					 NUM_CLASSES, RANGPUR_MODE)


def one_hot(t):
	return F.one_hot(t, NUM_CLASSES).movedim(-1, 1).float()

def dice_loss(logits, target, eps=1e-6):
	probs, onehot = torch.softmax(logits, 1), one_hot(target)
	dims = (0,) + tuple(range(2, logits.ndim)) # everything except the class axis
	inter = (probs * onehot).sum(dims) # Intersection between prediction and ground-truth
	denom = probs.sum(dims) + onehot.sum(dims)
	return 1 - ((2 * inter + eps) / (denom + eps)).mean()

@torch.no_grad()
def evaluate(model, loader, device, eps=1e-6):
	"""Pooled per-class Dice over the whole loader (same definition for 2D and 3D)."""
	model.eval()
	inter = torch.zeros(NUM_CLASSES, device=device)
	pred_n = torch.zeros(NUM_CLASSES, device=device)
	true_n = torch.zeros(NUM_CLASSES, device=device)
	for x, y in loader:
		x, y = x.to(device), y.to(device)
		p, t = one_hot(model(x).argmax(1)), one_hot(y)
		dims = (0,) + tuple(range(2, p.ndim))
		inter += (p * t).sum(dims)
		pred_n += p.sum(dims)
		true_n += t.sum(dims)
	return ((2 * inter + eps) / (pred_n + true_n + eps)).cpu().tolist()

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--model", choices=["can2d", "can3d"], default="can2d", help="Model variation to train")
	ap.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
	ap.add_argument("--lr", type=float, default=1e-3, help="Adam optimized learning rate")
	ap.add_argument("--scale", type=float, default=None, help="Scaling factor applied to image before the model. default 0.5")
	ap.add_argument("--batch", type=int, default=16, help="slices per batch (2D only)")
	ap.add_argument("--overfit", type=int, default=0, help="train+val on N items")
	ap.add_argument("--out", default="models", help="Ensure the output folder exists beforehand!")
	ap.add_argument("--data_root", default="/home/groups/comp3710/HipMRI_Study_open" if RANGPUR_MODE else ".")
	ap.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
	ap.add_argument("--width", type=int, default=24, help="Width of the network (default 24)")
	ap.add_argument("--seed", type=int, default=-1, help="Random seed, if >=0 the training is deterministic and seed is used")
	args = ap.parse_args()
	os.makedirs(args.out, exist_ok=True)

	if args.seed >= 0:
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	dim = 3 if args.model == "can3d" else 2
	if args.scale is None:
		args.scale = 0.5
	device = ("cuda" if torch.cuda.is_available()
			  else "mps" if torch.backends.mps.is_available() else "cpu")

	if dim == 3:
		splits = split_pairs(args.data_root, get_pairs(args.data_root))
		Data, bs = HipMRI3D, 1
	else:
		splits = {s: get_slices(args.data_root, s) for s in ("train", "val", "test")}
		Data, bs = HipMRI2D, args.batch
	print(args.model, {k: len(v) for k, v in splits.items()})
	if args.overfit:
		splits["train"] = splits["val"] = splits["train"][:args.overfit]

	if dim == 3:
		train_ds = Data(splits["train"], args.scale, augment=not args.overfit)   # augment train only
	else:
		train_ds = Data(splits["train"], args.scale)
	val_ds = Data(splits["val"], args.scale)

	# Multiple workers help load dataset faster, ensuring compute is the models limit
	kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0,
			  pin_memory=(device == "cuda"))
	train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, **kw)
	val_dl = DataLoader(val_ds, batch_size=bs, **kw)

	model = CAN(in_ch=1, num_classes=NUM_CLASSES, dim=dim).to(device)
	opt = torch.optim.Adam(model.parameters(), lr=args.lr)

	losses, val_dice, best = [], [], 0
	for ep in range(args.epochs):
		model.train()
		run = 0
		t0 = time.time()
		for x, y in train_dl:
			x, y = x.to(device), y.to(device)
			logits = model(x)
			loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
			opt.zero_grad()
			loss.backward()
			opt.step()
			run += loss.item()
		d = evaluate(model, val_dl, device)
		fg = sum(d[1:]) / (NUM_CLASSES - 1) # foreground mean (no background). 
		losses.append(run / len(train_dl))
		val_dice.append(fg)
		print(f"ep {ep+1}: loss {losses[-1]:.4f} val fg-dice {fg:.3f} "
			  f"per-class {[round(v, 2) for v in d]} ({time.time()-t0:.0f}s)")
		if fg > best:
			best = fg
			torch.save(model.state_dict(), os.path.join(args.out, f"{args.model}_best.pt"))

	fig, ax = plt.subplots(1, 2, figsize=(10, 4))
	ax[0].plot(losses)
	ax[0].set_title(f"{args.model} train loss")
	ax[1].plot(val_dice)
	ax[1].set_title("Val mean foreground Dice")
	fig.savefig(os.path.join(args.out, f"{args.model}_curves.png"))
	plt.close(fig)

if __name__ == "__main__":
	main()