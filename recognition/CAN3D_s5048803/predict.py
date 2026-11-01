"""
predict.py: Showing example usage of your trained model. It should load your saved model
weights, run inference on test cases, print out any results, and provide visualisations where applicable
(e.g., prediction overlays, generated samples, heatmaps). NumPy is allowed in this script for
visualization or loading data.

Usage:
	python predict.py --weights2d models/can2d_best.pt --weights3d models/can3d_best.pt
"""

import argparse, os
import numpy as np
import time
import torch
import resource, sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from modules import CAN
from dataset import (get_pairs, split_pairs, HipMRI3D, get_slices, HipMRI2D, key, slice_key,
					 NUM_CLASSES, FG_NAMES, RANGPUR_MODE)


def get_device():
	return ("cuda" if torch.cuda.is_available()
			else "mps" if torch.backends.mps.is_available() else "cpu")


def load_model(weights, dim, device):
	"""Rebuild the architecture, then load the trained weights into it."""
	model = CAN(in_ch=1, num_classes=NUM_CLASSES, dim=dim).to(device)
	model.load_state_dict(torch.load(weights, map_location=device))
	return model.eval()


def load_set(dim, root, scale, split="test"):
	if dim == 3:
		return HipMRI3D(split_pairs(root, get_pairs(root))[split], scale)
	return HipMRI2D(get_slices(root, split), scale)


def planes(t, axis=2):
	"""Stack of 2D planes cut along an axis (default 2, coronal), same layout as 2D slice"""
	return t.movedim(axis, 0).transpose(1, 2)


def confusion(pred, target):
	"""NUM_CLASSES x NUM_CLASSES counts. Rows = ground truth class, columns = predicted class."""
	idx = target.flatten() * NUM_CLASSES + pred.flatten()
	return torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_conf(conf):
	"""Calculate per-class segmentation metrics from confusion matrix"""
	tp = conf.diag().float()
	fp = conf.sum(0).float() - tp
	fn = conf.sum(1).float() - tp
	gt = (tp + fn).clamp(min=1)
	result = dict(dice=2 * tp / (2 * tp + fp + fn).clamp(min=1),
				  iou=tp / (tp + fp + fn).clamp(min=1),
				  over=100 * fp / gt,
				  under=100 * fn / gt)
	return {k: v[1:] for k, v in result.items()}


def metrics(pred, target):
	"""Get metrics from prediction and target"""
	return metrics_from_conf(confusion(pred, target))


@torch.no_grad()
def evaluate(model, dataset, dim, device):
	"""Pooled confusion matrix over a whole test set (batches of slices for 2D, one volume for 3D)."""
	conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)

	for x, y in DataLoader(dataset, batch_size=32 if dim == 2 else 1):
		pred = model(x.to(device)).argmax(1)
		conf += confusion(pred, y.to(device))
	return conf.cpu()


def print_table(name, metric_result):
	"""Print Dice, IoU, over -and under contouring for all foreground classes"""
	print(f"{name}, mean metrics:")
	for i, cname in enumerate(FG_NAMES):
		print(f"  {cname:>8}: Dice {metric_result['dice'][i]:.3f}  IoU {metric_result['iou'][i]:.3f}  "
			  f"over {metric_result['over'][i]:.1f}%  under {metric_result['under'][i]:.1f}%")
	print(f"  mean foreground: Dice {metric_result['dice'].mean():.3f}  IoU {metric_result['iou'].mean():.3f}\n")


# Plot a single slice's segmentation, metrics and difference plot (to be animated as gifs)
@torch.no_grad()
def plot_item(model, dataset, patient, device, dim, cls, index=32, batch=True, axis=2, show_total_error=True):
	x, y = dataset[patient]

	pred = model(x[None].to(device)).argmax(1)[0].cpu()

	if dim == 3:
		img = planes(x[0], axis)
		gt = planes(y, axis)
		pred = planes(pred, axis)
		img = img[index]
		gt = gt[index]
		pred = pred[index]
		name = f"CAN3D patient {patient}, slice {index}"
	else:
		img = x[0]
		gt = y
		name = f"CAN2D slice {patient}"

	metric_results = metrics(pred, gt)
	stats = (f"Dice {metric_results['dice'][cls-1]:.3f} | IoU {metric_results['iou'][cls-1]:.3f}\n"
			f"over {metric_results['over'][cls-1]:.1f}% | under {metric_results['under'][cls-1]:.1f}%")


	fig, ax = plt.subplots(1, 4, figsize=(12, 4))

	ax[0].imshow(img, cmap="gray")
	ax[0].set_title(f"{name} - MRI")

	ax[1].imshow(gt, cmap="tab10", vmin=0, vmax=9)
	ax[1].set_title("Ground truth")

	ax[2].imshow(pred, cmap="tab10", vmin=0, vmax=9)
	ax[2].set_title(f"Prediction ({FG_NAMES[cls-1]})\n{stats}")

	ax[3].imshow(img, cmap="gray")
	if show_total_error:
		# Difference overlay: red = total error
		ax[3].imshow((pred != gt).float(), cmap="Reds", alpha=0.5)
		ax[3].set_title("Difference (red = wrong)")
	else:
		# Difference overlay: red = false positive, blue = false negative
		fp = (pred == cls) & (gt != cls)
		fn = (gt == cls) & (pred != cls)
		overlay = torch.zeros(*img.shape, 4)
		overlay[fp] = torch.tensor([1.0, 0.0, 0.0, 0.6])
		overlay[fn] = torch.tensor([0.0, 0.4, 1.0, 0.6])

		ax[3].imshow(overlay)
		ax[3].set_title(f"{FG_NAMES[cls-1]}: FP (red) / FN (blue)")

	for a in ax:
		a.axis("off")

	fig.tight_layout()

	folder = f"results/{dim}d_slices/" if batch else "results/"
	suffix = f"_ax{axis}" if dim == 3 else ""
	os.makedirs(folder, exist_ok=True)
	fig.savefig(f"./{folder}can{dim}d_slice_{index}{suffix}.png", dpi=200)
	plt.close(fig)


def plot_dice_comparison(m2d, m3d, path):
	"""Grouped bar plot: per-class Dice of CAN2D next to CAN3D."""
	x = np.arange(len(FG_NAMES))
	fig, ax = plt.subplots(figsize=(8, 4))
	ax.bar(x - 0.2, m2d["dice"].numpy(), 0.4, label="CAN2D")
	ax.bar(x + 0.2, m3d["dice"].numpy(), 0.4, label="CAN3D")
	ax.set_xticks(x)
	ax.set_xticklabels(FG_NAMES)
	ax.set_ylabel("Dice")
	ax.set_ylim(0.5, 1)
	ax.set_title("Per-class Dice on the test set")
	ax.legend()
	fig.tight_layout()
	fig.savefig(path)
	plt.close(fig)


def mem_mb(device):
	"""Memory in MB: CUDA = peak allocated; MPS = memory held by the Metal driver; CPU = peak process RSS."""
	if device == "cuda":
		return torch.cuda.max_memory_allocated() / 1024**2
	if device == "mps":
		return torch.mps.driver_allocated_memory() / 1024**2
	rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss      # bytes on macOS, KB on Linux
	return rss / 1024**2 if sys.platform == "darwin" else rss / 1024


@torch.no_grad()
def time_mem_inference(model, x, device, warmup=3, runs=10):
	"""Mean/std seconds for one forward pass of x (shaped [B, C, ...]) and peak GPU memory in MB."""
	x = x.to(device)
	sync = torch.cuda.synchronize if device == "cuda" else (torch.mps.synchronize if device == "mps" else (lambda: None))
	if device == "mps":
		torch.mps.empty_cache()
	for _ in range(warmup): # warm-up
		model(x)
	sync()
	if device == "cuda":
		torch.cuda.reset_peak_memory_stats()   # count only the timed runs, starting from weights + input
	times = []
	peak_mem = 0.0
	for _ in range(runs):
		sync()
		t0 = time.perf_counter()
		model(x)
		sync()
		times.append(time.perf_counter() - t0)
		peak_mem = max(peak_mem, mem_mb(device))
	t = torch.tensor(times)
	return t.mean().item(), t.std().item(), peak_mem


def num_parameters(model, dim):
	total_params = sum(p.numel() for p in model.parameters())
	print(f"Total parameters {dim}D: {total_params:,}")


@torch.no_grad()
def slice_metrics(pred, y):
	results = {
		"dice": [],
		"iou": [],
		"over": [],
		"under": []
	}

	for layer in range(len(y)):
		metric_results = metrics(pred[layer], y[layer])

		for name in results:
			results[name].append(metric_results[name])

	return {
		name: torch.stack(values)
		for name, values in results.items()
	}


def plot_slice_metrics(results2d, results3d, has_prostate, cls, x, vol):
	metrics_to_plot = ["dice", "iou", "over", "under"]
	titles = ["Dice", "IoU", "Over-contouring", "Under-contouring"]

	fig, ax = plt.subplots(2, 2, figsize=(12, 8))
	ax = ax.flatten()
	for a, metric, title in zip(ax, metrics_to_plot, titles):
		values2d = results2d[metric][:, cls - 1].numpy()
		values3d = results3d[metric][:, cls - 1].numpy()
		
		if (metric == "dice"):
			a.axhline(y=0.7, color='r')

		a.plot(x, values2d, color='blue', label="CAN2D")
		a.plot(x, values3d, color='green', label="CAN3D")

		# Plot the prostate ground truth area
		prostate_segment = np.flatnonzero(has_prostate)
		lo = x[prostate_segment[0]]
		hi = x[prostate_segment[-1]]
		a.axvspan(lo, hi, color="orange", alpha=0.2, label="Prostate GT")

		a.set_title(f"Per-layer {title}")
		a.set_xlabel("Slice")
		a.set_ylabel("%" if metric in ("over", "under") else metric)
		a.grid()
		a.legend()

		if metric in ("dice", "iou"):
			a.set_ylim(0, 1)

	for a in ax:
		a.xaxis.set_major_locator(MaxNLocator(integer=True))

	fig.tight_layout()
	fig.savefig(f"./results/can_depth_comparison_vol{vol}.png", dpi=200)
	plt.close(fig)


@torch.no_grad()
def compare_slice_metrics(runs, root, vol, cls, device, scale2d):
	ds3 = runs[3]["dataset"]

	matched = []

	# Match the 2d slices with the 3d volumes
	for i, pair in enumerate(ds3.pairs):
		slices = matching_slices(root, pair)

		if slices:
			matched.append((i, slices))

	i3, slices = matched[vol]
	x3, y3 = ds3[i3]

	pred3 = runs[3]["model"](
		x3[None].to(device)
	).argmax(1)[0].cpu()

	ds2 = HipMRI2D(slices, scale2d)

	x2 = torch.stack([ds2[i][0] for i in range(len(ds2))])
	y2 = torch.stack([ds2[i][1] for i in range(len(ds2))])

	pred2 = runs[2]["model"](
		x2.to(device)
	).argmax(1).cpu()

	y2 = y2[..., :64] # Drop the padding
	pred2  = pred2[..., :64]
	
	nums = torch.tensor([int(slice_key(p[0])[2]) for p in slices])

	v = planes(y3)
	score = (y2[:, None] == v[None]).float().mean((2, 3))   # (n2d, n_planes)
	idx = score.argmax(1)

	y3 = planes(y3)[idx]
	pred3 = planes(pred3)[idx]

	has_prostate = (y2 == cls).flatten(1).any(1).numpy() 

	results2d = slice_metrics(pred2, y2)
	results3d = slice_metrics(pred3, y3)

	plot_slice_metrics(
		results2d,
		results3d,
		has_prostate,
		cls,
		nums,
		vol
	)


def matching_slices(root, pair3d):
	"""The keras test slices from the same patient and week as one 3D volume, ordered by slice number."""
	pid, week = key(pair3d[0])
	slices = [p for p in get_slices(root, "test")
		  if slice_key(p[0])[0] == pid and slice_key(p[0])[1] == week]
	return sorted(slices, key=lambda p: int(slice_key(p[0])[2]))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--weights2d", default="./models/can2d_best.pt", help="path to the CAN2D .pt file")
	ap.add_argument("--weights3d", default="./models/can3d_best.pt", help="path to the CAN3D .pt file")
	ap.add_argument("--scale2d", type=float, default=0.5, help="must match CAN2D training")
	ap.add_argument("--scale3d", type=float, default=0.5, help="must match CAN3D training")
	ap.add_argument("--cls", type=int, default=5, help="Class to plot metrics of (5=prostate)")
	ap.add_argument("--out", default="results", help="Folder to put model results")
	ap.add_argument("--data_root", default="/home/groups/comp3710/HipMRI_Study_open" if RANGPUR_MODE else ".")
	ap.add_argument("--show_total_error", type='store_true', default=False, help="True -> plot total error in sliceprediction, False -> plot fp / fn in sliceprediction")
	args = ap.parse_args()
	os.makedirs(args.out, exist_ok=True)
	device = get_device()

	# Evaluate both models' Dice, IoU, over -and under contouring for every class (except background)
	runs = {} # Holds dataset and metrics of entire testset for both models
	for dim, weights, scale, split in ((3, args.weights3d, args.scale3d, "test"),
	                                   (2, args.weights2d, args.scale2d, "test"),
	                                   (3, args.weights3d, args.scale3d, "val"),
	                                   (2, args.weights2d, args.scale2d, "val")):
		model = load_model(weights, dim, device)
		dataset = load_set(dim, args.data_root, scale, split)

		num_parameters(model, dim)

		metric_results = metrics_from_conf(evaluate(model, dataset, dim, device)) # Evaluate runs on whole testset
		print_table(f"CAN{dim}D {split}", metric_results)
		if split == "test":
			runs[dim] = dict(label=f"CAN{dim}D", model=model, dataset=dataset, metrics=metric_results)

	# Run both models on a complete volume / set of slices
	pairs = load_set(3, args.data_root, args.scale3d).pairs # 3D pairs, to get patient/week
	slices = matching_slices(args.data_root, pairs[0]) # sorted, one patient/week only
	ds_patient = HipMRI2D(slices, args.scale2d)
	for i in range(len(ds_patient)):
		plot_item(runs[2]["model"], ds_patient, i, device, 2, args.cls, i, show_total_error=args.show_total_error)
	for i in range(128):
		plot_item(runs[3]["model"], runs[3]["dataset"], 0, device, 3, args.cls, i, show_total_error=args.show_total_error)

	# Run CAN3D on the 3 different axis
	plot_item(runs[3]["model"], runs[3]["dataset"], 1, device, 3, args.cls, 30, False, axis=0, show_total_error=False)
	plot_item(runs[3]["model"], runs[3]["dataset"], 1, device, 3, args.cls, 61, False, axis=1, show_total_error=False)
	plot_item(runs[3]["model"], runs[3]["dataset"], 1, device, 3, args.cls, 61, False, axis=2, show_total_error=False)

	# Plot per-slice metrics
	for vol in range(3):
		compare_slice_metrics(
		runs,
		args.data_root,
		vol,
		args.cls,
		device,
		args.scale2d
	)

	# per-class Dice, 2D vs 3D
	plot_dice_comparison(runs[2]["metrics"], runs[3]["metrics"], os.path.join(args.out, "dice_comparison.png"))

	# Run inference time and memory usage for both models
	x3d = runs[3]["dataset"][0][0][None] # (1, 1, D, H, W): one test volume
	x2d = x3d[0].permute(1, 0, 2, 3) # (D, 1, H, W): the same volume as D slices in one batch

	for name, model, x in (("CAN3D", runs[3]["model"], x3d), ("CAN2D", runs[2]["model"], x2d)):
		mean, std, mem = time_mem_inference(model, x, device)
		print(f"{name}: {mean*1000:.1f} +/- {std*1000:.1f} ms per volume, peak {mem:.0f} MB")

if __name__ == "__main__":
	main()