import time
import torch
import numpy as np
import random
import torch.nn.functional as F
from pathlib import Path
import argparse

from torch import Tensor
from torch.utils.data import DataLoader
from dataset import SEED, LABELS, BATCH_SIZE_2D, get_dataloader_2d, get_dataloader_3d, BATCH_SIZE_3D, verify_splits
from modules import dsc_per_class, ImprovedUnet, ImprovedUnet3D, iou_per_class, fpr_per_class, fnr_per_class, pr_per_class
from utils import write_report, plot_2d_inference, plot_3d_inference

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)

# ======================================================================================================================
# =                                               Score Summary Helpers                                                =
# ======================================================================================================================
def _one_hot(labels: Tensor, K: int) -> Tensor:
    """
    Converts [N, (D), H, W] to [N, K, (D), H, W]
    :param labels: [N, (D), H, W] labels
    :param K: Number of classes
    :return: Mask with one-hot encoding
    """
    return F.one_hot(labels.long(), K).movedim(-1, 1).float()


@torch.no_grad()
def compute_metrics(predictions: Tensor, targets: Tensor, K: int = len(LABELS)) -> dict[str, Tensor]:
    """
    Label maps [N, K, H, W] (2D) or [N, K, D, H, W] (3D). Called after processing every batch.
    Evaluates:
        DSC (+ empty),\n
        DSC (present),\n
        DSC (pooled),\n
        IoU (pooled),\n
        FPR (pooled),\n
        FNR (pooled),\n
        Precision,
        Recall
    Every value returned has shape [K].
    :param predictions: Predictions
    :param targets: Targets (Ground Truth)
    :param K: Number of classes
    :return: Evaluation metrics
    """

    present = targets.flatten(2).sum(2) > 0     # Flattens to [N, K, dims] then sum to get tensor of shape [N, K]
                                                # Like creating a mask to only filter out scores where the class of
                                                # interest exists in the ground truth
    dsc = dsc_per_class(predictions, targets)   # [N, K] per slice
    dsc_wempty = dsc.mean(0)
    dsc_present = (dsc * present).sum(0) / present.sum(0).clamp(min=1)

    # Change [N, K, dims] to [K, N, dims]
    # Unsqueeze to make it [1, K, N, dims]. Now can be treated as a tensor with batch size of 1.
    predictions = predictions.movedim(0, 1).unsqueeze(0)
    targets = targets.movedim(0, 1).unsqueeze(0)

    dsc_pooled = dsc_per_class(predictions, targets)[0]
    iou_pooled = iou_per_class(predictions, targets)[0]
    pixel_fpr = fpr_per_class(predictions, targets)[0]
    pixel_fnr = fnr_per_class(predictions, targets)[0]
    pixel_pr, pixel_re = pr_per_class(predictions, targets)

    return {
        "n_present"         : present.sum(0),
        "dsc_wempty"        : dsc_wempty,
        "dsc_present"       : dsc_present,
        "dsc_pooled"        : dsc_pooled,
        "iou_pooled"        : iou_pooled,
        "pixel_fpr"         : pixel_fpr,
        "pixel_fnr"         : pixel_fnr,
        "precision"         : pixel_pr[0],
        "recall"            : pixel_re[0],
    }


def metrics_to_string(metrics: dict) -> str:
    lines = [
        f"{'class':<13}"
        f"{'#present':>9}"
        f"{'Dice(+empty)':>14}"
        f"{'Dice(present)':>15}"
        f"{'Dice(pooled)':>14}"
        f"{'IoU(pooled)':>13}"
        f"{'pixelFPR':>10}"
        f"{'pixelFNR':>10}"
        f"{'precision':>10}"
        f"{'recall':>10}"
    ]
    for k, name in enumerate(LABELS):
        lines.append(
            f"{name:<13}"
            f"{int(metrics['n_present'][k]):>9}"
            f"{metrics['dsc_wempty'][k]:>14.3f}"
            f"{metrics['dsc_present'][k]:>15.3f}"
            f"{metrics['dsc_pooled'][k]:>14.3f}"
            f"{metrics['iou_pooled'][k]:>13.3f}"
            f"{metrics['pixel_fpr'][k]:>10.4f}"
            f"{metrics['pixel_fnr'][k]:>10.4f}"
            f"{metrics['precision'][k]:>10.4f}"
            f"{metrics['recall'][k]:>10.4f}"
        )
    return "\n".join(lines) + "\n"


# ======================================================================================================================
# =                                               Inference (Test) Loop                                                =
# ======================================================================================================================
@torch.no_grad()
def run_inference2d(model: ImprovedUnet, loader: DataLoader, device: torch.device, batch_size: int) -> tuple:
    """
    Run inference with 2D Improved UNet
    :param model: 2D Improved UNet
    :param loader: Test set dataloader
    :param device: CUDA or CPU
    :param batch_size: Test batch size
    :return: Inference results
    """
    predictions_all, targets_all = list(), list()
    inference_times = list()
    results = []

    print("Running inference...")
    model.eval()
    for bidx, (test_images, test_masks) in enumerate(loader):
        start_time = time.time()

        test_images = test_images.to(device)
        test_masks = test_masks.to(device)
        predictions = model(test_images)

        end_time = time.time()
        inference_times.append(end_time - start_time)

        predictions, test_masks = predictions.argmax(1).to(torch.uint8), test_masks.argmax(1).to(torch.uint8)
        predictions, test_masks = _one_hot(predictions, len(LABELS)), _one_hot(test_masks, len(LABELS))
        results.append((test_masks, predictions))

        predictions_all.append(predictions)
        targets_all.append(test_masks)

    plot_2d_inference(results, device, len(loader), batch_size)
    metrics = compute_metrics(torch.cat(predictions_all), torch.cat(targets_all))
    table = metrics_to_string(metrics)
    print(table)

    inference_times_str = (f"Mean Inference Time: {np.mean(inference_times):.2f}s\n"
                           f"Max Inference Time: {np.max(inference_times):.2f}s\n")

    return table, inference_times_str


@torch.no_grad()
def run_inference3d(model: ImprovedUnet3D, loader: DataLoader, device: torch.device, batch_size: int) -> tuple:
    """
    Run inference with 3D Improved UNet
    :param model: 3D Improved UNet
    :param loader: Test set dataloader
    :param device: CUDA or CPU
    :param batch_size: Test batch size
    :return: Inference results
    """
    predictions_all, targets_all = list(), list()
    inference_times = list()
    results = []

    print("Running inference...")
    model.eval()
    for batch_idx, (image, target) in enumerate(loader):
        start_time = time.time()
        print(f"Batch: {batch_idx + 1}/{len(loader)}")
        image, target = image.to(device), target.to(device)
        predictions = model(image)
        end_time = time.time()
        inference_times.append(end_time - start_time)

        predictions, target = predictions.argmax(1).to(torch.uint8), target.argmax(1).to(torch.uint8)
        predictions, target = _one_hot(predictions, len(LABELS)), _one_hot(target, len(LABELS))
        results.append((target, predictions))

        predictions_all.append(predictions)
        targets_all.append(target)

    plot_3d_inference(results, device)
    metrics = compute_metrics(torch.cat(predictions_all), torch.cat(targets_all))
    table = metrics_to_string(metrics)
    print(table)

    inference_times_str = (f"Mean Inference Time p/V: {np.mean(inference_times):.2f}s\n"
                           f"Max Inference Time p/V: {np.max(inference_times):.2f}s\n")

    return table, inference_times_str


# ======================================================================================================================
# =                                                     Main Guard                                                     =
# ======================================================================================================================
def main(argsv):
    model_dir = Path("trained_model/")
    model_dir.mkdir(parents=True, exist_ok=True)

    if argsv.model == "3D":
        # For 3D Improved UNet
        model = ImprovedUnet3D(in_channels=1, out_channels=len(LABELS))
        model = model.to(device)
        model_path = model_dir / r"unet_improved_3D.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_dataloader = get_dataloader_3d("test")
        inference_result = run_inference3d(model, test_dataloader, device, BATCH_SIZE_3D["test"])
    elif argsv.model == "2D":
        # For 2D Improved UNet
        model = ImprovedUnet(in_channels=1, out_channels=len(LABELS))
        model = model.to(device)

        model_path = model_dir / r"unet_improved_dsc.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))

        test_dataloader = get_dataloader_2d("test")
        inference_result = run_inference2d(model, test_dataloader, device, BATCH_SIZE_2D["test"])
    else:
        raise Exception("Unknown model")

    # Write to file
    out_path = Path(f"trained_model/inference_results_{argsv.model}.txt")
    write_report(out_path, inference_result)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type = str, required = True)
    args = ap.parse_args()

    output_dir = Path("images/")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not verify_splits(args.model):
        raise Exception("Please retrain data with the new splits")
    main(args)
