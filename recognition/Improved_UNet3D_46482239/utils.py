import math
import random
import numpy as np
from tqdm import tqdm

import dataset
import glob
import os
import torch
import pandas as pd
import matplotlib as mpl
import matplotlib.animation as animation
from torch import Tensor
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
from pathlib import Path

from dataset import SEED, LABELS
from modules import dsc_per_class

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ======================================================================================================================
# =                                                 Reporting Helpers                                                  =
# ======================================================================================================================
def write_report(out_path: Path, metrics: tuple) -> None:
    """
    Write report to a txt file
    :param out_path: Path/dir to save the .txt report
    :param metrics: The metrics to write into the report
    """
    report = str()
    divider_length = 100

    for metric in metrics:
        report = report + metric
        report = report + "=" * divider_length + "\n\n"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    report_file = open(out_path, "w")
    report_file.write(report)
    report_file.close()


# ======================================================================================================================
# =                                                  Plotting Helpers                                                  =
# ======================================================================================================================
def plot_training(out_path: Path, t_loss: list[float], v_loss: list[float], t_score: list[float], v_score: list[float]) -> None:
    """
    Plot training loss vs validation loss and training dice scores vs validation dice scores.
    :param out_path: Path/dir to save the plot
    :param t_loss: Training DSC+CE losses
    :param v_loss: Validation DSC+CE losses
    :param t_score: Training DSC+CE scores
    :param v_score: Validation DSC+CE scores
    """

    # Plot training loss vs validation loss
    fig, ax1 = plt.subplots(figsize=(700, 700, "px"))
    ax1.plot(t_loss, color='tab:red', label='Training DSC+CE Loss')
    ax1.plot(v_loss, color='tab:orange', label='Validation DSC+CE Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Plot training dice score vs validation dice score
    ax2 = ax1.twinx()
    ax2.plot(t_score, color='tab:blue', label='Training DSC+CE Score')
    ax2.plot(v_score, color='tab:green', label='Validation DSC+CE Score')
    ax2.set_ylabel('DSC+CE Score', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    fig.legend(loc="lower left")

    plt.title("U-Net Segmentation Training Curve")
    fig.tight_layout()

    fig.savefig(out_path)
    plt.show()


def plot_2d_inference(results: list[tuple[Tensor, Tensor]], device: torch.device, loader_len: int, batch_size: int) -> None:
    """
    Visualisation of the inference results for the 2D Improved U-Net segmentation model.
    Incl:
        predicted segmentation of a random slice from each batch,
        cumulative error between ground truth and prediction masks for each of the random slices.
        ground truth vs predicted masks for each of the classes for each of the random slice from each batch + error.
    :param results: Ground truth and predicted masks for each batch
    :param device: CUDA or CPU
    :param loader_len: length of the dataloader
    :param batch_size: Batch size
    """
    output_dir = Path("images/")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_cum_cols = 4
    fig_cum_rows = int(math.ceil(loader_len / fig_cum_cols))
    fig_cum, axes_cum = plt.subplots(fig_cum_rows, fig_cum_cols, figsize=(1540, 1540, "px"))
    fig_cum_diff, axes_cum_diff = plt.subplots(fig_cum_rows, fig_cum_cols, figsize=(1540, 1540, "px"))
    four_colour_cmap = mpl.colormaps['jet'].resampled(len(LABELS))

    err_cmap = plt.get_cmap("Spectral_r")
    err_cmap = LinearSegmentedColormap.from_list(f"", err_cmap([0.15, 0.5, 0.85])).resampled(3)
    cum_cmap = plt.get_cmap("Reds").resampled(2)

    for bidx, (test_masks, predictions) in enumerate(results):
        fig_main, axes_main = plt.subplots(3, len(LABELS), figsize=(1440, 900, "px"))
        sample_no = random.randint(0, len(test_masks) - 1)

        diff_in_masks = test_masks[sample_no] - predictions[sample_no]

        # For plotting the gnd vs preds + fp & fn
        for label_idx in range(len(LABELS)):
            test_mask_pixels = test_masks[sample_no][label_idx].cpu()
            reproduced_mask_pixels = predictions[sample_no][label_idx].cpu()
            diff_in_mask_pixels = -1 * diff_in_masks[label_idx].cpu()

            axes_main[0, label_idx].set_title(LABELS[label_idx], fontsize=12)
            # Ground truth mask
            axes_main[0, label_idx].imshow(test_mask_pixels.squeeze(), cmap="grey")
            axes_main[0, label_idx].axis("off")

            # Predicted mask
            axes_main[1, label_idx].imshow(reproduced_mask_pixels.squeeze(), cmap="grey")
            axes_main[1, label_idx].axis("off")

            # Difference in mask
            im = axes_main[2, label_idx].imshow(diff_in_mask_pixels.squeeze(), cmap=err_cmap, vmin=-1.0, vmax=1.0, interpolation="nearest")
            fig_main.colorbar(im, orientation='vertical', ticks = [-1, 0, 1])
            axes_main[2, label_idx].axis("off")
            fig_main.supylabel('Difference        |        Reconstructed        |        Ground Truth', ha='left',
                               va='center', fontsize=18)

        fig_main.suptitle(f"Random Sample ({sample_no + 1}/{batch_size}) from batch {bidx + 1}", fontsize=14)
        fig_main.tight_layout()
        out_path = output_dir / f"rand_recon_sample_batch_{bidx + 1}.png"
        fig_main.savefig(out_path)

        # For plotting the wrong predictions of the whole scan
        diff_in_masks_cum = torch.sum(torch.abs(diff_in_masks), dim=0)
        cum_rows, cum_cols = int(bidx // fig_cum_cols), bidx % fig_cum_cols
        cum_sub_title = f"Cum Diff Batch {bidx + 1}"

        # Cumulative difference masks
        im_cum_diff = axes_cum_diff[cum_rows, cum_cols].imshow(diff_in_masks_cum.cpu().squeeze(), cmap=cum_cmap, vmin=0.0,
                                                               vmax=1.0, interpolation="nearest")
        fig_cum_diff.colorbar(im_cum_diff, orientation='vertical', ticks = [0, 1])
        axes_cum_diff[cum_rows, cum_cols].axis("off")
        axes_cum_diff[cum_rows, cum_cols].set_title(cum_sub_title, fontsize=12)

        # For plotting the predicted segmentation
        cum_mask = torch.zeros_like(predictions[sample_no][0]).to(device)
        for class_idx in range(predictions[sample_no].shape[0]):
            class_mask = predictions[sample_no][class_idx]

            # Ceiling the class
            class_mask_ceil = torch.where(
                class_mask > 0.5,
                torch.ceil(class_mask) * class_idx,
                class_mask
            )
            cum_mask = torch.add(cum_mask, class_mask_ceil)
        im_cum = axes_cum[cum_rows, cum_cols].imshow(cum_mask.cpu().squeeze(), cmap=four_colour_cmap, vmin=0.0,
                                                     vmax=len(LABELS))
        fig_cum.colorbar(im_cum, orientation='vertical')
        axes_cum[cum_rows, cum_cols].axis("off")
        axes_cum[cum_rows, cum_cols].set_title(cum_sub_title, fontsize=12)

    fig_cum_diff.suptitle("Cumulative Difference Between GND Truth and Reconstructed Masks", fontsize=18)
    out_path = output_dir / f"cum_diff_in_masks_{loader_len}_batches.png"
    fig_cum_diff.savefig(out_path)

    fig_cum.suptitle("MRI Masks Segmented", fontsize=18)
    out_path = output_dir / f"cum_masks_{loader_len}_batches.png"
    fig_cum.savefig(out_path)
    plt.show()
    plt.close()


def plot_3d_inference(results: list[tuple[Tensor, Tensor]], device: torch.device) -> None:
    """
    Visualisation of the inference results for the 3D Improved U-Net segmentation model.
    Incl:
        predicted
    :param results: Ground truth and predicted masks for each batch
    :param device: CUDA or CPU
    :param loader_len: length of the dataloader
    """
    output_dir = Path("images/")
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 1  # We use batch size of 1 for testing 3D volumes

    p_sup_gt, p_sup_pd = list(), list()
    p_m_gt, p_m_pd = list(), list()
    p_inf_gt, p_inf_pd = list(), list()

    for volume_no, (predicted, ground_truth) in enumerate(results):
        # Make both masks [K, D, H, W]
        predicted = torch.squeeze(predicted, dim=0).cpu().numpy()
        ground_truth = torch.squeeze(ground_truth, dim=0).cpu().numpy()

        diff_in_masks = ground_truth - predicted
        fp = (diff_in_masks == -1).astype(int)      # False Positives
        fn = (diff_in_masks == 1).astype(int)       # False Negatives
        K, D, H, W = ground_truth.shape

        # 3D scatter plot?
        scatter_3d(output_dir, fp, fn, K, volume_no)

        # Sliding w.r.t depth
        # Essentially a gif of a 2D slice at each depth.
        # Also pool together superior, middle, and inferior slices for precessing later.
        superior_gt, superior_pd, middle_gt, middle_pd, inferior_gt, inferior_pd = sliding_plot(output_dir, device, volume_no, predicted, ground_truth)
        # All returned should be of shape [1, K, H, D]
        p_sup_gt.append(superior_gt)
        p_sup_pd.append(superior_pd)
        p_m_gt.append(middle_gt)
        p_m_pd.append(middle_pd)
        p_inf_gt.append(inferior_gt)
        p_inf_pd.append(inferior_pd)

    p_sup = (torch.cat(p_sup_gt, dim=0), torch.cat(p_sup_pd, dim=0))
    p_m = (torch.cat(p_m_gt, dim=0), torch.cat(p_m_pd, dim=0))
    p_inf = (torch.cat(p_inf_gt, dim=0), torch.cat(p_inf_pd, dim=0))
    plot_prostate(output_dir, p_sup, p_m, p_inf)


def scatter_3d(output_dir: Path, fp: np.ndarray, fn: np.ndarray, K: int, volume_no: int) -> None:
    """
    Helper method to plot the 3D scatter plot to visualise each mask
    :param output_dir:
    :param fp: False positive in predicted masks
    :param fn: False negative in predicted masks
    :param K: Number of classes
    :param volume_no: Which volume this instance is out of the entire test set.
    """

    fig = plt.figure(figsize=(1980, 500, "px"))
    for channel in range(K):
        ax = fig.add_subplot(1, K, channel + 1, projection='3d')
        # Extract coordinates where false positives occur
        z_fp, y_fp, x_fp = np.where(fp[channel] == 1)
        if len(x_fp) > 0:
            ax.scatter(x_fp, y_fp, z_fp, label=f'FP Ch {channel}', alpha=0.6, s=1)

        # Extract coordinates where false negatives occur
        z_fn, y_fn, x_fn = np.where(fn[channel] == 1)
        if len(x_fn) > 0:
            ax.scatter(x_fn, y_fn, z_fn, label=f'FN Ch {channel}', alpha=0.6, s=1)

        ax.set_xlabel("Width (W)", fontsize=7)
        ax.set_ylabel("Height (H)", fontsize=7)
        ax.set_zlabel("Depth (D)", fontsize=7)
        ax.set_title(f"Prediction error for {LABELS[channel]}", fontsize=8)
        ax.legend()

    fig.subplots_adjust(wspace=1.0, hspace=1.0)
    fig.tight_layout()
    plt.suptitle(f"3D Inference Error Plot - Volume {volume_no + 1}", fontsize=11)
    plt.savefig(output_dir / f"volume_{volume_no + 1}_scatter.png", dpi=100)
    # plt.show()
    plt.close()


def sliding_plot(output_dir: Path, device: torch.device, v_no: int, predicted: np.ndarray, ground_truth: np.ndarray) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Takes a 3D predicted and ground truth mask of shape [K, D, H, W] and creates a .gif of the two 2D segmentation as it
    slides along a chosen dimension.
    Gif created following matplotlib documentation [5]
    :param output_dir: Path to the output directory
    :param device: CUDA or CPU
    :param v_no: Volume number
    :param predicted: Predicted mask
    :param ground_truth: Ground truth mask
    :return: The selected superior, middle, and inferior scans for this volume.
    """
    predicted = torch.from_numpy(predicted).to(device)
    ground_truth = torch.from_numpy(ground_truth).to(device)

    # Superior to inferior end.
    dims = (3, 0, 2, 1) # shape = [W, K, H, D]
    predicted = torch.permute(predicted, dims)
    ground_truth = torch.permute(ground_truth, dims)
    # Can treat it as 2D scans with batch size of D|H|W.

    dsc_scores = dsc_per_class(predicted, ground_truth).cpu() # [W, K] per slice
    W, K = dsc_scores.shape

    channel = 5 # Prostate label no.
    prostate_slices = list()    # Indices where prostate is present in GT

    fig_gifs, ax_gifs = plt.subplots(1, 3)  # 1 row of 3 gifs, showing the GT, predictions, and diff
    imgs_gif = list()
    cmap_whole = plt.get_cmap("nipy_spectral")
    cmap_delta = plt.get_cmap("cool")
    print(f"Rendering frames for Volume {v_no + 1} gif...")

    gnd_truth = ground_truth.argmax(1).to(torch.uint8)  # Should give [W, H, D]
    pred = predicted.argmax(1).to(torch.uint8)          # Should give [W, H, D]
    delta = -1 * (gnd_truth - pred)                     # Result includes fp and fn
    for layer in range(W):
        if ground_truth[layer][channel].sum() > 0:
            # This slice actually contains the prostate!
            prostate_slices.append(layer)

        # Create frame (image)
        gt = gnd_truth[layer].cpu().numpy()     # Should give [H, D]
        pd = pred[layer].cpu().numpy()          # Should give [H, D]
        dm = delta[layer].cpu().numpy()         # Should give [H, D]

        img_gt = ax_gifs[0].imshow(gt, cmap = cmap_whole, animated=True, vmin=0, vmax=6)
        img_pd = ax_gifs[1].imshow(pd, cmap = cmap_whole, animated=True, vmin=0, vmax=6)
        img_dm = ax_gifs[2].imshow(dm, cmap = cmap_delta, animated=True)
        if layer == 0:
            img_gt = ax_gifs[0].imshow(gt, cmap = cmap_whole, animated=True, vmin=0, vmax=6)
            img_pd = ax_gifs[1].imshow(pd, cmap = cmap_whole, animated=True, vmin=0, vmax=6)
            img_dm = ax_gifs[2].imshow(dm, cmap = cmap_delta, animated=True)
        imgs_gif.append((img_gt, img_pd, img_dm),)

    for ax in ax_gifs:
        ax.axis("off")

    ax_gifs[0].set_title("Ground Truth", fontsize=12)
    ax_gifs[1].set_title("Predicted", fontsize=12)
    ax_gifs[2].set_title("Errors", fontsize=12)

    gif_duration = 15000 # ms
    intvl = int(gif_duration / W)
    fps = int(1000 / intvl)
    anim_gif = animation.ArtistAnimation(fig_gifs, imgs_gif, interval=intvl, repeat_delay=0, repeat=True)

    # plt.show()
    gif_writer = animation.PillowWriter(fps = fps, metadata=dict(artist='Me'), bitrate=1800)
    anim_gif.save(output_dir / f"masks_{v_no + 1}.gif",writer=gif_writer)
    plt.close()

    # Now filter prostate slices
    # Extract slices
    gt = torch.index_select(ground_truth, dim = 0, index = torch.tensor(prostate_slices).to(device))   # [N, K, H, D]
    pd = torch.index_select(predicted, dim = 0, index = torch.tensor(prostate_slices).to(device))      # [N, K, H, D]

    s_gt_1, i_gt_1 = gt[0], gt[-1]              # [K, H, D]
    s_pd_1, i_pd_1 = pd[0], pd[-1]              # [K, H, D]

    mid_index = len(prostate_slices) // 2
    m_gt, m_pd = gt[mid_index], pd[mid_index]   # [K, H, D]

    superior_gt = torch.Tensor(s_gt_1).unsqueeze(dim=0).to(device)
    superior_pd = torch.Tensor(s_pd_1).unsqueeze(dim=0).to(device)
    middle_gt = torch.Tensor(m_gt).unsqueeze(dim=0).to(device)
    middle_pd = torch.Tensor(m_pd).unsqueeze(dim=0).to(device)
    inferior_gt = torch.Tensor(i_gt_1).unsqueeze(dim=0).to(device)
    inferior_pd = torch.Tensor(i_pd_1).unsqueeze(dim=0).to(device)

    return superior_gt, superior_pd, middle_gt, middle_pd, inferior_gt, inferior_pd


def plot_prostate(out_dir: Path, sup: tuple[Tensor, Tensor], mid: tuple[Tensor, Tensor], inf: tuple[Tensor, Tensor]) -> None:
    """
    Plots a subset of the slices where the prostate is present.
    1 at superior, 1 at inferior, 1 in the middle. For all three, we show the GT, PD, and Error (FP & FN).
    3 positions, each with 3 plots.
    Bladder and prostate are in contact when going through the slices top-down.
    Inputs should be tuples with 2 tensors each of [B, K, H, D]
    :param out_dir: The output directory to save the plots
    :param sup: Superior slices where the prostate is present; pooled together across the whole set
    :param mid: Middle slices where the prostate is present; pooled together across the whole set
    :param inf: Inferior slices where the prostate is present; pooled together across the whole set
    """
    channel = 5

    sup_gt, sup_pd = sup    # Both tensors should be [B, K, H, D]
    mid_gt, mid_pd = mid    # Both tensors should be [B, K, H, D]
    inf_gt, inf_pd = inf    # Both tensors should be [B, K, H, D]

    sup_dsc = dsc_per_class(sup_pd, sup_gt) # [B, K]
    mid_dsc = dsc_per_class(mid_pd, mid_gt) # [B, K]
    inf_dsc = dsc_per_class(inf_pd, inf_gt) # [B, K]

    sup_worst = torch.argmin(sup_dsc, dim = 0)[channel]  # [K]
    mid_worst = torch.argmin(mid_dsc, dim = 0)[channel]  # [K]
    inf_worst = torch.argmin(inf_dsc, dim = 0)[channel]  # [K]

    superior_pd = sup_pd[sup_worst]
    superior_gt = sup_gt[sup_worst]
    middle_pd = mid_pd[mid_worst]
    middle_gt = mid_gt[mid_worst]
    inferior_pd = inf_pd[inf_worst]
    inferior_gt = inf_gt[inf_worst]

    superior_delta = -1 * (superior_gt - superior_pd)
    middle_delta = -1 * (middle_gt - middle_pd)
    inferior_delta = -1 * (inferior_gt - inferior_pd)

    fig_rows = 3    # GT, PD, Error
    fig_cols = 3    # As per the docstring
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(800, 1000, "px"))
    cmap_whole = plt.get_cmap("nipy_spectral")
    cmap_delta = plt.get_cmap("Spectral_r").resampled(3)

    plot_data = (
        (superior_gt, superior_pd, superior_delta),
        (middle_gt, middle_pd, middle_delta),
        (inferior_gt, inferior_pd, inferior_delta),
    )
    for row, scan_positon in enumerate(plot_data):
        for col, scan in enumerate(scan_positon):
            img_data = scan[channel].cpu().numpy()
            cmap = cmap_whole if col < len(scan_positon) - 1 else cmap_delta
            if col == len(scan_positon) - 1:
                img = axes[row, col].imshow(img_data, cmap=cmap, vmin=-1.0, vmax=1.0, interpolation="nearest")
                fig.colorbar(img, orientation="vertical", ticks = [-1, 0, 1])
            else:
                img = axes[row, col].imshow(img_data, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
            axes[row, col].axis("off")
    fig.supylabel(
        'Inferior        |        Middle        |        Superior',
        ha='left',
        va='center',
        fontsize=20
    )
    fig.supxlabel("Ground Truth    |    Predicted    |    Error", fontsize=20)
    plt.tight_layout()
    plt.savefig(out_dir / "smi_slice.png", dpi=200)
    plt.close()

    dice_output = (
        f"Superior worst DSC:\t{sup_dsc[sup_worst][channel]}\n"
        f"Middle worst DSC:\t{mid_dsc[mid_worst][channel]}\n"
        f"Inferior worst DSC:\t{inf_dsc[inf_worst][channel]}\n"
    ),
    out_path = Path(r"trained_model/prostate_dice.txt")
    write_report(out_path, dice_output)


# ======================================================================================================================
# =                                                   Helper Objects                                                   =
# ======================================================================================================================
class VRAMTracker:
    def __init__(self):
        self.peak_allocated = list()
        self.peak_reserved = list()

    def update(self):
        """
        Tracks peak allocated and peak reserved VRAMs
        Called after evaluating each batch.
        :return: None
        """
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # In MB
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)  # In MB

        self.peak_allocated.append(peak_allocated)
        self.peak_reserved.append(peak_reserved)

    def evaluate(self) -> tuple[float, float,float, float]:
        """
        Evaluate the max and mean VRAM statistics
        :return: Tuple of statistics
        """
        max_peak_allocated = max(self.peak_allocated)
        max_peak_reserved = max(self.peak_reserved)
        mean_peak_allocated = np.mean(self.peak_allocated)
        mean_peak_reserved = np.mean(self.peak_reserved)

        return max_peak_allocated, max_peak_reserved, mean_peak_allocated, mean_peak_reserved


class DataAudit:
    def __init__(self):
        self.image_files = sorted(glob.glob(os.path.join(dataset.DATASET_3D_TYPES["raw"], "*.nii*")))
        self.mask_files = sorted(glob.glob(os.path.join(dataset.DATASET_3D_TYPES["mask"], "*.nii*")))

        self.raw_scan_shapes        = list()
        self.mask_scan_shapes       = list()
        self.raw_mask_match         = list()
        self.mask_num_channels      = list()
        self.mask_channels_present  = list()
        self.patient_ids            = list()
        self.raw_file_names         = list()
        self.mask_file_names        = list()
        self.patient_count          = dict()
        self.to_csv = None

    def __repr__(self) -> str:
        if self.to_csv is None:
            raise "Data audit not yet evaluated!"
        out_str = f"Number of unique patients: {len(self.patient_count.keys())}\n"
        out_str = out_str + str(self.to_csv)
        return out_str

    def __str__(self) -> str:
        return self.__repr__()

    def evaluate(self):
        """
        Performs the data audit
        """
        assert len(self.image_files) == len(self.mask_files), "Data Mismatch!"

        patient_ids = list()
        raw_file_names, mask_file_names = list(), list()
        patient_count = dict()
        for index in range(len(self.image_files)):
            raw_file_name, mask_file_name = self.image_files[index].split("\\")[-1], self.mask_files[index].split("\\")[-1]
            raw_file_names.append(raw_file_name)
            mask_file_names.append(mask_file_name)
            patient_id = raw_file_name.split("_")[0]    # Extracts the patient id from the file name
            patient_ids.append(patient_id)
            patient_count[patient_id] = patient_count.get(patient_id, 0) + 1

        raw_scan_shapes = list()
        mask_scan_shapes = list()
        raw_mask_match = list()
        mask_num_channels = list()
        mask_channels_present = list()
        for patient_id in patient_count.keys():
            # shape for both tensors:
            #   raw shape   [H, W, D]
            #   mask shape  [H, W, D, K]
            for raw_scan, mask_scan in zip(dataset.RAW_COMPILED[patient_id], dataset.MASKS_COMPILED[patient_id]):
                raw_scan = torch.from_numpy(raw_scan)
                mask_scan = torch.from_numpy(mask_scan)

                raw_scan_shapes.append(raw_scan.shape)
                mask_labels = mask_scan.argmax(dim = 3)  # Also converts to [H, W, D]
                mask_scan_shapes.append(mask_labels.shape)

                mask_num_channels.append(mask_scan.shape[-1])
                labels = torch.unique(mask_labels)
                mask_channels_present.append(labels)

                raw_mask_match.append(raw_scan.shape == mask_labels.shape)


        assert (len(raw_file_names) == len(mask_file_names) ==
                len(raw_scan_shapes) == len(mask_scan_shapes) ==
                len(raw_mask_match) == len(mask_num_channels) == len(mask_channels_present)
            ), "Data Mismatch!"

        self.raw_scan_shapes        = raw_scan_shapes
        self.mask_scan_shapes       = mask_scan_shapes
        self.raw_mask_match         = raw_mask_match
        self.mask_num_channels      = mask_num_channels
        self.mask_channels_present  = mask_channels_present
        self.patient_ids            = patient_ids
        self.raw_file_names         = raw_file_names
        self.mask_file_names        = mask_file_names
        self.patient_count          = patient_count

        data = {
            "patient_id":               self.patient_ids,           # 0
            "raw_file_name":            self.raw_file_names,        # 1
            "mask_file_name":           self.mask_file_names,       # 2
            "raw_shape":                self.raw_scan_shapes,       # 3
            "mask_shape":               self.mask_scan_shapes,      # 4
            "shape_match":              self.raw_mask_match,        # 5
            "num_channels":             self.mask_num_channels,     # 6
            "mask_channels_present":    self.mask_channels_present, # 7
        }
        self.to_csv = pd.DataFrame(data)
        output_dir = Path("audit_out/")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.to_csv.to_csv(output_dir / "data_audit.csv", index=True)


# ======================================================================================================================
# =                                                     Main Guard                                                     =
# ======================================================================================================================
if __name__ == "__main__":
    audit = DataAudit()
    audit.evaluate()
    print(audit)