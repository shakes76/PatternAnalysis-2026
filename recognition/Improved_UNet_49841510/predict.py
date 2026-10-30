from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import HipMRIDataset
from modules import UNet, AttentionUNet, ResidualUNet, ResidualAttentionUNet
import time
import csv

def calculate_class_metrics(prediction, target, class_id):
    valid_mask = target != IGNORE_INDEX

    pred_mask = (prediction == class_id) & valid_mask
    true_mask = (target == class_id) & valid_mask

    tp = (pred_mask & true_mask).sum().item()
    fp = (pred_mask & ~true_mask).sum().item()
    fn = (~pred_mask & true_mask).sum().item()

    return tp, fp, fn

NUM_CLASSES = 6
BATCH_SIZE = 6
TARGET_SIZE = (256, 144)
IGNORE_INDEX = -100
MODEL_TYPE = "residual"
LOSS_TYPE = "weighted_ce"
MASK_COLORS = [
    "#222222",  # Class 0
    "#4477AA",  # Class 1
    "#66CC99",  # Class 2
    "#EEAA44",  # Class 3
    "#CC6677",  # Class 4
    "#AA66CC",  # Class 5
]

mask_cmap = ListedColormap(MASK_COLORS)

VIS_CASE_ID = "042"
VIS_WEEK = 0
VIS_SLICE_ID = 13

DATA_ROOT = Path(
    r"D:\academic\Semester 2 2026\COMP3710"
    r"\data_for_report(private)\keras_slices_data"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

if MODEL_TYPE == "baseline":
    CHECKPOINT_PATH = (
        CHECKPOINT_DIR / "best_unet_baseline_full_trained.pth"
    )

elif MODEL_TYPE == "attention":
    CHECKPOINT_PATH = (
        CHECKPOINT_DIR / "best_attention_unet_full.pth"
    )

elif MODEL_TYPE == "residual" and LOSS_TYPE == "weighted_ce":
    CHECKPOINT_PATH = (
        CHECKPOINT_DIR / "best_residual_weighted_ce_unet_full.pth"
    )



elif MODEL_TYPE == "residual":
    CHECKPOINT_PATH = (
        CHECKPOINT_DIR / "best_residual_unet_full.pth"
    )
elif MODEL_TYPE == "residual_attention":
    CHECKPOINT_PATH = (
        CHECKPOINT_DIR / "best_residual_attention_unet_full.pth"
    )


else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

test_dataset = HipMRIDataset(
    DATA_ROOT,
    "test",
    target_size=TARGET_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))

if MODEL_TYPE == "baseline":
    model = UNet(
        in_channels=1,
        num_classes=NUM_CLASSES
    )

elif MODEL_TYPE == "attention":
    model = AttentionUNet(
        in_channels=1,
        num_classes=NUM_CLASSES
    )

elif MODEL_TYPE == "residual":
    model = ResidualUNet(
        in_channels=1,
        num_classes=NUM_CLASSES
    )

elif MODEL_TYPE == "residual_attention":
    model = ResidualAttentionUNet(
        in_channels=1,
        num_classes=NUM_CLASSES
    )

model = model.to(device)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded checkpoint epoch:", checkpoint["epoch"])
print("Checkpoint validation loss:", checkpoint["validation_loss"])
print("Model type:", MODEL_TYPE)
print("Checkpoint:", CHECKPOINT_PATH)
print("Loaded checkpoint epoch:", checkpoint["epoch"])
print("Checkpoint validation loss:", checkpoint["validation_loss"])

if "model_type" in checkpoint:
    assert checkpoint["model_type"] == MODEL_TYPE

total_tp = [0] * NUM_CLASSES
total_fp = [0] * NUM_CLASSES
total_fn = [0] * NUM_CLASSES
case_metrics = {}
slice_results = []
confusion_matrix = torch.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=torch.int64,
    device=device
)
total_inference_time = 0.0
total_inference_images = 0


model.eval()
warmup_images, _ = next(iter(test_loader))
warmup_images = warmup_images.to(device)

with torch.inference_mode():
    for _ in range(5):
        _ = model(warmup_images)

if device.type == "cuda":
    torch.cuda.synchronize(device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(device))
    torch.cuda.reset_peak_memory_stats(device)

with torch.inference_mode():

    for batch_idx, (images, masks) in enumerate(test_loader):

        images = images.to(device)
        masks = masks.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start_time = time.perf_counter()

        logits = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        end_time = time.perf_counter()

        batch_inference_time = end_time - start_time

        total_inference_time += batch_inference_time
        total_inference_images += images.shape[0]

        predictions = torch.argmax(logits, dim=1)
        # Exclude padding pixels
        valid_mask = masks != IGNORE_INDEX

        true_labels = masks[valid_mask]
        predicted_labels = predictions[valid_mask]

        # Convert pairs of labels into matrix indices
        combined_indices = (
            true_labels * NUM_CLASSES + predicted_labels
        )

        batch_counts = torch.bincount(
            combined_indices,
            minlength=NUM_CLASSES * NUM_CLASSES
        )

        confusion_matrix += batch_counts.reshape(
            NUM_CLASSES, NUM_CLASSES
        )

        if batch_idx == 0:
            print("Input images:", images.shape)
            print("Output logits:", logits.shape)
            print("Predicted masks:", predictions.shape)
            print("Ground truth masks:", masks.shape)

        for c in range(NUM_CLASSES):

            tp, fp, fn = calculate_class_metrics(
                predictions, masks, c
            )

            total_tp[c] += tp
            total_fp[c] += fp
            total_fn[c] += fn

        start_index = batch_idx * BATCH_SIZE

        for i in range(images.shape[0]):

            dataset_index = start_index + i

            image_path, _ = test_dataset.pairs[dataset_index]

            case_id = image_path.name.split("_")[1]

            if case_id not in case_metrics:
                case_metrics[case_id] = {
                    "tp": [0] * NUM_CLASSES,
                    "fp": [0] * NUM_CLASSES,
                    "fn": [0] * NUM_CLASSES
                }
            filename_parts = image_path.name.split("_")
            week_id = int(filename_parts[3])
            slice_id = int(filename_parts[5].replace(".nii.gz", ""))
            for c in range(NUM_CLASSES):

                tp, fp, fn = calculate_class_metrics(
                    predictions[i],
                    masks[i],
                    c
                )

                case_metrics[case_id]["tp"][c] += tp
                case_metrics[case_id]["fp"][c] += fp
                case_metrics[case_id]["fn"][c] += fn
                denominator = 2 * tp + fp + fn

                slice_dice = (
                    2 * tp / denominator
                    if denominator > 0 else float("nan")
                )

                

                slice_results.append({
                    "case_id": case_id,
                    "week": week_id,
                    "slice_id": slice_id,
                    "class_id": c,
                    "dice": slice_dice,
                    "ground_truth_pixels": tp + fn,
                    "predicted_pixels": tp + fp
                })
print("Case IDs:", sorted(case_metrics.keys()))
print("Number of cases:", len(case_metrics))

print("===== Test Set Evaluation =====")

valid_pixel_count = sum(
    total_tp[c] + total_fn[c]
    for c in range(NUM_CLASSES)
)

predicted_pixel_count = sum(
    total_tp[c] + total_fp[c]
    for c in range(NUM_CLASSES)
)

print("Test valid pixels:", valid_pixel_count)
print("Test predicted pixels:", predicted_pixel_count)

for c in range(NUM_CLASSES):

    tp = total_tp[c]
    fp = total_fp[c]
    fn = total_fn[c]

    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn

    dice = (
        2 * tp / dice_denominator
        if dice_denominator > 0 else float("nan")
    )

    iou = (
        tp / iou_denominator
        if iou_denominator > 0 else float("nan")
    )

    print(
        f"Class {c}: "
        f"Dice = {dice:.4f}, "
        f"IoU = {iou:.4f}"
    )


print("\n===== Per-Case Evaluation =====")
case_dice_values = {c: [] for c in range(NUM_CLASSES)}
for case_id, metrics in sorted(case_metrics.items()):

    print(f"\nCase {case_id}")

    for c in range(NUM_CLASSES):

        tp = metrics["tp"][c]
        fp = metrics["fp"][c]
        fn = metrics["fn"][c]

        dice_denominator = 2 * tp + fp + fn
        iou_denominator = tp + fp + fn

        dice = (
            2 * tp / dice_denominator
            if dice_denominator > 0 else float("nan")
        )
        case_dice_values[c].append(dice)
        iou = (
            tp / iou_denominator
            if iou_denominator > 0 else float("nan")
        )

        print(
            f"Class {c}: "
            f"Dice = {dice:.4f}, IoU = {iou:.4f}"
        )

for c in range(NUM_CLASSES):

    assert total_tp[c] == sum(
        metrics["tp"][c] for metrics in case_metrics.values()
    )

    assert total_fp[c] == sum(
        metrics["fp"][c] for metrics in case_metrics.values()
    )

    assert total_fn[c] == sum(
        metrics["fn"][c] for metrics in case_metrics.values()
    )

print("Per-case aggregation check passed!")

print("\n===== Confusion Matrix =====")

cm = confusion_matrix.cpu().numpy()

print(cm)

assert cm.sum() == valid_pixel_count

for c in range(NUM_CLASSES):
    assert cm[c, c] == total_tp[c]
    assert cm[c, :].sum() == total_tp[c] + total_fn[c]
    assert cm[:, c].sum() == total_tp[c] + total_fp[c]

print("Confusion matrix consistency check passed!")

class4_to_class5 = cm[4, 5]
class5_to_class4 = cm[5, 4]

print("Class 4 predicted as Class 5:", class4_to_class5)
print("Class 5 predicted as Class 4:", class5_to_class4)

class4_total = cm[4, :].sum()

if class4_total > 0:
    confusion_rate = class4_to_class5 / class4_total

    print(
        f"Class 4 -> Class 5 rate: "
        f"{confusion_rate:.2%}"
    )
row_sums = cm.sum(axis=1, keepdims=True)

cm_normalized = np.divide(
    cm,
    row_sums,
    out=np.zeros_like(cm, dtype=float),
    where=row_sums != 0
)

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(
    cm_normalized,
    cmap="Blues",
    vmin=0,
    vmax=1
)

ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Ground Truth Class")
ax.set_title(f"{MODEL_TYPE.capitalize()} U-Net Confusion Matrix")

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        value = cm_normalized[i, j]

        ax.text(
            j, i,
            f"{value:.2f}",
            ha="center",
            va="center",
            color="white" if value > 0.5 else "black"
        )

fig.colorbar(im, ax=ax, label="Fraction of True Class")

plt.tight_layout()
EXPERIMENT_NAME = f"{MODEL_TYPE}_{LOSS_TYPE}"
CM_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "visualizations"
    / f"{EXPERIMENT_NAME}_confusion_matrix.png"
)

plt.savefig(
    CM_PATH,
    dpi=200,
    bbox_inches="tight"
)

print("Confusion Matrix saved:", CM_PATH)

plt.show()

print("\n===== Worst Per-Slice Dice =====")

TARGET_CLASS = 5  # Prostate

filtered_results = [
    record
    for record in slice_results
    if record["class_id"] == TARGET_CLASS
    and record["ground_truth_pixels"] >= 50
    and not np.isnan(record["dice"])
]

sorted_results = sorted(
    filtered_results,
    key=lambda record: record["dice"]
)

print("Matched slices:", len(filtered_results))

for record in sorted_results[:10]:
    print(
        f"Case {record['case_id']}, "
        f"Week {record['week']}, "
        f"Slice {record['slice_id']}: "
        f"Dice = {record['dice']:.4f}, "
        f"GT Pixels = {record['ground_truth_pixels']}, "
        f"Pred Pixels = {record['predicted_pixels']}"
    )
if not sorted_results:
    raise ValueError(
        f"No valid failure cases found for Class {TARGET_CLASS}"
    )

worst_case = sorted_results[3]

VIS_CASE_ID = worst_case["case_id"]
VIS_WEEK = worst_case["week"]
VIS_SLICE_ID = worst_case["slice_id"]

print(
    "\nSelected failure case:",
    VIS_CASE_ID,
    VIS_WEEK,
    VIS_SLICE_ID
)
print("\n===== Mean Per-Case Dice =====")

for c in range(NUM_CLASSES):
    values = np.array(case_dice_values[c], dtype=float)

    valid_values = values[~np.isnan(values)]

    mean_dice = (
        valid_values.mean()
        if len(valid_values) > 0 else float("nan")
    )

    print(f"Class {c}: Mean Dice = {mean_dice:.4f}")

average_inference_time = (
    total_inference_time / total_inference_images
)

print("\n===== Inference Time =====")
print(f"Total model forward time: {total_inference_time:.4f} s")
print(f"Average inference time per image: {average_inference_time * 1000:.4f} ms")
print(f"Measured images: {total_inference_images}")

if device.type == "cuda":
    torch.cuda.synchronize(device)

    peak_memory = (
        torch.cuda.max_memory_allocated(device)
        / (1024 ** 2)
    )

    print(f"Peak Inference GPU Memory: {peak_memory:.2f} MiB")

expected_filename = (
    f"case_{VIS_CASE_ID}_week_{VIS_WEEK}"
    f"_slice_{VIS_SLICE_ID}.nii.gz"
)

case_files = [
    image_path.name
    for image_path, _ in test_dataset.pairs
    if image_path.name.startswith(f"case_{VIS_CASE_ID}_")
]

print("Available examples:", case_files[:10])
print("Total files for this case:", len(case_files))
selected_index = None

for i, (image_path, _) in enumerate(test_dataset.pairs):
    if image_path.name == expected_filename:
        selected_index = i
        break
    

if selected_index is None:
    raise ValueError(
        f"Cannot find Case {VIS_CASE_ID}, Slice {VIS_SLICE_ID}"
    )
image, mask = test_dataset[selected_index]
input_image = image.unsqueeze(0).to(device)

model.eval()

with torch.inference_mode():
    logits = model(input_image)
    prediction = torch.argmax(logits, dim=1)
image_np = image.squeeze(0).cpu().numpy()
mask_np = mask.cpu().numpy()
prediction_np = prediction.squeeze(0).cpu().numpy()

target_class = TARGET_CLASS

true_region = mask_np == target_class

predicted_labels_in_region = prediction_np[true_region]

labels, counts = np.unique(
    predicted_labels_in_region,
    return_counts=True
)

print("\n===== Misclassification Analysis =====")

print(f"Case {VIS_CASE_ID}, Slice {VIS_SLICE_ID}")
print(f"Ground Truth Class {target_class} pixels:", true_region.sum())

for label, count in zip(labels, counts):
    print(f"Predicted as Class {label}: {count} pixels")

valid_mask = mask_np != IGNORE_INDEX
mask_display = np.ma.masked_where(
    ~valid_mask,
    mask_np
)

prediction_display = np.ma.masked_where(
    ~valid_mask,
    prediction_np
)

fig, axes = plt.subplots(1, 3, figsize=(13, 5))

axes[0].imshow(image_np, cmap="gray")
axes[0].set_title("MRI Input")

axes[1].imshow(
    mask_display,
    cmap=mask_cmap,
    vmin=0,
    vmax=NUM_CLASSES - 1
)
axes[1].set_title("Ground Truth")

axes[2].imshow(
    prediction_display,
    cmap=mask_cmap,
    vmin=0,
    vmax=NUM_CLASSES - 1
)
axes[2].set_title("Prediction")

for ax in axes:
    ax.axis("off")

fig.suptitle(
    f"Case {VIS_CASE_ID}, Week {VIS_WEEK}, Slice {VIS_SLICE_ID}"
)

plt.tight_layout()
VIS_DIR = Path(__file__).resolve().parent / "results" / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

output_path = (
    VIS_DIR
    / f"{MODEL_TYPE}_{LOSS_TYPE}_case_{VIS_CASE_ID}_week_{VIS_WEEK}_slice_{VIS_SLICE_ID}.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

print("Visualization saved:", output_path)
plt.show()
print("Total per-slice records:", len(slice_results))

EXPERIMENT_NAME = f"{MODEL_TYPE}_{LOSS_TYPE}"

CSV_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / f"{EXPERIMENT_NAME}_slice_metrics.csv"
)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "case_id",
            "week",
            "slice_id",
            "class_id",
            "dice",
            "ground_truth_pixels",
            "predicted_pixels"
        ]
    )

    writer.writeheader()
    writer.writerows(slice_results)

print("Slice metrics saved:", CSV_PATH)