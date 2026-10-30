import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import HipMRIDataset
from modules import UNet, AttentionUNet, ResidualUNet, ResidualAttentionUNet
import numpy as np
import nibabel as nib
import random


def calculate_class_metrics(prediction, target, class_id):
    valid_mask = (target != IGNORE_INDEX)
    pred_mask = (prediction == class_id) & valid_mask
    true_mask = (target == class_id) & valid_mask

    tp = (pred_mask & true_mask).sum().item()
    fp = (pred_mask & ~true_mask).sum().item()
    fn = (~pred_mask & true_mask).sum().item()

    return tp, fp, fn
# 1. Configuration
MODEL_TYPE = "residual" ## Switching Standard 2D-UNet/Improved 2D-UNet
NUM_CLASSES = 6
BATCH_SIZE = 6
LEARNING_RATE = 1e-4
LOSS_TYPE = "weighted_ce"
DATA_ROOT = Path(
        r"D:\academic\Semester 2 2026\COMP3710"
        r"\data_for_report(private)\keras_slices_data"
    )
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
RANDOM_SEED = 42
TRAIN_CLASS_COUNTS = torch.tensor(
    [
        64897313,
        253682431,
        46951538,
        7296009,
        1340454,
        1599295
    ],
    dtype=torch.float32
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = f"{MODEL_TYPE}_{LOSS_TYPE}"
BEST_MODEL_PATH = CHECKPOINT_DIR / f"best_{EXPERIMENT_NAME}_unet_full.pth" ##Watch the file!!


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}_unet_training.csv" ##Watch the file!!
NUM_EPOCHS = 30
MAX_BATCHES = 2
PATIENCE = 5
MIN_DELTA = 0.001
TARGET_SIZE = (256, 144)
IGNORE_INDEX = -100

# 2. Load dataset
train_dataset = HipMRIDataset(DATA_ROOT, "train", target_size=TARGET_SIZE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
validation_dataset = HipMRIDataset(
    DATA_ROOT,
    "validate",
    target_size=TARGET_SIZE
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# 3. Create model, loss function and optimizer


if MODEL_TYPE == "baseline":
    model = UNet(in_channels=1, num_classes=NUM_CLASSES)
elif MODEL_TYPE == "attention":
    model = AttentionUNet(in_channels=1, num_classes=NUM_CLASSES)
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
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

model = model.to(device)

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Model Type:", MODEL_TYPE)
print("Total Parameters:", total_params)
print("Trainable Parameters:", trainable_params)

if LOSS_TYPE == "weighted_ce":

    frequencies = (
        TRAIN_CLASS_COUNTS / TRAIN_CLASS_COUNTS.sum()
    )

    class_weights = 1.0 / torch.sqrt(frequencies)

    class_weights = (
        class_weights / class_weights.mean()
    )

    class_weights = class_weights.to(device)

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=-100
    )

    print("Class weights:", class_weights.cpu())

else:
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=-100
    )

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
print("===== Training Configuration =====")
print("Model:", MODEL_TYPE)
print("Device:", device)
print("Train samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))
print("Batch Size:", BATCH_SIZE)
print("Learning Rate:", LEARNING_RATE)
print("Maximum Epochs:", NUM_EPOCHS)
print("Checkpoint:", BEST_MODEL_PATH)
print("History:", HISTORY_PATH)



if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

val_images, val_masks = next(iter(validation_loader))

print("===== Validation Batch =====")
print("Images:", val_images.shape)
print("Masks:", val_masks.shape)
print("Unique labels:", torch.unique(val_masks))


best_validation_loss = float("inf")
best_epoch = 0
history = []

no_improvement_count = 0
for epoch in range(NUM_EPOCHS):

    model.train()
    total_loss = 0.0
    batch_count = 0

    for batch_idx, (images, masks) in enumerate(train_loader):

        #if batch_idx >= MAX_BATCHES:
        #    break

        # 1. Check mask labels
        assert images.shape[1:] == (1, *TARGET_SIZE)
        assert masks.shape[1:] == TARGET_SIZE

        valid_pixels = masks != IGNORE_INDEX
        valid_labels = masks[valid_pixels]
        assert valid_labels.numel() > 0
        assert valid_labels.min().item() >= 0
        assert valid_labels.max().item() < NUM_CLASSES
        # 2. Move data to GPU
        images = images.to(device)
        masks = masks.to(device)

        # 3. Clear gradients
        optimizer.zero_grad()

        # 4. Forward Pass
        logits = model(images)

        # 5. Calculate Loss
        loss = criterion(logits, masks)

        # 6. Backward Pass
        loss.backward()

        # 7. Update parameters
        optimizer.step()

        # 8. Record Batch Loss
        total_loss += loss.item()
        batch_count += 1

    average_loss = total_loss / batch_count

    print(
        f"Epoch {epoch + 1}: "
        f"Average Loss = {average_loss:.4f}"
    )

    
    model.eval()
    validation_loss = 0.0
    validation_count = 0
    total_tp = [0] * NUM_CLASSES
    total_fp = [0] * NUM_CLASSES
    total_fn = [0] * NUM_CLASSES
    with torch.no_grad():

        for images, masks in validation_loader:
            
            # Check valid mask labels, excluding padding
            valid_pixels = masks != IGNORE_INDEX
            valid_labels = masks[valid_pixels]

            assert valid_labels.numel() > 0
            assert valid_labels.min().item() >= 0
            assert valid_labels.max().item() < NUM_CLASSES

            # Move tensors to GPU
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass only
            logits = model(images)

            # Calculate validation loss
            loss = criterion(logits, masks)

            prediction = torch.argmax(logits, dim=1)

            for c in range(NUM_CLASSES):
                tp, fp, fn = calculate_class_metrics(prediction, masks, c)

                total_tp[c] += tp
                total_fp[c] += fp
                total_fn[c] += fn

            validation_loss += loss.item()
            validation_count += 1



    valid_pixel_count = sum(
        total_tp[c] + total_fn[c]
        for c in range(NUM_CLASSES)
    )

    print("Validation valid pixels:", valid_pixel_count)

    predicted_pixel_count = sum(
        total_tp[c] + total_fp[c]
        for c in range(NUM_CLASSES)
    )

    print("Validation predicted pixels:", predicted_pixel_count)


    epoch_dice = {}
    epoch_iou = {}
    for c in range(NUM_CLASSES):
        tp = total_tp[c]
        fp = total_fp[c]
        fn = total_fn[c]
        predicted_pixels = tp + fp
        ground_truth_pixels = tp + fn

        print(
            f"Class {c}: "
            f"Predicted Pixels = {predicted_pixels}, "
            f"Ground Truth Pixels = {ground_truth_pixels}"
        )

        denominator_dice = 2 * tp + fp + fn
        denominator_iou = tp + fp + fn

        dice = (
            2 * tp / denominator_dice
            if denominator_dice > 0 else float("nan")
        )


        iou = (
            tp / denominator_iou
            if denominator_iou > 0 else float("nan")
        )
        epoch_dice[f"dice_{c}"] = dice

        epoch_iou[f"iou_{c}"] = iou

        print(f"Class {c}: Dice = {dice:.4f}, IoU = {iou:.4f}")

    average_validation_loss = validation_loss / validation_count


    print(
        f"Epoch {epoch + 1}: "
        f"Validation Loss = {average_validation_loss:.4f}"
    )

    # Calculate Validation Macro Dice
    valid_dice = [
        epoch_dice[f"dice_{c}"]
        for c in range(NUM_CLASSES)
        if not np.isnan(epoch_dice[f"dice_{c}"])
    ]

    macro_dice = (
        float(np.mean(valid_dice))
        if valid_dice else float("nan")
    )

    print(f"Validation Macro Dice: {macro_dice:.4f}")


    if average_validation_loss < best_validation_loss - MIN_DELTA:

        best_validation_loss = average_validation_loss
        best_epoch = epoch + 1
        no_improvement_count = 0
        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_loss": best_validation_loss,
                "num_classes": NUM_CLASSES,
                "random_seed": RANDOM_SEED,
                "model_type": MODEL_TYPE,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "target_size": TARGET_SIZE,
            },
            BEST_MODEL_PATH
        )

        print(
            f"Best model saved! "
            f"Epoch = {best_epoch}, "
            f"Validation Loss = {best_validation_loss:.4f}"
        )

    else:
        no_improvement_count += 1

        print(
            f"No significant improvement: "
            f"{no_improvement_count}/{PATIENCE}"
        )


    epoch_record = {
        "epoch": epoch + 1,
        "train_loss": average_loss,
        "val_loss": average_validation_loss,
        "val_macro_dice": macro_dice,
    }

    epoch_record.update(epoch_dice)
    epoch_record.update(epoch_iou)

    history.append(epoch_record)


    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys()
        )

        writer.writeheader()
        writer.writerows(history)
    if no_improvement_count >= PATIENCE:
        print(
            f"Early stopping triggered at epoch {epoch + 1}. "
            f"Best epoch: {best_epoch}"
        )
        break

if device.type == "cuda":
    torch.cuda.synchronize(device)

    peak_memory = (
        torch.cuda.max_memory_allocated(device)
        / (1024 ** 2)
    )

    print(f"Peak GPU Memory: {peak_memory:.2f} MiB")

# Use your existing DATA_ROOT
#validation_dataset = HipMRIDataset(DATA_ROOT, "validate")
#test_dataset = HipMRIDataset(DATA_ROOT, "test")

#for name, dataset in [
#    ("Validation", validation_dataset),
#    ("Test", test_dataset)
#]:
#    labels = set()

#    for _, mask_path in dataset.pairs:
#        mask = np.asanyarray(nib.load(str(mask_path)).dataobj)
#        labels.update(np.unique(mask).tolist())

#    print(f"{name} labels:", sorted(labels))


print("Best Epoch:", best_epoch)
print("Best Validation Loss:", best_validation_loss)
print("Checkpoint exists:", BEST_MODEL_PATH.is_file())

print("History file exists:", HISTORY_PATH.is_file())
print("Recorded epochs:", len(history))
print("History path:", HISTORY_PATH)