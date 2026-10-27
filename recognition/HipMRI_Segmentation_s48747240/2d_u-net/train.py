"""
train.py

Training script for the baseline 2D U-Net on HipMRI prostate segmentation.

Imports the model from modules.py and the data pipeline from dataset.py,
trains, validates, evaluates on the test split, and saves loss/Dice plots
plus the trained model weights.

Run on Rangpur inside an interactive GPU session or via a Slurm batch script
(see the course's Rangpur HPC documentation).
"""

import os
import sys
import time
import matplotlib
matplotlib.use("Agg")  # no display on a cluster node
import matplotlib.pyplot as plt
import tensorflow as tf

from modules import (build_unet_2d, build_baseline_2d, build_improved_unet_2d,
                     dice_coefficient, combined_loss, combined_loss_per_class)
from dataset import load_hipmri_2d

# ---- Config ----
NUM_CLASSES = 6          
INPUT_SHAPE = (256, 128, 1)  
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "checkpoints"
PLOTS_DIR = "plots"
EARLY_STOP = False  # set True for a fast smoke-test run on a handful of slices

# Usage: python train.py [unet|baseline]   (default: unet)
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "unet"
CKPT_PREFIX = {"unet": "unet2d", "baseline": "baseline2d",
               "improved": "improved2d",
               "improved_pcd": "improved2d_pcd"}[MODEL_NAME]

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading data...")
    train_ds, val_ds, test_ds = load_hipmri_2d(batch_size=BATCH_SIZE, early_stop=EARLY_STOP)

    print("Building model...")
    if MODEL_NAME == "baseline":
        model = build_baseline_2d(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    elif MODEL_NAME in ("improved", "improved_pcd"):
        model = build_improved_unet_2d(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    else:
        model = build_unet_2d(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    loss_fn = combined_loss_per_class if MODEL_NAME == "improved_pcd" else combined_loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=[dice_coefficient, "accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, f"{CKPT_PREFIX}_best.keras"),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient", mode="max", patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6,
        ),
    ]

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_history(history)

    print("Evaluating on test set...")
    test_results = model.evaluate(test_ds, return_dict=True)
    print("Test results:", test_results)

    model.save(os.path.join(CHECKPOINT_DIR, f"{CKPT_PREFIX}_final.keras"))
    print(f"Saved final model to {CHECKPOINT_DIR}/{CKPT_PREFIX}_final.keras")
    # ---- Resource profiling for the README table ----
    print("Parameters:", model.count_params())
    mem = tf.config.experimental.get_memory_info("GPU:0")
    print(f"Peak GPU memory: {mem['peak'] / 1e9:.2f} GB")

    # Inference latency per slice (one warm-up call first)
    x = next(iter(test_ds))[0][:1]
    model(x, training=False)
    t0 = time.time()
    for _ in range(50):
        model(x, training=False)
    print(f"Inference: {(time.time() - t0) / 50 * 1000:.1f} ms per slice")


def plot_history(history):
    """Plot and save training/validation loss and Dice coefficient curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["dice_coefficient"], label="train dice")
    axes[1].plot(history.history["val_dice_coefficient"], label="val dice")
    axes[1].set_title("Dice Coefficient")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f"training_curves_{CKPT_PREFIX}.png")
    fig.savefig(out)
    print(f"Saved training curves to {out}")


if __name__ == "__main__":
    main()
