"""
train.py  (3D HipMRI)
 
Usage: python train.py [baseline3d|improved3d]
Set SMOKE=1 in the environment for a tiny test run.
"""
import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
 
from modules import build_unet_3d, build_improved_unet_3d, combined_loss, mean_dice_fg
from dataset import load_hipmri_3d
 
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "improved3d"
CKPT = {"baseline3d": "unet3d", "improved3d": "improved3d"}[MODEL_NAME]
SMOKE = os.environ.get("SMOKE") == "1"
INPUT_SHAPE = (128, 128, 64, 1)
BATCH_SIZE = 2
EPOCHS = 2 if SMOKE else 100
LEARNING_RATE = 5e-4
CHECKPOINT_DIR, PLOTS_DIR = ("checkpoints_smoke", "plots_smoke") if SMOKE else ("checkpoints", "plots")
 
 
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Model: {MODEL_NAME}")
    train_ds, val_ds, test_ds, _ = load_hipmri_3d(BATCH_SIZE, max_volumes=16 if SMOKE else None)
 
    build = build_unet_3d if MODEL_NAME == "baseline3d" else build_improved_unet_3d
    model = build(input_shape=INPUT_SHAPE)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss=combined_loss, metrics=[mean_dice_fg])
    model.summary()
 
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(CHECKPOINT_DIR, f"{CKPT}_best.keras"),
                                           monitor="val_mean_dice_fg", mode="max",
                                           save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_mean_dice_fg", mode="max",
                                         patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8,
                                             min_lr=1e-6),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    plot_history(history)
 
    print("Evaluating on test set...")
    print("Test results:", model.evaluate(test_ds, return_dict=True))
    model.save(os.path.join(CHECKPOINT_DIR, f"{CKPT}_final.keras"))
 
    # Resource profiling for the README table
    print("Parameters:", model.count_params())
    print(f"Peak GPU memory: {tf.config.experimental.get_memory_info('GPU:0')['peak'] / 1e9:.2f} GB")
    x = next(iter(test_ds))[0]
    model(x, training=False).numpy()  # warm-up; .numpy() forces the GPU to finish
    t0 = time.time()
    for _ in range(10):
        model(x, training=False).numpy()  # .numpy() so the timing includes GPU completion
    print(f"Inference: {(time.time() - t0) / 10 * 1000:.1f} ms per volume")
 
 
def plot_history(h):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(h.history["loss"], label="train"); ax[0].plot(h.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(h.history["mean_dice_fg"], label="train"); ax[1].plot(h.history["val_mean_dice_fg"], label="val")
    ax[1].set_title("Mean foreground Dice"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"training_curves_{CKPT}.png"))
 
 
if __name__ == "__main__":
    main()
