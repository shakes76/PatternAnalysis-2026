"""
modules.py  (3D HipMRI)

Plain 3D U-Net baseline (Cicek et al., 2016) and 3D Improved U-Net
(Isensee et al., 2018): residual context modules, instance norm + leaky ReLU,
strided-conv downsampling, localization modules and deep supervision.
Pure TensorFlow/Keras only.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------- baseline 3D U-Net ----------
def conv_block(x, filters, name):
    for i in (1, 2):
        x = layers.Conv3D(filters, 3, padding="same", name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    return x


def build_unet_3d(input_shape=(128, 128, 64, 1), num_classes=6, base_filters=16):
    inputs = layers.Input(shape=input_shape, name="input_volume")
    skips, x = [], inputs
    for lvl in range(4):
        skip = conv_block(x, base_filters * 2 ** lvl, f"enc{lvl}")
        skips.append(skip)
        x = layers.MaxPooling3D(2, name=f"enc{lvl}_pool")(skip)
    x = conv_block(x, base_filters * 16, "bottleneck")
    for lvl in range(3, -1, -1):
        f = base_filters * 2 ** lvl
        x = layers.Conv3DTranspose(f, 2, strides=2, padding="same", name=f"dec{lvl}_up")(x)
        x = layers.Concatenate(name=f"dec{lvl}_concat")([x, skips[lvl]])
        x = conv_block(x, f, f"dec{lvl}")
    out = layers.Conv3D(num_classes, 1, activation="softmax", name="segmentation_output")(x)
    return Model(inputs, out, name="UNet3D")


# ---------- 3D Improved U-Net ----------
def norm_act(x, name):
    """Instance norm (GroupNorm, one group per channel) + leaky ReLU."""
    x = layers.GroupNormalization(groups=x.shape[-1], name=f"{name}_in")(x)
    return layers.LeakyReLU(negative_slope=0.01, name=f"{name}_lrelu")(x)


def context_module(x, filters, name, dropout=0.3):
    """Pre-activation residual block with spatial dropout between the convs."""
    h = norm_act(x, f"{name}_a")
    h = layers.Conv3D(filters, 3, padding="same", name=f"{name}_conv1")(h)
    h = layers.SpatialDropout3D(dropout, name=f"{name}_drop")(h)
    h = norm_act(h, f"{name}_b")
    h = layers.Conv3D(filters, 3, padding="same", name=f"{name}_conv2")(h)
    return layers.Add(name=f"{name}_add")([x, h])


def build_improved_unet_3d(input_shape=(128, 128, 64, 1), num_classes=6,
                           base_filters=16, dropout=0.3):
    inputs = layers.Input(shape=input_shape, name="input_volume")
    x = layers.Conv3D(base_filters, 3, padding="same", name="stem_conv")(inputs)
    skips = []
    for lvl in range(5):
        f = base_filters * 2 ** lvl
        if lvl > 0:
            x = layers.Conv3D(f, 3, strides=2, padding="same", name=f"enc{lvl}_down")(x)
        x = context_module(x, f, f"enc{lvl}", dropout)
        skips.append(x)

    x = skips[-1]
    segs = []
    for lvl in range(3, -1, -1):
        f = base_filters * 2 ** lvl
        x = layers.UpSampling3D(2, name=f"dec{lvl}_up")(x)
        x = layers.Conv3D(f, 3, padding="same", name=f"dec{lvl}_upconv")(x)
        x = norm_act(x, f"dec{lvl}_up")
        x = layers.Concatenate(name=f"dec{lvl}_concat")([skips[lvl], x])
        x = layers.Conv3D(f, 3, padding="same", name=f"dec{lvl}_loc1")(x)
        x = norm_act(x, f"dec{lvl}_loc1")
        x = layers.Conv3D(f, 1, name=f"dec{lvl}_loc2")(x)
        x = norm_act(x, f"dec{lvl}_loc2")
        if lvl <= 2:  # deep supervision from the 3 highest-resolution levels
            segs.append(layers.Conv3D(num_classes, 1, name=f"seg{lvl}")(x))

    s = segs[0]
    for i, nxt in enumerate(segs[1:]):
        s = layers.UpSampling3D(2, name=f"seg_up{i}")(s)
        s = layers.Add(name=f"seg_add{i}")([s, nxt])
    out = layers.Activation("softmax", name="segmentation_output")(s)
    return Model(inputs, out, name="ImprovedUNet3D")


# ---------- losses / metrics (per-class, so small classes are not swamped) ----------
def per_class_dice(y_true, y_pred, smooth=1.0):
    axes = [0, 1, 2, 3]
    inter = tf.reduce_sum(y_true * y_pred, axis=axes)
    denom = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    return (2.0 * inter + smooth) / (denom + smooth)


def combined_loss(y_true, y_pred):
    """Mean per-class soft Dice loss + categorical cross-entropy."""
    dice_loss = 1.0 - tf.reduce_mean(per_class_dice(y_true, y_pred))
    return dice_loss + tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)


def mean_dice_fg(y_true, y_pred):
    """Mean soft Dice over foreground classes 1..5 (monitoring metric)."""
    return tf.reduce_mean(per_class_dice(y_true, y_pred)[1:])
