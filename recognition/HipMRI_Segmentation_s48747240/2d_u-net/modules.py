"""
modules.py

Model components for HipMRI prostate segmentation.

Contains a standard 2D U-Net, used as:
  (a) the Easy-difficulty deliverable while the pipeline is first verified, and
  (b) the baseline model for the mandatory benchmarking comparison against
      the final (Hard-difficulty) 3D model.

Pure TensorFlow/Keras only - no NumPy dependency, per assignment constraints.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters, name_prefix, use_bn=True):
    """Two 3x3 convolutions, each with optional batch norm, then ReLU."""
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv1")(x)
    if use_bn:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv2")(x)
    if use_bn:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)
    return x


def encoder_block(x, filters, name_prefix, use_bn=True):
    """Conv block followed by 2x2 max pooling; returns (skip, pooled)."""
    skip = conv_block(x, filters, name_prefix, use_bn)
    pooled = layers.MaxPooling2D(2, name=f"{name_prefix}_pool")(skip)
    return skip, pooled


def decoder_block(x, skip, filters, name_prefix, use_bn=True):
    """Transposed conv upsample, concat with skip connection, then conv block."""
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same",
                                name=f"{name_prefix}_upconv")(x)
    x = layers.Concatenate(name=f"{name_prefix}_concat")([x, skip])
    x = conv_block(x, filters, name_prefix, use_bn)
    return x


def build_unet_2d(input_shape=(256, 256, 1), num_classes=6, base_filters=32,
                  use_bn=True, name="UNet2D"):
    """
    Standard 2D U-Net (Ronneberger et al., 2015) for multi-class segmentation.

    Args:
        input_shape: (H, W, C) of input MRI slices.
        num_classes: number of segmentation label classes (one-hot channels).
        base_filters: number of filters in the first encoder block; doubles
            at each depth level as is standard for U-Net.

    Returns:
        A compiled-free tf.keras.Model. Compile it in train.py with your
        chosen loss (e.g. Dice loss or categorical crossentropy + Dice) and
        optimizer.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Encoder
    skip1, pool1 = encoder_block(inputs, base_filters, "enc1", use_bn)
    skip2, pool2 = encoder_block(pool1, base_filters * 2, "enc2", use_bn)
    skip3, pool3 = encoder_block(pool2, base_filters * 4, "enc3", use_bn)
    skip4, pool4 = encoder_block(pool3, base_filters * 8, "enc4", use_bn)

    # Bottleneck
    bottleneck = conv_block(pool4, base_filters * 16, "bottleneck", use_bn)

    # Decoder
    up4 = decoder_block(bottleneck, skip4, base_filters * 8, "dec4", use_bn)
    up3 = decoder_block(up4, skip3, base_filters * 4, "dec3", use_bn)
    up2 = decoder_block(up3, skip2, base_filters * 2, "dec2", use_bn)
    up1 = decoder_block(up2, skip1, base_filters, "dec1", use_bn)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax",
                             name="segmentation_output")(up1)

    return Model(inputs, outputs, name=name)


def build_baseline_2d(input_shape=(256, 128, 1), num_classes=6):
    """Reduced baseline: half the filters (16 vs 32) and no batch norm."""
    return build_unet_2d(input_shape, num_classes, base_filters=16,
                         use_bn=False, name="Baseline2D")


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Per-batch Dice coefficient averaged across classes; used as a metric."""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    """1 - Dice coefficient, usable as a standalone or combined loss term."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """Dice loss + categorical crossentropy, a common choice for segmentation."""
    cce = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    return dice_loss(y_true, y_pred) + cce


def norm_act(x, name):
    """Instance normalisation (GroupNorm with one group per channel) + leaky ReLU."""
    channels = x.shape[-1]
    x = layers.GroupNormalization(groups=channels, name=f"{name}_in")(x)
    return layers.LeakyReLU(negative_slope=0.01, name=f"{name}_lrelu")(x)


def context_module(x, filters, name, dropout=0.3):
    """Pre-activation residual block: two 3x3 convs with dropout between them."""
    h = norm_act(x, f"{name}_a")
    h = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(h)
    h = layers.Dropout(dropout, name=f"{name}_drop")(h)
    h = norm_act(h, f"{name}_b")
    h = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(h)
    return layers.Add(name=f"{name}_add")([x, h])


def build_improved_unet_2d(input_shape=(256, 128, 1), num_classes=6,
                           base_filters=16, dropout=0.3):
    """
    2D Improved U-Net (Isensee et al., 2018): residual context modules,
    instance norm + leaky ReLU, strided-conv downsampling, localization
    modules, and deep supervision by summing upsampled segmentation outputs
    from the three highest-resolution decoder levels.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Encoder: 5 levels (16, 32, 64, 128, 256 filters by default)
    x = layers.Conv2D(base_filters, 3, padding="same", name="stem_conv")(inputs)
    skips = []
    for level in range(5):
        filters = base_filters * 2 ** level
        if level > 0:
            x = layers.Conv2D(filters, 3, strides=2, padding="same",
                              name=f"enc{level}_down")(x)
        x = context_module(x, filters, f"enc{level}", dropout)
        skips.append(x)

    # Decoder with localization modules; collect deep-supervision outputs
    x = skips[-1]
    seg_outputs = []
    for level in range(3, -1, -1):
        filters = base_filters * 2 ** level
        x = layers.UpSampling2D(2, name=f"dec{level}_up")(x)
        x = layers.Conv2D(filters, 3, padding="same", name=f"dec{level}_upconv")(x)
        x = norm_act(x, f"dec{level}_up")
        x = layers.Concatenate(name=f"dec{level}_concat")([skips[level], x])
        x = layers.Conv2D(filters, 3, padding="same", name=f"dec{level}_loc1")(x)
        x = norm_act(x, f"dec{level}_loc1")
        x = layers.Conv2D(filters, 1, name=f"dec{level}_loc2")(x)
        x = norm_act(x, f"dec{level}_loc2")
        if level <= 2:
            seg_outputs.append(
                layers.Conv2D(num_classes, 1, name=f"seg{level}")(x))

    # Deep supervision: upsample lower-resolution outputs and add
    s = seg_outputs[0]
    for i, nxt in enumerate(seg_outputs[1:]):
        s = layers.UpSampling2D(2, name=f"seg_up{i}")(s)
        s = layers.Add(name=f"seg_add{i}")([s, nxt])
    outputs = layers.Activation("softmax", name="segmentation_output")(s)

    return Model(inputs, outputs, name="ImprovedUNet2D")


def multiclass_dice_loss(y_true, y_pred, smooth=1e-6):
    """Per-class soft Dice (averaged over classes), unlike the global dice_loss."""
    axes = [0, 1, 2]
    inter = tf.reduce_sum(y_true * y_pred, axis=axes)
    denom = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    return 1.0 - tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))


def combined_loss_per_class(y_true, y_pred):
    """Per-class Dice loss + categorical crossentropy."""
    cce = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    return multiclass_dice_loss(y_true, y_pred) + cce
