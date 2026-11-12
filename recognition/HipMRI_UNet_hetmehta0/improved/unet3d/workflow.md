# Improved 3D U-Net — Residual Blocks (milestone D)

Status: implemented and unit-tested on CPU. **Not trained. No results yet.**
Project-level research question and hypothesis: see `architeture/workflow.md` (unchanged).

## 1. Motivation and hypothesis

The research question asks whether an improved 3D U-Net gives enough benefit over the
standard 3D U-Net to justify its extra cost. Of the two candidate changes in the project
workflow (residual blocks, deep supervision), only **residual blocks** are tested, so that
any difference can be attributed to a single change.

- **IF** each double-convolution block of the standard 3D U-Net is replaced by a residual block,
- **THEN** validation mean foreground Dice may improve at a small parameter and memory cost,
- **BECAUSE** the identity/projection shortcut gives gradients a direct path through every block,
  which can ease optimisation of a deep volumetric network trained with small batches.

This is a hypothesis to be tested, not an expected outcome.

## 2. Difference from the standard 3D U-Net

| | Standard (`modules.UNet3D`) | Residual (`ResUNet3D`) |
|---|---|---|
| Block | `ReLU(BN(Conv(ReLU(BN(Conv(x))))))` | `ReLU(F(x) + shortcut(x))`, with F = Conv-BN-ReLU-Conv-BN |
| Shortcut | none | identity if widths match, else 1x1x1 Conv + BN |
| Everything else | — | identical, inherited from `UNet3D` |

The rest is shared code, not a copy: four encoder stages (32, 64, 128, 256 channels),
2x2x2 max-pooling, bottleneck width (256 or 512), 2x2x2 transposed-convolution upsampling,
long skip concatenation `[skip, upsampled]`, 1x1x1 head, six-class raw logits, input `[B, 1, D, H, W]`.

Because every block except the 256 -> 256 bottleneck changes width, all shortcuts are 1x1x1
projections except that one identity shortcut in `resunet3d_256` (none in `resunet3d_512`).

| Variant | Standard parameters | Residual parameters | Added |
|---|---|---|---|
| bottleneck 256 | 14,975,238 | 15,194,278 | +219,040 (+1.5%) |
| bottleneck 512 | 22,578,438 | 22,929,574 | +351,136 (+1.6%) |

## 3. Implementation and verification

- `residual_blocks.py`: `ResidualBlock3D`, `ResUNet3D` (subclass of `UNet3D` that swaps
  each `DoubleConv3D` for a `ResidualBlock3D` with the same widths), `build_resunet3d`.
- `unet3d_256.py`, `unet3d_512.py`: thin entry points (variants `resunet3d_256`, `resunet3d_512`).
- `train3d.py --model resunet3d_256|resunet3d_512`: same training script as the standard models.
- `tests/test_improved3d.py` (CPU) covers:
  - identity and projection shortcuts, and that the shortcut is really added;
  - input/output shapes at 64^3 and 96^3, and stage shapes;
  - skip-connection order, widths and parameter counts against the standard model;
  - finite gradients;
  - identical patch plans for the standard and residual runs.

Still to do (each needs approval before running on the GPU):
1. memory probe;
2. overfit check on 4 patches;
3. full training of **one** variant.

## 4. Controlled experiment and evaluation plan

Only the architecture changes. Everything else matches the standard 3D run it is compared with:
- **Data:** frozen patient-level split (26/6/6 patients, split SHA-256 `4da18222…`), with the test split unused.
- **Loss:** 0.5 Dice + 0.5 cross-entropy.
- **Optimiser:** AdamW, same learning rate and weight decay.
- **Sampling:** seed 3710, 96^3 class-balanced patches with `fg_prob = 0.33`, same epochs and
  patches per epoch, same batch size.
- **Validation:** whole-volume sliding window with 50% overlap; `best.pt` chosen by global mean
  foreground Dice on validation.

The patch plan depends only on the seed, so both models see the same patches in the same order.
The variant to train (256 or 512) is chosen after the standard 3D results, and the choice
and its reason are recorded here.

Compare: global and per-volume mean foreground Dice, per-class Dice, learning curves, training
time, peak GPU memory, parameter count, and inference time. The test split is evaluated only once
the model-selection procedure is frozen (milestone E).

## 5. Limitations and what results can establish

- **One seed and one run per model:** a small Dice difference may be within run-to-run variation
  (cuDNN and AMP are not bitwise deterministic). It shows a difference under this protocol,
  not a general or statistically significant improvement.
- **Validation set:** only 6 patients (34 scans), so per-class Dice for small structures
  (prostate, rectum) is noisy.
- **Confounding:** the residual model has about 1.5% more parameters, so a gain is not purely from
  the shortcut mechanism.
- **Hyperparameters:** tuned for neither model; both use the standard settings.
- **No change does not mean residual blocks never help:** a null result says nothing about other
  settings (longer training, other learning rates, deep supervision).
- **No claim without a run:** nothing in this file is a result until a run directory with
  `summary.json` exists.
