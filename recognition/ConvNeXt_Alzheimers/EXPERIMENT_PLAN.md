# Step 8 — regularized CNN comparison and ConvNeXt-Tiny

## Evidence that motivated this step

The supplied initial baseline selected epoch 7 and stopped at epoch 12. Validation
patient accuracy was 80/102 (78.43%), patient macro-F1 0.729247, AUROC 0.832675,
AD recall 17/33 (51.52%) and NC recall 63/69 (91.30%). At that epoch, online
training slice accuracy was 96.17% versus validation slice accuracy 69.10%.
This is consistent with overfitting and uneven class performance; it is NOT a
proof of a data-pipeline bug. The majority-NC comparator achieved 67.65% accuracy
but zero AD recall. The baseline is informative and must be preserved.

The curve also has substantial epoch-to-epoch validation variation. A single
aggregate report cannot determine exactly which anatomical features were learned,
prove calibration, or establish why a specific image was misclassified.

## Predeclared next experiments — two runs, not an open-ended search

1. B1: exactly the existing BaselineCNN, retrained from random weights using the
   shared regularized settings below.
2. C1: a full ConvNeXt-Tiny architecture with one-channel MRI stem and two-class
   output, trained from random weights using the same settings.

Do NOT modify the original experiment, tune the test set, change patient splits,
regenerate a different seed because the result looks poor, or automatically choose
a worse baseline to make ConvNeXt look better. All runs remain in the report.
The better of B0 and B1 under the already-defined validation selection rule is the
candidate CNN for later final comparison; the B1/C1 pair supplies the matched-recipe
comparison. Report the additional validation use honestly.

## Shared settings

- Fixed finalized manifests; NC=0 and AD=1.
- Unchanged greyscale preprocessing: 224x224 tensor, range [-1,1]. No new image
  augmentation, thresholds, class weights or pretrained initialization.
- Physical batch 8, accumulated over 4 microbatches (effective batch 32, except
  the final partial window). Both models use the same physical batch size.
- Mean slice cross-entropy, uniformly shuffled slices. Patients with more scans
  still contribute more training images. This is not patient-balanced training.
- AdamW, maximum learning rate 0.0003. First 3 epochs linearly warm up; subsequent
  learning rates decay on a cosine schedule to 0.000001 at epoch 30.
- Weight decay 0.05 on kernels/matrices only; no decay on biases, one-dimensional
  normalization weights or LayerScale. Gradient norm clipping at 5.
- Up to 30 epochs. Early stopping after 8 unimproved epochs, but not before
  completing 12 epochs (unless the runtime budget is reached).
- Soft runtime budget 10,800 seconds per run, checked between complete epochs;
  an epoch can overrun that budget. Budget-stopped runs are labelled accordingly.
- Float32; no AMP or TF32. CUDA; zero loader workers initially. Seed 42.
- Validation: mean slice p(AD) within scan, then mean scan p(AD) within patient.
- Select checkpoint on patient macro-F1; lower patient log loss breaks exact
  ties, then earliest. Report slice, scan and patient metrics. Threshold 0.5.
- Test images are never decoded by this stage. Validation is development data,
  not an unbiased final performance estimate.

B1 changes several settings versus B0 together: learning rate/schedule, weight
regularization, physical batch size and early-stopping limits. Therefore it is
not an isolated causal test of weight decay. The architecture's own regularizers
also differ: the CNN retains dropout=0.25; ConvNeXt uses stochastic depth=0.1.
Gradient accumulation does not reproduce BatchNorm's large-batch statistics.
A shared recipe is a useful comparison, not a claim that it optimizes both models.

## Architecture

ConvNeXt-Tiny uses stage depths (3,3,9,3), widths (96,192,384,768), a stride-4
stem, channel LayerNorm, depthwise 7x7 convolution, 4x channel expansion via
linear layers, GELU, LayerScale=1e-6, residual connections, and linearly increasing
stochastic depth up to 0.1. Spatial averaging followed by LayerNorm feeds the
2-class head. The one-channel/two-class version has 27,818,594 trainable parameters.
The unchanged CNN has 359,922. No model library or downloaded weights are used.

Reference: Liu et al., *A ConvNet for the 2020s*, CVPR 2022,
https://arxiv.org/abs/2201.03545 (assessment reference [7]). The code is a PyTorch
implementation of the paper's architecture, with the MRI input/output adaptation;
it is not copied from or silently substituted with a pretrained library model.

## Limits, resource evidence and storage

The small CUDA check verifies training and checkpoint I/O, not generalisation.
Linux/CPU synthetic tests are not evidence of Windows/CUDA or ADNI performance.
Training records actual peak allocator memory, model-only batch-1 latency,
end-to-end validation time and wall runtime. Memory is not the total Windows GPU
usage; model-only latency is not patient-level throughput. Results on another
precision/hardware configuration require a separately labelled comparison.

All checkpoints/patient predictions stay in COMP3710-private, outside Git and
OneDrive. Only aggregate metrics/curves and reviewed code enter the repository.
Do not edit the previous JSON reports or split hashes. The installer saves the
original Step 7 source files so the original run remains reproducible.

Checkpoints are saved each epoch, but automatic training resume is NOT implemented.
On interruption or out-of-memory failure, preserve outputs and diagnose before
starting another run. No improvement or 0.80 held-out-test accuracy is guaranteed.

## AI disclosure

ChatGPT assisted with this plan, implementation, synthetic tests and report draft.
Review the model blocks, accumulation weighting, schedule, grouping and checkpoint
selection, run the local tests, and accurately document your own checks. Do not
substitute an automated pass marker for understanding the submitted work.
