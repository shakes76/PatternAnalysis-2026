# Frozen VAE feature diagnostic — 7 October 2026

Read102 deterministic patients per train/validation reference (33AD/69NC), using
the established patient_selection routine. No test image. Compare64 images each
from original SD, adapted SD CFG7.5, and the simple VAE. Preserve all samples.

Use the existing frozen SD1.5 VAE posterior mean at256, float32, no latent sampling.
Pool its4x32x32 spatial map to4x8x8 and flatten256 features; retain RGB generated
inputs so color artifacts are not silently removed. Real images use the existing
black-padding, replicated grayscale and[-1,1] preprocessing. Baseline128 outputs
are bilinearly enlarged to256; native resolution remains a confound.

Distances are unstandardized feature RMSE. Report generated->matched train and
validation nearest distances and reverse validation->generated distances. Also
report validation->train real-reference context. Reverse distance measures finite
sample reference coverage, not a calibrated percentage.64 generated vs102 real
references differ in count: additionally use a deterministic32AD/32NC train subset
for equal-count reverse-distance context. Report classes separately. Retain the
nearest-neighbor identities only in private records; no privacy guarantee follows.

t-SNE: all396 points jointly, fixed seed3710, perplexity30, PCA initialization,
1000 iterations, Euclidean distance in these256 features. Marker denotes source;
color denotes AD/NC metadata for real samples and requested prompt for generated
samples, not verified generated disease state. Do not calculate coverage from
2D projection or interpret its visual density as the original density.

Shared SD VAE features favor neither a clinical interpretation nor an independent
quality benchmark. Encoder/domain bias, pooling, alignment, and sample count limit
the claims. This addresses one permitted latent-distance diagnostic; SSIM,
synthetic-to-training-image interpolation, and final test evaluation remain open.
The earlier initial-noise interpolation does not satisfy that entire requirement.
