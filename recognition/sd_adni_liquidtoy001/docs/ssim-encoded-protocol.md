# SSIM and encoded endpoints — 7 October 2026

Before scoring, fix256 grayscale SSIM, scikit-image0.25.2, data_range1,
Gaussian weights,sigma1.5,population covariance. Report all192 generated samples
against same-label validation references under two pairing rules: nearest in the
previously extracted VAE features, and a deterministic reference chosen by
SHA256(label:seed) modulo the label's reference count (identical across models).
Neither pairing supplies ground truth for an unconditional generated image.
Nearest selection is optimistically biased; the fixed reference is arbitrary,
not an image-reconstruction target. No best-SSIM search. Report mean/median/range,
fraction above0.6 and class groups as descriptive observations, not a clinical pass.
Real validation->same-label training feature-neighbor SSIM supplies context.
Native128 baseline is enlarged; background and alignment affect every score.

Encoded interpolation: fixed adapted samples NC37215 and AD37220, each paired
with its same-label nearest among102 training references in the frozen VAE feature
space. Encode full RGB generated image and replicated-gray training image with
posterior means; interpolate unscaled VAE codes linearly at0,.25,.5,.75,1, decode
with frozen VAE. This is the VAE subspace, not a reverse diffusion/noise inversion.
Endpoints are reconstructions, not originals. Display original endpoints alongside
all five decoded frames. Record endpoint reconstruction errors. No medical or
privacy guarantee is inferred from a continuous decoder path. No test images.

Reference API: https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html
Original patient references, intermediate images and per-image records stay private.
