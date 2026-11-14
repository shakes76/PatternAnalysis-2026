# VAE reconstruction diagnostic

Recorded before execution, 2 October 2026 (Australia/Brisbane).

Data acceptance job 623310 completed with exit 0:0 in 40 seconds. All 30,520
JPEGs decoded, with zero cross-split patient/scan/hash overlap. There are
476/102/102 patients and 21,280/4,540/4,700 slices in train/validation/test.
The existing manifest is reused; the held-out test images are not evaluated.

The first model diagnostic uses only the VAE of
`stable-diffusion-v1-5/stable-diffusion-v1-5`, revision
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, from the ungated SD 1.5 mirror.
Its model card identifies CreativeML OpenRAIL-M and a lossy autoencoder.
This is a candidate component check, not a final fine-tuning design decision.

Create a project-local venv inheriting the existing torch environment to avoid
duplicating CUDA packages. Install additional pinned packages only in that venv,
run pip check and record the complete resolved environment. Download only the
VAE config and float32 safetensors (roughly 335 MB), on a CPU allocation.
Require 2 GiB free before downloading; the current home has approximately 12 GiB
free. No cleanup or modification of the unrelated Transformer environment.

For each of train and validation, select two distinct patients per class using
seeded SHA-256 ordering. Within each selected patient take the middle available
slice of the first scan. Selection occurs before reconstruction, never by output
quality. Run the same eight slices at 256 and 512 pixels, batch size one,
float32, posterior mode (deterministic). Encode then decode directly: latent
scaling is not applied between these two VAE operations; a diffusion pipeline
would apply it before denoising and undo it before decoding.

Save original/reconstruction/absolute-error triptychs locally with anonymous
sample labels and a fixed error display scale (5x). Measure paired RGB MSE and
PSNR on [0,1] images after clipping decoder outputs, and separately record the
fraction clipped. These are reconstruction diagnostics, not SSIM, generation
quality, population estimates, or medical validation. Record raw finite checks,
peak GPU allocation, timing, source hashes, manifest hash and model provenance.
Keep sample identities and images in ignored outputs; do not publish to the site.

Sources:
- https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- https://huggingface.co/docs/diffusers/api/models/autoencoderkl
