# Cohort and simple baseline protocol — 5 October 2026

Before execution: original SD and step20000 SD each generate64 uncurated images,
AD/NC x seeds37200–37231, DDIM30 eta0 guidance7.5 at256. Match within-resolution
initial noise; use float32 scheduler input as verified by the parity experiment.
Base uses freshly initialized zero-output LoRA, not the learned adapter values.

Audit full training and validation references at grayscale64 with pixel RMSE,
plus class-count-matched patient subsets (one deterministic middle-ordered slice
per patient, matching the validation patient counts). No learned preprocessing
and no test images. Full reference sets are unequal; distances are not directly
comparable privacy scores. Report pairwise generated RMSE overall and per label.
Exact copied references must retrieve distance0 as an actual positive control.
Private neighbor records retain identities locally/remote only; only aggregates
may enter Git. Source and generated hashes checked. No arbitrary-pair SSIM/FID.

Simple baseline: independently initialized convolutional VAE at128 pixels,
64 latent dimensions, grayscale input, one-hot class conditioning in decoder.
MSE summed over pixels plus analytical KL summed over latents, divided by batch;
coefficient1 for KL. Adam1e-3, batch32,5 epochs over training data, seed3710.
Five convolutional reductions128->4; symmetric decoder. Prior generation uses
standard normal latents, same predeclared seed labels but not paired SD latent
noise (different architectures). Save complete state every epoch. A short real
forward/backward/save/reload acceptance gate precedes a longer comp3710 job.
Native resolutions differ, so64-pixel metrics are exploratory, not a fair
matched-capacity final benchmark. No pretrained VAE weights in this baseline.

Budget: cohort GPU cap20min; CPU reference audit cap1h; baseline cap2h, oneGPU.
No new downloads. Never equate job success, clean images or distant neighbors
with anatomy, diversity or privacy acceptance. Student review remains pending.
