# Paired generation pilot — 4 October 2026

Pre-run protocol. This is a bounded exploration, not the final baseline comparison.
Use the accepted SD1.5 revision, manifest and preprocessing from the smoke run.
Select 16 training patients per class by the existing deterministic hash ordering,
one middle ordered slice per patient. This selection is not anatomical alignment.
Cycle seeded shuffled permutations of these 32 indices for exactly 200 updates.
Keep rank4, AdamW lr1e-4, batch1, 256 pixels, frozen VAE/CLIP/base UNet,
float32 weights and bfloat16 UNet autocast. Hold all test data untouched.

Save full state at 100 and restart a fresh process for the remaining 100 updates.
Keep the previously accepted eight-step diagnostic unchanged. This new pilot
does not separately prove continuous-versus-resumed equivalence at 200 steps.

At steps 0, 100 and 200, generate all eight label/seed combinations:
AD and NC, seeds 37100–37103. Match initial latent hashes across stages.
Use the same training prompts, DDIM30, eta0, guidance1 (conditional prediction
without classifier-free extrapolation), 256 pixels. No seed or image selection
after seeing results. Sampling preserves training RNG and train/eval mode.
Step0 uses zero-initialized LoRA output and therefore unchanged base predictions.

Budget: one A100 allocation, maximum20min, existing model files only; reserve
at least1GiB persistent free space. Store images/checkpoints privately outside Git.
Record finite loss/gradient checks, actual parameter changes, memory, timings,
full sampling configuration and artifact hashes. Compare grids visually without
claiming anatomical accuracy, disease conditioning, generalization or privacy.
No arbitrary-pair SSIM or tiny-sample FID will be used as a quality verdict.

The assessment Section2.4 still requires a separately implemented simple baseline,
distribution comparisons, embeddings, failure analysis and memorization auditing.
LoRA implementation scope remains a tutor-check-off topic. Student review of
AI-assisted implementation is pending.
