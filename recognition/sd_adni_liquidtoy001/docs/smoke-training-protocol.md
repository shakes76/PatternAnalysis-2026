# Bounded LoRA training and restart acceptance

Pre-run protocol, 4 October 2026. This is engineering acceptance, not a quality
experiment or a decision that LoRA alone satisfies the assessment.

Reuse the accepted patient manifest and SD 1.5 revision from VAE provenance.
Keep the VAE environment intact. Create `.venv-train` inheriting the existing
torch environment, with separately pinned diffusion/PEFT/Transformers packages.
CPU setup resolves dependencies before installation and downloads only UNet,
text encoder, tokenizer and scheduler assets. It reuses the existing VAE.
Require at least 6 GiB free before setup, record actual file hashes and versions.

First repeat VAE reconstruction with raw and clipped MSE, and intensity-based
background/foreground diagnostics (threshold 0.02 in [0,1], not segmentation).
Report signed extrema and overshoot magnitude, not just clipped pixel counts.

Short training uses the real pretrained UNet, frozen VAE and CLIP text encoder,
rank-4 LoRA on attention to_q/to_k/to_v/to_out.0, AdamW lr=1e-4, constant schedule,
batch one, 256 pixels, seed 3710, float32 weights and bfloat16 autocast. The first
eight deterministic patient-selected training examples are used; no test images.
Minimal text conditioning distinguishes AD and NC without invented pathology.
Use posterior sampling, model latent scaling and checkpoint prediction_type.

Run A: eight updates continuously. Run B: four updates, save and exit. Run C:
new Python process restoring B, continue to step eight. All on one bounded GPU
allocation of at most 20 minutes in a100-test. Keep loss and gradient-finiteness,
parameter-change, GPU allocation, timing, checkpoint bytes and source hashes.

Checkpoint contains adapter parameters, optimizer, LR scheduler, completed step,
CPU/CUDA RNG, fixed ordered input indices and a configuration/data/model identity
contract. Write via a temporary file and atomic rename. Reject a mismatched
contract. This small fixed-order diagnostic has no shuffled sampler/workers and
does not claim general production crash/signal recovery.

Acceptance compares resumed final adapters, optimizer and scheduler with A, as
well as loss/noise/timestep/input traces. Float tensors use atol=1e-6, rtol=1e-5;
discrete state and RNG must match exactly. Loss decreasing is not an acceptance
criterion over eight steps. Include a negative control: corrupt the saved contract
and verify rejection. A process exit alone is not successful acceptance.

Sources: https://huggingface.co/docs/diffusers/training/lora and
https://huggingface.co/docs/diffusers/api/models/autoencoderkl .
