# Inference diagnosis — 5 October 2026

Keep the trained checkpoint fixed at20000, with no optimizer updates. Check
model/VAE provenance identities and adapter parameter names before loading.
Use the official StableDiffusionPipeline and DDIM30, eta0, same embeddings,
fp32 weights, bf16 denoising, fp32 VAE decode and identical quantization.
Compare four historical256/guidance1 outputs (AD/NC x seeds37100/37101) by
pixel arrays. Preserve differences; do not force a pass or assume matching
rounded pixels prove identical intermediate latents.

Then run the fixed2x2 settings:256/512 and guidance1/7.5 on the same four
label/seed combinations,16 images in total. Same-resolution pairs share exact
initial-noise hashes. Across resolutions, the same seed has a different tensor
shape and does not establish pixel-aligned paired noise. Empty negative prompt.
All images retained, no quality-based sample selection.

This diagnostic distinguishes some sampling effects from training effects; it
does not validate anatomical fidelity, disease specificity or privacy. Conditional
dropout was not used in training, so CFG results require empirical inspection.
Use one bounded20min A100 job, no new model downloads, no held-out data.
