# Full training-split experiment — 4 October 2026

Pre-run commitment: one 20,000-update experiment, fresh pinned SD1.5 weights;
do not initialize from the 32-patient pilot. Use the accepted 21,280 training
slices (476 patients) only, in a seed3710 shuffled permutation. At batch1,
20,000 updates cover about94% of one slice epoch, not many full epochs.
The sampler is slice-uniform, not patient-uniform; patients with more scans have
more weight. All slice positions remain included and anatomical alignment is
not asserted. Validation and test data are not read by this training script.

Keep pilot architecture/optimization: 256 pixels, frozen VAE/CLIP/base UNet,
attention LoRA rank4,797184 trainable parameters, AdamW1e-4, constant LR,
gradient clipping1.0, fp32 weights/bf16 autocast. This is a first longer study,
not a claim of convergence or sufficient final assessment scope.

Every2000 updates save a separate atomic full checkpoint (adapter, optimizer,
scheduler, RNG, step, data order and identity contract), then sample fixed probes
and restart the next stage in a new process. Keep all ten checkpoints. Previous
completed stages survive later-stage failure; at most one2000-step stage may be
lost on interruption. Signal-triggered emergency saving is not implemented.
Generation follows the pilot's exact eight label/seed probes, DDIM30, guidance1,
eta0,256 pixels; include step0. Save all88 images without cherry-picking.

Submit a new eight-step continuous-versus4+4 restart GPU gate, including actual
state comparison and deliberate corruption rejection. The long job uses Slurm
afterok dependency, so it can queue now but cannot train before the gate succeeds.
If the gate fails, hold the long run for repair; do not treat submission as success.

Resources: one A100, two CPUs, comp3710 account/partition,4h wall-time cap.
Live QoS inspection reports comp3710 MaxWall12h. Current home free8.6GiB;
require1GiB free at startup. No new model download. Budget <1GiB new artifacts,
including checkpoints, trace logs and images; measure actual totals afterwards.
The pilot timing is only a rough sizing input, not a completion guarantee.

Preserve source snapshot in a dedicated remote run directory with model/data/env
symlinks. Do not mutate that snapshot while queued or running. Local commits and
private backup only; no public course push. Training completion, sample quality,
memorization audit and simple baseline comparison are separate acceptance steps.
