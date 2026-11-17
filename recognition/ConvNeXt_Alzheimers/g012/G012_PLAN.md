# G0/G1/G2 scan-supervision study — seed 42

Predeclared implementation-phase plan, 27 October 2026. The user runs the real
smokes/training/replays; Codex executes synthetic tests and read-only checks only.
The adopted 01_IMPLEMENT_AND_RETURN_COMMANDS.md protocol supersedes the research
report's conditional G2 sequence and two-patient G1–G0 margin. Exactly G0, G1 and
G2 are planned together, each once at seed 42. A1 remains selected throughout.

The machine-readable G012_PLAN.json binds runtime/import-closure source bytes,
approved manifest identities, exact A1 control measurements and configurations.
It excludes itself, report prose, tests and post-run handbacks from its source
hash set, avoiding circular or changing checkpoint prerequisites. The private
prepared study binds the plan, explicit reference paths/hashes, initial tensor
digest and all 30 deterministic order/augmentation schedules. No trained A1
weights initialize G0/G1/G2. Implementation and plan must be committed first.

## Losses and predictions

For K positive slices with raw logits z and hard scan label y:

- G0: mean negative true-class log-softmax over slices; scan probabilities are
  mean slice softmax probabilities.
- G1: negative [logsumexp of true-class slice log-softmax minus log K]; scan
  probabilities are mean slice softmax probabilities. The aggregate stays in
  the gradient graph. No cross-entropy on probability values as if they were logits.
- G2: hard-label cross-entropy on mean raw slice logits; scan probabilities are
  softmax of those mean logits, in training diagnostics, selection, saved
  predictions, replay and comparison.

Every patient probability is the mean of its scan AD probabilities; threshold
is >=0.5. G2 does not average patient logits. Slice metrics are diagnostic only.
The raw G0/G1/G2 objective values have different scales and are plotted separately.
G1–G0 isolates the loss in matched grouped batching; G2 also changes aggregation.
G0–historical-A1 changes batching/update count and is not an isolated loss effect.

## Fixed data and optimization

Use only approved membership: 1,061 training scans/476 patients/21,220 slices,
233 validation scans/102 patients/4,660 slices, exactly 20 slices per scan. Check
whole-scan coverage, unique slice indices and one patient/label/partition per
scan. Unexpected counts fail before model training. Pooling supports general K
in synthetic math tests; no padding, omissions, reshuffling partitions or resampling.
The test manifest is read only by existing structural/hash/disjointness checks;
no test images, predictions or panels are study inputs. Original source folder
names do not determine eligibility after approved regrouping.

One complete scan per forward/backward microbatch; two sequential scan graphs
per optimizer update, each freed after backward. A final one-scan window uses
denominator one. All 1,061 scans are visited once: 531 updates/epoch, versus
historical A1's 664. No LR scaling. Identical seed-42 random ConvNeXt-Tiny tensors,
scan order and independent per-slice +/-5-degree angles across all variants.
Separate CPU generators determine order and angles; a separate deterministic
epoch stream controls stochastic depth. Validation preserves global RNG state.

Reuse the unchanged one-channel/two-class ConvNeXt, grayscale aspect-preserving
preprocessing and A1 AdamW groups/defaults, decay 0.05 on eligible tensors, clip
global norm 5 after accumulation, float32, original deterministic settings,
stochastic depth and zero workers. Original A1 LR function: epochs 1–3 warmup
to 3e-4; epoch 4 peak; epoch 30 floor 1e-6. No rejected epoch-12 schedule.
Maximum 30/minimum 12 epochs, patience 8. Select validation patient accuracy,
then macro-F1, then lower ordinary patient NLL; retain earliest exact tie.
Selected epochs before 12 remain eligible; epoch 13 is never forced.

No pretrained initialization, smoothing, auxiliary/focal loss, attention, EMA,
ensemble, calibration, threshold fitting, additional seeds or automatic adoption.

## Execution and recovery

User-run GPU smoke for each variant uses the same fixed three complete training
scans (two NC and one AD), 60 slices, two updates including the one-scan tail.
It uses the full model and real preprocessing and is separately marked non-candidate.
All three smokes and the bound historical A1 regression must pass before training.
Real-run weights/RNG are reset; smoke weights are never reused. The historical
regression reuses the exact prior clean validation-runtime receipt and rechecks
its frozen source/checkpoint/manifest bindings and all cached validation outputs.
It runs no new A1 inference. Each NEW selected checkpoint requires a full native
validation replay with absolute probability tolerance 2e-5 and identical classes
at slice, scan and patient levels; retain G2 slice logits privately.

Use existing V2 locks/atomic completed-epoch saves with previous valid state,
and the existing LeaseBudget/WindowsJob helpers. Save model/optimizer, global
Python/NumPy/Torch/CUDA RNG, custom order/angle generator states/digests and next
epoch position, LR schedule position, full history, selection and a live-budget
snapshot. The external cumulative ledger remains authoritative; recovery never
restores it backwards. Repeat interrupted partial epochs within the SAME attempt.
Unreadable latest state may recover the previous valid state; no usable state
is a blocker, never fresh training. Completed receipt hashes are checked and
completed training/replay is skipped on repeated commands.

Cap supervised active execution at 10,800 seconds per variant and 32,400 total.
Include initialization, validation, I/O, repeated partial work AND the separately
commanded selected-checkpoint replay under the same cumulative ledger. Exclude
idle time, preparation and non-candidate smoke, whose resources are separately
reported. Five-second prepaid leases conservatively charge unknown crash work.
Cap exhaustion is WALL_LIMIT_INCONCLUSIVE, never protocol completion. Do not
extend limits, change batches/precision or create replacement attempts.

Require 12 GiB free on the study output volume. GPU smoke records actual allocated,
reserved, total and sampled free device memory. A 4.5 GiB allocated peak is only
a warning. Refuse OOM or sampled free memory below max(1 GiB,10% device capacity);
report the blocker without changing model/bag/precision. These implementation
headroom guards are feasibility constraints, not an accuracy hypothesis.

## Development screen and outputs

Each of G0/G1/G2 is eligible for further confirmation at >=86/102 correct,
>=22/33 AD detections, macro-F1 >= the exact A1 reference
**0.8012151782643586**, and completed integrity/replay. No two-patient margin over
G0 is required to acknowledge a candidate. Separately report paired differences
versus A1 and G0 and whether G1/G2 outperform G0. One seed on reused validation
patients cannot establish repeatability, loss attribution beyond its control,
generalization, clinical safety or a test-score gain. No automatic promotion.

Report all four models, completion/stop reason, selected/completed epochs, full
patient/class/native-scan metrics, ordinary NLL/Brier, diagnostic slice metrics,
all trajectories and fixed last-five COMPLETED epoch means/population SD/range.
NLL deterioration >=0.02 versus A1/G0 is a review warning, not an undisclosed gate.
Keep incomplete/blocked outcomes and near misses. Store full private histories,
weights/optimizer/recovery, logits/predictions, resources, schedule digests and
replay receipts locally. Export only the allowlisted aggregate ZIP; no IDs,
image paths, probabilities/logits by patient, raw images, states or private paths.
Do not publish pending results or alter the selected-model README/interface.

AI assistance: Codex implementation, tests and command drafting. This scoped
verification is not a proof of complete correctness. REAL EXPERIMENTS HAVE NOT RUN
as part of implementation; the next user prompt will separately authorize analysis.
