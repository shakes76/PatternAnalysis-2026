# Project Plan and Feasibility Review Outline

**Historical outline below was recorded on 2 October. Current status is in the 7 October coverage ledger at the end; old pending items are not current claims.**

Recorded: 2026-10-02 (Australia/Brisbane).

This is a planning record, not a completed feasibility review or evidence of tutor check-off.

## Research question

How does domain adaptation change structural fidelity, diversity, and evidence of memorization in generated ADNI brain MRI slices?

## Intended user and scope

The user is a researcher evaluating synthetic MRI cohorts. Study 2D image generation and its limitations. Do not infer clinical safety or privacy guarantees from image appearance or nearest-neighbor checks alone.

## Proposed acceptance criteria

1. Produce an auditable patient-level split manifest, with zero known patient overlap. If identifiers are unavailable or ambiguous, resolve the split design before training.
2. Execute a real forward/backward/update step and a save/restart/resume check with finite loss and gradients. Record restored optimizer, scheduler, step, and relevant random/data state.
3. Compare an implemented simple generator, an unadapted SD model, and an adapted SD model under one documented evaluation protocol.
4. Report uncurated generated samples, distribution/coverage measurements, nearest-neighbor audits, and 3–5 representative failures. Specify how SSIM reference pairs are formed; do not treat arbitrary image pairs as a validated generation metric.
5. Measure peak GPU and host memory, step time, and checkpoint size. Demonstrate durable checkpoint storage within verified available space and completion or recovery within the verified Slurm wall-time limit.

These criteria are proposals. Numerical quality targets and resource budgets must be finalized using the course requirements and actual data/resource evidence.

## Execution sequence

1. Read-only Rangpur audit: current partition/account limits, available persistent storage, environment compatibility, and dataset layout/identity metadata.
2. Data audit: patient grouping, image dimensions/orientation, intensity handling, duplicates, and label availability. Split patients before selecting slices. Fit any learned preprocessing only on training data.
3. VAE reconstruction check: compare real slices with reconstructions and error maps to identify compression limitations.
4. Short training and recovery test: choose a versioned pretrained checkpoint and run a bounded experiment; measure resources before estimating a full run.
5. Baseline and adaptation experiments: select a simple generative baseline, fix evaluation settings, and vary one justified adaptation factor at a time.
6. Final evaluation/report: reserve held-out test data for the finalized protocol, write limitations and recommendations, verify reproduction, then prepare the PR and PDF.

## Rangpur storage and scheduling

Use CPU jobs for environment setup and preprocessing, GPU jobs for training/evaluation, and the login node for lightweight operations and job submission. Keep reconstructible caches in job scratch; keep recovery checkpoints in verified persistent storage and back up completed artifacts locally.

Historical user records report an A100 PCIe 40GB, a 12-hour course partition, and constrained home storage. These are not current resource guarantees. Verify them before writing final job resource directives. GPU memory and Slurm `--mem` host-memory allocation are distinct.

## Open decisions and risks

- Confirm the permitted pretrained-library scope and whether LoRA is sufficient for the main SD implementation.
- Determine whether the course data has reliable patient identities and a usable conditional label/slice description.
- Select the exact base-model identifier/revision and verify access and terms.
- Select and implement the simple baseline; unadapted SD alone does not replace the requested simple generative baseline.
- Check diffusion dependencies against the available PyTorch environment without disrupting other projects.
- Budget persistent storage for optimizer state and overlapping files during atomic checkpoint replacement.
- Validate signal handling for the actual launcher/process tree; previous project recovery tests do not validate this trainer.
- Confirm the midpoint consultation date and final submission time.

## Preliminary evidence

The [2 October audit](rangpur-audit-2026-10-02.md) found 30,520 grayscale JPEG slices from 1,526 scans and 680 patients, with metadata available for patient grouping. The supplied train/test directories share 216 patients and must not be described as patient-independent. Home has approximately 2.20 GiB free; a durable storage budget is still unresolved. No baseline smoke test or model training has occurred. Add the final split design and resource budget before presenting the one-page feasibility review.

## 2026-10-07 coverage of the requirements and next steps (status on that date)

| Requirement | Evidence at the time | Status / gap |
| --- | --- | --- |
| Patient isolation and data protocol | 476/102/102 patients, full hashing and load checks | accepted; test split not used for evaluation |
| SD and an independent simple baseline | LoRA 20k steps, conditional VAE 5 epochs | run; scope of LoRA and pretrained libraries to be confirmed with the tutor |
| Training stability and resources | strict 8-step resume equality, 20k-step records, memory and time | evidence exists; not a full uninterrupted 20k comparison |
| SSIM and distribution measures | two paired SSIM variants, pixel and VAE-feature nearest neighbours in both directions | computed; SD mean below 0.6, must not be reported as meeting the target |
| t-SNE/UMAP | 396-point joint t-SNE, coloured by metadata and prompt label | generated; shared-encoder bias, does not show clinical labels are correct |
| Complete fixed-seed grids | 64 images for each of three models, 192 for the CFG sweep | inspected; all outputs kept |
| 3-5 failure cases | grey background, colour artefacts, blurry baseline | 3 kinds observed; SD's own failure cases and root causes still to be refined by the student; risk probes not to be passed off as failures |
| Nearest neighbours and interpolation | pixel and feature neighbours, noise paths, real/generated VAE-code paths | evidence exists; cannot prove absence of memorisation or safety |
| One-page feasibility review and check-off | feasibility-review-2026-10-07.pdf (local, not in the repository) | one-page English draft made; student review and tutor record not done |
| Final code entry points | dataset.py and diagnostic scripts | modules.py/train.py/predict.py not unified; reproduction environment list still to be written |
| README, PDF, final PR | local incremental commits and backups, AI_USAGE | formal README report, PDF and PR not done; must not be described as submitted |
| Final test evaluation | test split sealed | to run only after a fixed protocol, never opened early for tuning |

Priority: the student reviews the one-page material and completes the consultation first; in parallel, consolidate the existing code into final entry points with reproduction notes. Sampling searches should not keep expanding. Any further training needs a single hypothesis, a matched budget and an explicit stopping condition.

### Points the student should be able to explain at the consultation

1. Why do the 216 patients shared across the original folders cause leakage, and how does the patient-level split fix it?
2. What does SD predict, which parameters does LoRA update, and why are the other components frozen?
3. Why does the VAE score a higher SSIM despite being blurrier? Why must the highest-SSIM pairing not be picked?
4. What can the t-SNE plot and the nearest-neighbour audit each not prove?
5. Keep observations (grey background) separate from hypothesised causes (prior, sampling, undertraining).

The consultation record is still empty; no tutor approval or completion date may be filled in before it happens. (Translated from the original Chinese note on 23 October 2026; the check-off took place on 15 October, see the README's feasibility review.)
