# Proposed title

Add reproducible ConvNeXt Alzheimer's interfaces and report slice-level results

# Proposed description

This contribution adds an explicit-input preparation, A1 training and independent prediction route for the ADNI Alzheimer's prototype. It retains the patient-independent partitions, CNN baseline, selected checkpoint and progressive experiment history. The custom ConvNeXt-Tiny is trained from random weights; no pretrained initialization is used.

The retained A1 model (`scratch_augmented`, seed 42, epoch 13) records **3,157/4,640 test slices correct (68.04%)**, below the required 80%, versus CNN's 3,193/4,640 (68.81%). The user-supplied Ed clarification identifies individual 2D slices as the assessment unit. A1 was originally selected by patient-validation accuracy, macro-F1 and ordinary patient log loss; that selection is unchanged. Patient-level calibration and referral are supplementary analyses, not calibrated slice policies.

## Implementation and evidence

- `dataset.py prepare` reproduces the approved manifests into a new directory from explicit data inputs, without author-only pointer files.
- `train.py a1` calls the checked A1 recipe; `predict.py infer` accepts explicit checkpoint/data/manifest inputs and optional frozen patient rules. Strict source, architecture, state and manifest checks remain in place.
- Exact historical runtime bytes and [replay documentation](HISTORICAL_REPLAY.md) preserve checkpoint provenance. No datasets, weights, patient manifests, private predictions or MRI panels are included.
- The [report](README.md) retains the feasibility proposal, baseline/resource comparisons, five reviewed patient cases, negative schedule/G012/G3 results and AI disclosure. The [slice review](SLICE_ALIGNMENT_REVIEW.md) found no substantial quick gain supported by retained checkpoints and saved validation evidence; missing metrics and weights are explicit.

## Verification and limits

Earlier submission verification passed fifteen synthetic interface tests, byte-identical manifest reproduction and two 32-training-slice, one-update smoke checks across a line-ending correction. The corrected clean-code export reproduced all 4,660 slice, 233 scan and 102 patient validation probabilities exactly, within the existing 2e-5 tolerance and with unchanged decisions. These checks used the existing pinned Python 3.11.9 / PyTorch 2.7.1+cu126 environment; a separate dependency installation and full retraining were not tested.

The later slice-alignment review used saved scores and checkpoint metadata without model execution; test reporting used the existing aggregate summary. Prior test exposure, repeated validation use, correlated slices and absent test-slice Brier remain disclosed. The 80% requirement is unmet; no marking waiver or clinical readiness is claimed.

The README discloses ChatGPT and Codex assistance with implementation, checks, analysis and writing, distinguishing it from author-confirmed review. Method references: [ConvNeXt](https://arxiv.org/abs/2201.03545), [temperature scaling](https://arxiv.org/abs/1706.04599), [gated attention used in G3](https://arxiv.org/abs/1802.04712) and the supplied Version 2.5 specification.
