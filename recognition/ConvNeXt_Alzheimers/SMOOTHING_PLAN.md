# Controlled ConvNeXt scratch-training experiments: S1 and S2

## Purpose and prior evidence

The existing A1 seed-42 reference remains unchanged (selected epoch 13, patient
validation accuracy 85/102, AD detected 22/33). Existing A1 diagnostics found
confidently wrong patients, substantial training/validation gaps and variation
between epochs. These observations motivate hypotheses, not established causes.
No pretrained weights, new data, relabelling or replacement partitions are used.

## Predeclared experiments

| Setting | A1 reference | S1 | S2 |
|---|---:|---:|---:|
| Uniform label smoothing | 0 | 0.10 | 0.10 |
| Maximum learning rate | 0.0003 | 0.0003 | 0.0001 |
| Linear warmup epochs | 3 | 3 | 3 |
| Cosine floor reached | epoch 30 | epoch 30 | epoch 12 |
| Floor | 0.000001 | 0.000001 | 0.000001 |
| After floor | stop at configured maximum | same | hold floor |

All other recorded A1 settings are copied and checked: custom ConvNeXt-Tiny,
random seed 42, one-channel 224x224 preparation, +/-5 degree training-only rotation,
no patient/class weighting, physical batch 8, accumulation 4, AdamW, weight decay
0.05 on kernels/matrices (not biases or one-dimensional parameters), clipping 5,
30 maximum epochs, 12 minimum epochs, patience 8, float32, no TF32 or AMP, zero
loader subprocess workers. Initialization, shuffle and augmentation RNG handling
follow the existing V2 trainer. Warmup reaches the peak in epoch 3; the cosine
phase starts at the peak in epoch 4. S2 reaches the floor in epoch 12 inclusive.

## Loss and measurements

For two classes, q = (1-epsilon)*one_hot(y) + epsilon/2 with epsilon=0.10 gives
0.95 for the correct training class and 0.05 for the other. L = -sum(q*log_softmax).
The implementation returns per-example loss and divides its sum by the true
accumulation-window sample count, including the short final batch. There is no
batch-dependent weighting renormalization. Stored class labels are never changed.

Record both the smoothed training objective and ordinary training log loss.
Validation and patient metrics use original hard labels and unchanged probabilities.
The smoothed training objective must not be compared directly with an old
unsmoothed objective as evidence of improvement. Validation probabilities are
mean(slice probabilities within scan), then mean(scan probabilities within patient).
The AD cutoff remains 0.5; no temperature/referral/threshold search occurs here.

Checkpoint selection: highest validation patient accuracy, then highest macro-F1,
then lowest ordinary patient log loss; keep the earliest exact tie. The stopping
rule is unchanged. S1 tests loss smoothing; S2 additionally changes both peak LR
and decay timing. S2 does not isolate those two schedule components from each other.

## Selection safeguard and scope endpoint

A candidate passes the provisional promotion screen only if it matches/exceeds
A1 patient accuracy without detecting fewer AD patients. Among eligible candidates,
compare accuracy, macro-F1 and ordinary log loss, and inspect AUROC and errors.
This is an engineering safeguard, not an assessment requirement or clinical
standard. Passing the screen is not automatic final selection. Preserve A1 even
when new runs do not improve it; report negative experiments.

Exactly two new configurations are supported by the CLI. No new seed search or
confirmation run is started automatically. After these results, select using
development evidence and integrate the selected model, calibration and report.
At most one separately declared confirmation run could follow a promising change;
it is not authorized or implemented by this package.

## Evaluation and privacy boundaries

The previous Version 1 test has already been exposed. These experiments use only
training and validation images, with no new test predictions. Existing partition
manifests are read to verify grouping and hashes; test images are not decoded.
Do not describe selected validation scores as fresh test results or proof of the
0.80 test target. Preserve patient/scan/slice reporting and disclose previous test
exposure in any later revised evaluation. No epoch, seed, threshold or patient
selection may be based on another query to the old test during this stage.

Private checkpoints and per-patient predictions remain outside Git and OneDrive.
The results bundle uses an explicit allowlist of aggregate reports/curves/history;
it excludes patient identities, prediction tables and checkpoints. Aggregate
reporting is not a formal anonymisation guarantee; review before sharing.

## Recovery and provenance

Save full model, optimizer, Python/NumPy/PyTorch/CUDA RNG states, history, selected
model and learning schedule after each completed epoch. Atomic replacement keeps
one prior checkpoint; an unfinished epoch repeats. Recovery rejects configuration,
source, manifest or environment changes. The same run command starts, resumes or
verifies-and-skips a completed run. Complete-epoch runtime omits interrupted work,
checkpoint/report I/O and downtime; GPU memory is PyTorch allocator usage, not
Windows total. Private state files are trusted local artifacts, not arbitrary
external checkpoints to load.

## Assistance and references

ConvNeXt architecture: Liu et al., *A ConvNet for the 2020s*, CVPR 2022
(assessment reference [7]). The implementation remains in existing modules.py.
The two experimental choices above are proposed engineering settings, not course
mandates. This plan refers to the supplied COMP3710 Version 2.5 classification
requirements; no claim of full marks or a guaranteed accuracy increase is made.

ChatGPT assisted with this additive implementation, synthetic tests and report
drafts. The student must review the code, run the local checks, explain the loss
and schedule, and record actual verification in the AI disclosure.
