# Frozen-model resource measurements and direct prediction example

This is a measurement task, not a new training experiment. It reuses frozen
2D epoch-5 and 3D refinement-epoch-11 weights and all 27 paired case identities.
The Slurm partition/account/GPU/1.5h template is unchanged, with no additional
CPU or memory request. Cluster operations use shared tmux `codex-714278`.

## Resource definitions

`UNet2D_s4984160/support/profile_inference.py` measures one model at a time on
the same GPU, removing the previous model from GPU and clearing cached blocks
between modes. Inputs are preprocessed and cached on CPU before timing.
The three rows are 2D strict FP32, 3D strict FP32, and 3D BF16. TF32 is disabled
for the matched FP32 comparison. Existing segmentation scores are not changed.

- One 2D case is all 60 padded native slices, streamed in batches of 16;
  one 3D case is the full resized 64×128×128 volume, batch size one.
- Three whole-case warmups on the first case, then three measurements per case.
- Forward-only milliseconds sum all batch forwards for the case, synchronized
  around each forward; model loading, preprocessing and transfers are excluded.
- Prediction milliseconds include transfers, forward, argmax and output copy,
  but still exclude disk loading, preprocessing and initial weight loading.
- Peak PyTorch allocated/reserved GiB include the resident model and inference
  tensors/cache, not CUDA driver/context or training memory. These are explicitly
  defined process-local allocation measures, not total GPU usage from other jobs.
- Aggregates are mean/median of per-case medians over 27 cases. Peak memory is
  the maximum across measured cases/repeats. The raw per-case CSV is retained.

Same hardware/precision controls do not remove differences in coverage,
resolution, normalization, capacity or training. Report these pipeline costs
without attributing the entire difference to convolution dimension.

Outputs are `results/resource_JOBID/{summary.json,per_case_profile.csv,profile_complete.txt}`
under the 2D cluster directory. Measured values should fill the report only
after the job completes and the recorded checksums/configuration are checked.

Job **717015** has now completed. The [archived measurements and direct example](../saved_results/resource_results/README.md)
contain the checked results; the report uses the new isolated profiles. The
earlier 3D-only timing remains in its original archive for provenance.

## 3D training curves from existing CSV

`support/plot_training_curves.py` reads the 15 initial and 11 refinement epochs.
It draws training objective and validation prostate Dice without any model
inference. Loss weights/optimizer differ between phases, so lines are separate;
the CSV did not log validation loss and no such curve is fabricated.

```bash
python -m support.plot_training_curves \
  --scratch-history support/saved_results/training_results/scratch_history.csv \
  --refine-history support/saved_results/test_results/history.csv \
  --output-dir results/replot_3d --selected-refine-epoch 11
```

## Direct checkpoint prediction

From the 3D project directory, on a GPU node with the private course files:

```bash
python predict.py \
  --checkpoint results/refine_v2/best_model.pt \
  --mri /home/groups/comp3710/HipMRI_Study_open/semantic_MRs/B037_Week0_LFOV.nii.gz \
  --mask /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/B037_Week0_SEMANTIC.nii.gz \
  --expected-sha256 77ecc9f051d5d1213c2e62474d582050fe5b425a08182e3d2becc73540737156 \
  --device cuda --precision bf16 --threads 1 --output-dir results/direct_example
```

This labelled example loads model weights with strict configuration matching,
uses `dataset.py` preprocessing, performs inference, prints Dice/HD95 and saves
`prediction.png`, `summary.json`, resized-grid `prediction_dhw.npy` and a
completion marker. The mask is used for evaluation/view selection, not model
input. Predictions are resized D,H,W arrays, not original-resolution clinical
contours. No aggregate result or new model selection is inferred from one case.
CPU FP32 is available for demonstration; original BF16 scores need not be
bit-identical to another device/precision. If local SciPy is unavailable,
HD95 is explicitly unavailable rather than zero.

The previous `--input-dir ... --output-dir ...` cached-render command continues
to work and does not repeat inference. The resource job also executes the
direct GPU example after profiling, using an isolated uploaded source snapshot.
