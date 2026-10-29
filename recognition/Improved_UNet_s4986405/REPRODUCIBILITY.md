# Reproducing the HipMRI comparison

**29 October update:** the student reports compulsory tutor check-off completed. The original numeric evidence below is retained as a historical record; [README](README.md) and [anatomical analysis](ANATOMICAL_LABEL_ANALYSIS.md) document the subsequent image-based prostate = ID 5 interpretation and direct baseline comparison.

This guide records the completed validation experiment and the submitted fixed-model test protocol. Commands that launch experiments are reference procedures for deliberate reproduction. Job 705243 is recorded as COMPLETED / 0:0, but its result files were not collected. The user-requested local fixed-model evaluation has completed; see TEST_RESULTS.md. Preserve its source snapshot, original model runtime and weights.

## Observed environment

| Component | Observed version |
|---|---|
| Python | 3.11.16, GCC 14.3.0 |
| PyTorch imported runtime | 2.14.0+cu130 |
| PyTorch distribution metadata | 2.14.0 |
| PyTorch CUDA build | 13.0 |
| NumPy | 2.4.6 |
| NiBabel | 5.4.2 |
| Matplotlib | 3.11.2 |
| Training GPU | NVIDIA A100-PCIE-40GB |
| Allocated-node driver observed during GPU checks | 590.48.01 |

The Python/package metadata was printed on Rangpur and saved to `$HOME/hipmri_runs/runtime_environment_705243.json`; that original metadata file was not part of the local input download. A separate exact local runtime record is committed with the test results. PyTorch runtime/CUDA and training GPU observations also appear in completed training records. Distribution metadata and `torch.__version__` are distinct observations. The environment is not a tested clean installation, complete dependency lock or guarantee that a particular package index supplies the same CUDA wheel. `requirements.txt` is an unpinned dependency list; NumPy is also used directly by loading/visualization code.

The actual interpreter is `$HOME/miniconda3/envs/torch/bin/python`. For the examples below, activate the existing environment:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
export HIPMRI_DATA=/home/groups/comp3710/HipMRI_Study_open/keras_slices_data
```

MRI data access and the original checkpoint files are prerequisites. A fresh install from these version observations has not been validated. Do not infer A100 execution from a login-node CUDA warning; the completed experiment records identify allocated compute nodes separately.

## Source revisions and snapshots

| Purpose | Exact revision / location |
|---|---|
| Training and evaluator source | `bc8a6a4688f50770f71de24a56dff8aad43204c7` |
| Original runtime snapshot | `$HOME/hipmri_comparisons/bc8a6a4688f50770f71de24a56dff8aad43204c7` |
| Held-out runner source | `bc601d92de5fb3da982b43d96de34afcd544a94c` |
| Submitted held-out snapshot | `$HOME/hipmri_heldout/bc601d92de5fb3da982b43d96de34afcd544a94c_fixed` |

Exporting an exact project tree must run Git at the repository root. For example, the following produces an archive only; it does not launch training or test evaluation:

```bash
training_source=bc8a6a4688f50770f71de24a56dff8aad43204c7
git -C "$HOME/COMP3710/PatternAnalysis-2026" archive \
  --format=tar --output="$HOME/hipmri_training_source_reproduction.tar" \
  "$training_source:recognition/Improved_UNet_s4986405"
tar -tf "$HOME/hipmri_training_source_reproduction.tar"
```

Check that the archive lists project files at its root. Transfer and extract into a fresh destination on Rangpur. Run from that snapshot and record its exact source revision. The submitted snapshots must remain unchanged; do not extract over an active or completed experiment source directory. Original provenance records identify runtime file SHA256 values in addition to the Git revision.

## Full training procedure

`comparison_protocol.json` fixes the budget, split counts, preprocessing, seed and validation-only checkpoint choice. `run_comparison.py --dry-run` prints the training/checkpoint/three-validation command plan without launching children or creating a run directory. From a fresh exported training snapshot:

```bash
export HIPMRI_SOURCE_COMMIT=bc8a6a4688f50770f71de24a56dff8aad43204c7
python run_comparison.py \
  --architecture UNet2D \
  --source-commit "$HIPMRI_SOURCE_COMMIT" \
  --output-dir "$HOME/hipmri_runs/reproduction_plan_UNet2D" \
  --dry-run
bash -n comparison_train.sbatch
```

For a deliberate new training experiment, `sbatch comparison_train.sbatch` launches the two model tasks. It requests one A100 and one CPU per task, four hours each; the scheduler may serialize them. Each task writes to `comparison_v1_ARRAY_JOB_ID_TASK_ID_ARCHITECTURE` under `$HOME/hipmri_runs`. Successful completion requires Slurm COMPLETED / 0:0, a completed result record and `Full training and validation comparison task: PASS`. Existing output directories are refused; automatic resume is not implemented. No extra training is needed to collect the existing experiment.

The completed experiment is array 704715. Its baseline directory is `comparison_v1_704715_0_UNet2D`; Improved is `comparison_v1_704715_1_ImprovedUNet2D`. Each contains the exact protocol, a data filename/size manifest, provenance, training configuration/history/profile/checkpoints, three validation evaluation directories and a completed comparison record. `training/best.pt` is selected by minimum full-validation loss; it is not chosen from test scores.

## Recreate tables and validation overlays

From a project checkout containing the reporting tools, with both original experiment directories and the frozen runtime still available:

```bash
python compare_results.py \
  --baseline-run "$HOME/hipmri_runs/comparison_v1_704715_0_UNet2D" \
  --improved-run "$HOME/hipmri_runs/comparison_v1_704715_1_ImprovedUNet2D" \
  --output-dir "$HOME/hipmri_runs/validation_report_reproduced"

python failure_analysis.py \
  --baseline-run "$HOME/hipmri_runs/comparison_v1_704715_0_UNet2D" \
  --improved-run "$HOME/hipmri_runs/comparison_v1_704715_1_ImprovedUNet2D" \
  --output-dir "$HOME/hipmri_runs/validation_overlays_reproduced" \
  --device cpu
```

Use new output names. Reporting audits saved completed records; real qualitative analysis additionally reads selected images/masks and reloads the original fixed checkpoints. CPU full-matrix difference flags versus the reference CUDA pass are retained. These commands do not establish a new training replication.

For a single validation example, the prediction CLI accepts a numeric ID for the diagnostic overlay. ID 3 here is simply an example, not a prostate assignment:

```bash
python predict.py \
  --checkpoint "$HOME/hipmri_runs/comparison_v1_704715_1_ImprovedUNet2D/training/best.pt" \
  --image-dir "$HIPMRI_DATA/keras_slices_validate" \
  --mask-dir "$HIPMRI_DATA/keras_slices_seg_validate" \
  --index 120 --label-id 3 --device cpu \
  --output-dir "$HOME/hipmri_runs/example_prediction_reproduced"
```

Outputs are `prediction.png`, a reference-aligned `prediction.nii.gz`, and `prediction.json` with native shape, checkpoint epoch, all-label overlap and selected-ID TP/FP/FN counts. Keep the generated NIfTI and weights outside Git. Normalization comes from the saved checkpoint. The example needs reference masks and is an evaluation/visualization CLI, not a standalone clinical application.

## Completed local test execution

The user requested local execution to avoid the queue. Both existing checkpoint SHA256 values, selected epochs, evaluator/model sources, full 540-slice split, three repeats, interleaved model order and warmup count were preserved. Execution paths and hardware changed explicitly under `hipmri-heldout-numeric-local-v1`. Downloaded before/after accounting records actually show remote job 705243 already completed; this report does not claim a successful cancellation or reviewed A100 test results.

The tested Windows environment used Python 3.12.14, PyTorch 2.14.0+cu130 / CUDA build 13.0, NumPy 2.4.6, NiBabel 5.4.2 and Matplotlib 3.11.2 on an RTX 4070 Laptop GPU, driver 617.42. A fresh isolated local virtual environment was installed and used for the actual six-pass evaluation; the original Linux training installation was not independently rebuilt. Direct installation commands used were:

```powershell
python -m venv .venv-hipmri
& ./.venv-hipmri/Scripts/python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
& ./.venv-hipmri/Scripts/python.exe -m pip install numpy==2.4.6 nibabel==5.4.2 matplotlib==3.11.2
```

For independent reproduction from the original model/evaluator project revision, use the fixed weights and a local data root containing `keras_slices_test` and `keras_slices_seg_test`. This Windows example creates six new evaluation directories and does not retrain or upload:

```powershell
$localData = 'C:/path/to/local_test_inputs'
$baselineWeight = 'C:/path/to/original_UNet2D_best.pt'
$improvedWeight = 'C:/path/to/original_ImprovedUNet2D_best.pt'
$localOutput = 'C:/path/to/new_test_results'
foreach ($trial in 1..3) {
  foreach ($architecture in @('UNet2D', 'ImprovedUNet2D')) {
    $weight = if ($architecture -eq 'UNet2D') { $baselineWeight } else { $improvedWeight }
    & ./.venv-hipmri/Scripts/python.exe -u evaluate.py --checkpoint $weight --data-root $localData --split test --output-dir "$localOutput/$architecture/test_$trial" --device cuda --max-samples 0 --warmup-steps 3
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed; preserve outputs and inspect the error' }
  }
}
```

Run from the exact original source revision bc8a6a4688f50770f71de24a56dff8aad43204c7. Verify source/weight hashes against the recorded protocol before inference. The actual workspace orchestration additionally audited saved training evidence, complete patient/filename coverage, source bytes and all test input content hashes before/after stages, and then recomputed all six reports and compared every CSV cell. Its exact executed wrapper, SHA256 records, protocol and observed dependency freeze are archived under `results/test_local_705243`; that wrapper has recorded workspace paths and is evidence, not a general standalone project CLI.

Local final marker: `Local fixed-model full held-out test comparison: PASS`. All six evaluation records have completed 540/540 coverage; the reviewed summary, five CSVs, audit record and two figures are linked from TEST_RESULTS.md. The complete raw local evidence, MRI/masks and weights are retained outside the Git submission. Local GPU times and A100 validation times have different hardware scopes.

## Verification scope and reproducibility limits

The saved tests exercise numeric metrics, model gradients/shapes, invalid input rejection, paired transformations, checkpoint reloads, native-shape batches and guarded orchestration. They use temporary synthetic inputs and, for orchestration, mocked child processes. Tests were run in the existing Rangpur environment; this guide does not claim a fresh full-suite run after final documentation edits. The completed real validation experiment, five real diagnostic images and completed local six-pass held-out evaluation provide separate empirical evidence. Nine original held-out tests and nine local migration/extraction checks also passed in the local environment before actual test execution.

Committed result copies contain derived summaries, CSVs and PNGs. Four validation CSV copies were normalized to LF with every parsed cell unchanged; `summary.json` retains both reviewed-copy hashes and original export hashes. The new held-out CSV exporter explicitly writes LF. Record checksums distinguish artifact consistency from bitwise reproducibility of training or full-dataset content identity.

Exact recreation of the saved comparison needs the original raw data, checkpoints, source snapshots and runtime records, which are not all distributed in Git. Independent users can reproduce the method with a fresh experiment but are not guaranteed identical GPU weights/timings. One seed and small patient cohorts limit generalization. The anatomical mapping, prostate-specific acceptance criterion, tutor check-off, and whole-device memory bound remain unresolved.
