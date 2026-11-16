# Reproducing the HipMRI workflow

## Code, environment and data

On Rangpur, choose an unused clone directory:

```bash
git clone --branch topic-recognition https://github.com/MoyuOlder/PatternAnalysis-2026.git hipmri_reproduction
cd hipmri_reproduction/recognition/HipMRI_ImprovedUNet_s4983404
source "$HOME/venv/bin/activate"
export OMP_NUM_THREADS=2
```

The existing environment uses Python 3.11.13, PyTorch 2.14.0+cu130,
NumPy 2.4.6, nibabel 5.4.2, matplotlib 3.11.1 and SciPy 1.17.1.
SciPy is needed by `analyse_failures.py` and `tests/test_boundaries.py`.
No pytest or nilearn is required. These are measured environment versions,
not a portable installer: other machines need a compatible PyTorch build.
The same pins are collected in `requirements.txt` for `pip install -r`; that
file records what the frozen results used, it does not remove the build caveat.
On a CPU-only machine, explicitly select `--device cpu`.

The shared data must be available at
`/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`, containing the six
`keras_slices_{train,validate,test}` and `keras_slices_seg_{train,validate,test}`
folders. Python entry points accept `--root` for a different location; the
provided Slurm scripts use the course path. Do not move or modify shared data.

## CPU checks and saved-weight inference

```bash
python -m unittest discover -s tests -v
python predict.py --device cpu --split validate --count 3 \
  --checkpoint "$HOME/hipmri_report_20261020_v4/runs/improved_full_679896/best.pt" \
  --output runs/improved_cpu_preview
```

The test suite includes model, dataset, boundary and prediction checks. The prediction command writes
three `pred_*.nii.gz` masks and three four-panel PNGs. Masks preserve the input
grid/affine and contain uint8 label IDs 0..5. Prediction loads the architecture
and width from the checkpoint; it does not train a new model. `--split test`
can be used for a test-case demonstration, without selecting or tuning the model.

The checkpoint above belongs to the student's Rangpur account. Git contains no
weights, and other accounts are not assumed to have permission to read this path.
Use an authorized copy via `--checkpoint`, or retrain as below. The corresponding
basic checkpoint is `runs/basic_full_679897/best.pt` under the same retained root.
CPU preview time is not the GPU latency reported in the experiment table.

For new 2D slices without segmentation labels, use `--input` with a single
`.nii`/`.nii.gz` file or a directory of slices:

```bash
python predict.py --device cpu --checkpoint /path/to/best.pt \
  --input /path/to/mri_slices --count 5 --output runs/new_mri_preview
```

This mode uses the same per-image normalization and saves masks plus two-panel
input/prediction figures. True 3D volumes are rejected; singleton axes such as
`(H, W, 1)` are preserved in the saved mask. `--count` selects up to that many
evenly spaced files from the sorted directory. Optional `--labels /path/to/masks`
restores four-panel comparisons: mask names match image names, except a leading
`case_` becomes `seg_`. Without `--input`, the existing course-split mode is unchanged.

For a small training/save/load check rather than an accuracy experiment:

```bash
python train.py --device cpu --model basic --width 4 --epochs 1 \
  --batch-size 2 --workers 0 --limit 4 --val-limit 4 --run runs/basic_cpu_smoke
python predict.py --device cpu --checkpoint runs/basic_cpu_smoke/best.pt \
  --split validate --count 3 --output runs/basic_smoke_preview
```

Repeat with `--model improved` and distinct run/output directory names if needed.
The small run is diagnostic; its metrics must not replace the frozen results.
All output directories must be new. Choose another name on repeat execution.

## Full training and evaluation

Check current partition/account availability before scheduling GPU work. From
the project directory, submit each model using Slurm (not direct GPU Python
commands on the login node):

```bash
sbatch submit_train.sh basic
sbatch submit_train.sh improved
```

Record the two returned job IDs. Each job uses the documented 20-epoch recipe
and writes `runs/<model>_full_<jobid>/` with `best.pt`, `history.json`,
`learning_curves.png`, `config.json`, `environment.json`, `resources.json`,
and `train.log`. Slurm writes `full_<jobid>.out`. The safe time limit can stop
training before epoch 20; inspect the history before claiming all epochs ran.

After successful training and validation-only selection, evaluate a chosen
checkpoint with its actual path (replace `JOBID` below):

```bash
sbatch submit_evaluate.sh basic "runs/basic_full_JOBID/best.pt"
sbatch submit_evaluate.sh improved "runs/improved_full_JOBID/best.pt"
```

The evaluation script accepts the model and checkpoint instead of historical job
IDs. It writes `runs/<model>_test_<evaluation-jobid>/evaluation.json`, resources
and configuration, plus `runs/<model>_validation_<evaluation-jobid>/` for boundary
analysis. The checkpoint itself now supplies `--model` and `--width`; pass either
flag only to confirm it, and a conflicting value is refused before any output exists.

The original test set has already been evaluated. Reproduction must not turn
test results into new tuning decisions; keep the original result tables intact.
Exact retraining equality is not guaranteed across hardware/software versions.
The reproducibility checks use validation inputs and do not rerun frozen test scoring.

## Verification on 24 October 2026

Tracked project files were exported into the fresh Rangpur directory
`~/hipmri_reproduce_20261024_final`, without local caches, run outputs or `_work`
files. Checks used the existing environment above and explicit CPU execution.

| Check | Observed result |
|---|---|
| Unit tests | All six passed |
| Basic and Improved short training | Each completed one epoch with four training and four validation slices; checkpoint and learning-curve outputs saved |
| Original saved weights | Both formal checkpoints loaded and predicted three validation slices each |
| Newly saved weights | Both short-run checkpoints loaded and predicted the same three validation slices each |
| Output verification | All 12 masks matched a fresh checkpoint-loaded forward pass exactly; native shape/affine, uint8 storage and IDs 0..5 verified; all 12 PNGs readable |
| Slurm script checks | Shell syntax passed; evaluation wrapper rejected missing arguments with exit code 2 |

The short-run label-5 Dice was zero for both models; these runs only verify the
training/save/load path and are not evidence of learning. The original 20-epoch
results remain the authoritative accuracy measurements. No new GPU job or frozen
test evaluation was run for this verification, so the new evaluation wrapper's
GPU execution was not remeasured. The login-node CUDA-driver warning emitted
during CPU gradient tests did not prevent the checks from passing.

### Unlabeled prediction pass

The prediction changes were checked the same day in the separate fresh directory
`~/hipmri_reproduce_20261024_unlabeled_v2`, using the same environment and
explicit CPU execution:

| Check | Observed result |
|---|---|
| Unit tests | All nine passed, including the three prediction checks added in this pass |
| Synthetic unlabeled input | `--input` on one saved slice without `--labels` produced a `pred_*.nii.gz` mask with the source shape/affine, uint8 storage and IDs 0..5, plus a two-panel PNG |
| Labeled comparison | Adding `--labels` restored the four-panel figure and wrote the same mask as the unlabeled run |
| Real checkpoint | `improved_full_679896/best.pt` predicted one validation slice through `--input` with no labels; the saved mask matched the earlier labeled prediction for that file exactly |
| Rejected input | A 3D volume, non-finite pixels, a non-NIfTI file, a missing `--labels` directory, non-positive `--count` and `--labels` without `--input` all raised errors instead of writing masks |

The six-test row in the first table describes the earlier checkout; nine was the
count after the prediction pass, and fourteen after the start-up-guard pass below.

### Start-up guards

The same fresh directory `~/hipmri_reproduce_20261024_unlabeled_v2` was reused, so
the guards were checked against the same environment and explicit CPU execution.
Checks now run in this order: arguments, checkpoint architecture, patient splits,
model construction, and only then the new run directory.

| Check | Observed result |
|---|---|
| Unit tests | All fourteen passed, adding four driver guards and one checkpoint-loader check |
| Rejected settings | Non-positive epochs, batch size, width, limit and validation limit, negative workers, zero learning rate and zero time budget each raised before any output |
| Architecture source | `improved_full_679896/best.pt` resolved to `improved 16` and `basic_full_679897/best.pt` to `basic 16` with no `--model`/`--width` flags; both weight sets loaded into the rebuilt models |
| Conflicting flags | `--model basic` against the improved checkpoint raised `--model=basic conflicts with checkpoint improved`; matching flags were still accepted |
| No half-built run | A missing dataset root raised `FileNotFoundError` and a conflicting `--width` raised `ValueError`, both leaving the requested run directory absent |
| Evaluation end to end | A synthetic one-slice root plus a width-4 checkpoint wrote `config.json` (`improved`, 4), `evaluation.json` (one sample) and `resources.json` without arch flags |
| Device guard | `--device cuda` on the login node raised `CUDA requested but unavailable; pass --device cpu` in `predict.py` and `analyse_failures.py`, and no output directory was created |
| Prediction regression | The unlabeled and labeled prediction paths still wrote a matching mask and panel for one validation slice after the shared loader change |

The module MD5 sums of `train.py`, `predict.py`, `modules.py`, `analyse_failures.py`
and `tests/test_models.py` on the remote copy matched the local files exactly, and
`tests/test_train.py` had already matched at `a1da948611dde556b579f03ccc8ffd47`.

`tests/test_metrics.py` was added afterwards to pin the metric conventions, which
brings the suite to twenty-one tests. All twenty-one passed on a fresh clone of
`topic-recognition` at `2a62df5` once that one file was copied in, with the local
and remote copies of `tests/test_metrics.py` matching at
`1cde89f4dafd7326d0254c6e0c42a500`.

Detailed local evidence is retained in `_work/reproduce_final_1.txt`,
`_work/test_unlabeled_1.txt`, `_work/item2_check.txt` and `_work/item2_final.txt`
outside the submission repository. The isolated remote run directories retain
their outputs.
