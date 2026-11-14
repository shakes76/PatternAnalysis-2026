# Usage Guide

<div align="center">

[README](../README.md) • [Installation](./INSTALLATION.md)

</div>

---

This page covers how to rebuild everything in the report: the dataset split, every trained model, the validation scores, the frozen test evaluation, the latency measurements, the failure-case panels and all figures. All commands are run from the project folder (`yolo_lesion_47041585/`). Install the project and fetch the data first (see [Installation](./INSTALLATION.md), it also covers running project files without `uv`).

## Pipeline overview

| # | Step | Command | Main outputs | Needed by |
|--:|:--|:--|:--|:--|
| 1 | [Audit and split](#1-run-the-data-audit) | `uv run python -m yolo_lesion_47041585.audit` | `artifacts/isic2018/` | everything |
| 2 | [Train](#2-train-the-models) | `uv run train --config configs/<config>.yaml` | `runs/<condition>-s<seed>/` | 3, 4, 6, 7 |
| 3 | [Score on validation](#3-score-the-runs-on-validation) | `uv run python scripts/score.py` | `assets/scores.md`, `artifacts/scores.json` and CSVs | 4, 8 |
| 4 | [Freeze and test](#4-freeze-the-thresholds-and-evaluate-on-test) | `uv run python scripts/final_test.py` | `assets/selection.json`, `assets/test-results.*` |6, 8 |
| 5 | [Latency](#5-measure-latency) | `uv run python scripts/profile_latency.py` | `artifacts/latency*.json` | report only |
| 6 | [Failure cases](#6-draw-the-failure-cases) | `uv run python scripts/autopsy.py` | `artifacts/autopsy/` | report only |
| 7 | [Figures](#7-regenerate-the-figures) | `uv run python scripts/visualise/<name>.py` | `assets/*.png` | report only |

Run the steps in this order. Steps 1–2 and the test run in step 4 need a GPU. The rest can run on a CPU if the detection caches exist (see [Notes](#notes-and-troubleshooting)).

> [!IMPORTANT]
> `artifacts/`, `runs/` and `weights/` are not committed (see `.gitignore`). A fresh clone has none of them, so you must rerun steps 1–2 to get any result beyond the committed `assets/`.

## 1. Run the data audit

`audit.py` acts as an interface for `dataset.py`. It scours the dataset for invalid records. This includes missing or invalid ISIC identifiers, patient identifiers, diagnosis labels, images and masks. The remaining data is used to build independence groups for maintaining patient-independence between sets before splitting the dataset into training, validation and test sets (**70/15/15**). It is also responsible for deriving bounding boxes from the training masks and exporting the dataset in a format usable for YOLO training.

```bash
uv run python -m yolo_lesion_47041585.audit [--source dataset] [--output artifacts/isic2018]
```

By default, data is read from `dataset/` and written to `artifacts/isic2018`. The output folder must be new or empty.

| File | Contents |
|:-----|:---------|
| `yolo/` | `dataset.yaml`, `images/` and `labels/` for `train`, `val` and `test` |
| `split_manifest.csv` | Split, patient, class and box for every retained image |
| `cohort.csv`, `exclusions.csv` | Retained images, and excluded images with reasons |
| `summary.json` | Counts and hashes of the raw metadata and split |

**Check your result.** With the default seed (`47041585`) and a 70/15/15 split you should get 1,259 retained records split 881 / 189 / 189 (train / val / test), with about 19% malignant in each split and no patient in more than one split. `summary.json` records the SHA-256 of the raw metadata and the split manifest so you can compare against `assets/selection.json` (`dataset.raw_metadata_sha256`, `dataset.split_manifest_sha256`).

> [!IMPORTANT]
> Images in `yolo/` are symlinks to absolute paths. **Run this on the machine you will train on.** Do not copy `artifacts/` between machines.

<sub><i>In hindsight, `audit.py` should probably be a script rather than a module. But that's okay.</i></sub>

## 2. Train the models

Each run takes one condition config from `configs/`. `configs/base.yaml` is always applied first, so a condition file only lists what it changes.

```bash
uv run train --config configs/a0.yaml [--seed N] [--device D] [--pilot]

# on Rangpur, from the project folder
sbatch scripts/train.slurm configs/a0.yaml [--seed N] [--pilot]
```

The slurm script takes the config path as its **first positional argument** and passes any other flags through.

| Flag | Description |
|:-----|:------------|
| `--config PATH` | Condition config (required for `uv run train` only) |
| `--seed N` | Training seed. Default `47041585`. The data split is unaffected |
| `--pilot` | 3-epoch smoke test. Saved as `...-pilot/` in `runs/` |
| `--device D` | Ultralytics device (`0`, `cpu`, `mps`). Default `0` |

Official weights (`weights/yolo11n.pt`, `weights/yolo11m.pt`) are downloaded to `weights/` on first use.

### The conditions in the report

| Condition | Config |
|:--|:--|
| `B0` | `configs/b0.yaml` |
| `A0` | `configs/a0.yaml` |
| `A0-wd` | `configs/a0_wd.yaml` |
| `O35`, `O50` | `configs/o35.yaml`, `configs/o50.yaml` |
| `A0-m`, `A0-m-wd` | `configs/a0_yolo11m.yaml`, `configs/a0_yolo11m_wd.yaml` |
| `A0-cls1`, `A0-cls2` | `configs/a0_cls1.yaml`, `configs/a0_cls2.yaml` |
| `A0-ep200` | `configs/a0_ep200.yaml` |
| `A0-rot` | `configs/a0_rot.yaml` |

### Reproduce every run

All 11 conditions were run with three seeds (`47041585`, `47041586`, `47041587`), 33 runs in total:

```bash
for cfg in b0 a0 a0_wd o35 o50 a0_yolo11m a0_yolo11m_wd a0_cls1 a0_cls2 a0_ep200 a0_rot; do
  for seed in 47041585 47041586 47041587; do
    sbatch scripts/train.slurm configs/$cfg.yaml --seed $seed
  done
done
```

A YOLO11n run takes about 11 minutes on an A100, `A0-ep200` about 19 and the YOLO11m runs about 24. Each should fit the 30-minute limit in `train.slurm` (the YOLO11m runs only just). All 33 runs took about 7.5 GPU hours. Each run writes `runs/<condition>-s<seed>/` containing:

| File | Contents |
|:--|:--|
| `weights/best.pt`, `last.pt` | Checkpoints. `best.pt` is used everywhere below |
| `results.csv` | Per-epoch metrics (used for the training curves) |
| `result.json` | Elapsed time, per-epoch time, peak VRAM, best validation mAP |
| `config.resolved.yaml`, `args.yaml`, `metadata.json`, `sampling.csv` | Exact settings and provenance |

> [!NOTE]
> `configs/mps.yaml` layers on a condition to run on an Apple GPU (`--config configs/a0.yaml --config configs/mps.yaml`). It uses a smaller batch size and is **not** comparable with the Rangpur runs. It was only used for local pilots.

## 3. Score the runs on validation

`score.py` scores every run's best checkpoint on the validation split and writes the tables and plot inputs. It applies no rule and chooses nothing.

```bash
# full sweep: every condition and seed found in runs/
uv run python scripts/score.py --device 0 --tau-m 0.07 --primary-seed 47041585

# or on Rangpur
sbatch scripts/score.slurm --tau-m 0.07 --primary-seed 47041585
```

| Output | Contents |
|:--|:--|
| `assets/scores.md`, `artifacts/scores.md` | Tables: by condition, difference from `A0`, every run, patient-level 95% intervals, trade-off, chosen point, clear guard |
| `artifacts/scores.json` | The same results as data (read by the figure scripts) |
| `artifacts/tradeoff.csv`, `artifacts/results-by-condition.csv` | Plot inputs |
| `artifacts/detection-cache/val/` | Cached detections, so reruns do not repeat inference |

`--tau-m 0.07` adds the "chosen operating point" table for `A0` and sets the clear-guard table at that threshold. Leave out `--tau-m` for the tables without it. Use `--conditions` and `--seeds` to score a subset.

**The `A0` vs `A0-cls2` trade-off used in the report** is written to a separate folder so it does not overwrite the main outputs:

```bash
uv run python scripts/score.py --device 0 \
  --conditions A0 A0-cls2 --baseline A0 --candidate A0-cls2 --primary-seed 47041585 \
  --assets artifacts/cls2 --tables-dir artifacts/cls2 --output artifacts/cls2/scores.json
```

Check the scoring code with `uv run python scripts/score.py --self-check`.

## 4. Freeze the thresholds and evaluate on test

This step is guarded so the test split is only evaluated once. First record the thresholds, using validation only:

```bash
uv run python scripts/final_test.py freeze --candidate A0 --baseline B0 \
  --seed 47041585 --tau-m 0.07 --final

# or on Rangpur
sbatch scripts/final_test.slurm freeze --candidate A0 --baseline B0 --seed 47041585 --tau-m 0.07 --final
```

This writes `assets/selection.json`: the candidate's `tau_m` (0.07), the baseline's `tau_m` (0.01, found at the same validation benign burden), the checkpoint SHA-256 hashes and the dataset hashes. Without `--final` the record is saved but the test run refuses to use it.

Run a dry run on validation (any number of times, writes to `artifacts/dry-run/`), then the single test evaluation:

```bash
uv run python scripts/final_test.py test --split val     # dry run
uv run python scripts/final_test.py test                 # the one test evaluation

# or: sbatch scripts/final_test.slurm test --split val  /  sbatch scripts/final_test.slurm test
```

| Output | Contents |
|:--|:--|
| `assets/test-results.md`, `assets/test-results.json` | Operating-point results with patient-level 95% intervals, routing counts and per-class AP |
| `artifacts/routes/<split>-<condition>.csv` | Per-image routing decisions (used for the routing figure) |

> [!WARNING]
> `test` refuses to run on the test split unless the record is final, the checkpoints still match their hashes, and no test result exists yet. `--again` overwrites an existing result. **Do not use it** if you want the test split evaluated only once. To reproduce the report without re-running the test, use the committed `assets/test-results.*` and `assets/selection.json`.

## 5. Measure latency

```bash
uv run python scripts/profile_latency.py --device 0 \
  --checkpoint runs/A0-s47041585/weights/best.pt --tau-m 0.07 --output artifacts/latency.json

uv run python scripts/profile_latency.py --device 0 \
  --checkpoint runs/B0-s47041585/weights/best.pt --tau-m 0.01 --output artifacts/latency-b0.json

# or on Rangpur, passing the same options to the slurm script
sbatch scripts/latency.slurm --checkpoint runs/A0-s47041585/weights/best.pt --tau-m 0.07 --output artifacts/latency.json
```

It times 100 validation images after a 20-image warmup at batch size 1 from a decoded image in memory to the final routing decision. It records the hardware and software versions. Use the same GPU for both models to make the results comparable. The reported numbers were measured with `--tau-m 0.19` (the script default). The threshold does not affect timing.

## 6. Draw the failure cases

```bash
uv run python scripts/autopsy.py                     # validation split (default)
uv run python scripts/autopsy.py --split test        # test split, needs a final record
```

For each failure category (baseline misses candidate finds, candidate misses baseline finds, benign false alert, poor localisation, missed malignancy) it picks the clearest case at the frozen thresholds and writes a side-by-side panel to `artifacts/autopsy/`. The script only suggests cases. You will need to choose which to keep and copy them to `assets/autopsy/`.

## 7. Regenerate the figures

Each script reads the outputs above and writes a PNG to `assets/`. These are the commands used for the figures in the README:

| README figure | Command | Needs |
|:--|:--|:--|
| Figure 3 (box areas) | `uv run python scripts/visualise/box_areas.py` | step 1 |
| Figure 4, 12–15 (failure cases) | step 6, copy the chosen panels into `assets/autopsy/` | step 6 |
| Figure 5 (results by condition) | `uv run python scripts/visualise/conditions.py` | step 3 |
| Figure 6 (training curves) | `uv run python scripts/visualise/training_curves.py --conditions B0 A0 A0-ep200` | step 2 |
| Figure 7 (validation intervals) | `uv run python scripts/visualise/intervals.py --source validation` | step 3 |
| Figure 8 (`A0` vs `A0-cls2` trade-off) | `uv run python scripts/visualise/tradeoff.py --csv artifacts/cls2/tradeoff.csv --baseline A0 --candidate A0-cls2 --baseline-colour tab:blue --candidate-colour tab:purple --mark-limits --output assets/tradeoff-cls2.png` | step 3 (cls2 command) |
| Figure 9 (test intervals) | `uv run python scripts/visualise/intervals.py --source test` | step 4 |
| Figure 10 (test routing) | `uv run python scripts/visualise/routes.py --split test` | step 4 |
| Figure 11 (trade-off with test points) | `uv run python scripts/visualise/tradeoff.py --tau-m 0.07 --test-results assets/test-results.json` | steps 3 and 4 |
| Score distributions (not in the README body) | `uv run python scripts/visualise/score_distributions.py` and `... --split test` | steps 2 and 4 |

All figures share one plain style set in `helpers/visualisation.py` (`use_report_style`). Each script has a settings block at the top for colours, sizes and labels and `--help` to list its options.

## Run inference on your own images

```bash
uv run predict --checkpoint runs/A0-s47041585/weights/best.pt --images path/to/images --output path/to/new_folder
```

The output folder must be new or empty. It writes `detections.csv` (every detection above the confidence floor, with columns `image_id, x_center, y_center, width, height, class_id, class_name, confidence`) and an `overlays/` folder of images showing detections at or above `--overlay-conf` (default 0.25). It needs no masks or labels. `predict` does not apply the routing rule, so it gives boxes and confidences only, not clear, monitor or urgent decisions.

## Notes and troubleshooting

- **Determinism**: Runs set `deterministic: true` but results will not match bit for bit on different hardware, drivers or library versions. Expect small differences from the reported numbers. The frozen checkpoints are identified by SHA-256 in `assets/selection.json`.
- **Data version**: Rangpur provides `_x2` training images and masks that are not on the public ISIC archive, so a run from the public files will differ slightly (see [Installation](./INSTALLATION.md)).
- **Slurm**: The scripts request one GPU on the `comp3710` partition for 30 minutes, load `cuda/11.1` and write `*slurm-*.out` logs (ignored by git). Run `sbatch` from the project folder.
- **CPU and cached detections**: Scoring and the figure scripts reuse `artifacts/detection-cache/`, so they only need a GPU the first time a checkpoint is scored. Pass `--device cpu` to run without one (much slower).
