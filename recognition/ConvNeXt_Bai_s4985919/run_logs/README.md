# run_logs: original evidence for the numbers in the report

Everything in this folder was written automatically by the project scripts during my own runs on Rangpur (and, where stated, the EAIT GPU Lab) and is committed unchanged. It contains settings and aggregate metrics only: no MRI images, no model weights, and no scan or subject IDs.

| Folder / file | What it is | Produced by | Used in README section |
|---|---|---|---|
| `convnext/<run>/`, `cnn/<run>/` | One folder per training run (13 runs). `config.json` = all training settings; `history.json` = training loss and validation metrics after every epoch; `best_meta.json` = which epoch was kept and why, plus fingerprints (SHA-256 hashes) of the train/validation/test file lists, so a checkpoint can only be evaluated on exactly the split it was trained for. Early runs have no `best_meta.json` (written before that feature existed). | `train.py` | 9 (experiment history) |
| `summary.csv` | One line per run: settings, best validation accuracy and the epoch that was saved. | `collect_runs.py` | 9 |
| `evaluation/analysis_convnext_ensemble.json`, `evaluation/analysis_cnn.json` | Every validation and test number for the final models on the provided split: accuracy, AUROC, confusion matrix, calibration (ECE, Brier, reliability bins), reject rule, subgroup accuracy, failure cases, parameter count, latency and GPU memory. | `analyze.py` | 2, 5, 6, 7, 8 |
| `evaluation/<project>__<output folder>__<split>_metrics.json` | Metric files of earlier evaluations (slice, scan and patient level), e.g. the strict-split test run of run 3. | `predict.py` | 2, 9 |
| `slurm/slurm_training_jobs.txt` | Slurm accounting (`sacct`) for the training jobs: job ID, state and elapsed time. | Rangpur `sacct` | 8 (training time) |
| `slurm/slurm_job_to_run.txt` | First line of each training job's log: which run (input mode, seed, run name) each job ID trained. | `run_recipe.slurm` | 8 |
| `slurm/slurm_job_history.txt` | Full list of my Rangpur jobs since 1 Oct 2026, including interactive sessions and failed or cancelled attempts. | Rangpur `sacct` | – (provenance) |
| `diagnostics/development/` | Development audit of the fixed run-3 model on training and validation subjects only (no test data): `report.md` = readable summary; `summary.json` = metrics with patient-level bootstrap intervals; `check_summary.json` = data checks (subject overlap, slice numbering). | `diagnose_development.py` | 9 |
| `diagnostics/brightness/` | Brightness-sensitivity experiment on 280 validation slices (70 subjects): `report.md` = readable summary; `condition_metrics.csv` = metrics per brightness factor and class (empty cells: a metric is undefined when a class is absent); `summary.json` = the same with bootstrap intervals. | `test_brightness_sensitivity.py` | 9 |
| `adni_v2/PROTOCOL.md` | Evaluation protocol for the corrected subject-disjoint split, committed before any test evaluation on that split. | written by me | 10 |
| `adni_v2/runs/<run>/` | `config.json`, `history.json` and `best_meta.json` of the six corrected-split runs (ConvNeXt and MatchedCNN, seeds 42/43/44). | `train.py` | 10 |
| `adni_v2/evaluation/*_metrics.json` | Validation and test metrics (slice, scan and patient level) of the two 3-seed ensembles. | `predict.py` | 10 |
| `adni_v2/evaluation/analysis_*_ensemble.json` | Calibration, reject rule, subgroup accuracy (scan year, sequence), failure cases and resource profile of the two ensembles. | `analyze.py` | 10 |
| `adni_v2/evaluation/eval_v2_job.log` | Console output of the single evaluation job (Rangpur job 686840), including the fingerprint checks. | `predict.py`, `analyze.py` | 10 |
| `adni_v2/slurm_v2_training_jobs.txt` | Slurm accounting and job-to-run mapping for the two corrected-split runs trained on Rangpur. | Rangpur `sacct` | 10 |
| `adni_v2/gpulab_environment.txt`, `adni_v2/gpulab_train_*.log` | Machine, GPU and software versions of the EAIT GPU Lab instance, and per-epoch training logs of the four runs trained there (per-batch progress lines removed). | GPU Lab shell, `train.py` | 10 |