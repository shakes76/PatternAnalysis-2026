# Experiment logs (appendix to section 5b of the main README)

Text logs, loss plots and job scripts from the experiments I ran after the tutor feedback. No model files and no data are included. Paths inside the logs are Rangpur paths. The main README ([section 5b](../README.md)) gives the results; this folder is the supporting evidence. Selection was done on the validation split only; test numbers are in the same files but were not used to choose anything.

Each run folder holds `predict.txt` (all metrics, triage tiers, bootstrap, per-source tables), `train_baseline.log` and `train_siamese.log` (one line per epoch), and loss plots. Some folders hold only some of these.

## Which folder is which trial

| Trial in section 5b | Folder |
|---|---|
| 1. Control on Rangpur | `ctrl_code/` (training logs and loss plots), `exp_code/outputs/ctrl/` (`predict.txt`, `diagnose.txt`) |
| 2. Strong augmentation | `exp_code/outputs/aug/` |
| 3. Learning rate 0.0003 | `exp_code/outputs/lr3e4/` |
| 4. 60 epochs | `exp_code/outputs/ep60/` (also has `diagnose.txt`) |
| 5. 100 epochs | `exp_code/outputs/ep100/` |
| 6. 100 epochs, lr 0.0003 | `exp_code/outputs/ep100lr3e4/` |
| 7. Gray-world colour correction | `exp2_code/outputs/pre/` |
| 8. Diagnostic | `diagnose.txt` in `exp_code/outputs/ctrl/` and `exp_code/outputs/ep60/` (and the same checkpoints re-scored in `exp6_code/outputs/`) |
| 9. Number of nearest references k | `exp_code/outputs/ep60_k1_n200/`, `ep60_k3_n200/`, `ep60_k10_n200/`, `ep60_k20_n200/`, `ep60_k5_n400/`, `ep60_k10_n400/` |
| 10. Prototype score | `exp6_code/outputs/ctrl_prototype_n0/`, `ctrl_prototype_n200/`, `ep60_prototype_n0/`, `ep60_prototype_n200/` (and the 5-nearest, every-training-image versions `ctrl_knn_n0/`, `ep60_knn_n0/`); 100 epochs: `exp7_code/outputs/ep100_proto/`, `ep100lr3e4_proto/` |
| 11. Margin 2.0 | `exp6_code/outputs/m20/` |
| 12. Same-source pairs | `exp6_code/outputs/same/` |
| 13. Sweep of 34 scoring settings | `exp7_code/sweeps/` (`ctrl.txt`, `ep60.txt`, `ep100.txt`, `ep100lr3e4.txt`, `m20.txt`, `same.txt`) |
| 14. Seeds 7 and 123, 30 epochs | `seeds_lab/` (lab T4 training logs and plots); re-scored on Rangpur in `exp7_code/outputs/lab_s7_*` and `lab_s123_*` |
| 15. Seeds 7 and 123, 100 epochs | `exp7_code/outputs/ep100_s7/`, `ep100_s123/` (Siamese and baseline at learning rate 0.001); baselines at learning rate 0.0003: `bl_lr3e4_s7/`, `bl_lr3e4_s123/`, `bl_lr3e4_s42/` |
| 16. Paired bootstrap | Last section of each `predict.txt` |
| 17. Triage tiers, 3 seeds | "Triage tiers on test" block of the `predict.txt` or `predict_prototype.txt` in the folders of trial 15 |

## Final run with per-epoch metrics
`final_run_rangpur/` holds the logs and plots from a 30-epoch re-run of the control (seed 42) on Rangpur, made after I added per-epoch metrics to `train.py` (job script `exp13.sh` is not included because it was typed on Rangpur; it only runs `train.py` for each model with `--epochs 30 --num_workers 3`). `train_*.log` have one line per epoch with the validation metric; `baseline_metrics.png` and `siamese_metrics.png` plot it next to the loss plots. Its losses match the earlier control in `ctrl_code/` exactly.

## Job scripts
`exp1.sh` to `exp11.sh` and `ctrl.sh` are the Slurm scripts I submitted on Rangpur, in order. The code that made the runs is in the parent folder (`train.py`, `predict.py`, `diagnose.py`, `sweep_k.py`); the options are listed in section 3 of the main README.

## Notes
- The folder names `exp_code`, `exp2_code`, `exp6_code` and `exp7_code` are the names of the code copies I used on Rangpur, kept as they were so the paths in the logs still match.
- `predict_knn.txt` is the original 5-nearest score and `predict_prototype.txt` the prototype score for the same trained models.
