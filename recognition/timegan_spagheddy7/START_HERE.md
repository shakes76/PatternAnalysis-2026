# TimeGAN submission guide

Start with [README.pdf](README.pdf), the submitted report, or its [Markdown version](README.md). The notebooks contain the saved supporting results; viewing them does not require the dataset or model weights.

## Suggested reading order

| File | What to look for |
| --- | --- |
| [timegan_results.ipynb](timegan_results.ipynb) | Main evidence. The opening assessment separates the final held-out test from validation, followed by the completed volume-conditioning pair. Earlier experiments remain below as the historical record. |
| [ar_results.ipynb](ar_results.ipynb) | VAR baseline, validation selection and comparison protocol. |
| [feasibility_README.md](feasibility_README.md) | Original feasibility proposal, also included in the main report. |
| [commitlog_README.md](commitlog_README.md) | Development record. |
| [data_checks.ipynb](data_checks.ipynb) | Original data-integrity and preprocessing checks; optional rerun requires the raw LOBSTER files. |

## Exact evidence

- [Final held-out test scores](reports/test/final_test_20261106.json): 15 frozen fitted models, three generation seeds each, 1,838 test windows. This is the only final-test result bundle.
- [Completed volume-pair validation study](reports/timegan/volume_recovery_study.json): fixed-final and secondary selected checkpoints, matched controls, diversity diagnostics and figures.
- [Historical experiment ledger](reports/timegan/tuning.json): earlier validation fits and references to their study reports. Its historical flags describe those studies at the time they were saved.
- [CPU and GPU resource profile](reports/resources/resource_profile_20261105.json): the short profiling measurements used in the report, with [method and provenance](evaluation/resource_profile/PROVENANCE.txt). These are not full-run peak-memory measurements.

Historical anchor-only models and the expanded-context volume pair have different information budgets. Their tables are kept separate. Generation-seed variation measures sampling from a fitted model, not retraining uncertainty. None of the frozen test models meets all absolute quality targets.

## Optional verification

From this project directory, the following command checks saved test hashes, means, standard deviations and target flags. It uses only standard Python and needs no raw data, weights or PyTorch:

```text
python evaluation/verify_final_test.py
```

The [test-evaluation note](evaluation/USAGE.txt) explains the frozen plan and archived evaluator. Archived evaluation/profiling scripts retain their original paths and guards; they are provenance, not commands required for marking.

The result notebooks run from their saved report bundles with `requirements-notebook.txt` installed. Leave `RECOMPUTE_GRID = False` in the AR notebook. To run the small synthetic regression suite with `requirements.txt` installed:

```text
python -m unittest discover -s . -p "test_*.py" -v
```

## Source layout

| Files | Responsibility |
| --- | --- |
| `dataset.py`, `protocol.py` | Chronological data preparation, training-only scaling and conditioning contracts. |
| `modules.py`, `train.py`, `predict.py` | Model components, training/recovery and checkpointed generation. |
| `evaluation.py` | Shared validity, distribution, ACF and depth-SSIM metrics. |
| `tune_ar.py`, `tune_timegan.py` | Recorded validation experiment and selection workflows. |
| `ar_report.py`, `timegan_report.py` | Saved result bundles and diagnostic figures. |
| `test_*.py` | Synthetic regression checks for the data, models, metrics and recovery mechanics. |

The root `image*.png` files are used by the submitted report. Figures and JSONs under `reports/` support the notebooks and experiment ledger. Raw CSVs, weights, generated arrays, local run logs and source archives are excluded from Git under `outputs/`; they are not needed to inspect the saved results. The existing AI usage disclosures remain in the report and data-check notebook.
