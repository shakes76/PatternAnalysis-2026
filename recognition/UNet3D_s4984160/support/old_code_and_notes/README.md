# Historical experiments and submitted source provenance

This directory preserves distinct earlier work, not extra copies of current
model/data/evaluation code. Use the project-root README for the current model.

- `early_train.py`: earlier BatchNorm training implementation, superseded by
  the final two-stage InstanceNorm experiment.
- `early_smoke_test.py` and `scripts/`: historical pilot entry points and
  original submission scripts. Their old paths/configurations are records,
  not the current execution instructions.
- `experiments/`: three original pilot/full-epoch notes.
- `data_audit_en.md` and `data_audit_zh.md`: original audit notes from the
  earlier Hard-scope plan; that plan was superseded by the Normal model.
- `source_snapshots.json`: hashes and original Git paths for the three
  submitted source snapshots, before duplicate repository copies were removed.

The exact old files remain retrievable from the recorded Git commit with
`git show COMMIT:PATH`. Current model, dataset and training math were not
changed during folder cleanup. The refinement source is now root `train.py`;
the scratch source is `support/train_phase1/scratch.py`. The relocated evaluator tests
use a project-root import path so discovery works from `support/check_code/`.

Slurm `#SBATCH` directives were preserved; current scripts only changed their
Python entry-point paths. No remote files, checkpoints or submitted cluster
jobs were modified by this local/GitHub cleanup.
