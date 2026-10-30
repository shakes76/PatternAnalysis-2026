# Supporting files

Start with the five Python entry points and the [project README](../README.md).

| Folder | Contents |
| --- | --- |
| `train_phase1/` | Phase 1 training code |
| `submit_jobs/` | Slurm submission scripts |
| `check_code/` | Synthetic code checks |
| `how_to_run/` | Usage, evaluation and alignment instructions |
| `saved_results/training_results/` | Original training CSV/configuration and curves |
| `saved_results/test_results/` | Original test metrics and prediction figures |
| `old_code_and_notes/` | Previous work and historical handoff notes |
| `saved_results/resource_results/` | Paired resource profiles and direct prediction example |
| `compare_results/` | Verified model comparison JSON/CSV |

Folder names were simplified for navigation. Saved measurements retain their
original contents; historical cluster snapshot names in raw records are not
new local paths. Cluster files were not changed by this reorganization.

`python -m support.prepare_comparison` verifies archived counts and writes JSON/CSV
under `compare_results/`. It does not edit the manually reviewed report.
