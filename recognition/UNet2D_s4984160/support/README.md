# Supporting files

Start with the five Python entry points and the [project README](../README.md).

| Folder | Contents |
| --- | --- |
| `submit_jobs/` | Slurm submission scripts |
| `check_code/` | Synthetic code checks |
| `how_to_run/` | Usage, evaluation and alignment instructions |
| `saved_results/training_results/` | Original training CSV/configuration and curves |
| `saved_results/test_results/` | Original test metrics and prediction figures |
| `old_code_and_notes/` | Previous work and historical handoff notes |

Folder names were simplified for navigation. Saved measurements retain their
original contents; historical cluster snapshot names in raw records are not
new local paths. Cluster files were not changed by this reorganization.

`patient_split.json` preserves the recorded patient split. `evaluation_metrics.py`
contains metric helpers; `profile_inference.py` measures both frozen models.
