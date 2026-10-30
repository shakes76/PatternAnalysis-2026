# Supporting scripts

Run these modules **from the FLAN-T5 project root**, with its environment active:

```bash
python -m scripts.inspection.inspect_data
python -m scripts.inspection.inspect_tokens
python -m scripts.inspection.inspect_lora
python -m scripts.inspection.inspect_predictions
python -m scripts.analysis.select_failure_candidates --split final
python -m scripts.setup.prepare_cache
python -m scripts.reporting.plot_results
```

`inspection/` holds data/token/prediction exploration. `diagnostics/` preserves
the original smoke tests and two-example training exercise. `analysis/` holds
candidate selection and inference profiling. `reporting/` renders the results
figure. `setup/` prepares the Hugging Face cache.

Using `python -m` keeps root imports (`dataset`, `modules`, `config`) available.
Do not launch relocated files as `python scripts/inspection/inspect_data.py`.
Scripts that read results or write checkpoints use the project root, not their
new script directory. Existing `results/`, `checkpoints/` and `.venv/` have not
been moved or deleted.

On Rangpur, use a CPU allocation for setup/data processing and a GPU allocation
for model execution. Job scripts live in `jobs/`; submit from the project root:

```bash
sbatch jobs/train/train_lora_full_data.sh
sbatch jobs/evaluate/evaluate_lora_final.sh
sbatch jobs/profile/profile_inference.sh
```

The reorganised training jobs use a one-process `torchrun` launcher. This is
single-GPU execution, not distributed multi-GPU training. Previously reported
experiments used the earlier `python -u train.py` launcher. No model was retrained
during the reorganisation.

Upload the reorganised folders before using these new paths remotely. Existing
remote files are not changed by local moves; do not mix an old and new layout
without updating the supporting scripts.
