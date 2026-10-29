# R6a-1: frozen final-test cluster run

Copy the committed project folder to `/home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104`. Keep the frozen protocol and code files unedited. The nine selected `.pt` files and split metadata must be under `/home/Student/s4982104/comp3710-data/adni` in their R4 source locations. Before the first JPEG is read, the entry point checks the protocol, code, split metadata, all nine source sets and actual checkpoint bytes, plus checkpoint model/epoch/seed metadata.

On the login node, submit from the project directory:

```bash
cd /home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104
sbatch run_r6_final.slurm /home/Student/s4982104/comp3710-env/bin/python
```

The batch job uses one A100, two CPUs, up to 12 hours. The script creates `~/comp3710-data/adni/r6_final_JOBID/` and a companion `_evidence.tar.gz`. It runs targeted tests before inference, saves environment details and GPU logs, then runs all nine checkpoints under the frozen settings. A failed job keeps `run/summary.json` with status `failed`, `run/failure.log`, and any earlier raw prediction files. A successful job has status `passed` and 9×2720 slice reads.

Return the full evidence archive under `final_report/adni_manifest_local/r6_final/`. If the archive is too large, return all of these together: job `.out/.err`, `exit_code.txt`, `pip_freeze.txt`, GPU logs, tests, `run/summary.json`, `run/protocol.json`, all nine `run/*_seed*/slice_predictions.csv` and `prediction_summary.json`, `run/evaluation.json`, all metric/referral CSVs, and `run/patient_tables/`. Do not include `.pt` or JPEG files. Preserve the original filenames and folder structure. The patient files stay in the private local evidence folder and out of Git.

Acceptance checks: `exit_code=0`, status `passed`, all input/output hashes, exact 136 test patients (44 AD, 92 NC) and 20 slices each, consistent patient and model IDs, no changed weights, fixed temperatures/thresholds, and independently recomputed metrics. If the test AD recall is below 80%, record it as observed. A technical failure can be corrected with a documented rerun; the test outcomes cannot be used to tune the model.

AI helped prepare this entry point, tests and guide. Real test performance will be recorded only after the cluster evidence is reviewed.

The four R5 validation summary hashes in the protocol document the accepted development decisions. Their original evidence stays in the private local `adni_manifest_local` folder; R6 cluster preflight verifies the committed protocol and its required code, split and checkpoint inputs. Rechecking the four original summaries is part of the local R5e evidence audit, not a new test-time fitting step. A failure midway through one checkpoint preserves completed earlier checkpoints and a failure log; it does not mark the run passed.
