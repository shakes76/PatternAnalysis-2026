# Fixed final retrospective evaluation

This closes the existing study. It does not train, select a checkpoint, sweep thresholds or create a new blind test. The frozen closeout plan selects geometry n, seed 3710, confidence strictly above 0.40. The final result is in `recognition/YOLO_s4905914/final_candidate_reassessment_20261022.json`.

The first A100 allocation request (686313) was cancelled while pending: elapsed 00:00:00, no assigned node. The scheduler estimated a start on the following morning. `device_decision.json` records the decision to use local Apple M2 CPU/FP32 before seeing any inference result. The one completed evaluation is `outputs/final_reassessment_cpu_20261022`; `evaluate_a100.sh` documents the cancelled route, not a second completed evaluation.

Restore the original checkpoint, selected config and lock, decision records, all split manifests, original photo audit and exact reassessment photographs before running the commands below. Source data and model weights are excluded from Git. The evaluation verifies photo and selection hashes and refuses to overwrite its output directory. Do not regenerate missing identity IDs.

From the research workspace root, with the pinned requirements installed:

```bash
python recognition/YOLO_s4905914/evaluate_locked_candidate.py \
  --root . --image-root /path/to/original_photos \
  --output outputs/new_fixed_final/evaluation --device cpu --verify-only

OMP_NUM_THREADS=4 MPLBACKEND=Agg YOLO_OFFLINE=true \
python recognition/YOLO_s4905914/evaluate_locked_candidate.py \
  --root . --image-root /path/to/original_photos \
  --output outputs/new_fixed_final/evaluation --device cpu
```

The completed result is already cached; no repeated inference is needed for the report. To verify that completed run, restore its `COMPLETED.txt`, bundle manifest, device decision, hardware/environment record and command timing alongside the evaluation cache, then run:

```bash
python recognition/YOLO_s4905914/audit_final_reassessment.py \
  --run outputs/final_reassessment_cpu_20261022
python recognition/YOLO_s4905914/build_final_report.py
python recognition/YOLO_s4905914/export_readme_pdf.py \
  --output output/pdf/COMP3710_Project3_Report.pdf
```

The final audit independently replays fixed-threshold detection and classification/IoU calculations. AP remains the pinned library result. The CPU command timing includes file checks, AP validation and one-image prediction calls; it is not A100 inference throughput or full application latency.

`prepare_bundle.py` creates an isolated package from the locked inputs and refuses to replace an existing package. The bundle is hashed in `bundle_manifest.json`. Its generated `code/` and `locked_inputs/` folders are excluded from Git and from the course source package. The course source package includes the code, definitions, result summaries and provenance, not photographs, training labels or weights.
