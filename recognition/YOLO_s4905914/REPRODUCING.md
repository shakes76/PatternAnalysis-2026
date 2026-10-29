# Reproducing the final coursework study

This is the current execution guide (28 October 2026). `RUNBOOK.md` also preserves earlier development notes. The original `experiment.json` is a disabled planning draft; do not enable or use it for training. The final model is geometry YOLOv8n, seed 3710, with its existing validation lock at confidence **> 0.40**. Nothing here retunes that policy.

## 1. Obtain code and the separate review inputs

```bash
git clone --branch topic-recognition --single-branch \
  https://github.com/lucaaaas-glich/PatternAnalysis-2026.git
cd PatternAnalysis-2026
python3 recognition/YOLO_s4905914/restore_study_sources.py \
  --destination ../comp3710-reproduction
cd ../comp3710-reproduction
```

Restoration verifies the recorded study-source hashes and creates a new directory. It copies source/configuration files, not data or weights. Use an unused destination. Do not rerun the historical `prepare_study.py` scripts: the frozen configs and source snapshots are already restored.

For course review, contact the student through UQ email, **s4905914@student.uq.edu.au**, and request **COMP3710_s4905914_review_inputs_20261028.zip**. This prepared archive is held separately from Git; there is no public download URL. It contains:

- The exact metadata and mask-derived box snapshots, grouping manifest, historical split records, and all 1,098 labels with the fixed train/validation/reassessment CSVs.
- Original n/s and final geometry-n `best.pt`, the associated `run.json` records and validation selection locks.
- A size/SHA-256 manifest for every included file.

It excludes source photographs/masks, the original COCO weights and complete raw prediction caches. Those caches are not needed for the training/prediction commands below. The archive does not claim to reproduce the original mask-to-box conversion from raw masks; it supplies the audited box snapshot used by this study. Do not add this archive, its extracted inputs, or trained weights to the public Git repository.

The archive has 1,119 payload files, size **32,589,053 bytes**, and SHA-256:

```text
307b312aab2865537ca1b4fa0e5a3b1b152e00b975ed32e937099b2770515144
```

Place the archive in the restored workspace, then verify it before extraction:

```bash
python3 recognition/YOLO_s4905914/transfer.py verify \
  --bundle COMP3710_s4905914_review_inputs_20261028.zip
python3 -m zipfile -e COMP3710_s4905914_review_inputs_20261028.zip .
python3 recognition/YOLO_s4905914/transfer.py verify --directory .
```

Also compare the archive's SHA-256 with the value above (`shasum -a 256` on macOS or `sha256sum` on Linux). The verifier checks every member and rejects unsafe names or unexpected members. Extract only into the fresh workspace, before creating new outputs. Existing historical configurations and records retain their original approval-status fields; current course confirmations are stated in the report.

## 2. Environment and photographs

The measured training environment used Python 3.13.16, PyTorch 2.8.0, torchvision 0.23.0 and Ultralytics 8.3.221 on Linux/A100. Install in an isolated environment, using the tested constraints on Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r recognition/YOLO_s4905914/requirements.txt \
  -c recognition/YOLO_s4905914/requirements-linux-a100.lock.txt
python -m pip check
python -m unittest discover -s recognition/YOLO_s4905914 -p 'test_*.py'
```

For macOS CPU prediction, install `requirements.txt` without the Linux constraints; the earlier local checks used Python 3.13.13. Use a separate environment for `requirements-report.txt` if exporting PDFs. Hardware, package builds and nondeterministic operations can prevent bit-identical retraining even with the recorded seed.

On Rangpur, use the unchanged course photographs at:

```text
/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2
```

Access requires a UQ account with permission to the course dataset and the school network/VPN. On another machine, obtain these same files through the course's permitted access route and set `IMAGE_ROOT` to their directory. The public ISIC source is https://challenge.isic-archive.com/data/ and the metadata collection is https://api.isic-archive.com/collections/63/ ; these identify provenance, not byte-identical substitutes for the recorded school photos and metadata snapshot. Do not silently replace inputs with resized or freshly downloaded variants. The staging step checks decoding and dimensions; the additional recorded-content check below verifies the final train/validation photograph bytes.

## 3. Validate inputs without training

```bash
CODE=recognition/YOLO_s4905914
STUDY=studies/known_lesion_optimization_20261021
FINAL_CONFIG="$STUDY/n_geometry.json"
PREPARED=data/known_lesion_v1
IMAGE_ROOT=/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2

python "$CODE/train.py" plan \
  --config "$CODE/experiment_provisional.json" --prepared "$PREPARED"
python "$STUDY/code/train.py" plan \
  --config "$FINAL_CONFIG" --prepared "$PREPARED"
```

The baseline plan must show 768 training, 165 validation and 165 reassessment images. The final-study plan intentionally reads only 768 training and 165 validation images. Both have 60 epochs and 2,880 optimizer calls. These checks validate labels, classes, hashes and known-group separation; they neither load photographs nor start training.

Optional replay of the prepared labels and grouped membership:

```bash
python "$CODE/prepare_dataset.py" --sources sources.json --protocol PROTOCOL.json \
  --output outputs/replayed_labels
python "$CODE/plan_group_splits.py" \
  --metadata data/source_snapshots/collection_63_metadata.csv \
  --historical-splits data/source_snapshots/historical_splits \
  --output outputs/replayed_groups --seed 3710
```

Use unused output directories. The three regenerated CSVs and labels should match `data/known_lesion_v1`; `outputs/replayed_groups/known_lesion/manifest.csv` should match `data/source_snapshots/known_lesion_manifest.csv`. Missing patient IDs remain missing.

## 4. Run the saved final model without retraining

This route needs the private review inputs and one photograph, but no GPU allocation. It uses the already selected checkpoint and threshold:

```bash
FINAL_RUN=outputs/known_lesion_optimization_20261021_681452/n_geometry
python "$CODE/predict.py" --config "$FINAL_CONFIG" \
  --checkpoint "$FINAL_RUN/train_n/fit/weights/best.pt" \
  --lock "$FINAL_RUN/val_n/selection_lock.json" \
  --image "$IMAGE_ROOT/ISIC_0012653.jpg" \
  --output outputs/review_prediction --device cpu
```

`ISIC_0012653` is a training image used here only to check the inference interface, not as new accuracy evidence. Replace it with any permitted dermoscopic image for a demonstration. Outputs are `prediction.json` and `prediction.jpg`. The code rejects a changed config/checkpoint pair and reads confidence > 0.40 from the lock. The final checkpoint SHA-256 is `a1bd556186a1dc98d4b1d731fd59897e3084a70f6484e6b700586e6a09ef794c`.

## 5. Reproduce baseline and final training

Run this only inside a school-approved GPU allocation. These are Python commands, not scheduler requests. Download the original initialization weights; their hashes are checked automatically:

```bash
python "$CODE/fetch_weights.py" --output weights/original_coco
```

For the original n/s comparison, use the enabled provisional config and one shared staging directory:

```bash
BASE_CONFIG="$CODE/experiment_provisional.json"
python "$CODE/dataset.py" --prepared "$PREPARED" --config "$BASE_CONFIG" \
  --image-root "$IMAGE_ROOT" --output data/review_baseline_stage
for MODEL in n s; do
  python "$CODE/train.py" fit --config "$BASE_CONFIG" --prepared "$PREPARED" \
    --stage data/review_baseline_stage --weights-dir weights/original_coco \
    --model "$MODEL" --output "outputs/review_baseline_$MODEL" --device 0
  python "$CODE/train.py" evaluate --config "$BASE_CONFIG" --prepared "$PREPARED" \
    --stage data/review_baseline_stage \
    --checkpoint "outputs/review_baseline_$MODEL/fit/weights/best.pt" \
    --split val --output "outputs/review_baseline_${MODEL}_val" --device 0
done
```

For final geometry n, use the **frozen study-specific code**. Its config has two split hashes; passing it to the three-split baseline training loader would be incorrect.

```bash
python "$STUDY/code/dataset.py" --prepared "$PREPARED" --config "$FINAL_CONFIG" \
  --image-root "$IMAGE_ROOT" --output data/review_geometry_stage
python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'studies/known_lesion_optimization_20261021/code')
from runtime import stage_content_digest
original = Path('outputs/known_lesion_optimization_20261021_681452/n_geometry/train_n/run.json')
assert stage_content_digest('data/review_geometry_stage') == json.loads(original.read_text())['staged_image_content_sha256']
print('Recorded final train/validation photograph bytes match.')
PY
python "$STUDY/code/train.py" fit --config "$FINAL_CONFIG" --prepared "$PREPARED" \
  --stage data/review_geometry_stage --weights-dir weights/original_coco \
  --model n --output outputs/review_geometry_n --device 0
python "$STUDY/code/train.py" evaluate --config "$FINAL_CONFIG" --prepared "$PREPARED" \
  --stage data/review_geometry_stage \
  --checkpoint outputs/review_geometry_n/fit/weights/best.pt \
  --split val --output outputs/review_geometry_n_val --device 0
```

Each new run writes its own checkpoint, history, metrics and validation lock. It does not overwrite the report model or guarantee identical results. Do not combine a retrained checkpoint with the old lock or use reassessment results to choose a replacement. The study-specific loader intentionally prevents reassessment in this development route. Historical frozen reassessment procedures are in `studies/final_reassessment_20261022/RUNBOOK.md`; reproducing those full cache-based audits requires additional original result files from the student, beyond this input bundle.

## 6. Rebuild the report and maintain the review package

The public checkout contains the metric summaries and figures used by the report. To export the current reviewed prose without replacing it:

```bash
python "$CODE/export_readme_pdf.py" \
  --output output/pdf/COMP3710_Project3_Report.pdf
```

`build_final_report.py` regenerates README from the report builder and summaries; run it only when intentionally regenerating the source. It can overwrite manual README edits. Neither report command trains a model. Review `REPRODUCTION_CHECKS_20261028.json` for the checks actually performed on this documentation update.

For the author, the prepared private inputs can be rebuilt from the complete research workspace with `reviewer_bundle.py --output <new-path.zip>`. `reviewer_bundle_20261028.json` records the published bundle identity; the older general transfer bundle is not the final review bundle. For a new data/checkpoint bundle, use a new filename and update the guide and checksums instead of silently replacing the recorded file.
