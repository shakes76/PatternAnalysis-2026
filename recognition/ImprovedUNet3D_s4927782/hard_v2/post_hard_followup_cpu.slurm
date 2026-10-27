#!/bin/bash
#SBATCH --job-name=hipmri-hard-results
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$SLURM_SUBMIT_DIR/artifacts/pycache"
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/artifacts/matplotlib"
DEMO3_V1_PROJECT=/home/Student/s4927782/comp3710-demo3
DEMO3_PYTHON="$DEMO3_V1_PROJECT/.venv/bin/python"
"$DEMO3_PYTHON" - <<'PY'
from pathlib import Path
import json,os,subprocess,hashlib,time
root=Path('.');job=os.environ['DEMO3_HARD_JOB_ID'];parts=[];row=''
for attempt in range(6):
    lines=subprocess.check_output(['/opt/slurm/bin/sacct','-j',job,'-X','-n','-P','--format=JobIDRaw,State%24,ElapsedRaw,ExitCode,NodeList%64,AllocTRES%200'],text=True).strip().splitlines()
    if lines:
        row=lines[0];parts=[x.strip() for x in row.split('|')]
        if parts[0]==job and parts[1]=='COMPLETED' and parts[3]=='0:0':break
    time.sleep(2)
else:raise RuntimeError('Hard formal job not verified complete: '+row)
runner='hard_followup_gpu.slurm'
receipt=dict(job_id=int(job),state=parts[1],elapsed_seconds=int(parts[2]),exit_code=parts[3],node_list=parts[4],alloc_tres=parts[5],same_allocation=True,provenance_source='slurm_sacct_and_frozen_runner',runner_file=runner,runner_sha256=hashlib.sha256((root/runner).read_bytes()).hexdigest(),raw_sacct_row=row)
target=root/'artifacts/hard_formal_execution_receipt.json'
with target.open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
PY
for split in validation test; do
  extra=()
  if [[ "$split" == test ]]; then extra=(--retrospective-test-ack); fi
  "$DEMO3_PYTHON" -u compare_hard_followup.py --review-root . \
    --v1-project "$DEMO3_V1_PROJECT" --artifacts-root artifacts \
    --scope-plan hard_followup_protocol.json --split "$split" \
    --output-dir "artifacts/hard_common_${split}" "${extra[@]}"
done
"$DEMO3_PYTHON" plot_hard_followup.py --report artifacts/hard_common_test/comparison.json \
  --output-dir artifacts/hard_results_figures
"$DEMO3_PYTHON" plot_hard_failure_gallery.py --v1-project "$DEMO3_V1_PROJECT" \
  --artifacts-root artifacts --report artifacts/hard_common_test/comparison.json \
  --output-dir artifacts/hard_failure_gallery
"$DEMO3_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('.');sel=json.loads((root/'artifacts/hard_selection.json').read_text());cp=root/'artifacts/hard_followup/improved3d/best.pt'
assert set(sel['models'])=={'improved3d'} and sel['models']['improved3d']['completed_epochs']==30
assert hashlib.sha256(cp.read_bytes()).hexdigest()==sel['models']['improved3d']['checkpoint_sha256']
PY
# Paired infer_v2's optional freeze requires two new models. Its image-only
# checkpoint/history guards remain valid; the genuine Hard-only choice was
# verified immediately above and is linked separately below, never fabricated.
"$DEMO3_PYTHON" infer_v2.py \
  --image /home/groups/comp3710/HipMRI_Study_open/semantic_MRs/B006_Week0_LFOV.nii.gz \
  --checkpoint artifacts/hard_followup/improved3d/best.pt --protocol protocol_v2.json \
  --project-root "$DEMO3_V1_PROJECT" --device cpu \
  --output-dir artifacts/hard_deployment/improved3d
"$DEMO3_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('.');d=r/'artifacts/hard_deployment/improved3d';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((d/'receipt.json').read_text());selection=json.loads((r/'artifacts/hard_selection.json').read_text())
assert receipt['checkpoint_sha256']==selection['models']['improved3d']['checkpoint_sha256']
assert receipt['image_only_inference'] and receipt['mask_or_manifest_read'] is False
out=dict(status='completed',purpose='TRAIN B006 image-only deployment/native-geometry check; no new generalisation evidence',scope_plan_sha256=h(r/'hard_followup_protocol.json'),selection_freeze_sha256=h(r/'artifacts/hard_selection.json'),checkpoint_sha256=receipt['checkpoint_sha256'],image_only_receipt_sha256=h(d/'receipt.json'),original_failed_pair_preserved=True)
with (d/'hard_scope_receipt.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
PY
