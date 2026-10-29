"""C0a only: verify manifests, one real training JPEG, and a CUDA tensor operation."""
import argparse
import csv, hashlib, json, os, platform, subprocess, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--split-dir', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
if not __debug__:
    raise RuntimeError('Run without -O: integrity assertions must remain enabled')
source = args.split_dir.expanduser().resolve(strict=True)
out = args.output_dir.expanduser().resolve()
if out == source or source in out.parents:
    raise ValueError('Outputs must be outside the split input directory')
import torch
from PIL import Image, ImageFile, __version__ as pillow_version
if out.exists():
    raise FileExistsError('Output exists; do not overwrite evidence')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest = source / 'split_samples.csv'
summary_path = source / 'split_summary.json'
evidence_hash = sha(summary_path)
evidence = json.loads(summary_path.read_text())
assert evidence['status'] == 'passed'
manifest_hash = sha(manifest)
assert manifest_hash == evidence['output_sha256']['split_samples'], 'CSV hash mismatch'
with manifest.open(newline='') as f:
    reader = csv.DictReader(f)
    assert reader.fieldnames == ['file_path','original_split','image_id','patient_id','scan_date','label','split']
    rows = list(reader)
counts = Counter(r['split'] for r in rows)
assert counts == {'train':8160,'validation':2720,'test':2720}
assert len({r['file_path'] for r in rows}) == 13600
patients = {s: {r['patient_id'] for r in rows if r['split'] == s} for s in counts}
assert {s:len(p) for s,p in patients.items()} == {'train':408,'validation':136,'test':136}
assert not (patients['train'] & patients['validation'] or patients['train'] & patients['test'] or patients['validation'] & patients['test'])
assert set(r['label'] for r in rows) == {'AD','NC'}
row_number, row = next((i,r) for i,r in enumerate(rows,1) if r['split'] == 'train')
p = Path(row['file_path'])
assert p.is_file(), 'Selected training image is not a file'
ImageFile.LOAD_TRUNCATED_IMAGES = False
with Image.open(p) as img:
    assert img.format == 'JPEG'
    img.verify()
with Image.open(p) as img:
    img.load()
    image_result = dict(split='train', selection_rule='First train row in verified CSV order',
                        manifest_data_row_number=row_number, label=row['label'],
                        image_format=img.format, width=img.width, height=img.height,
                        mode=img.mode, channels=len(img.getbands()), pixel_extrema=img.getextrema(),
                        fully_decoded=True, source_image_sha256=sha(p))
assert os.environ.get('SLURM_JOB_ID'), 'Must run in SLURM allocation'
assert torch.cuda.is_available(), 'CUDA unavailable'
a = torch.tensor([[1.,2.],[3.,4.]], device='cuda')
b = torch.tensor([[5.,6.],[7.,8.]], device='cuda')
c = a @ b
torch.cuda.synchronize()
assert torch.equal(c.cpu(), torch.tensor([[19.,22.],[43.,50.]]))
assert torch.isfinite(c).all().item()
smi = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip().splitlines()
assert sha(manifest) == manifest_hash and sha(summary_path) == evidence_hash
out.mkdir(mode=0o700)
result = dict(stage='C0a', status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
              scope='Environment, manifest integrity, and one real training image only; no dataset loader, batches, or training.',
              python_version=platform.python_version(), python_executable=sys.executable,
              torch_version=torch.__version__, pillow_version=pillow_version,
              torch_cuda_build_version=torch.version.cuda,
              cudnn_version=torch.backends.cudnn.version(),
              cuda_available=True, gpu_name=torch.cuda.get_device_name(0), nvidia_smi=smi,
              environment_action='Reused existing independent Rangpur environment; no packages installed this run.',
              installation_commands_executed=[], historical_installation_command=None,
              historical_installation_note='Original installation command was not verified; no reconstruction claimed.',
              slurm_job_id=os.environ['SLURM_JOB_ID'], hostname=platform.node(),
              split_manifest_sha256=manifest_hash, split_summary_sha256=evidence_hash,
              split_counts=dict(counts), patient_counts={s:len(v) for s,v in patients.items()},
              real_images_fully_decoded=1, images_read_by_split={'train':1,'validation':0,'test':0},
              training_sample=image_result, tensor_check=dict(device=str(c.device),dtype=str(c.dtype),shape=list(c.shape),result=c.cpu().tolist(),passed=True),
              preprocessing='Not selected or applied in C0a; decoded native image only.',
              probe_script_sha256=sha(Path(__file__)))
with (out/'environment_check_summary.json').open('x') as f:
    json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
(out/'environment_check_summary.json').chmod(0o600)
print(json.dumps(result,sort_keys=True))
