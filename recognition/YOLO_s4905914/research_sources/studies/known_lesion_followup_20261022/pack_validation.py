"""Stream exactly the locked validation images, never the reassessment split."""
import csv,hashlib,json,sys,tarfile
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[1]
p=json.loads((S/'plan.json').read_text())
manifest=R/'data/known_lesion_v1/val.csv'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==p['manifest_sha256']['val']
rows=list(csv.DictReader(manifest.open()));assert len(rows)==165
audit=json.loads((R/'outputs/known_lesion_optimization_20261021_681452/n_geometry/inputs/staged/audit.json').read_text())
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
    for row in rows:
        path=Path(row['image_path'])
        assert path.stem==row['image_id'] and row['lesion_id']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['files'][row['image_id']]['file_sha256']
        archive.add(path,arcname=path.name,recursive=False)
