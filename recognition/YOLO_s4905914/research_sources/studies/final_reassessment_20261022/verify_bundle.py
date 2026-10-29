"""Verify the final-evaluation bundle before installing packages or inference."""
import hashlib,json
from pathlib import Path
study=Path(__file__).resolve().parent
manifest=json.loads((study/'bundle_manifest.json').read_text())
for rel,expected in manifest['files'].items():
    p=study/rel
    if not p.resolve().is_relative_to(study.resolve()):raise ValueError('Unsafe bundle path')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Changed bundle input: '+rel)
print('Verified',len(manifest['files']),'fixed bundle files; no inference performed.')
