"""Refresh the explicit public package inventory after integration-document edits.

Never include tensor weights, raw medical images, cached arrays, private logs or
execution automation. The manifest deliberately omits itself, not source files.
"""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
FORBIDDEN = {'.pt', '.pth', '.npz', '.npy', '.nii', '.gz', '.ckpt', '.h5', '.hdf5', '.zip', '.tar'}
EXCLUDED = {'PUBLICATION_MANIFEST.json', 'INTEGRATION_CHECK.json'}

def main():
    files = []
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name in EXCLUDED:
            continue
        relative = path.relative_to(HERE).as_posix()
        if path.suffix in FORBIDDEN or relative.startswith(('logs/', 'documentation_history/')):
            raise ValueError('Non-public package member: ' + relative)
        if path.suffix not in {'.json', '.md', '.py', '.png', '.slurm'}:
            raise ValueError('Unapproved extension: ' + relative)
        if path.name in {'ASSISTANCE_LOG.jsonl', 'AUTOMATION_STATE.json'}:
            raise ValueError('Private record must not enter publication')
        data = path.read_bytes()
        files.append({'path': relative, 'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    manifest = {'format': 'hipmri_hard_v2_public_file_inventory_v1',
                'scope': 'Completed scientific source/metadata copies and separate integration-only helpers; no training or image inference during packaging.',
                'original_code_and_protocols': 'See INTEGRATION_ORIGINS.json for byte-exact source mapping.',
                'historical_gate': 'Unchanged old protocol/output fields are historical evidence, not a current assignment requirement.',
                'raw_mri_masks_predictions_weights_in_git': False, 'files': files}
    (HERE / 'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'files': len(files), 'bytes': sum(x['size_bytes'] for x in files)}))

if __name__ == '__main__':
    main()
