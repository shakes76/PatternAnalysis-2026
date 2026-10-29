"""Build the private review inputs; verify them with transfer.py, never upload.

Keep this archive outside Git. It contains fixed metadata/labels and fitted
weights but no source photographs. Code and frozen configurations come from
the public course checkout. No model is loaded and no inference is run here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from transfer import MANIFEST, regular_file, verify_bundle

CODE = Path('recognition/YOLO_s4905914')
BASELINE = Path('outputs/provisional_20261017_669143')
FINAL = Path('outputs/known_lesion_optimization_20261021_681452/n_geometry')


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    checksum = output.with_suffix('.zip.sha256')
    if output.suffix != '.zip' or output.exists() or checksum.exists():
        raise ValueError('Choose unused .zip and .zip.sha256 paths')
    paths = [Path('sources.json')]
    for folder in ('data/source_snapshots', 'data/known_lesion_v1'):
        paths.extend(p.relative_to(root) for p in (root / folder).rglob('*')
                     if p.is_file())
    for folder, model in ((BASELINE, 'n'), (BASELINE, 's'), (FINAL, 'n')):
        paths.extend([folder / f'train_{model}/fit/weights/best.pt',
                      folder / f'train_{model}/run.json',
                      folder / f'val_{model}/selection_lock.json'])
    decision = json.loads((root / CODE / 'known_optimization_decision_20261021.json').read_text())
    final_lock = json.loads((root / decision['selection_lock_path']).read_text())
    final_bytes = regular_file(root, decision['checkpoint_path']).read_bytes()
    sha = lambda data: hashlib.sha256(data).hexdigest()
    if not (sha(final_bytes) == decision['checkpoint_sha256'] == final_lock['checkpoint_sha256']
            and final_lock['threshold'] == decision['confidence'] == 0.4
            and final_lock['config_sha256'] == sha((root / decision['config_path']).read_bytes())):
        raise ValueError('Final checkpoint, config or validation lock does not match the decision')
    payload = {p.as_posix(): regular_file(root, p.as_posix()).read_bytes()
               for p in sorted(set(paths))}
    if any(Path(name).suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff') for name in payload):
        raise ValueError('Do not put source photographs or masks in this bundle')
    manifest = {'format_version': 1, 'purpose': 'private_course_review_inputs',
                'contains_source_photographs_or_masks': False,
                'contains_metadata_labels_and_fitted_weights': True,
                'upload_to_git': False,
                'files': {name: {'sha256': sha(data), 'bytes': len(data)}
                          for name, data in payload.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x') as archive:
        for name, data in {**payload, MANIFEST: (json.dumps(manifest, indent=2)+'\n').encode()}.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    result = verify_bundle(output)
    bundle_sha = sha(output.read_bytes())
    checksum.write_text(f'{bundle_sha}  {output.name}\n')
    return {**result, 'archive_name': output.name, 'archive_sha256': bundle_sha,
            'archive_bytes': output.stat().st_size,
            'final_checkpoint_sha256': decision['checkpoint_sha256'],
            'final_confidence': final_lock['threshold'],
            'code_and_configs': 'Restore from the matching public course checkout',
            'photographs': 'Obtain the unchanged school ISIC2018 Task1-2 Training Input x2 files'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root, args.output), indent=2))
