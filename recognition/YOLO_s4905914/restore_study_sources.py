"""Restore the recorded research layout from the course folder into a new workspace.

Course Git files stay within recognition/YOLO_s4905914. Historical experiment
scripts are preserved byte-for-byte so their original provenance hashes remain
meaningful. No dataset, source metadata, cached raw prediction or model weight
is restored by this helper; these must be supplied separately as documented.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def restore(destination):
    code = Path(__file__).resolve().parent
    sources = code/'research_sources'
    manifest = json.loads((sources/'manifest.json').read_text())
    if destination.exists():
        raise ValueError('Choose a new destination; existing work is not overwritten')
    if destination.resolve().is_relative_to(code):
        raise ValueError('The new workspace must be outside this source module')
    for relative, expected in manifest['files'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Unsafe relative source path')
        source = sources/relative
        if source.is_symlink() or not source.resolve().is_relative_to(sources.resolve()):
            raise ValueError('Unsafe source member')
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('Source hash differs: '+relative)
    destination.mkdir(parents=True)
    target = destination/'recognition/YOLO_s4905914'
    shutil.copytree(code, target, ignore=shutil.ignore_patterns('research_sources','__pycache__','*.pyc'))
    for relative in manifest['files']:
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sources/relative, target)
    (destination/'RESTORED_SOURCE_PROVENANCE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Restored source workspace:', destination)
    print('Supply the recorded data snapshots, manifests, cached outputs and weights separately.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();restore(args.destination.resolve())
