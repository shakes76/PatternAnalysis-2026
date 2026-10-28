"""Record exact package versions from the actual Rangpur conda environment."""
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

names = ['torch', 'torchvision', 'numpy', 'pillow', 'matplotlib']
pkgs = {}
for name in names:
    try:
        pkgs[name] = version(name)
    except PackageNotFoundError:
        raise RuntimeError(f'Missing required package: {name}')
info = {'python': sys.version.split(' ')[0], 'packages': pkgs}
Path('environment_versions.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
Path('requirements.txt').write_text(''.join(f'{n}=={v}\n' for n,v in pkgs.items()), encoding='utf-8')
print(json.dumps(info, indent=2))
print('Created environment_versions.json and requirements.txt')
