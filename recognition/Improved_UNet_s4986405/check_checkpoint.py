"""Inspect one saved run and check a strict model reload on CPU."""
import argparse
import csv
import json
from pathlib import Path

import torch

from model_factory import model_from_checkpoint
from training_io import load_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    directory = args.run_dir.expanduser().resolve()
    last = load_checkpoint(directory / 'last.pt')
    best = load_checkpoint(directory / 'best.pt')
    history = json.loads((directory / 'history.json').read_text(encoding='utf-8'))
    with (directory / 'history.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(history) != last['epoch'] or len(rows) != len(history):
        raise RuntimeError('Epoch/history row count mismatch')
    if history != last['history']:
        raise RuntimeError('Checkpoint and JSON histories differ')
    best_entry = min(history, key=lambda entry: entry['validation']['loss'])
    if best['epoch'] != best_entry['epoch'] or last['best_epoch'] != best_entry['epoch']:
        raise RuntimeError('Best checkpoint does not match minimum validation loss')
    if any(last[key] != best[key] for key in ('architecture', 'model_config')):
        raise ValueError('Best/latest checkpoint model metadata mismatch')
    model = model_from_checkpoint(best).cpu()
    if model.in_channels != 1 or model.num_classes != 6:
        raise ValueError('Expected one MRI input channel and six numerical label IDs')
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 1, 32, 32))
    if tuple(logits.shape) != (1, 6, 32, 32) or not torch.isfinite(logits).all():
        raise RuntimeError('Invalid reloaded model output')
    print('Run directory:', directory)
    print('Architecture:', best['architecture'])
    print('History epochs:', len(history))
    print('Latest epoch:', last['epoch'])
    print('Best epoch:', best['epoch'])
    print('Best validation loss:', last['best_val_loss'])
    print('Partial run:', last['config']['partial_run'])
    print('History consistency: PASS')
    print('Checkpoint reload: PASS')


if __name__ == '__main__':
    main()
