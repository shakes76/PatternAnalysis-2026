"""Persist isolated runs, checkpoints and strict JSON/CSV epoch histories."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import torch


SOURCE_FILES = (
    'train.py', 'training_io.py', 'plot_history.py', 'gpu_profile.py', 'modules.py', 'dataset.py',
    'losses.py', 'metrics.py', 'check_splits.py', 'shape_batching.py', 'model_factory.py',
)


def source_hashes(directory):
    return {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def _atomic_write(path, writer, binary=False):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        options = {} if binary else {'encoding': 'utf-8', 'newline': ''}
        with os.fdopen(descriptor, 'wb' if binary else 'w', **options) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        copied = value.__class__((key, _cpu_copy(item)) for key, item in value.items())
        if hasattr(value, '_metadata'):
            copied._metadata = _cpu_copy(value._metadata)
        return copied
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _result_record(result):
    loss = float(result['loss'])
    if not math.isfinite(loss):
        raise ValueError('Recorded loss must be finite')
    record = {'loss': loss}
    for key in ('samples', 'batches', 'pixels'):
        record[key] = int(result[key])
        if record[key] < 1:
            raise ValueError('Recorded counts must be positive')
    record['confusion'] = result['confusion'].detach().cpu().tolist()
    for key in ('dice', 'iou'):
        values = result[key].detach().cpu().tolist()
        if len(values) != 6:
            raise ValueError('Expected six per-class metric values')
        if any(math.isinf(float(value)) for value in values):
            raise ValueError('Infinite metrics cannot be recorded')
        record[key] = [None if math.isnan(float(value)) else float(value) for value in values]
    return record


def _history_csv(history):
    rows = []
    for entry in history:
        row = {'epoch': entry['epoch']}
        for phase in ('train', 'validation'):
            result = entry[phase]
            for key in ('loss', 'samples', 'batches', 'pixels'):
                row[f'{phase}_{key}'] = result[key]
            for metric in ('dice', 'iou'):
                for label, value in enumerate(result[metric]):
                    row[f'{phase}_{metric}_{label}'] = value
        rows.append(row)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


class RunRecorder:
    """A fresh output directory owns one run; best means lowest validation loss.

    Files are replaced atomically individually, not as a multi-file transaction.
    last.pt embeds its history as well as model, optimizer and RNG snapshots.
    """
    def __init__(self, directory, config):
        # Validate JSON compatibility before creating the run directory.
        self.config = json.loads(json.dumps(config, allow_nan=False))
        if 'model_config' not in self.config:
            raise ValueError('config must include model_config')
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.history = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.environment = {
            'python': platform.python_version(), 'torch': str(torch.__version__),
            'torch_cuda_build': torch.version.cuda,
        }
        metadata = {'config': self.config, 'environment': self.environment}
        _atomic_write(self.directory / 'config.json',
                      lambda handle: json.dump(metadata, handle, indent=2, allow_nan=False))

    def record(self, epoch, model, optimizer, train_result, val_result, generator):
        if epoch != len(self.history) + 1:
            raise ValueError('Epochs must be consecutive and begin at 1')
        entry = {
            'epoch': epoch, 'train': _result_record(train_result),
            'validation': _result_record(val_result),
        }
        improved = entry['validation']['loss'] < self.best_val_loss
        if improved:
            self.best_val_loss = entry['validation']['loss']
            self.best_epoch = epoch
        self.history.append(entry)
        rng_state = {'torch_cpu': torch.get_rng_state(), 'shuffle': generator.get_state()}
        if next(model.parameters()).device.type == 'cuda':
            rng_state['torch_cuda'] = torch.cuda.get_rng_state_all()
        checkpoint = {
            'schema_version': 1, 'epoch': epoch,
            'architecture': self.config.get('architecture', 'UNet2D'),
            'model_config': self.config['model_config'],
            'model_state': _cpu_copy(model.state_dict()),
            'optimizer_state': _cpu_copy(optimizer.state_dict()),
            'config': self.config, 'environment': self.environment,
            'history': self.history, 'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch, 'rng_state': _cpu_copy(rng_state),
        }
        if improved:
            _atomic_write(self.directory / 'best.pt', lambda handle: torch.save(checkpoint, handle), binary=True)
        _atomic_write(self.directory / 'last.pt', lambda handle: torch.save(checkpoint, handle), binary=True)
        _atomic_write(self.directory / 'history.json',
                      lambda handle: json.dump(self.history, handle, indent=2, allow_nan=False))
        csv_text = _history_csv(self.history)
        _atomic_write(self.directory / 'history.csv', lambda handle: handle.write(csv_text))
        return improved


def load_checkpoint(path):
    """Load the project's dictionary checkpoint onto CPU with restricted loading."""
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != 1:
        raise ValueError('Unsupported checkpoint schema')
    return checkpoint
