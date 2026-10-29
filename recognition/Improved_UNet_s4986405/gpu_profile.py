"""Record PyTorch GPU allocator peaks for a bounded training run."""
import json
import math
import os
from pathlib import Path
import socket
import time


GIB = 1024 ** 3


def build_memory_report(allocated_bytes, reserved_bytes, total_bytes, elapsed_seconds):
    for name, value in (('allocated', allocated_bytes), ('reserved', reserved_bytes), ('total', total_bytes)):
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} bytes must be a non-negative integer')
    if allocated_bytes == 0 or total_bytes == 0:
        raise ValueError('GPU training must have positive allocated and total memory')
    if (isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0):
        raise ValueError('Elapsed seconds must be finite and non-negative')
    return {
        'peak_allocated_bytes': allocated_bytes, 'peak_reserved_bytes': reserved_bytes,
        'device_total_bytes': total_bytes, 'peak_allocated_gib': allocated_bytes / GIB,
        'peak_reserved_gib': reserved_bytes / GIB, 'device_total_gib': total_bytes / GIB,
        'run_wall_seconds_including_io_and_plot': float(elapsed_seconds),
    }


def save_gpu_profile(run_dir, device, started_at):
    import torch
    if device.type != 'cuda':
        raise ValueError('GPU profiling requires a CUDA device')
    torch.cuda.synchronize(device)
    metrics = build_memory_report(
        torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device),
        torch.cuda.get_device_properties(device).total_memory, time.perf_counter() - started_at,
    )
    directory = Path(run_dir).expanduser().resolve()
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))['config']
    history = json.loads((directory / 'history.json').read_text(encoding='utf-8'))
    report = {
        **metrics, 'device_name': torch.cuda.get_device_name(device), 'logical_device': str(device),
        'allocator_backend': torch.cuda.memory.get_allocator_backend(),
        'node': socket.gethostname(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'torch_version': str(torch.__version__), 'torch_cuda_build': torch.version.cuda,
        'architecture': config['architecture'], 'model_config': config['model_config'],
        'partial_run': config['partial_run'], 'completed_epochs': len(history),
        'processed_samples_per_epoch': {
            phase: [entry[phase]['samples'] for entry in history]
            for phase in ('train', 'validation')
        },
        'memory_scope': 'PyTorch tensor/allocator peaks, not whole-device NVIDIA-SMI memory',
        'time_scope': 'Run wall time includes data preparation, checkpoint I/O and plotting; not inference latency',
    }
    with (directory / 'gpu_profile.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    return report
