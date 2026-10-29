"""Measure finite forward/backward steps of CAN3D on one training volume.

This is a GPU feasibility/timing diagnostic, not a trained-model evaluation.
Only the training split is opened. Never extrapolate accuracy from this run.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import time

import torch

from can3d import CAN3D
from dataset import DEFAULT_DATA_ROOT, HipMRIVolumeDataset, PILOT_SHAPE_DHW
from experiment import file_sha256, get_device, save_json, seed_everything
from modules import segmentation_loss


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('runs/split_seed42.json'))
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    device = get_device('cuda')
    seed_everything(42)
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = HipMRIVolumeDataset(args.manifest, 'train', args.data_root, PILOT_SHAPE_DHW)
    sample = dataset[0]
    image, target = sample['image'][None].to(device), sample['target'][None].to(device)
    model = CAN3D().to(device)
    criterion = segmentation_loss('dice-ce')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    durations, losses, peak = [], [], 0
    for step in range(5):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        if logits.shape != target.shape:
            raise ValueError('Output grid differs from target.')
        loss = criterion(logits, target)
        loss.backward()
        if (not torch.isfinite(loss) or not torch.isfinite(logits).all()
                or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())):
            raise ValueError('Nonfinite output, loss or gradient.')
        optimizer.step()
        del logits
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        peak = max(peak, torch.cuda.max_memory_allocated()/2**20)
        losses.append(loss.item())
        if step >= 2:
            durations.append(elapsed)
        print(f'Training-only diagnostic step {step+1}/5: loss={loss.item():.6f}, seconds={elapsed:.3f}', flush=True)
    report = dict(model='can3d_one_down', input_shape=list(image.shape), target_shape=list(target.shape),
                  case_id=sample['case_id'], split='train', diagnostic_only=True,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  peak_training_allocated_mib=peak, measured_step_seconds=durations,
                  median_step_seconds=statistics.median(durations), losses=losses,
                  measured_steps_after_warmup=3, warmup_steps=2,
                  manifest_sha256=file_sha256(args.manifest),
                  source_sha256={name: file_sha256(Path(__file__).with_name(name))
                      for name in ['can3d.py','can3d_smoke.py','modules.py','dataset.py','experiment.py']},
                  gpu_name=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                  hostname=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  training_recipe='Adam lr0.001; one full resized training volume; unit Dice+CE',
                  timing_scope='GPU-resident training tensors through Adam step; includes finite checks, excludes data loading; same case repeated',
                  limitation='Not an epoch-time or convergence measurement; no validation/test results; no trained checkpoint saved')
    save_json(args.output/'smoke.json',report)
    print(json.dumps({key:report[key] for key in ['parameter_count','peak_training_allocated_mib','median_step_seconds']}), flush=True)
    print('CAN3D_SMOKE_PASSED (feasibility only; not trained-model accuracy).', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
