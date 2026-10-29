"""Frozen-checkpoint test accuracy and repeated complete-case inference profiling.

No optimizer or parameter updates. One process/model, same saved patient split
and predeclared settings. Test results are for reporting, never further tuning.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import time

import nibabel as nib
import numpy as np
import torch

from dataset import DEFAULT_DATA_ROOT, HipMRIVolumeDataset, LABEL_NAMES
from experiment import (aggregate_cases, file_sha256, get_device, hard_overlap,
                        load_checkpoint, predict_volume, save_json, seed_everything)
from predict import save_panels
from validate_best import low_scoring_patients


def measure_case(model, image, device, batch_size, warmups, repeats):
    """Wall time: preprocessed CPU image -> complete CPU hard prediction.

    Includes transfers, model compute and argmax. Excludes disk, preprocessing,
    metric computation and plots. Allocated/reserved peaks exclude CUDA context
    and other processes; they are not total board-memory measurements.
    """
    for _ in range(warmups):
        predict_volume(model, image, device, batch_size).argmax(0)
    durations, allocated, reserved = [], [], []
    previous = None
    for _ in range(repeats):
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        prediction = predict_volume(model, image, device, batch_size).argmax(0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        durations.append(time.perf_counter()-start)
        if previous is not None and not torch.equal(prediction, previous):
            raise RuntimeError('Repeated hard predictions differ; inspect determinism before reporting.')
        previous = prediction
        if device.type == 'cuda':
            allocated.append(torch.cuda.max_memory_allocated()/2**20)
            reserved.append(torch.cuda.max_memory_reserved()/2**20)
    return prediction, dict(inference_seconds=statistics.median(durations),
                            inference_repeat_seconds=durations,
                            peak_allocated_mib=max(allocated) if allocated else None,
                            peak_reserved_mib=max(reserved) if reserved else None,
                            repeated_predictions_identical=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('final_evaluation_plan.json'))
    parser.add_argument('--model', choices=['can3d','unet2d'], required=True)
    parser.add_argument('--manifest', type=Path, default=Path('runs/split_seed42.json'))
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--comparison-results', type=Path,
                        help='Use the frozen CAN3D report to render identical baseline cases.')
    parser.add_argument('--final-test', action='store_true')
    args = parser.parse_args(argv)
    if not args.final_test:
        parser.error('Frozen held-out evaluation requires --final-test; do not tune from results.')
    if args.model == 'unet2d' and args.comparison_results is None:
        parser.error('Baseline comparison requires the frozen CAN3D report for matched example cases.')
    plan = json.loads(args.plan.read_text())
    if plan.get('plan_version') != 1:
        raise ValueError('Unknown final evaluation plan.')
    for name, digest in plan['code_sha256'].items():
        if file_sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f'Frozen evaluation code differs: {name}')
    manifest_sha = file_sha256(args.manifest)
    if manifest_sha != plan['manifest_sha256']:
        raise ValueError('The fixed patient split changed.')
    spec = plan['models'][args.model]
    checkpoint_path = Path(spec['checkpoint'])
    if file_sha256(checkpoint_path) != spec['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed.')
    protocol = plan['protocol']
    if min(protocol['warmups_per_case'], protocol['repeats_per_case'],
           protocol['slice_batch_size'], protocol['example_count']) < 1:
        raise ValueError('Warmup, repeat, batch and example counts must be positive.')
    torch.set_num_threads(1)
    seed_everything(plan['seed'])
    device = get_device(args.device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint['config']
    if (config['mode'] != 'fit' or config['model'] != args.model
            or checkpoint['epoch'] != spec['epoch']
            or config['seed'] != plan['seed']
            or config['manifest_sha256'] != manifest_sha
            or config['shape_dhw'] != plan['shape_dhw']):
        raise ValueError('Frozen checkpoint identity, epoch or grid differs.')
    del checkpoint  # No optimizer state is put on the GPU or used here.
    data = HipMRIVolumeDataset(args.manifest, 'test', args.data_root, tuple(plan['shape_dhw']))
    if (len(data) != plan['test_case_count']
            or len({x['patient_id'] for x in data.pairs}) != plan['test_patient_count']):
        raise ValueError('Test cohort differs from the declared plan.')
    cases, comparison = [], None
    plan_sha = file_sha256(args.plan)
    if args.comparison_results:
        comparison = json.loads(args.comparison_results.read_text())
        if (args.model != 'unet2d' or comparison['model'] != 'can3d'
                or comparison['plan_sha256'] != plan_sha
                or comparison['checkpoint_sha256'] != plan['models']['can3d']['checkpoint_sha256']):
            raise ValueError('Comparison examples must come from the frozen CAN3D evaluation.')
        if {x['case_id'] for x in comparison['cases']} != {x['case_id'] for x in data.pairs}:
            raise ValueError('Models must evaluate exactly the same test cases.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'predictions').mkdir()
    (args.output/'frozen_plan.json').write_bytes(args.plan.read_bytes())
    batch = protocol['slice_batch_size'] if args.model == 'unet2d' else 1
    print(f'Frozen test: model={args.model}, epoch={spec["epoch"]}, cases={len(data)}, device={device}',flush=True)
    for index in range(len(data)):
        sample = data[index]
        prediction, profile = measure_case(model, sample['image'], device, batch,
                                           protocol['warmups_per_case'], protocol['repeats_per_case'])
        result = hard_overlap(prediction, sample['mask'])
        result.update(case_id=sample['case_id'],patient_id=sample['patient_id'],**profile)
        cases.append(result)
        mask_xyz = prediction.numpy().transpose(2,1,0).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask_xyz,sample['affine_xyz'].numpy()),
                 args.output/'predictions'/f'{sample["case_id"]}_prediction_RAS.nii.gz')
        save_json(args.output/'progress.json',dict(model=args.model,completed_cases=len(cases),
                                                  total_cases=len(data),cases=cases))
        print(f'{index+1}/{len(data)} {sample["case_id"]}: prostate Dice={result["dice"][5]}, median seconds={profile["inference_seconds"]:.4f}',flush=True)
    report = aggregate_cases(cases)
    selected = (low_scoring_patients(cases,protocol['example_count']) if comparison is None else
                [next(c for c in cases if c['case_id']==x['case_id']) for x in comparison['selected_examples']])
    peak_a = [x['peak_allocated_mib'] for x in cases if x['peak_allocated_mib'] is not None]
    peak_r = [x['peak_reserved_mib'] for x in cases if x['peak_reserved_mib'] is not None]
    report.update(model=args.model,split='test',cases=cases,label_names=LABEL_NAMES,
                  checkpoint_sha256=spec['checkpoint_sha256'],checkpoint_epoch=spec['epoch'],
                  manifest_sha256=manifest_sha,plan_sha256=plan_sha,shape_dhw=plan['shape_dhw'],
                  protocol=protocol,selected_examples=selected,training_config=config,
                  selection_scope='Lowest CAN3D prostate Dice; one scan per distinct test patient; same cases for baseline',
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  mean_case_median_seconds=statistics.mean(x['inference_seconds'] for x in cases),
                  median_case_median_seconds=statistics.median(x['inference_seconds'] for x in cases),
                  peak_allocated_mib=max(peak_a) if peak_a else None,
                  peak_reserved_mib=max(peak_r) if peak_r else None,
                  device=str(device),gpu_name=torch.cuda.get_device_name() if device.type=='cuda' else None,
                  hostname=socket.gethostname(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  python=platform.python_version(),torch_version=str(torch.__version__),
                  nibabel_version=nib.__version__,numpy_version=np.__version__,cpu_threads=torch.get_num_threads(),
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  timing_scope='Preprocessed CPU image -> complete CPU hard prediction; includes transfers, model and argmax; excludes disk/preprocessing/metrics/plots',
                  memory_scope='PyTorch peak allocated/reserved after per-case warmup; model plus inference tensors; excludes driver/context and other processes',
                  accuracy_scope='All frozen test cases; patient-macro primary, scan-macro secondary; no training or post-processing changes')
    save_json(args.output/'test_metrics.json',report)
    (args.output/'examples').mkdir()
    for item in selected:
        index = next(i for i,x in enumerate(data.pairs) if x['case_id']==item['case_id'])
        sample = data[index]
        saved = nib.load(args.output/'predictions'/f'{item["case_id"]}_prediction_RAS.nii.gz')
        prediction = torch.from_numpy(np.asarray(saved.dataobj).transpose(2,1,0).copy()).long()
        save_panels(sample,prediction,args.output/'examples'/f'{item["case_id"]}_prostate.png')
    print(f'{args.model}: patient-macro test prostate Dice={report["patient_macro_dice"][5]:.6f}, IoU={report["patient_macro_iou"][5]:.6f}',flush=True)
    print(f'{args.model}: mean case-median seconds={report["mean_case_median_seconds"]:.4f}, peak allocated MiB={report["peak_allocated_mib"]}',flush=True)
    print('FROZEN_TEST_MODEL_COMPLETED (no optimizer or training updates; do not tune from these results).',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
