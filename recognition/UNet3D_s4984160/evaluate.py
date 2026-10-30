"""Evaluate the frozen 3D checkpoint; no training or test-driven selection."""
import argparse
import csv
from importlib.metadata import version
import json
import platform
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from dataset import HipMRIDataset, get_data_pairs, split_by_patient
from support.evaluation_utils import (boundary_regions, checkpoint_sha256, class_metrics,
                              geometry, hd95, load_model, select_examples, summarize)


def save_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_split(data_root, name, manifest_path):
    all_pairs = get_data_pairs(Path(data_root))
    train, validation, test = split_by_patient(all_pairs, seed=42)
    frozen = json.loads(Path(manifest_path).read_text())
    for key, pairs in [('train',train), ('validation',validation)]:
        observed = {(p[0],p[1].name) for p in pairs}
        expected = {(r['patient'],r['case']) for r in frozen[key]}
        if observed != expected:
            raise ValueError(f'{key} differs from the frozen training manifest')
    train_ids, val_ids, test_ids = ({p[0] for p in group} for group in (train,validation,test))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError('Patient leakage detected')
    return validation if name == 'validation' else test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['validation','test'], default='test')
    parser.add_argument('--split-manifest', required=True)
    parser.add_argument('--data-root', default='/home/groups/comp3710/HipMRI_Study_open')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--timing-repeats', type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Run on a Slurm GPU node')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This evaluation matches the BF16 validation precision and requires BF16 support')
    if args.timing_repeats < 1:
        raise ValueError('timing-repeats must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    started = time.perf_counter()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model, checkpoint = load_model(args.checkpoint, 'cuda')
    shape = tuple(checkpoint['training_config']['target_shape'])
    pairs = select_split(args.data_root, args.split, args.split_manifest)
    if not pairs:
        raise ValueError('Empty evaluation split')
    data = HipMRIDataset(pairs, target_shape=shape)
    digest = checkpoint_sha256(args.checkpoint)
    print(f'Frozen checkpoint: {args.checkpoint}; SHA256={digest}', flush=True)
    print(f'Split={args.split}; scans={len(pairs)}; patients={len({p[0] for p in pairs})}', flush=True)
    all_rows, shared_rows, edge_rows, timings, example_cache = [], [], [], [], {}
    peak_gb = 0.
    with torch.inference_mode():
        for index, (patient, mri_path, _) in enumerate(pairs):
            image, target_tensor = data[index]
            gpu_image = image.unsqueeze(0).cuda()
            if index == 0:
                for _ in range(3):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        model(gpu_image)
                torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(args.timing_repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(gpu_image)
                torch.cuda.synchronize()
                durations.append((time.perf_counter()-start)*1000)
            prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            target = target_tensor.numpy().astype(np.uint8)
            peak_gb = max(peak_gb,torch.cuda.max_memory_allocated()/1024**3)
            latency = float(np.median(durations)); timings.append(latency)
            geo = geometry(mri_path, shape)
            distance, distance_status = hd95(prediction==5,target==5,geo['spacing_dhw_mm'])
            in_bladder = int(((prediction==5) & (target==3)).sum())
            in_rectum = int(((prediction==5) & (target==4)).sum())
            rows = class_metrics(prediction,target)
            for row in rows:
                row.update(case=mri_path.name, patient_id=patient,
                           hd95_mm=distance if row['class_id']==5 else None,
                           hd95_status=distance_status if row['class_id']==5 else '',
                           prostate_fp_in_bladder=in_bladder if row['class_id']==5 else 0,
                           prostate_fp_in_rectum=in_rectum if row['class_id']==5 else 0,
                           inference_latency_ms=latency)
            all_rows.extend(rows)
            shared = geo['shared_target_w_indices']
            if not shared:
                raise ValueError('No overlap with the mapped course 2D slice band')
            for row in class_metrics(prediction[:,:,shared],target[:,:,shared]):
                shared_rows.append(dict(row,case=mri_path.name,patient_id=patient))
            for row in boundary_regions(prediction,target,geo['original_axcodes'][2]):
                edge_rows.append(dict(row,case=mri_path.name,patient_id=patient))
            example_cache[mri_path.name] = (image[0].numpy(),target,prediction,
                dict(case=mri_path.name,patient_id=patient,split=args.split,**geo,
                     target_shape=list(shape),prostate_dice=rows[5]['dice'],
                     prostate_hd95_mm=distance,hd95_status=distance_status))
            print(f'{index+1}/{len(pairs)} {mri_path.name}: prostate Dice={rows[5]["dice"]}; HD95={distance} ({distance_status})', flush=True)
            del logits, gpu_image

    prostate_rows = [r for r in all_rows if r['class_id']==5]
    selected = select_examples(prostate_rows)
    cache_dir = output/'prediction_inputs'
    cache_dir.mkdir()
    for row in selected:
        image,target,prediction,meta = example_cache[row['case']]
        meta['selection_reason'] = row['reason']
        filename = row['case'].replace('.nii.gz','')+'.npz'
        np.savez_compressed(cache_dir/filename,image=image,target=target,prediction=prediction,
                            metadata=np.asarray(json.dumps(meta)))
        row['cache_file'] = filename
    (cache_dir/'manifest.json').write_text(json.dumps(dict(
        checkpoint=args.checkpoint,checkpoint_sha256=digest,split=args.split,cases=selected),indent=2),encoding='utf-8')
    save_csv(output/'per_case_metrics.csv',all_rows)
    save_csv(output/'shared_2d_region_metrics.csv',shared_rows)
    if edge_rows:
        save_csv(output/'boundary_extent_metrics.csv',edge_rows)
    hd = [r['hd95_mm'] for r in prostate_rows if r['hd95_status']=='ok']
    classes = summarize(all_rows)
    bladder_gt,rectum_gt = classes[3]['gt_voxels'],classes[4]['gt_voxels']
    summary = dict(
        checkpoint=args.checkpoint,checkpoint_sha256=digest,
        checkpoint_epoch=checkpoint['epoch'],model_config=checkpoint['model_config'],
        cumulative_training_epochs=checkpoint['training_config'].get('source_epoch',0)+checkpoint['epoch'],
        split=args.split,scans=len(pairs),patients=len({p[0] for p in pairs}),
        target_shape=list(shape),class_metrics=classes,
        prostate_mean_hd95_mm=float(np.mean(hd)) if hd else None,
        prostate_hd95_valid_cases=len(hd),prostate_hd95_invalid_cases=len(pairs)-len(hd),
        hd95_definition='P95 pooled bidirectional six-connected surface distances; resized-space approximate mm',
        prostate_fp_in_bladder=sum(r['prostate_fp_in_bladder'] for r in prostate_rows),
        prostate_fp_in_rectum=sum(r['prostate_fp_in_rectum'] for r in prostate_rows),
        bladder_gt_voxels=bladder_gt,rectum_gt_voxels=rectum_gt,
        shared_2d_region=dict(original_nifti_axis=1,original_indices_inclusive=[98,157],
            note='Same resized 3D grid, selecting W by nearest-mask source indices; partial band, not full-volume evaluation',
            class_metrics=summarize(shared_rows)),
        boundary_region_note='First/last two GT-bearing depth planes: extent proxies, not independently verified clinical landmarks',
        parameters=sum(p.numel() for p in model.parameters()),
        peak_allocated_gpu_gb=peak_gb,mean_forward_latency_ms=float(np.mean(timings)),
        median_forward_latency_ms=float(np.median(timings)),timing_repeats=args.timing_repeats,
        timing_note='BF16; 3 warmups; per-case median of forward timings; excludes loading, metrics, transfer and plots',
        hardware=dict(gpu=torch.cuda.get_device_name(),threads=args.threads),
        software=dict(python=platform.python_version(),torch=torch.__version__,
                      numpy=np.__version__,nibabel=nib.__version__,
                      scipy=version('scipy'),matplotlib=version('matplotlib')),
        elapsed_seconds=time.perf_counter()-started,
        evaluation_note='No training or checkpoint changes. Do not tune on these held-out test results.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    (output/'evaluation_complete.txt').write_text('Evaluation completed successfully.\n')
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
