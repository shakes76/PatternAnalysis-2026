"""Frozen 2D test evaluation and exact shared-grid comparison with frozen 3D.

Native 60-slice scores and resized 30-plane scores are separate. Shared GT
must match voxel-for-voxel. Checkpoints are hash checked before/after inference.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import time

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import HipMRI2DDataset, get_data_pairs, load_patient_split
from modules import UNet2D
from support.evaluation_metrics import class_metrics, geometry, hd95, summarize


FIXED_CASES = ['B037_Week0_LFOV.nii.gz', 'L011_Week2_LFOV.nii.gz',
               'H017_Week3_LFOV.nii.gz', 'K042_Week0_LFOV.nii.gz',
               'H007_Week0_LFOV.nii.gz']
SHAPE = (64, 128, 128)
SHARED = slice(49, 79)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def freeze_checkpoint(source, destination, expected_sha256):
    """Preserve an exact read-only copy; never replace a different snapshot."""
    source, destination = Path(source), Path(destination)
    if digest(source) != expected_sha256:
        raise ValueError('Selected checkpoint source hash differs')
    if not destination.exists():
        destination.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as stream, destination.open('xb') as target:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                target.write(chunk)
    if digest(destination) != expected_sha256:
        raise ValueError('Frozen checkpoint hash differs')
    destination.chmod(0o444)
    return destination


def native_dhw(stack):
    """[slice, reversed X, Z] -> [Z, X, slice], after removing padding."""
    stack = torch.as_tensor(np.ascontiguousarray(stack))
    if stack.ndim != 3 or stack.shape[0] != 60 or stack.shape[1] != 256:
        raise ValueError(f'Expected [60,256,Z], got {tuple(stack.shape)}')
    if stack.shape[2] not in (128, 144):
        raise ValueError('Unexpected original Z size')
    return stack.flip(1).permute(2, 1, 0).contiguous()


def align_shared(stack):
    """Nearest hard-label mapping; flip before resize to avoid parity shift."""
    volume = native_dhw(stack)
    return F.interpolate(volume[None, None].float(), size=(64, 128, 30),
                         mode='nearest')[0, 0].to(torch.uint8).numpy()


def load_reference_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_reference_model(code_dir, checkpoint, device='cuda'):
    # Isolate the reference architecture module from the 2D `modules` import.
    module = load_reference_module(code_dir/'modules.py', 'frozen_3d_components')
    config = checkpoint['model_config']
    if config != dict(base_channels=8, normalization='instance'):
        raise ValueError('Unexpected reference model configuration')
    model = module.UNet3D(in_channels=1, num_classes=6, base_channels=8)
    def replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.BatchNorm3d):
                setattr(parent, name, nn.InstanceNorm3d(
                    child.num_features, eps=child.eps, affine=True,
                    track_running_stats=False))
            else:
                replace(child)
    replace(model)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.eval().to(device)


def metric_rows(prediction, target, case, patient, spacing):
    distance, status = hd95(prediction == 5, target == 5, spacing)
    bladder = int(((prediction == 5) & (target == 3)).sum())
    rectum = int(((prediction == 5) & (target == 4)).sum())
    rows = class_metrics(prediction, target)
    for row in rows:
        prostate = row['class_id'] == 5
        row.update(case=case, patient_id=patient,
                   hd95_mm=distance if prostate else None,
                   hd95_status=status if prostate else '',
                   prostate_fp_in_bladder=bladder if prostate else 0,
                   prostate_fp_in_rectum=rectum if prostate else 0)
    return rows


def scope_summary(rows):
    prostate = [r for r in rows if r['class_id'] == 5]
    distances = [r['hd95_mm'] for r in prostate if r['hd95_status'] == 'ok']
    return dict(class_metrics=summarize(rows), scans=len(prostate),
                patients=len({r['patient_id'] for r in prostate}),
                prostate_mean_hd95_mm=float(np.mean(distances)) if distances else None,
                prostate_hd95_valid_cases=len(distances),
                prostate_hd95_invalid_cases=len(prostate)-len(distances),
                prostate_fp_in_bladder=sum(r['prostate_fp_in_bladder'] for r in prostate),
                prostate_fp_in_rectum=sum(r['prostate_fp_in_rectum'] for r in prostate))


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--reference-checkpoint', required=True)
    p.add_argument('--reference-sha256', required=True)
    p.add_argument('--reference-code-dir', required=True)
    p.add_argument('--reference-results', required=True)
    p.add_argument('--split-manifest', required=True)
    p.add_argument('--data-dir', default='/home/groups/comp3710/HipMRI_Study_open')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--threads', type=int, default=1)
    return p


@torch.inference_mode()
def main(argv=None):
    args = parser().parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError('batch-size and threads must be positive')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Use a BF16-capable Slurm GPU node for the frozen 3D reference')
    torch.set_num_threads(args.threads); torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    started = time.perf_counter()
    for path, expected in [(args.checkpoint, args.expected_sha256),
                           (args.reference_checkpoint, args.reference_sha256)]:
        if digest(path) != expected:
            raise ValueError(f'Checkpoint SHA256 mismatch: {path}')
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=False)
    (out/'evaluation_config.json').write_text(json.dumps(vars(args),indent=2)+'\n',encoding='utf-8')
    frozen = out/'frozen_checkpoint.pt'
    freeze_checkpoint(args.checkpoint, frozen, args.expected_sha256)
    checkpoint = torch.load(frozen, map_location='cpu', weights_only=True)
    split_document = json.loads(Path(args.split_manifest).read_text(encoding='utf-8'))
    (out/'patient_split.json').write_text(json.dumps(split_document,indent=2)+'\n',encoding='utf-8')
    if checkpoint['epoch'] != 5 or checkpoint.get('smoke_only'):
        raise ValueError('Expected formal epoch-5 checkpoint')
    if checkpoint['config']['patient_split'] != split_document:
        raise ValueError('Checkpoint split differs from supplied manifest')
    groups = load_patient_split(args.split_manifest)
    model = UNet2D(**checkpoint['model_config']).eval().cuda()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    ref_dir = Path(args.reference_code_dir)
    ref_data = load_reference_module(ref_dir/'dataset.py', 'frozen_3d_dataset')
    all_reference = ref_data.get_data_pairs(Path(args.data_dir))
    reference_groups = ref_data.split_by_patient(all_reference, seed=42)
    for key, pairs in zip(('train', 'validation', 'test'), reference_groups):
        if {patient[1:] for patient, _, _ in pairs} != groups[key]:
            raise ValueError(f'2D/3D patient split mismatch: {key}')
    pairs = reference_groups[2]
    if len(pairs) != 27 or len(groups['test']) != 6:
        raise ValueError('Expected 27 scans from six test patients')
    reference_summary = json.loads((Path(args.reference_results)/'summary.json').read_text())
    if reference_summary['checkpoint_sha256'] != args.reference_sha256:
        raise ValueError('Archived 3D reference checkpoint differs')
    archived_full = {(r['case'], int(r['class_id'])): r for r in
                     _read_csv(Path(args.reference_results)/'per_case_metrics.csv')}
    archived_shared = {(r['case'], int(r['class_id'])): r for r in
                       _read_csv(Path(args.reference_results)/'shared_2d_region_metrics.csv')}
    ref_checkpoint = torch.load(args.reference_checkpoint, map_location='cpu', weights_only=True)
    if tuple(ref_checkpoint['training_config']['target_shape']) != SHAPE:
        raise ValueError('Unexpected frozen 3D preprocessing shape')
    reference_model = make_reference_model(ref_dir, ref_checkpoint)
    code_root = Path(__file__).resolve().parent
    source_hashes = {name:digest(code_root/name) for name in
                     ('evaluate.py','predict.py','support/evaluation_metrics.py','dataset.py','modules.py')}
    source_hashes.update({f'reference_3d/{name}':digest(ref_dir/name)
                          for name in ('dataset.py','modules.py')})
    reference_dataset = ref_data.HipMRIDataset(pairs, target_shape=SHAPE)
    records = get_data_pairs(args.data_dir)
    by_scan = {}
    for record in records:
        if record['patient_id'] in groups['test']:
            by_scan.setdefault((record['patient_id'], record['week_id']), []).append(record)
    expected_keys = {(patient[1:], int(re.search(r'_Week(\d+)_', mri.name)[1]))
                     for patient, mri, _ in pairs}
    if set(by_scan) != expected_keys:
        raise ValueError('2D/3D test scan identity mismatch')
    rows_native, rows_shared, rows_reference, paired, caches = [], [], [], [], []
    private = out/'native_predictions'; private.mkdir()
    cache_dir = out/'prediction_inputs'; cache_dir.mkdir()
    print(f'Frozen 2D: epoch=5 SHA256={args.expected_sha256}; 27 scans / 1620 slices', flush=True)
    for index, (patient, mri_path, _) in enumerate(pairs):
        week = int(re.search(r'_Week(\d+)_', mri_path.name)[1])
        samples = sorted(by_scan[(patient[1:], week)], key=lambda r: r['slice_id'])
        if [r['slice_id'] for r in samples] != list(range(60)):
            raise ValueError(f'Missing/duplicate slice in {mri_path.name}')
        dataset = HipMRI2DDataset(args.data_dir, samples=samples)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        prediction_slices, target_slices = [], []
        for images, targets, metadata in loader:
            # FP32 matches 2D training validation; no mixed-precision substitution.
            predictions = model(images.cuda()).argmax(1).cpu()
            for i in range(len(images)):
                width = int(metadata['original_w'][i])
                prediction_slices.append(predictions[i, :, :width].numpy().astype(np.uint8))
                target_slices.append(targets[i, :, :width].numpy().astype(np.uint8))
        native_prediction, native_target = np.stack(prediction_slices), np.stack(target_slices)
        np.savez_compressed(private/(mri_path.stem.replace('.nii','')+'.npz'),
                            prediction=native_prediction, target=native_target)
        geo = geometry(mri_path, SHAPE)
        if geo['original_shape'][1] != 256 or geo['shared_target_w_indices'] != list(range(49,79)):
            raise ValueError('Source geometry differs from fixed shared-grid protocol')
        if geo['original_shape'][0] != 256 or geo['original_shape'][2] != native_target.shape[2]:
            raise ValueError('Original 3D dimensions differ from native 2D slices')
        native_spacing = nib.affines.voxel_sizes(nib.load(str(mri_path)).affine)[[2,0,1]]
        native_rows = metric_rows(native_dhw(native_prediction).numpy(),
                                  native_dhw(native_target).numpy(), mri_path.name,
                                  patient, native_spacing)
        rows_native.extend(native_rows)
        aligned_prediction, aligned_target = align_shared(native_prediction), align_shared(native_target)
        image, reference_target = reference_dataset[index]
        target = reference_target.numpy().astype(np.uint8)
        shared_target = target[:, :, SHARED]
        mismatch = int(np.count_nonzero(aligned_target != shared_target))
        if mismatch:
            (out/'alignment_failure.json').write_text(json.dumps(dict(
                case=mri_path.name, mismatched_voxels=mismatch), indent=2))
            raise ValueError(f'{mri_path.name}: shared GT differs at {mismatch} voxels')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            reference_prediction = reference_model(image[None].cuda()).argmax(1)[0].cpu().numpy().astype(np.uint8)
        # Recomputed reference must reproduce the archived overlap counts.
        for scope, pred, truth, archived in [
                ('full', reference_prediction, target, archived_full),
                ('shared', reference_prediction[:,:,SHARED], shared_target, archived_shared)]:
            for row in class_metrics(pred, truth):
                frozen_row = archived[(mri_path.name, row['class_id'])]
                if any(row[k] != int(frozen_row[k]) for k in ('tp','fp','fn')):
                    raise ValueError(f'{mri_path.name}: recomputed 3D {scope} counts differ from archive')
        shared_rows = metric_rows(aligned_prediction, shared_target, mri_path.name,
                                  patient, geo['spacing_dhw_mm'])
        reference_rows = metric_rows(reference_prediction[:,:,SHARED], shared_target,
                                     mri_path.name, patient, geo['spacing_dhw_mm'])
        rows_shared.extend(shared_rows); rows_reference.extend(reference_rows)
        touches = bool((shared_target[:,:,0] == 5).any() or (shared_target[:,:,-1] == 5).any())
        paired.append(dict(case=mri_path.name, patient_id=patient, gt_mismatch_voxels=0,
                           gt_touches_shared_W_boundary=touches,
                           prediction_2d_touches_shared_W_boundary=bool(
                               (aligned_prediction[:,:,0] == 5).any() or (aligned_prediction[:,:,-1] == 5).any()),
                           prediction_3d_touches_shared_W_boundary=bool(
                               (reference_prediction[:,:,49] == 5).any() or (reference_prediction[:,:,78] == 5).any()),
                           native_2d_dice=native_rows[5]['dice'], shared_2d_dice=shared_rows[5]['dice'],
                           shared_3d_dice=reference_rows[5]['dice'],
                           delta_2d_minus_3d=shared_rows[5]['dice']-reference_rows[5]['dice']))
        if mri_path.name in FIXED_CASES:
            full_prediction = np.full(SHAPE, 255, dtype=np.uint8)
            full_prediction[:,:,SHARED] = aligned_prediction
            metadata = dict(case=mri_path.name, original_axcodes=geo['original_axcodes'],
                            native_2d_dice=native_rows[5]['dice'],
                            shared_2d_dice=shared_rows[5]['dice'], shared_3d_dice=reference_rows[5]['dice'],
                            shared_2d_hd95_mm=shared_rows[5]['hd95_mm'],
                            shared_3d_hd95_mm=reference_rows[5]['hd95_mm'])
            name=mri_path.name.replace('.nii.gz','.npz')
            np.savez_compressed(cache_dir/name, image=image[0].numpy(), target=target,
                                prediction_2d=full_prediction, prediction_3d=reference_prediction,
                                metadata=np.asarray(json.dumps(metadata)))
            caches.append(dict(case=mri_path.name, cache_file=name))
        print(f'{index+1}/27 {mri_path.name}: native2D={native_rows[5]["dice"]:.5f}; '
              f'shared2D={shared_rows[5]["dice"]:.5f}; shared3D={reference_rows[5]["dice"]:.5f}; GT identical', flush=True)
    for path, expected in [(args.checkpoint,args.expected_sha256),
                           (args.reference_checkpoint,args.reference_sha256), (frozen,args.expected_sha256)]:
        if digest(path) != expected:
            raise ValueError('Checkpoint changed during evaluation')
    if {row['case'] for row in caches} != set(FIXED_CASES):
        raise ValueError('Fixed paired cases missing')
    write_csv(out/'native_2d_per_case_metrics.csv', rows_native)
    write_csv(out/'shared_2d_per_case_metrics.csv', rows_shared)
    write_csv(out/'reference_3d_shared_metrics.csv', rows_reference)
    write_csv(out/'paired_cases.csv', paired)
    native_summary = scope_summary(rows_native)
    native_dice = native_summary['class_metrics'][5]['mean_case_dice']
    summary = dict(checkpoint=str(frozen), checkpoint_sha256=args.expected_sha256,
                   checkpoint_epoch=5, model_config=checkpoint['model_config'],
                   reference_checkpoint_sha256=args.reference_sha256,
                   test_evaluated=True, patients=6, scans=27, slices=1620,
                   parameters_2d=sum(p.numel() for p in model.parameters()),
                   native_2d=native_summary, shared_2d=scope_summary(rows_shared),
                   shared_3d=scope_summary(rows_reference),
                   native_2d_prostate_target=0.75,
                   native_2d_target_met=bool(native_dice >= 0.75),
                   shared_gt_all_voxels_equal=True, reference_counts_match_archive=True,
                   shared_boundary_touches=dict(
                       gt_scans=sum(r['gt_touches_shared_W_boundary'] for r in paired),
                       prediction_2d_scans=sum(r['prediction_2d_touches_shared_W_boundary'] for r in paired),
                       prediction_3d_scans=sum(r['prediction_3d_touches_shared_W_boundary'] for r in paired)),
                   shared_grid=[64,128,30], shared_W_indices=list(range(49,79)),
                   native_scope='All 60 unpadded supplied slices; native spacing from original 3D affine',
                   hd95_definition='P95 pooled bidirectional six-connected surface distances; shared spacing approximate',
                   precision=dict(model_2d='FP32', model_3d='BF16'),
                   gpu=torch.cuda.get_device_name(),
                   software=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__),
                   source_files_sha256=source_hashes,
                   elapsed_seconds=time.perf_counter()-started,
                   note='No training/tuning. Different coverage, resolution, preprocessing and model capacity remain disclosed.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (cache_dir/'manifest.json').write_text(json.dumps(dict(cases=sorted(
        caches,key=lambda r:FIXED_CASES.index(r['case']))),indent=2),encoding='utf-8')
    (out/'evaluation_complete.txt').write_text('Frozen native/shared test evaluation completed.\n')
    print(json.dumps(dict(native_2d_dice=native_dice,target_met=summary['native_2d_target_met'],
                          shared_2d_dice=summary['shared_2d']['class_metrics'][5]['mean_case_dice'],
                          shared_3d_dice=summary['shared_3d']['class_metrics'][5]['mean_case_dice']),indent=2),flush=True)
    return out


def _read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


if __name__ == '__main__':
    main()
