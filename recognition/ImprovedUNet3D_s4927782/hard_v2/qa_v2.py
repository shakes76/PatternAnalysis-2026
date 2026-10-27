"""Real TRAIN-only gate before the single finite v2 formal comparison.

Independently reconstruct image preprocessing, exercise complete image-only
inference/native restoration with untrained models, then require fixed-budget
TRAIN-crop memorisation in BOTH model families. No VAL or TEST pixel is read.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

import infer_native as adapter
import small_fit
import v2_pipeline as v2


def validate_original_through(old, plan):
    """Bind the preceding v1 check to the actual frozen sources and models."""
    required = plan['required_original_sha256']
    review = plan['required_review_sha256']
    if (old.get('status') != 'passed'
            or old.get('selected_case_keys') != plan['qa']['through_cases']
            or old.get('manifest_sha256') != required['artifacts/cache64/manifest.json']
            or old.get('original_modules_sha256') != required['modules.py']
            or old.get('original_utils_sha256') != required['utils.py']
            or old.get('adapter_sha256') != review['infer_native.py']
            or old.get('verifier_sha256') != review['real_pipeline_check.py']
            or old.get('scope') != dict(raw_images=2, cached_train_masks=2,
                                      models_per_case=2, test_arrays_read=0, raw_masks_read=0)
            or old.get('generalisation_evidence') is not False):
        raise ValueError('Original TRAIN through-check source/model/scope identity is not passed')
    cases = old.get('cases', [])
    if (len(cases) != 2 or [[r.get('patient'), r.get('visit')] for r in cases] != plan['qa']['through_cases']):
        raise ValueError('Original TRAIN through-check cases differ from the fixed QA scope')
    for case in cases:
        if (case.get('split') != 'train' or case.get('preprocessing_elementwise_equal') is not True
                or set(case.get('models', {})) != set(v2.MODELS)):
            raise ValueError('Original TRAIN image-only parity check is incomplete')
        for name in v2.MODELS:
            measured = case['models'][name]
            expected = required[f'artifacts/full_seed42/{name}/best.pt']
            if (measured.get('checkpoint_sha256') != expected
                    or measured.get('cache_prediction_elementwise_equal') is not True
                    or measured.get('native_affine_matches_image') is not True):
                raise ValueError('Original TRAIN checkpoint/prediction/native-geometry check is incomplete')


def check_real_sampler_batch(model_name, manifest_path, manifest, plan, deps, device):
    """One real minibatch/backward QA; no optimizer and no candidate is saved.

    Limit the new training dataset's volume inventory to the same two fixed
    TRAIN visits before any __getitem__ call. Metadata may contain the remaining
    TRAIN class counts; no other TRAIN/VAL/TEST pixel arrays are opened here.
    """
    cfg = plan['configuration']
    allowed = {tuple(key) for key in plan['qa']['through_cases']}
    dataset = v2.SampledTrainingDataset(manifest_path, model_name, cfg, deps, manifest)
    dataset.volumes.records = [r for r in dataset.volumes.records
                              if (r['patient'], r['visit']) in allowed]
    if ({(r['patient'], r['visit']) for r in dataset.volumes.records} != allowed
            or len(dataset.volumes.records) != len(allowed)):
        raise ValueError('Real sampler QA must contain precisely the two fixed TRAIN visits')
    dataset.volumes.cache.clear()
    dataset.set_epoch(1)
    size = cfg['train_batch_size_2d'] if model_name == 'unet2d' else cfg['train_batch_size_3d']
    loader = torch.utils.data.DataLoader(dataset, batch_size=size, shuffle=False,
                                        num_workers=0, drop_last=False)
    batch = next(iter(loader))
    images = batch['image'].to(device)
    targets = batch['mask'].to(device)
    expected_spatial = tuple(cfg['target_shape_dhw'][1:]) if model_name == 'unet2d' else tuple(cfg['patch_shape_dhw'])
    if (tuple(images.shape) != (size, 1, *expected_spatial)
            or tuple(targets.shape) != (size, *expected_spatial)
            or not bool(torch.isfinite(images).all()) or targets.dtype != torch.int64
            or int(targets.min()) < 0 or int(targets.max()) >= cfg['num_classes']):
        raise ValueError('Real sampled TRAIN minibatch has invalid shape/intensities/classes')
    counts = np.sum([r['class_voxel_counts'] for r in manifest['records'] if r['split'] == 'train'], axis=0).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError('TRAIN metadata has undefined loss weights')
    weights = 1 / np.sqrt(counts / counts.sum())
    weights /= weights.mean(); weights = np.clip(weights, .05, 10); weights /= weights.mean()
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    v2.seed_everything(plan['qa']['fit_seed'])
    model = v2.create_model(model_name, cfg, deps).to(device).train()
    amp = bool(cfg['amp'] and device.type == 'cuda')
    with torch.autocast(device_type=device.type, enabled=amp):
        output = model(images)
        loss = deps.modules.segmentation_loss(output, targets, class_weights=weights)
    if (tuple(output['logits'].shape) != (size, cfg['num_classes'], *expected_spatial)
            or not bool(torch.isfinite(loss))):
        raise ValueError('Real sampled TRAIN forward/loss is invalid')
    # No scaler or optimizer is created: this gate checks an unscaled finite
    # backward pass without introducing an extra model-training experiment.
    loss.backward()
    if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
           for parameter in model.parameters()):
        raise ValueError('Real sampled TRAIN backward has missing/nonfinite gradients')
    auxiliary = []
    if model_name == 'improved3d':
        if len(output['aux_logits']) != 2 or len(model.aux_heads) != 2:
            raise ValueError('The Hard model is missing its two deep-supervision heads')
        for head in model.aux_heads:
            total = float(head.weight.grad.detach().abs().sum())
            if not np.isfinite(total) or total <= 0:
                raise ValueError('A real sampled TRAIN auxiliary head has no nonzero finite gradient')
            auxiliary.append(total)
    actual_indices = sorted(set(batch['record_index'].tolist()))
    result = dict(passed=True, purpose='One-batch development software QA only; no optimizer/candidate',
                  inventory_case_keys=[list(key) for key in sorted(allowed)],
                  actual_batch_case_keys=[[dataset.volumes.records[i]['patient'], dataset.volumes.records[i]['visit']]
                                          for i in actual_indices],
                  epoch=1, image_shape=list(images.shape), target_shape=list(targets.shape),
                  observed_class_ids=[int(c) for c in torch.unique(targets).cpu().tolist()],
                  loss=float(loss.detach()), amp=amp, unscaled_backward=True,
                  all_parameter_gradients_present_and_finite=True,
                  auxiliary_weight_gradient_absolute_sums=auxiliary,
                  class_weight_source='All TRAIN new-cache manifest counts only; no other pixel arrays loaded',
                  test_or_validation_pixels_read=0)
    model.zero_grad(set_to_none=True)
    del model, output, loss, batch, images, targets, dataset, loader, weights
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--original-through',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    plan=v2.load_protocol(a.protocol)
    v2.verify_review_sources(plan,a.protocol.parent)
    deps=v2.load_dependencies(a.project_root,plan)
    manifest=v2.load_new_manifest(a.manifest,plan,deps)
    old=json.loads(a.original_through.read_text())
    validate_original_through(old,plan)
    if not torch.cuda.is_available(): raise ValueError('GPU QA was requested but CUDA is absent')
    torch.set_num_threads(1)
    device=torch.device('cuda')
    a.output_dir.mkdir(parents=True)
    report=dict(status='running',protocol_sha256=adapter.sha256(a.protocol),
                manifest_sha256=adapter.sha256(a.manifest),
                original_through_receipt_sha256=adapter.sha256(a.original_through),
                scope='Only two fixed TRAIN raw images/cache masks; no validation or test pixels; no selection',
                models={},cases=[],gpu_name=torch.cuda.get_device_name(0))
    def save():
        (a.output_dir/'qa_receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        volumes=v2.CachedVolumes(a.manifest,manifest,'train')
        chosen=[]
        for key in plan['qa']['through_cases']:
            i=next(i for i,r in enumerate(volumes.records) if [r['patient'],r['visit']]==key)
            r=volumes.records[i]
            im,mask,spacing=volumes.load(i)
            image_path=a.data_root/r['source_image']
            image,original,geometry=adapter.preprocess_image(image_path,tuple(plan['configuration']['target_shape_dhw']))
            if not np.array_equal(im,image): raise ValueError('Independent image-only new cache reconstruction differs')
            if not np.allclose(geometry['cache_affine_xyz'],r['resampled_affine_xyz'],rtol=0,atol=1e-8):
                raise ValueError('New cache affine differs')
            report['cases'].append(dict(patient=r['patient'],visit=r['visit'],image_elementwise_equal=True,
                                        source_image_sha256=adapter.sha256(image_path),cache_sha256=adapter.sha256(a.manifest.parent/r['cache_path'])))
            chosen.append((r,im,mask,original,geometry))
        # Fresh untrained models exercise full image-only prediction. Their
        # scores are plumbing checks, deliberately not candidate evidence.
        r,im,mask,original,geometry=chosen[0]
        for name in v2.MODELS:
            sampled_batch = check_real_sampler_batch(name,a.manifest,manifest,plan,deps,device)
            # Start complete untrained inference afresh from the same fixed
            # initialization after the backward graph/gradients are released.
            v2.seed_everything(plan['qa']['fit_seed'])
            model=v2.create_model(name,plan['configuration'],deps).to(device).eval()
            pred=v2.predict_image(model,torch.from_numpy(im).unsqueeze(0),name,device,plan['configuration']).numpy()
            if pred.shape!=im.shape or pred.min()<0 or pred.max()>=6: raise ValueError('Incomplete new-grid inference')
            own=small_fit.metrics(pred,mask)
            original_metrics=deps.utils.compute_case_metrics(pred,mask,6)
            for c,row in enumerate(own['per_class']):
                if row['dice']!=original_metrics['per_class'][c]['dice']: raise ValueError('Independent new-grid metric mismatch')
            adapter.save_outputs(a.output_dir/('untrained_full_image_'+name),pred,original,geometry,list(range(6)),
                                 dict(purpose='Untrained TRAIN pipeline QA only',candidate=False))
            report['models'][name]=dict(full_image_only_inference_passed=True,native_geometry_serialisation_passed=True,
                                       independent_metric_check_passed=True,sampler_batch_backward_passed=True,
                                       real_sampler_batch=sampled_batch,passed=False)
            save()
            del model
            gc.collect(); torch.cuda.empty_cache()
        subprocess.run([sys.executable,str(Path(__file__).with_name('small_fit.py')),
                        '--project-root',str(a.project_root),'--protocol',str(a.protocol),
                        '--manifest',str(a.manifest),'--output-dir',str(a.output_dir/'fit'),
                        '--device','cuda'],check=True)
        fit=json.loads((a.output_dir/'fit/fit_receipt.json').read_text())
        for name in v2.MODELS:
            report['models'][name]['passed']=fit['models'][name]['passed'] is True
            report['models'][name]['final_fit_prostate_dice']=fit['models'][name]['history'][-1]['prostate_dice']
            report['models'][name]['final_fit_foreground_dice']=fit['models'][name]['history'][-1]['foreground_dice']
        report['fit_receipt_sha256']=adapter.sha256(a.output_dir/'fit/fit_receipt.json')
        report['status']='passed' if fit['status']=='passed' and all(z['passed'] for z in report['models'].values()) else 'failed_fit_gate'
        report['generalisation_evidence']=False
        save()
        if report['status']!='passed': raise RuntimeError('Fixed real TRAIN fit gate was not passed; no formal training may start')
    except BaseException as error:
        if report['status']=='running': report['status']='failed'
        report['error']=f'{type(error).__name__}: {error}'
        save()
        raise
    return 0


if __name__=='__main__': main()
