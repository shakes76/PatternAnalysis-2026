"""Finite TRAIN-only memorisation QA, never a candidate-selection experiment.

Uses the preserved v1 model/loss implementations explicitly, and an independent
confusion matrix. A single labelled TRAIN crop is intentionally used to test
gradient, supervision and optimisation plumbing. Success says nothing about
generalisation. The final fixed step, not the best observed step, decides QA.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(pred, target, classes=6):
    p, t = np.asarray(pred, dtype=np.int64), np.asarray(target, dtype=np.int64)
    if p.shape != t.shape or not p.size or p.min() < 0 or p.max() >= classes or t.min() < 0 or t.max() >= classes:
        raise ValueError("Invalid predictions")
    confusion = np.bincount(classes * t.ravel() + p.ravel(), minlength=classes**2).reshape(classes, classes)
    rows = []
    for c in range(classes):
        tp = int(confusion[c, c]); gt = int(confusion[c].sum()); pr = int(confusion[:, c].sum())
        rows.append(dict(class_index=c, tp=tp, fp=pr-tp, fn=gt-tp,
                         dice=2*tp/(gt+pr) if gt+pr else None))
    defined = [x['dice'] for x in rows[1:] if x['dice'] is not None]
    return dict(per_class=rows, foreground_dice=sum(defined)/len(defined), prostate_dice=rows[5]['dice'])


def fit_passed(final, initial_loss, qa):
    return (initial_loss > 0 and final['prostate_dice'] is not None and
            final['prostate_dice']>=qa['minimum_fit_prostate_dice'] and
            final['foreground_dice']>=qa['minimum_fit_foreground_dice'] and
            final['loss']<=qa['maximum_final_initial_loss_ratio']*initial_loss)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--device', choices=['cuda', 'cpu'], required=True)
    a = ap.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    plan = json.loads(a.protocol.read_text()); cfg = plan['configuration']; q = plan['qa']
    for rel, expected in plan['required_original_sha256'].items():
        if sha(a.project_root / rel) != expected:
            raise ValueError('Original source changed: '+rel)
    for rel, expected in plan['required_review_sha256'].items():
        if sha(a.protocol.parent / rel) != expected:
            raise ValueError('Review source changed: '+rel)
    manifest = json.loads(a.manifest.read_text())
    if manifest['num_classes'] != 6 or manifest['label_ids'] != list(range(6)) or manifest['target_shape_dhw'] != cfg['target_shape_dhw']:
        raise ValueError('QA requires the exact declared six-class v2 grid')
    members = manifest['patient_splits']
    if any(set(members[u]) & set(members[v]) for u,v in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Patient overlap')
    records = sorted((r for r in manifest['records'] if r['split']=='train'), key=lambda r:(r['patient'],r['visit']))
    r = records[0]
    if [r['patient'], r['visit']] != q['fit_case']:
        raise ValueError('Unexpected lexicographic first TRAIN case')
    with np.load(a.manifest.parent/r['cache_path'], allow_pickle=False) as v:
        x = v['image'].copy(); y = v['mask'].copy()
    if r['patient'] not in members['train'] or x.shape != tuple(cfg['target_shape_dhw']) or y.shape != x.shape or not np.isfinite(x).all() or y.min()<0 or y.max()>=6 or not np.any(y==5):
        raise ValueError('Invalid real TRAIN data for fit QA')
    patch = np.asarray(cfg['patch_shape_dhw'])
    centre = np.rint(np.argwhere(y==5).mean(0)).astype(int)
    starts = np.minimum(np.maximum(centre-patch//2,0),np.asarray(y.shape)-patch)
    sl = tuple(slice(int(s),int(s+n)) for s,n in zip(starts,patch))
    x,y = x[sl].copy(),y[sl].copy()
    if min(x.shape)<8 or x.shape != tuple(patch) or not np.any(y==5):
        raise ValueError('Invalid TRAIN crop')
    # Import only frozen model/loss sources, retaining original provenance.
    sys.path.insert(0,str(a.project_root))
    from modules import UNet2D, ImprovedUNet3D, segmentation_loss
    device = torch.device(a.device)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    xt = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    yt = torch.from_numpy(y.astype(np.int64)).unsqueeze(0).to(device)
    counts = np.sum([z['class_voxel_counts'] for z in records],axis=0).astype(np.float64)
    w = 1/np.sqrt(counts/counts.sum()); w/=w.mean(); w=np.clip(w,.05,10); w/=w.mean()
    weights=torch.tensor(w,dtype=torch.float32,device=device)
    a.output_dir.mkdir(parents=True)
    result=dict(status='running',purpose='TRAIN-only finite memorisation QA; no generalisation claim',
                protocol_sha256=sha(a.protocol),manifest_sha256=sha(a.manifest),
                case=[r['patient'],r['visit']],crop_start_dhw=starts.tolist(),crop_shape_dhw=list(x.shape),
                crop_class_counts=np.bincount(y.ravel(),minlength=6).tolist(),
                cache_file_sha256=sha(a.manifest.parent/r['cache_path']),
                required_original_sha256=plan['required_original_sha256'],
                amp=False,seed=q['fit_seed'],steps=q['fit_steps'],models={},
                gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                torch_version=str(torch.__version__))
    def save():
        (a.output_dir/'fit_receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    for name, cls in [('unet2d',UNet2D),('improved3d',ImprovedUNet3D)]:
        torch.manual_seed(q['fit_seed']); np.random.seed(q['fit_seed'])
        model=cls(num_classes=6,base_channels=cfg['base_channels']).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=q['fit_lr'],weight_decay=cfg['weight_decay'])
        history=[]; start=time.perf_counter()
        def evaluate():
            model.eval()
            with torch.inference_mode():
                if name=='improved3d':
                    out=model(xt); loss=segmentation_loss(out,yt,class_weights=weights)
                    pred=out['logits'].argmax(1)[0].cpu().numpy()
                else:
                    parts=[]; losses=[]
                    for z in range(0,x.shape[0],16):
                        im=xt[0,0,z:z+16].unsqueeze(1); target=yt[0,z:z+16]
                        out=model(im); losses.append(float(segmentation_loss(out,target,class_weights=weights)))
                        parts.append(out['logits'].argmax(1).cpu().numpy())
                    pred=np.concatenate(parts); loss=sum(losses)/len(losses)
            model.train()
            return dict(loss=float(loss),**metrics(pred,y))
        history.append(dict(step=0,**evaluate()))
        for step in range(1,q['fit_steps']+1):
            opt.zero_grad(set_to_none=True)
            if name=='unet2d':
                # Deterministic cycling covers all crop slices every four steps.
                z=((step-1)*16)%x.shape[0]
                im=xt[0,0,z:z+16].unsqueeze(1); target=yt[0,z:z+16]
            else:
                im,target=xt,yt
            output=model(im); loss=segmentation_loss(output,target,class_weights=weights)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite fit loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad'],error_if_nonfinite=True)
            opt.step()
            if step%q['fit_measure_every']==0 or step==q['fit_steps']:
                row=dict(step=step,**evaluate()); history.append(row)
                print(name,step,'loss',row['loss'],'P5',row['prostate_dice'],'FG',row['foreground_dice'],flush=True)
        if device.type=='cuda': torch.cuda.synchronize()
        final=history[-1]
        passed=fit_passed(final,history[0]['loss'],q)
        result['models'][name]=dict(passed=passed,history=history,seconds=time.perf_counter()-start,
                                   final_loss_ratio=final['loss']/history[0]['loss'])
        save()
        del model,opt
        if device.type=='cuda': torch.cuda.empty_cache()
    result['status']='passed' if all(m['passed'] for m in result['models'].values()) else 'failed_fit_gate'
    result['generalisation_evidence']=False
    save()
    print('FIT_QA',result['status'],flush=True)


if __name__=='__main__': main()
