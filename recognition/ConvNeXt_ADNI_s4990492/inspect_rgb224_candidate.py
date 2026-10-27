"""Validation failure examples and optional local GPU timing; no test access."""
import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def failures(run, records, rows, verified):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    output = run/'validation_rgb224_failures'
    if output.exists():
        raise FileExistsError(output)
    selected, seen = [], set()
    for target, reason in [(1,'High-confidence AD predicted NC'),(0,'High-confidence NC predicted AD')]:
        candidates = sorted((r for r in rows if r['target']==target and r['prediction']!=target),
                            key=lambda r:(-r['confidence'],r['row_index']))
        count=0
        for r in candidates:
            patient=records[r['row_index']]['patient_id']
            if patient in seen:
                continue
            seen.add(patient)
            selected.append(dict(r,selection_reason=reason))
            count+=1
            if count==2:
                break
    for r in sorted((r for r in rows if r['prediction']!=r['target']),key=lambda r:(abs(r['prob_ad']-.5),r['row_index'])):
        if records[r['row_index']]['patient_id'] not in seen:
            selected.append(dict(r,selection_reason='Borderline incorrect prediction'))
            break
    if len(selected)!=5:
        raise ValueError('Unable to select five distinct-patient failures')
    fig,axes=plt.subplots(2,3,figsize=(12,9),layout='constrained')
    for ax in axes.flat:
        ax.set_axis_off()
    for ax,r in zip(axes.flat,selected):
        with Image.open(records[r['row_index']]['path']) as im:
            im=ImageOps.pad(im.convert('L'),(224,224),method=Image.Resampling.BILINEAR,color=0)
            ax.imshow(im,cmap='gray',vmin=0,vmax=255)
        ax.set_title(f"True {['NC','AD'][r['target']]} -> {['NC','AD'][r['prediction']]}\nConfidence {r['confidence']:.1%}; p(AD) {r['prob_ad']:.3f}\nValidation row {r['row_index']}",fontsize=11)
    fig.suptitle('Five validation failures at the default threshold (0.50)')
    output.mkdir()
    fig.savefig(output/'failure_examples.png',dpi=160)
    plt.close(fig)
    report={'split':'validation','checkpoint_sha256':verified['checkpoint_sha256'],
            'threshold':.5,'selection':'Two highest-confidence errors per true class, then one closest-to-0.5 error; five distinct patients globally',
            'display':'Grayscale after the same deterministic resize/padding as model input; before normalization',
            'limitations':'Selected examples are not representative error rates or evidence of diagnostic image features. Confidence is uncalibrated.',
            'examples':selected}
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    print('Saved:',output,flush=True)


def profile(run, manifest, verified):
    import torch
    from torch.utils.data import DataLoader
    from dataset_rgb224 import RGB224Dataset
    from modules import ConvNeXtTinyRGB
    output=run/'validation_rgb224_profile'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    path=run/'best_convnext.pth'
    if digest(path)!=verified['checkpoint_sha256']:
        raise ValueError('Checkpoint identity mismatch')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['config']['model']!='ConvNeXtTinyRGB' or checkpoint['config']['preprocessing']!=verified['preprocessing']:
        raise ValueError('Checkpoint configuration mismatch')
    model=ConvNeXtTinyRGB()
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    del checkpoint
    model.cuda().eval()
    data=RGB224Dataset(manifest,training=False)
    results=[]
    with torch.inference_mode():
        for bs in [1,8]:
            inputs=torch.stack([data[i][0] for i in range(bs)]).cuda()
            for _ in range(10):
                model(inputs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times=[]
            for _ in range(50):
                torch.cuda.synchronize()
                start=time.perf_counter()
                logits=model(inputs)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
            if not torch.isfinite(logits).all():
                raise ValueError('Non-finite outputs')
            results.append({'batch_size':bs,'warmup_batches':10,'timed_batches':50,
                            'median_forward_batch_ms':statistics.median(times),
                            'median_forward_ms_per_slice':statistics.median(times)/bs,
                            'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})
            del inputs,logits
        # A bounded sequential validation subset, distinct from preloaded forward timing.
        loader=DataLoader(data,batch_size=8,shuffle=False,num_workers=0)
        torch.cuda.synchronize()
        start=time.perf_counter()
        count=0
        from itertools import islice
        for images,_ in islice(loader,32):
            probabilities=model(images.cuda()).softmax(1).cpu()
            count+=len(probabilities)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
    report={'split':'validation','checkpoint_sha256':verified['checkpoint_sha256'],
            'gpu':torch.cuda.get_device_name(0),'torch_version':str(torch.__version__),
            'precision':'float32','parameter_count':sum(p.numel() for p in model.parameters()),
            'forward_only':results,
            'bounded_pipeline':{'images':count,'batch_size':8,'seconds':elapsed,'ms_per_slice':elapsed*1000/count},
            'timing_scope':'Forward-only: synchronized host timing, preloaded GPU inputs, no disk/transfer. Per-slice batch-8 value is amortized throughput cost, not single-request latency. Pipeline: first 256 validation images, file reading, RGB224 preprocessing, transfer, forward, softmax and CPU probability copy; excludes model startup, metrics and output writing. OS cache is uncontrolled.',
            'limits':'One local session; laptop load and power conditions affect timing. Memory is peak allocated, not total VRAM. No clinical speed claim.'}
    output.mkdir()
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    print('Saved:',output,flush=True)


def main():
    base=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['failures','profile'],required=True)
    args=parser.parse_args()
    run=base/'runs/local_rgb224_lr3e6_imagenet_20261026_201343_178908'
    manifest=base.parents[1]/'local_data/splits_local/val.csv'
    source=run/'validation_rgb224_evaluation'
    verified=json.loads((source/'metrics.json').read_text(encoding='utf-8'))
    if verified['split']!='validation' or digest(manifest)!=verified['manifest_sha256']:
        raise ValueError('Validation identity mismatch')
    if args.mode=='profile':
        profile(run,manifest,verified)
        return
    with manifest.open(newline='',encoding='utf-8-sig') as f:
        records=list(csv.DictReader(f))
    with (source/'predictions.csv').open(newline='',encoding='utf-8') as f:
        raw=list(csv.DictReader(f))
    rows=[{k:(int(v) if k in ['row_index','target','prediction','correct'] else float(v)) for k,v in r.items()} for r in raw]
    if len(rows)!=len(records) or len(rows)!=verified['images']:
        raise ValueError('Row count mismatch')
    for i,r in enumerate(rows):
        if r['row_index']!=i or r['target']!=int(records[i]['target']):
            raise ValueError('Row identity mismatch')
        if r['prediction']!=int(r['prob_ad']>.5) or abs(r['confidence']-max(r['prob_ad'],1-r['prob_ad']))>1e-6:
            raise ValueError('Prediction consistency mismatch')
    if abs(sum(r['target']==r['prediction'] for r in rows)/len(rows)-verified['accuracy'])>1e-10:
        raise ValueError('Accuracy mismatch')
    failures(run,records,rows,verified)


if __name__=='__main__':
    main()
