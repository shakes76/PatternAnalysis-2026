"""Isolated-model inference profiling on frozen cases; no training or selection."""
import argparse
from contextlib import nullcontext
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def measure_case(model,images,batch_size,precision,repeats):
    """Forward-only sum and prediction wall time for an entire scan.

    CPU images are already preprocessed. Prediction time includes CPU->GPU,
    forward, argmax and output copy to CPU, but excludes loading/preprocessing.
    Model weights are resident; every measurement has only one model on GPU.
    """
    results=[]
    for _ in range(repeats):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        begin=time.perf_counter();forward_ms=0.
        for start in range(0,len(images),batch_size):
            inputs=images[start:start+batch_size].cuda()
            torch.cuda.synchronize();t=time.perf_counter()
            context=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
            with context:logits=model(inputs)
            torch.cuda.synchronize();forward_ms+=(time.perf_counter()-t)*1000
            prediction=logits.argmax(1).cpu()
            del inputs,logits,prediction
        torch.cuda.synchronize()
        results.append(dict(forward_ms=forward_ms,prediction_ms=(time.perf_counter()-begin)*1000,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    return dict(forward_case_ms=statistics.median(r['forward_ms'] for r in results),
                prediction_case_ms=statistics.median(r['prediction_ms'] for r in results),
                peak_allocated_gib=max(r['peak_allocated_bytes'] for r in results)/1024**3,
                peak_reserved_gib=max(r['peak_reserved_bytes'] for r in results)/1024**3)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluator-code-dir',required=True)
    p.add_argument('--reference-code-dir',required=True)
    p.add_argument('--checkpoint-2d',required=True);p.add_argument('--checkpoint-3d',required=True)
    p.add_argument('--sha-2d',required=True);p.add_argument('--sha-3d',required=True)
    p.add_argument('--split-manifest',required=True)
    p.add_argument('--data-root',default='/home/groups/comp3710/HipMRI_Study_open')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--warmups',type=int,default=3);p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if min(args.batch_size,args.warmups,args.repeats)<1:raise ValueError('Positive batch/warmup/repeat counts required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16-capable CUDA GPU required')
    torch.set_num_threads(1);torch.manual_seed(42)
    torch.backends.cudnn.benchmark=True
    # Matched FP32 arithmetic policy; BF16 is a separate recorded deployment row.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(Path(args.evaluator_code_dir).resolve()))
    evaluator=load_module(Path(args.evaluator_code_dir)/'evaluate.py','frozen_2d_evaluator_for_profile')
    for source,expected in [(args.checkpoint_2d,args.sha_2d),(args.checkpoint_3d,args.sha_3d)]:
        if evaluator.digest(source)!=expected:raise ValueError('Checkpoint hash mismatch')
    checkpoint2=torch.load(args.checkpoint_2d,map_location='cpu',weights_only=True)
    checkpoint3=torch.load(args.checkpoint_3d,map_location='cpu',weights_only=True)
    split=json.loads(Path(args.split_manifest).read_text())
    if checkpoint2['epoch']!=5 or checkpoint2.get('smoke_only') or checkpoint2['config']['patient_split']!=split:
        raise ValueError('Unexpected 2D checkpoint or patient split')
    reference=load_module(Path(args.reference_code_dir)/'dataset.py','reference_data_for_profile')
    groups=reference.split_by_patient(reference.get_data_pairs(Path(args.data_root)),seed=42)
    for key,pairs in zip(('train','validation','test'),groups):
        if {patient[1:] for patient,_,_ in pairs}!=set(split[key]):raise ValueError('Patient split mismatch')
    pairs=groups[2]
    if len(pairs)!=27:raise ValueError('Expected all 27 frozen test scans')
    records=evaluator.get_data_pairs(args.data_root)
    lookup={}
    for row in records:lookup.setdefault((row['patient_id'],row['week_id']),[]).append(row)
    data3=reference.HipMRIDataset(pairs,target_shape=(64,128,128))
    cases2,cases3=[],[]
    import re
    for index,(patient,mri,_) in enumerate(pairs):
        week=int(re.search(r'_Week(\d+)_',mri.name)[1])
        sample=sorted(lookup[(patient[1:],week)],key=lambda r:r['slice_id'])
        if [r['slice_id'] for r in sample]!=list(range(60)):raise ValueError('Incomplete slice set')
        dataset=evaluator.HipMRI2DDataset(args.data_root,samples=sample)
        images=torch.stack([dataset[i][0] for i in range(60)])
        cases2.append((mri.name,images))
        image,_=data3[index];cases3.append((mri.name,image[None]))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    rows=[];profiles=[]
    for label,precision,cases in [('standard_2d_unet','fp32',cases2),
                                  ('standard_3d_unet','fp32',cases3),
                                  ('standard_3d_unet','bf16',cases3)]:
        torch.cuda.empty_cache()
        baseline=torch.cuda.memory_allocated()
        if label=='standard_2d_unet':
            model=evaluator.UNet2D(**checkpoint2['model_config'])
            model.load_state_dict(checkpoint2['model_state_dict'],strict=True)
            batch=args.batch_size
        else:
            model=evaluator.make_reference_model(Path(args.reference_code_dir),checkpoint3,device='cpu')
            batch=1
        parameters=sum(t.numel() for t in model.parameters())
        model=model.eval().cuda()
        measure_case(model,cases[0][1],batch,precision,args.warmups)
        mode=[]
        for case,images in cases:
            measured=measure_case(model,images,batch,precision,args.repeats)
            row=dict(model=label,precision=precision,case=case,batch_size=batch,
                     input_shape=json.dumps(list(images.shape)),**measured)
            rows.append(row);mode.append(row)
        profiles.append(dict(model=label,precision=precision,scans=len(mode),parameters=parameters,
                             forward_case_ms_mean=statistics.mean(r['forward_case_ms'] for r in mode),
                             forward_case_ms_median=statistics.median(r['forward_case_ms'] for r in mode),
                             prediction_case_ms_mean=statistics.mean(r['prediction_case_ms'] for r in mode),
                             prediction_case_ms_median=statistics.median(r['prediction_case_ms'] for r in mode),
                             peak_allocated_gib=max(r['peak_allocated_gib'] for r in mode),
                             peak_reserved_gib=max(r['peak_reserved_gib'] for r in mode),
                             non_model_allocated_bytes_before_load=baseline,batch_size=batch))
        print(json.dumps(profiles[-1]),flush=True)
        model.to('cpu');del model;torch.cuda.empty_cache()
    for source,expected in [(args.checkpoint_2d,args.sha_2d),(args.checkpoint_3d,args.sha_3d)]:
        if evaluator.digest(source)!=expected:raise ValueError('Checkpoint changed during profiling')
    with (out/'per_case_profile.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(profiles=profiles,gpu=torch.cuda.get_device_name(),torch_version=str(torch.__version__),
                 gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),
                 checkpoint_2d_sha256=args.sha_2d,checkpoint_3d_sha256=args.sha_3d,
                 warmups=args.warmups,repeats=args.repeats,threads=1,
                 tf32=False,no_training=True,models_measured_separately=True,
                 note='27 complete cases each; CPU loading/preprocessing excluded; prediction includes transfers and argmax. '
                      'PyTorch allocated/reserved peak includes resident model but excludes driver/context. '
                      '2D covers 60 native slices; 3D covers the resized full volume. These are different pipelines.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'profile_complete.txt').write_text('Isolated frozen-model profiles completed.\n')


if __name__=='__main__':
    main()
