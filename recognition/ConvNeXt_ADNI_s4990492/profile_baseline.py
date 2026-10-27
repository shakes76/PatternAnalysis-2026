"""Profile the preselected SimpleCNN using the main model's timing procedure.

Original baseline preprocessing is preserved; no test images are read.
"""
import json
import statistics
import time
from pathlib import Path
from evaluate_rgb224_validation import sha256 as digest


def profile(run, manifest, verified):
    import torch
    from torch.utils.data import DataLoader
    from dataset import ADNISliceDataset
    from modules import SimpleCNN
    output=run/'validation_baseline_profile'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    path=run/'best_baseline.pth'
    if digest(path)!=verified['checkpoint_sha256']:
        raise ValueError('Checkpoint identity mismatch')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['config']['model']!='SimpleCNN' or checkpoint['epoch']!=9:
        raise ValueError('Checkpoint configuration mismatch')
    model=SimpleCNN()
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    del checkpoint
    model.cuda().eval()
    data=ADNISliceDataset(manifest)
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
            'timing_scope':'Forward-only: synchronized host timing, preloaded GPU inputs, no disk/transfer. Per-slice batch-8 value is amortized throughput cost, not single-request latency. Pipeline: first 256 validation images, file reading, original-size grayscale [0,1] preprocessing, transfer, forward, softmax and CPU probability copy; excludes model startup, metrics and output writing. OS cache is uncontrolled.',
            'limits':'One local session; laptop load and power conditions affect timing. Memory is peak allocated, not total VRAM. No clinical speed claim.'}
    output.mkdir()
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    print('Saved:',output,flush=True)


def main():
    base=Path(__file__).resolve().parent
    run=base/'runs/baseline_20261022_211305_056381'
    verified=json.loads((run/'final_baseline_evaluation/metrics.json').read_text(encoding='utf-8'))
    manifest=base.parents[1]/'local_data/splits_local/val.csv'
    if digest(manifest)!=verified['manifest_sha256']['val']:
        raise ValueError('Validation manifest changed')
    if verified['model']!='SimpleCNN' or not verified['validation_reproduction_passed']:
        raise ValueError('Expected the validated baseline')
    if verified['checkpoint_sha256']!='593339ff8dd1dc992c55db6412611531bd347d3b30c04f76bf395d4f99a94e93':
        raise ValueError('Unexpected baseline checkpoint identity')
    print('Profiling validated SimpleCNN on validation inputs only.',flush=True)
    profile(run,manifest,verified)


if __name__=='__main__':
    main()
