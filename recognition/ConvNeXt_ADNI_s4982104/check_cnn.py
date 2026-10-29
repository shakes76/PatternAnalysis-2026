"""Tiny CNN forward/loss/backward smoke check, not a performance experiment."""
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import sys
import time

import PIL
import torch
from torch import nn
from modules import SimpleCNN
from dataset import PREPROCESSING
from check_dataloader import AuditedDataset, make_loader, validate_batch, require, sha256, code_version


def train_step(model, optimizer, images, labels):
    """Check a full optimization step and return finite numerical diagnostics."""
    require(bool(torch.isfinite(images).all()), 'Nonfinite input')
    require(labels.dtype == torch.int64 and labels.ndim == 1 and
            len(labels) == len(images) and bool(((labels == 0) | (labels == 1)).all()),
            'Labels must be int64 [B] with values 0 or 1')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before=[p.detach().clone() for p in model.parameters()]
    logits=model(images)
    require(tuple(logits.shape) == (len(labels),2) and bool(torch.isfinite(logits).all()),
            'Invalid logits')
    loss=nn.functional.cross_entropy(logits,labels)  # raw logits, no softmax
    require(bool(torch.isfinite(loss)), 'Nonfinite loss')
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad]
    require(all(g is not None and bool(torch.isfinite(g).all()) for g in grads),
            'Missing or nonfinite gradient')
    norm=float(torch.sqrt(sum(g.detach().square().sum() for g in grads)))
    require(norm > 0, 'All gradients are zero')
    optimizer.step()
    changed=any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
    require(changed, 'Optimizer did not update parameters')
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite parameters')
    return dict(loss=float(loss.detach()),gradient_norm=norm,parameters_changed=changed,
                logits_shape=list(logits.shape))


def check_checkpoint(model, images, path):
    """Compare eval logits before and after a state-dict round trip."""
    model.eval()
    with torch.no_grad():
        expected=model(images)
        torch.save(model.state_dict(),path)
        restored=SimpleCNN().to(images.device)
        restored.load_state_dict(torch.load(path,map_location=images.device,weights_only=True))
        restored.eval()
        actual=restored(images)
    require(torch.allclose(expected,actual,rtol=1e-5,atol=1e-6), 'Checkpoint prediction mismatch')
    return True


def run_smoke(manifest, summary, output_dir, device='cpu', batch_size=4, steps=2,
              seed=42, expected_lengths=None):
    require(1 <= steps <= 10, 'Smoke check steps must be between 1 and 10')
    require(batch_size > 0, 'batch_size must be positive')
    require(device in ('cpu','cuda'), 'Use cpu or cuda')
    require(device != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False
        torch.cuda.reset_peak_memory_stats()
    train=AuditedDataset(manifest,'train',summary)
    val=AuditedDataset(manifest,'validation',summary)
    # Constructing test metadata never reads test image pixels.
    test=AuditedDataset(manifest,'test',summary)
    lengths=dict(train=len(train),validation=len(val),test=len(test))
    require(expected_lengths is None or lengths == expected_lengths, 'Frozen cohort lengths mismatch')
    loader=make_loader(train,batch_size,seed)
    require(steps <= len(loader), 'Not enough distinct batches for requested steps')
    model=SimpleCNN().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    results=[]
    iterator=iter(loader)
    for step in range(steps):
        batch=next(iterator); validate_batch(batch,train)
        images=batch['image'].to(device); labels=batch['label'].to(device)
        if device == 'cuda': torch.cuda.synchronize()
        started=time.perf_counter()
        result=train_step(model,optimizer,images,labels)
        if device == 'cuda': torch.cuda.synchronize()
        result['seconds']=time.perf_counter()-started
        result['batch_size']=len(labels)
        results.append(result)
        logging.info('Step %d: %s',step+1,result)
    batch=next(iter(make_loader(val,batch_size,seed))); validate_batch(batch,val)
    images=batch['image'].to(device); labels=batch['label'].to(device)
    model.eval()
    with torch.no_grad():
        logits=model(images)
        require(tuple(logits.shape)==(len(labels),2) and bool(torch.isfinite(logits).all()), 'Invalid validation logits')
        loss=nn.functional.cross_entropy(logits,labels)
        require(bool(torch.isfinite(loss)), 'Nonfinite validation loss')
    # Peaks captured before reload, to exclude the second model instance.
    memory=(dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                 reserved_bytes=torch.cuda.max_memory_reserved()) if device=='cuda' else None)
    checkpoint=Path(output_dir)/'smoke_checkpoint.pt'
    equal=check_checkpoint(model,images,checkpoint)
    return dict(status='passed',error_count=0,device=device,
        gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,
        model='SimpleCNN',parameter_count=sum(p.numel() for p in model.parameters()),
        optimizer='Adam',learning_rate=0.001,seed=seed,batch_size=batch_size,
        steps=results,validation_loss=float(loss),validation_logits_shape=list(logits.shape),
        validation_grad_enabled=False,checkpoint_reload_equal=equal,
        dataset_lengths=lengths,preprocessing=PREPROCESSING,
        read_counts=dict(train=train.read_calls,validation=val.read_calls,test=test.read_calls,
                         unique_images=len(set(train.observed)|set(val.observed))),
        peak_cuda_memory=memory,
        scope='Tiny pipeline check only; no accuracy claim or convergence requirement',
        timing_scope='Synchronized optimization step including diagnostic checks; excludes loading/transfers; not a formal speed benchmark')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name in ['cnn_check_summary.json','cnn_check.log','smoke_checkpoint.pt']:
        if (args.output_dir/name).exists():parser.error('Evidence exists: use a new output directory')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(args.output_dir/'cnn_check.log'),logging.StreamHandler(sys.stdout)])
    result=dict(generated_at=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        versions=dict(python=platform.python_version(),torch=torch.__version__,pillow=PIL.__version__,cuda_build=torch.version.cuda),
        code_version=code_version())
    try:
        result['code_sha256']={n:sha256(Path(__file__).with_name(n)) for n in
                              ['modules.py','check_cnn.py','dataset.py','check_dataloader.py']}
        result['input_sha256']={n:sha256(path) for n,path in
                               [('split_samples',args.manifest),('split_summary',args.summary)]}
        result.update(run_smoke(args.manifest,args.summary,args.output_dir,args.device,
            args.batch_size,args.steps,args.seed,{'train':8160,'validation':2720,'test':2720}))
        logging.info('PASS: %s',result['read_counts'])
    except Exception as exc:
        result.update(status='failed',error_count=1,error=f'{type(exc).__name__}: {exc}')
        logging.exception('CNN check failed')
        raise
    finally:
        (args.output_dir/'cnn_check_summary.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
