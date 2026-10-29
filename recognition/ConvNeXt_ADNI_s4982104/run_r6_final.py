"""One frozen final-test inference pass per checkpoint, with complete failure evidence.

No training, temperature fitting, threshold search, or configurable inference knobs.
Preflight verifies all nine checkpoint bytes before any test image is opened.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

from final_evaluation import load_protocol, digest, manifest_records, aggregate_slices, evaluate
from run_r5_ensemble import require, sha

ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA='96b492906327273f8379d2e532434f57fe1b77a99ebc8b9151918ec9bc521fa8'
FROZEN_SOURCE_COMMIT='0967d26e4c23dbfbd513c496f69aae540f57af0b'


def safe_path(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    require(path.is_relative_to(root),'Path escapes data root')
    return path


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def preflight(root):
    # Missing private inputs should be reported before any source-version check.
    manifest_path = Path(root) / 'split_seed42_60_20_20/split_samples.csv'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Missing private test manifest: {manifest_path}')
    try:
        protocol=load_protocol(ROOT/'r5_final_protocol.json',PROTOCOL_SHA,ROOT)
    except ValueError as exc:
        if 'Code SHA mismatch:' not in str(exc):
            raise
        raise RuntimeError(
            'Frozen test protocol requires source commit ' + FROZEN_SOURCE_COMMIT +
            '; check out that commit for exact replay of the reported result') from exc
    for name,expected in protocol['split_files_sha256'].items():
        require(sha(safe_path(root,name))==expected,'Split hash mismatch: '+name)
    manifest=safe_path(root,'split_seed42_60_20_20/split_samples.csv')
    from dataset import ADNISliceDataset
    dataset=ADNISliceDataset(manifest,'test') # metadata only
    expected=manifest_records(dataset.records)
    require(digest(expected)==protocol['cohort']['manifest_records_sha256'],'Frozen test cohort mismatch')
    # Validate shape/identity without opening any image.
    fake=[dict(r,p_ad=.5,prediction=1) for r in expected]
    patients=aggregate_slices(fake,expected,20)
    require(len(patients)==136 and sum(r['label'] for r in patients)==44,'Wrong class count')
    import torch
    verified={}
    for entry in protocol['checkpoints']:
        source=safe_path(root,entry['source'])
        for name,expected_sha in entry['files'].items():
            require(sha(safe_path(source,name))==expected_sha,'Frozen source hash mismatch: '+str(source/name))
        checkpoint=source/'selection/best_patient_nll.pt'
        require(sha(checkpoint)==entry['checkpoint_sha256'],'Checkpoint hash mismatch')
        cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
        require(cp['epoch']==entry['epoch'] and cp['selection_rule']=='patient_nll','Checkpoint metadata mismatch')
        require(cp['config']['seed']==entry['seed'],'Checkpoint seed mismatch')
        convnext=entry['model']=='ConvNeXt-F'
        require(cp['config']['model']==('convnext_tiny' if convnext else 'SimpleCNN'),'Checkpoint model mismatch')
        if not convnext:require(cp['config']['arm']==entry['model'][-1],'Checkpoint CNN arm mismatch')
        del cp
        verified[entry['model']+'_'+str(entry['seed'])]=entry['checkpoint_sha256']
    return protocol,expected,verified


def environment():
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one visible CUDA GPU required')
    require('A100' in torch.cuda.get_device_name(0),'Expected A100 GPU')
    require(str(torch.__version__)=='2.14.0+cu130' and torch.version.cuda=='13.0','GPU runtime differs from accepted R4')
    versions={name:importlib.metadata.version(name) for name in ('numpy','torchvision','Pillow','matplotlib')}
    require(versions==dict(numpy='2.4.6',torchvision='0.29.0',Pillow='12.3.0',matplotlib='3.11.2'),'Pinned dependency mismatch')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch')
    torch.set_num_threads(2);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),packages=versions,
        hostname=platform.node(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        batch_size=4,tf32=False,amp=False,deterministic=True,cpu_threads=torch.get_num_threads())


def state_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def infer(model,loader,device):
    import torch
    model.eval();rows=[]
    with torch.inference_mode():
        for batch in loader:
            logits=model(batch['image'].to(device=device,dtype=torch.float32))
            require(tuple(logits.shape)==(len(batch['label']),2) and bool(torch.isfinite(logits).all()),'Invalid logits')
            probabilities=logits.softmax(dim=1)[:,1].cpu().tolist()
            for i,p in enumerate(probabilities):
                rows.append(dict(file_path=batch['file_path'][i],patient_id=batch['patient_id'][i],
                    image_id=batch['image_id'][i],label=int(batch['label'][i]),split='test',p_ad=p,prediction=int(p>=.5)))
    return rows


def predict_one(root,protocol,entry):
    import torch
    from dataset import ConvNeXtSliceDataset
    from train import CountedDataset
    from check_dataloader import make_loader
    from modules import build_model
    source=safe_path(root,entry['source']);path=source/'selection/best_patient_nll.pt'
    require(sha(path)==entry['checkpoint_sha256'],'Checkpoint changed after preflight')
    cp=torch.load(path,map_location='cpu',weights_only=True)
    require(cp['epoch']==entry['epoch'] and cp['selection_rule']=='patient_nll','Checkpoint epoch/selection mismatch')
    require(cp['config']['seed']==entry['seed'],'Checkpoint seed mismatch')
    convnext=entry['model']=='ConvNeXt-F';name='convnext_tiny' if convnext else 'simple_cnn'
    require(cp['config']['model']==('convnext_tiny' if convnext else 'SimpleCNN'),'Checkpoint model mismatch')
    if not convnext:require(cp['config']['arm']==entry['model'][-1],'Checkpoint CNN arm mismatch')
    model=build_model(name);model.load_state_dict(cp['model_state_dict'],strict=True);del cp
    before=state_digest(model);model=model.float().to('cuda').eval()
    manifest=safe_path(root,'split_seed42_60_20_20/split_samples.csv')
    dataset=(ConvNeXtSliceDataset if convnext else CountedDataset)(manifest,'test')
    rows=infer(model,make_loader(dataset,4,42),'cuda')
    require(state_digest(model)==before,'Inference modified parameters')
    require(dataset.read_calls==2720 and len(rows)==2720,'Incomplete test pass')
    result=dict(entry=entry,protocol_sha256=digest(protocol),inference=protocol['inference'],slices=rows)
    details=dict(image_reads=dataset.read_calls,split='test',state_sha256=before,parameters_unchanged=True,
        checkpoint_sha256=entry['checkpoint_sha256'],protocol_file_sha256=PROTOCOL_SHA)
    del model;gc.collect();torch.cuda.empty_cache()
    return result,details


def export_report(out,report):
    write_json(out/'evaluation.json',report)
    for name in ('metrics','seed_summary','paired_differences','referrals','reliability_bins'):
        rows=[{k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in report[name]]
        write_csv(out/(name+'.csv'),rows)
    folder=out/'patient_tables';folder.mkdir()
    for name,rows in report['patient_tables'].items():write_csv(folder/(name.replace('/','_')+'.csv'),rows)


def run(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();summary=dict(status='running',runs=[],command=sys.argv,
        protocol_file_sha256=PROTOCOL_SHA,scope='Frozen final test, no refitting or model selection',
        entrypoint_sha256=sha(Path(__file__)))
    write_json(out/'summary.json',summary)
    try:
        protocol,expected,verified=preflight(root)
        summary['verified_checkpoints']=verified
        write_json(out/'protocol.json',protocol)
        summary['environment']=environment();summary['protocol_digest']=digest(protocol)
        summary['code_sha256']=protocol['code_sha256']
        runs=[]
        for entry in protocol['checkpoints']:
            tag=entry['model']+'_seed'+str(entry['seed']);folder=out/tag;folder.mkdir()
            print('Predicting frozen test:',tag,flush=True)
            bundle,details=predict_one(root,protocol,entry)
            # Save raw output before validation so failures can be inspected.
            write_csv(folder/'slice_predictions.csv',bundle['slices'])
            write_json(folder/'prediction_bundle.json',bundle)
            aggregate_slices(bundle['slices'],expected,20)
            write_json(folder/'prediction_summary.json',dict(status='passed',**details,
                output_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
            runs.append(bundle);summary['runs'].append(tag);write_json(out/'summary.json',summary)
        report=evaluate(protocol,expected,runs);export_report(out,report)
        summary['output_sha256']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='summary.json'}
        summary['status']='passed';summary['image_reads']=sum(2720 for _ in runs)
        print('PASS: nine frozen test predictions and fixed evaluation complete',flush=True)
    except BaseException as exc:
        summary.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        (out/'failure.log').write_text(traceback.format_exc());raise
    finally:
        summary['elapsed_seconds']=time.perf_counter()-started;write_json(out/'summary.json',summary)
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    return p

if __name__=='__main__':
    args=parser().parse_args();run(args.data_root,args.output_dir)
