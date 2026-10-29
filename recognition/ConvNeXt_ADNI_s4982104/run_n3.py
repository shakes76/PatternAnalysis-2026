"""N3: fixed ConvNeXt three-epoch pilot, then independent validation prediction.

Reuses train.run_epoch and predict's patient aggregation/metrics; CNN entry
points and historical evidence are unchanged. No test image pixels are read.
"""
import argparse
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback
import PIL
import torch
import torchvision
from torch import nn
from dataset import ConvNeXtSliceDataset, CONVNEXT_PREPROCESSING, LABEL_MAP
from modules import build_model
from check_convnext import verify_weights, WEIGHT_NAME
from check_dataloader import require, sha256, code_version, make_loader
from train import run_epoch, class_weight_config, classification_metrics, write_json
from predict import predict_pass, aggregate_patients, write_csv


def protocol_config():
    return dict(model='convnext_tiny',num_classes=2,epochs=3,batch_size=4,seed=42,
        optimizer='AdamW',learning_rate=1e-4,weight_decay=0.01,precision='float32',
        full_finetuning=True,stochastic_depth_probability=0.1,class_weighting='balanced',
        preprocessing=deepcopy(CONVNEXT_PREPROCESSING),weight_name=WEIGHT_NAME,
        augmentation=False,early_stopping=False,scheduler=None,num_workers=0,drop_last=False,
        validation_loss_weighting='none',checkpoint_selection='minimum validation loss; earliest tie',
        aggregation='mean slice P(AD)',threshold=0.5,deterministic_algorithms=True,tf32=False)


def initialize_model(weights, weights_summary):
    digest=verify_weights(weights,weights_summary)
    model=build_model('convnext_tiny',num_classes=1000)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    torch.manual_seed(42)
    model.classifier[-1]=nn.Linear(768,2)
    nn.init.trunc_normal_(model.classifier[-1].weight,std=0.02)
    nn.init.zeros_(model.classifier[-1].bias)
    return model,digest


def load_checkpoint(path,expected_config,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    require(checkpoint.get('config')==expected_config,'Checkpoint configuration mismatch')
    for key,value in protocol_config().items():
        require(expected_config.get(key)==value,f'Unsupported model/preprocessing protocol: {key}')
    devices=[torch.cuda.current_device()] if device=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        model=build_model(expected_config['model'],num_classes=expected_config['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    return model.to(device).eval(),checkpoint


def state_digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        v=value.detach().cpu().contiguous()
        h.update(key.encode());h.update(str(v.dtype).encode());h.update(str(list(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def sync(device):
    if device=='cuda':torch.cuda.synchronize()


def memory(device):
    return dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved()) if device=='cuda' else None


def verify_input_hashes(manifest, summary, expected):
    require(dict(split_samples=sha256(manifest),split_summary=sha256(summary))==expected,
            'Training/prediction input hashes changed')


def validate_frozen_labels(records, split):
    labels={r['patient_id']:r['label'] for r in records}
    expected={'train':{'AD':133,'NC':275},'validation':{'AD':44,'NC':92},'test':{'AD':44,'NC':92}}
    require(dict(Counter(labels.values()))==expected[split], 'Frozen patient class counts mismatch')


def validate_rows(rows,records):
    require(len(rows)==len(records),'Prediction count mismatch')
    lookup={r['file_path']:r for r in records}
    require(len({r['file_path'] for r in rows})==len(rows),'Duplicate predictions')
    for r in rows:
        require(r['file_path'] in lookup,'Prediction outside fixed validation manifest')
        source=lookup[r['file_path']]
        require(r['label']==LABEL_MAP[source['label']] and all(r[k]==source[k] for k in ('patient_id','image_id')),'Prediction/manifest mismatch')


def run_pilot(manifest,summary,weights,weights_summary,output_dir,device='cuda',
              expected_lengths={'train':8160,'validation':2720,'test':2720},expected_slices=20,expected_patients=136):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    train_dir=out/'training';train_dir.mkdir();pred_dir=out/'prediction';pred_dir.mkdir()
    started=time.perf_counter();result=dict(status='running',generated_at=datetime.now(timezone.utc).isoformat())
    training=dict(status='running',history=[]);prediction=dict(status='not_started');datasets={};pdatasets={}
    def log(folder,message):
        print(message,flush=True)
        name='training.log' if folder==train_dir else 'prediction.log'
        with (folder/name).open('a') as f:f.write(message+'\n')
    try:
        require(device in ('cpu','cuda'),'Invalid device')
        require(device!='cuda' or torch.cuda.is_available(),'CUDA unavailable')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.manual_seed(42);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if device=='cuda':torch.cuda.manual_seed_all(42);torch.cuda.reset_peak_memory_stats()
        config=protocol_config();config['device']=device
        model,weight_digest=initialize_model(weights,weights_summary)
        config['weights_sha256']=weight_digest
        datasets={s:ConvNeXtSliceDataset(manifest,s,summary) for s in ('train','validation','test')}
        lengths={s:len(d) for s,d in datasets.items()}
        require(expected_lengths is None or lengths==expected_lengths,'Frozen cohort lengths mismatch')
        for split,ds in datasets.items():
            if expected_lengths is not None:validate_frozen_labels(ds.records,split)
            metadata=[dict(r,label=LABEL_MAP[r['label']],p_ad=0.) for r in ds.records]
            patients=aggregate_patients(metadata,expected_slices)
            if split=='validation':require(len(patients)==expected_patients,'Patient count mismatch')
        config['class_weight_details']=class_weight_config(datasets['train'].records,'balanced')
        write_json(train_dir/'config.json',config)
        versions=dict(python=platform.python_version(),torch=str(torch.__version__),torchvision=str(torchvision.__version__),pillow=PIL.__version__,cuda_build=torch.version.cuda)
        code={n:sha256(Path(__file__).with_name(n)) for n in ['run_n3.py','train.py','predict.py','modules.py','dataset.py','check_dataloader.py','check_convnext.py']}
        inputs=dict(split_samples=sha256(manifest),split_summary=sha256(summary))
        training.update(config=config,versions=versions,code_version=code_version(),code_sha256=code,input_sha256=inputs,
            gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,dataset_lengths=lengths,
            initial_state_sha256=state_digest(model),parameter_count=sum(p.numel() for p in model.parameters()),
            trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
        model=model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        cw=torch.tensor(config['class_weight_details']['weights'],dtype=torch.float32,device=device)
        loaders={s:make_loader(datasets[s],4,42) for s in ('train','validation')}
        probe=torch.linspace(-1,1,3*224*224,device=device).reshape(1,3,224,224)
        best_loss=float('inf');best_epoch=None
        for epoch in range(1,4):
            sync(device);epoch_start=time.perf_counter()
            before=state_digest(model)
            tm=run_epoch(model,loaders['train'],device,optimizer,cw)
            changed=before!=state_digest(model);require(changed,'No parameter update')
            optimizer.zero_grad(set_to_none=True)
            before=state_digest(model);vm=run_epoch(model,loaders['validation'],device)
            unchanged=before==state_digest(model) and all(p.grad is None for p in model.parameters())
            require(unchanged,'Validation changed state or gradients')
            sync(device);seconds=time.perf_counter()-epoch_start
            improved=vm['loss']<best_loss
            if improved:
                best_loss=vm['loss'];best_epoch=epoch
                with torch.no_grad():probe_logits=model(probe).cpu()
                torch.save(dict(config=config,model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch,validation_loss=best_loss,probe_logits=probe_logits),train_dir/'best.pt')
            row=dict(epoch=epoch,train=tm,validation=vm,training_parameters_changed=changed,validation_parameters_unchanged=unchanged,saved_best=improved,seconds=seconds)
            training['history'].append(row)
            with (train_dir/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            log(train_dir,f'Epoch {epoch}/3: train_loss={tm["loss"]:.6f} val_loss={vm["loss"]:.6f} seconds={seconds:.2f} best={improved}')
        training['peak_cuda_memory']=memory(device)
        del optimizer,model
        if device=='cuda':torch.cuda.empty_cache()
        restored,cp=load_checkpoint(train_dir/'best.pt',config,device)
        with torch.no_grad():actual=restored(probe).cpu()
        require(torch.allclose(actual,cp['probe_logits'],rtol=1e-5,atol=1e-6),'Checkpoint reload mismatch')
        require(cp['epoch']==best_epoch and cp['validation_loss']==best_loss,'Best metadata mismatch')
        del restored,cp,probe
        if device=='cuda':torch.cuda.empty_cache()
        training.update(status='passed',completed_epochs=3,best_epoch=best_epoch,best_validation_loss=best_loss,checkpoint_reload_equal=True,
            checkpoint_sha256=sha256(train_dir/'best.pt'),read_counts={s:d.read_calls for s,d in datasets.items()},elapsed_seconds=time.perf_counter()-started,
            timing_scope='Full epoch I/O, transfers, updates, validation and diagnostic checks; no inference-latency claim')
        write_json(train_dir/'training_summary.json',training)
        # Reconstruct model AND validation dataset; no training model survives into prediction.
        pred_started=time.perf_counter()
        verify_input_hashes(manifest,summary,inputs)
        pdatasets={s:ConvNeXtSliceDataset(manifest,s,summary) for s in ('train','validation','test')}
        verify_input_hashes(manifest,summary,inputs)
        val=pdatasets['validation'];loader=make_loader(val,4,42)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        passes=[];times=[]
        for attempt in range(2):
            require(sha256(train_dir/'best.pt')==training['checkpoint_sha256'],'Checkpoint hash changed')
            model,cp=load_checkpoint(train_dir/'best.pt',config,device);del cp
            before=state_digest(model);sync(device);t0=time.perf_counter()
            rows=predict_pass(model,loader,device);sync(device);times.append(time.perf_counter()-t0)
            require(before==state_digest(model),'Prediction changed state')
            validate_rows(rows,val.records);passes.append(rows);del model
        first,second=passes;maxdiff=0.
        for a,b in zip(first,second):
            require(all(a[k]==b[k] for k in ('file_path','patient_id','image_id','label','prediction')),'Repeat identities/classes mismatch')
            maxdiff=max(maxdiff,abs(a['p_ad']-b['p_ad']))
            require(abs(a['p_ad']-b['p_ad'])<=1e-6+1e-5*abs(b['p_ad']),'Repeat probabilities mismatch')
        patients=aggregate_patients(first,expected_slices);require(len(patients)==expected_patients,'Patient count mismatch')
        write_csv(pred_dir/'slice_predictions.csv',first);write_csv(pred_dir/'patient_predictions.csv',patients)
        metric=lambda rows:classification_metrics([r['label'] for r in rows],[r['p_ad'] for r in rows])
        prediction=dict(status='passed',config=config,versions=versions,code_sha256=code,input_sha256=inputs,
            checkpoint_sha256=training['checkpoint_sha256'],training_summary_sha256=sha256(train_dir/'training_summary.json'),
            slice_count=len(first),patient_count=len(patients),slice_metrics=metric(first),patient_metrics=metric(patients),
            repeat_equal=True,repeat_max_probability_difference=maxdiff,parameters_unchanged=True,manifest_exact_match=True,
            read_counts={s:d.read_calls for s,d in pdatasets.items()},peak_cuda_memory=memory(device),full_pass_seconds=times,
            timing_scope='Full validation pass with I/O/preprocessing/transfers; synchronized CUDA; not isolated inference latency',
            elapsed_seconds=time.perf_counter()-pred_started,output_sha256={n:sha256(pred_dir/n) for n in ['slice_predictions.csv','patient_predictions.csv']})
        write_json(pred_dir/'config.json',config)
        log(pred_dir,f'PASS: {len(first)} slices, {len(patients)} patients, repeat max diff={maxdiff}')
        result.update(status='passed',completed_epochs=3,prediction_repeat_equal=True,best_epoch=best_epoch,
            total_read_counts={s:datasets[s].read_calls+pdatasets[s].read_calls for s in datasets},
            weights_sha256=weight_digest,training_summary_sha256=sha256(train_dir/'training_summary.json'))
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        (out/'n3_error.log').write_text(traceback.format_exc());raise
    finally:
        if training['status']!='passed':
            training.update(status='failed',read_counts={s:d.read_calls for s,d in datasets.items()})
            write_json(train_dir/'training_summary.json',training)
        write_json(pred_dir/'prediction_summary.json',prediction)
        result['elapsed_seconds']=time.perf_counter()-started
        result['prediction_summary_sha256']=sha256(pred_dir/'prediction_summary.json')
        write_json(out/'n3_summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['manifest','summary','weights','weights-summary','output-dir']:p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    run_pilot(a.manifest,a.summary,a.weights,a.weights_summary,a.output_dir,a.device)

if __name__=='__main__':main()
