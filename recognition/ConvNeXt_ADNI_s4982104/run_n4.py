"""N4: one predeclared learning-rate stability control, then independent validation prediction.

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
        optimizer='AdamW',learning_rate=3e-5,weight_decay=0.01,precision='float32',
        full_finetuning=True,stochastic_depth_probability=0.1,class_weighting='balanced',
        preprocessing=deepcopy(CONVNEXT_PREPROCESSING),weight_name=WEIGHT_NAME,
        augmentation=False,early_stopping=False,scheduler=None,num_workers=0,drop_last=False,
        validation_loss_weighting='none',checkpoint_selection='minimum validation loss; earliest tie',
        aggregation='mean slice P(AD)',threshold=0.5,deterministic_algorithms=True,tf32=False)


from run_n3 import initialize_model, state_digest, sync, memory, verify_input_hashes, validate_frozen_labels, validate_rows


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


def read_reference(folder, manifest, summary):
    """Fail before MRI reads if the immutable N3 evidence/source chain differs."""
    folder=Path(folder)
    names=['n3_summary.json','training/config.json','training/training_summary.json',
           'training/epochs.jsonl','prediction/prediction_summary.json',
           'prediction/config.json','prediction/slice_predictions.csv','prediction/patient_predictions.csv']
    hashes={n:sha256(folder/n) for n in names}
    load=lambda n:json.loads((folder/n).read_text())
    overall=load('n3_summary.json');t=load('training/training_summary.json');pr=load('prediction/prediction_summary.json')
    require(all(x['status']=='passed' for x in (overall,t,pr)), 'N3 reference status not passed')
    require(overall['training_summary_sha256']==hashes['training/training_summary.json'] and
            overall['prediction_summary_sha256']==hashes['prediction/prediction_summary.json'], 'N3 reference summary hash mismatch')
    require(pr['training_summary_sha256']==hashes['training/training_summary.json'],'N3 reference provenance mismatch')
    require(t['config']==load('training/config.json')==pr['config']==load('prediction/config.json'),'N3 reference configuration mismatch')
    import run_n3
    require(all(t['config'].get(k)==v for k,v in run_n3.protocol_config().items()),'N3 reference protocol mismatch')
    for name,digest in t['code_sha256'].items():
        require(sha256(Path(__file__).with_name(name))==digest,'N3 reference source mismatch: '+name)
    require(pr['code_sha256']==t['code_sha256'],'N3 reference prediction source mismatch')
    verify_input_hashes(manifest,summary,t['input_sha256'])
    require(pr['input_sha256']==t['input_sha256'],'N3 reference input mismatch')
    history=[json.loads(line) for line in (folder/'training/epochs.jsonl').read_text().splitlines()]
    require(history==t['history'] and [x['epoch'] for x in history]==[1,2,3],'N3 reference history mismatch')
    best=min(history,key=lambda x:x['validation']['loss'])
    require(t['best_epoch']==best['epoch'] and t['best_validation_loss']==best['validation']['loss'],'N3 reference best epoch mismatch')
    require(t['checkpoint_sha256']==pr['checkpoint_sha256'],'N3 reference checkpoint chain mismatch')
    for name,digest in pr['output_sha256'].items():
        require(sha256(folder/'prediction'/name)==digest,'N3 reference CSV hash mismatch')
    # Local preflight need not have the weight; cluster reference includes it.
    if (folder/'training/best.pt').exists():
        require(sha256(folder/'training/best.pt')==t['checkpoint_sha256'],'N3 reference weight hash mismatch')
    return dict(folder=folder,training=t,prediction=pr,provenance=dict(path=str(folder),file_sha256=hashes,code_sha256=t['code_sha256']))


def check_pair_start(reference, actual):
    expected=dict(reference['config']);expected['learning_rate']=3e-5
    require(actual['config']==expected,'N3 paired configuration mismatch beyond learning rate')
    for key in ('versions','gpu_name','input_sha256','initial_state_sha256','parameter_count','trainable_parameter_count'):
        require(actual[key]==reference[key],'N3 paired start mismatch: '+key)


def validation_with_rows(model,loader,device):
    """Observe existing eval forwards, never run a second pass or touch RNG.

    Bind the observer to the current batch, not to an assumed CSV ordering.
    Capture detached logits for stable CE reconstruction when probabilities round to 0/1.
    """
    current={};rows=[]
    class ObservedLoader:
        dataset=loader.dataset
        def __iter__(self):
            for batch in loader:
                current['batch']=batch
                yield batch
    def capture(module,args,output):
        batch=current['batch'];logits=output.detach()
        probabilities=logits.softmax(1)[:,1].cpu().tolist()
        values=logits.cpu().tolist()
        for i,p in enumerate(probabilities):
            rows.append(dict(file_path=batch['file_path'][i],patient_id=batch['patient_id'][i],
                image_id=batch['image_id'][i],label=int(batch['label'][i]),p_ad=p,prediction=int(p>=.5),
                logit_nc=values[i][0],logit_ad=values[i][1]))
    handle=model.register_forward_hook(capture)
    try:metrics=run_epoch(model,ObservedLoader(),device)
    finally:handle.remove()
    calculated=classification_metrics([r['label'] for r in rows],[r['p_ad'] for r in rows])
    require(all(metrics[k]==v for k,v in calculated.items()),'Validation diagnostic metric mismatch')
    return metrics,rows


def save_diagnostics(out,epoch,rows,metrics,expected_slices,expected_patients):
    folder=out/'diagnostics'/f'epoch_{epoch}';folder.mkdir(parents=True)
    patients=aggregate_patients(rows,expected_slices)
    require(len(patients)==expected_patients,'Epoch diagnostic patient count mismatch')
    write_csv(folder/'slice_predictions.csv',rows);write_csv(folder/'patient_predictions.csv',patients)
    result=dict(slice_count=len(rows),patient_count=len(patients),slice_metrics=metrics,
        patient_metrics=classification_metrics([r['label'] for r in patients],[r['p_ad'] for r in patients]),
        additional_image_reads=0,source='same validation forward; detached output observer',
        output_sha256={n:sha256(folder/n) for n in ('slice_predictions.csv','patient_predictions.csv')})
    write_json(folder/'summary.json',result)
    return result


def flat_metrics(m):
    ad=m['per_class']['AD'];nc=m['per_class']['NC'];cm=m['confusion_matrix']
    return dict(accuracy=m['accuracy'],ad_recall=ad['recall'],nc_recall=nc['recall'],
        balanced_accuracy=(ad['recall']+nc['recall'])/2,auroc=m['auroc'],macro_f1=m['macro_f1'],
        ad_precision=ad['precision'],predicted_ad_fraction=(cm[0][1]+cm[1][1])/sum(map(sum,cm)))


def stability_signal(reference,current):
    span=lambda rows,name:max(x['per_class'][name]['recall'] for x in rows)-min(x['per_class'][name]['recall'] for x in rows)
    a=dict(final_ce=reference[-1]['loss'],ad_recall_range=span(reference,'AD'),nc_recall_range=span(reference,'NC'))
    b=dict(final_ce=current[-1]['loss'],ad_recall_range=span(current,'AD'),nc_recall_range=span(current,'NC'))
    checks={k:b[k]<a[k] for k in a}
    return dict(reference=a,current=b,strict_improvements=checks,joint_signal=all(checks.values()),
        interpretation='Diagnostic only; no automatic LR adoption; consider best-patient tradeoffs and slower learning',
        units='CE natural log; recall ranges 0-1 (multiply by 100 for percentage points)')


def finish_comparison(out,reference,training,prediction,result):
    old=reference['training'];require(len(training['history'])==3,'Incomplete control')
    check_pair_start(old,training)
    rows=[]
    for name,t in [('N3_lr1e4',old),('N4_lr3e5',training)]:
        for h in t['history']:
            rows.append(dict(run=name,epoch=h['epoch'],train_weighted_ce=h['train']['loss'],
                train_unweighted_ce=h['train']['unweighted_loss'],validation_ce=h['validation']['loss'],**flat_metrics(h['validation'])))
    write_csv(out/'epoch_comparison.csv',rows)
    best=[]
    for name,t,pr in [('N3_lr1e4',old,reference['prediction']),('N4_lr3e5',training,prediction)]:
        best.append(dict(run=name,best_epoch=t['best_epoch'],validation_ce=t['best_validation_loss'],**flat_metrics(pr['patient_metrics'])))
    write_csv(out/'best_patient_comparison.csv',best)
    patient_rows=[]
    for h in training['history']:
        e=h['epoch'];oldm=reference['prediction']['patient_metrics'] if e==old['best_epoch'] else None
        for name,m in [('N3_lr1e4',oldm),('N4_lr3e5',h['diagnostics']['patient_metrics'])]:
            patient_rows.append(dict(run=name,epoch=e,available=m is not None,
                reason='' if m else 'Not retained in N3; no reconstruction or extra image reads',
                **(flat_metrics(m) if m else {k:None for k in flat_metrics(prediction['patient_metrics'])})))
    write_csv(out/'epoch_patient_comparison.csv',patient_rows)
    for name,digest in reference['provenance']['file_sha256'].items():
        require(sha256(reference['folder']/name)==digest,'N3 reference changed during control')
    require(prediction['slice_metrics']=={k:training['history'][training['best_epoch']-1]['validation'][k] for k in prediction['slice_metrics']},'Best diagnostic/reloaded metrics mismatch')
    result.update(paired_checks=dict(initial_state_equal=True,all_epoch_orders_equal=True,
        other_config_equal=True,environment_equal=True,reference_unchanged=True),
        stability=stability_signal([h['validation'] for h in old['history']],[h['validation'] for h in training['history']]),
        comparison_sha256={n:sha256(out/n) for n in ('epoch_comparison.csv','best_patient_comparison.csv','epoch_patient_comparison.csv')},
        formal_protocol_frozen=False)


def run_control(manifest,summary,weights,weights_summary,output_dir,reference_dir,device='cuda',
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
        reference = read_reference(reference_dir, manifest, summary)
        result['reference'] = reference['provenance']
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
        code={n:sha256(Path(__file__).with_name(n)) for n in ['run_n4.py','run_n3.py','train.py','predict.py','modules.py','dataset.py','check_dataloader.py','check_convnext.py']}
        inputs=dict(split_samples=sha256(manifest),split_summary=sha256(summary))
        training.update(config=config,versions=versions,code_version=code_version(),code_sha256=code,input_sha256=inputs,
            gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,dataset_lengths=lengths,
            initial_state_sha256=state_digest(model),parameter_count=sum(p.numel() for p in model.parameters()),
            trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
        check_pair_start(reference['training'], training)
        model=model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        training['optimizer_actual'] = dict(learning_rate=optimizer.param_groups[0]['lr'],weight_decay=optimizer.param_groups[0]['weight_decay'],betas=list(optimizer.param_groups[0]['betas']),eps=optimizer.param_groups[0]['eps'])
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
            before=state_digest(model);vm,epoch_rows=validation_with_rows(model,loaders['validation'],device)
            validate_rows(epoch_rows,datasets['validation'].records)
            diagnostics=save_diagnostics(out,epoch,epoch_rows,vm,expected_slices,expected_patients)
            require(tm['order_sha256']==reference['training']['history'][epoch-1]['train']['order_sha256'],'N3 train batch order mismatch')
            require(vm['order_sha256']==reference['training']['history'][epoch-1]['validation']['order_sha256'],'N3 validation batch order mismatch')
            unchanged=before==state_digest(model) and all(p.grad is None for p in model.parameters())
            require(unchanged,'Validation changed state or gradients')
            sync(device);seconds=time.perf_counter()-epoch_start
            improved=vm['loss']<best_loss
            if improved:
                best_loss=vm['loss'];best_epoch=epoch
                with torch.no_grad():probe_logits=model(probe).cpu()
                torch.save(dict(config=config,model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch,validation_loss=best_loss,probe_logits=probe_logits),train_dir/'best.pt')
            row=dict(epoch=epoch,train=tm,validation=vm,diagnostics=diagnostics,training_parameters_changed=changed,validation_parameters_unchanged=unchanged,saved_best=improved,seconds=seconds)
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
        finish_comparison(out,reference,training,prediction,result)
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if pdatasets:
            prediction.update(status='failed',error=result['error'],read_counts={s:d.read_calls for s,d in pdatasets.items()})
        (out/'n4_error.log').write_text(traceback.format_exc());raise
    finally:
        if training['status']!='passed':
            training.update(status='failed',read_counts={s:d.read_calls for s,d in datasets.items()})
            write_json(train_dir/'training_summary.json',training)
        write_json(pred_dir/'prediction_summary.json',prediction)
        result['elapsed_seconds']=time.perf_counter()-started
        result['prediction_summary_sha256']=sha256(pred_dir/'prediction_summary.json')
        write_json(out/'n4_summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['manifest','summary','weights','weights-summary','output-dir','reference-dir']:p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    run_control(a.manifest,a.summary,a.weights,a.weights_summary,a.output_dir,a.reference_dir,a.device)

if __name__=='__main__':main()
