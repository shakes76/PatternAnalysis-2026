"""R3 CNN A/B entry: frozen V1 training protocol, patient-NLL primary selection.

A=ordinary CE; B=train-patient-balanced CE. Validation is always unweighted.
Defaults to three-epoch compatibility; no scheduling or automatic GPU submission.
Every validation forward is observed once; both selected weights are reloaded
and predicted twice after training. V1 files/runners remain unchanged.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import time
import traceback
import PIL
import torch
from modules import SimpleCNN
from dataset import PREPROCESSING, LABEL_MAP
from train import CountedDataset, run_epoch, class_weight_config, state_sha256, write_json
from check_dataloader import make_loader, require, sha256, code_version
from run_n3 import sync, memory, validate_frozen_labels
from run_n4 import validation_with_rows
from run_r2 import rng_state, same_rng, save_predictions
from patient_evaluation import evaluate_patients
from dual_checkpoint import DualCheckpointSelector, load_selected


def protocol(arm,epochs,seed,device):
    require(arm in ('A','B'),'arm must be A or B')
    require(type(epochs) is int and epochs in (3,30),'epochs must be 3 or 30')
    require(type(seed) is int and seed in (42,43,44),'seed must be 42,43,44')
    return dict(model='SimpleCNN',arm=arm,epochs=epochs,seed=seed,device=device,
        batch_size=16,learning_rate=.001,optimizer='Adam',weight_decay=0.,precision='float32',
        class_weighting='none' if arm=='A' else 'balanced',preprocessing=PREPROCESSING,
        augmentation=False,num_workers=0,drop_last=False,early_stopping=False,scheduler=None,
        validation_loss_weighting='none',checkpoint_selection='patient NLL primary; slice CE auxiliary; earliest tie',
        positive_class='AD=1',threshold=.5,metric_unit='patient',aggregation='mean of 20 slice probabilities',
        deterministic_algorithms=True)


def verify_rows(actual,expected):
    require(len(actual)==len(expected),'Reload row count mismatch')
    lookup={r['file_path']:r for r in expected}
    require(len(lookup)==len(expected),'Duplicate expected row')
    for a in actual:
        require(a['file_path'] in lookup,'Unexpected reload path');b=lookup[a['file_path']]
        for k in ('patient_id','image_id','label','prediction'):
            require(str(a[k])==str(b[k]),'Reload identity/class mismatch: '+k)
        p=float(b['p_ad']);require(abs(a['p_ad']-p)<=1e-6+1e-5*abs(p),'Reload probability mismatch')


def run_cnn(manifest,summary,output_dir,arm='A',epochs=3,seed=42,device='cuda',
            expected_lengths={'train':8160,'validation':2720,'test':2720},expected_slices=20):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',history=[],arm=arm);datasets={};prediction_sets={};start=time.perf_counter()
    try:
        config=protocol(arm,epochs,seed,device)
        require(device in ('cpu','cuda') and (device!='cuda' or torch.cuda.is_available()),'Device unavailable')
        inputs=dict(split_samples=sha256(manifest),split_summary=sha256(summary))
        source_names=('run_r3_cnn.py','train.py','modules.py','dataset.py','check_dataloader.py',
            'predict.py','run_n3.py','run_n4.py','run_r2.py','check_convnext.py','patient_evaluation.py','dual_checkpoint.py')
        provenance=dict(input_sha256=inputs,code_sha256={n:sha256(Path(__file__).with_name(n)) for n in source_names},pretrained=None)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
        if device=='cuda':
            torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True;torch.cuda.reset_peak_memory_stats()
        datasets={k:CountedDataset(manifest,k,summary) for k in ('train','validation','test')}
        lengths={k:len(d) for k,d in datasets.items()}
        require(expected_lengths is None or lengths==expected_lengths,'Frozen cohort lengths mismatch')
        for split,ds in datasets.items():
            if expected_lengths is not None:validate_frozen_labels(ds.records,split)
            evaluate_patients([dict(r,label=LABEL_MAP[r['label']],p_ad=0.) for r in ds.records],ds.records,expected_slices)
        config['class_weight_details']=class_weight_config(datasets['train'].records,config['class_weighting'])
        config['expected_slices_per_patient']=expected_slices
        weights=config['class_weight_details']['weights']
        weights=torch.tensor(weights,dtype=torch.float32,device=device) if weights is not None else None
        loaders={k:make_loader(datasets[k],16,seed) for k in ('train','validation')}
        model=SimpleCNN().to(device)
        require(all(p.dtype==torch.float32 for p in model.parameters()),'FP32 required')
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        result.update(config=config,provenance=provenance,code_version=code_version(),dataset_lengths=lengths,
            initial_state_sha256=state_sha256(model),parameter_count=sum(p.numel() for p in model.parameters()),
            versions=dict(python=platform.python_version(),torch=str(torch.__version__),pillow=PIL.__version__,cuda_build=torch.version.cuda),
            gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,
            optimizer_actual={k:v for k,v in optimizer.param_groups[0].items() if k!='params'},
            backend_flags=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                               cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32))
        write_json(out/'config.json',config)
        selector=DualCheckpointSelector(out/'selection',config,provenance,expected_slices)
        for epoch in range(1,epochs+1):
            sync(device);t0=time.perf_counter();before=state_sha256(model)
            tm=run_epoch(model,loaders['train'],device,optimizer,weights)
            require(before!=state_sha256(model),'Training did not update parameters')
            before=state_sha256(model)
            vm,rows=validation_with_rows(model,loaders['validation'],device)
            require(before==state_sha256(model),'Validation updated parameters')
            rng=rng_state();counts={k:d.read_calls for k,d in datasets.items()}
            evaluation=selector.consider(epoch,model,rows,datasets['validation'].records,vm['loss'])
            diag=save_predictions(out/'diagnostics'/f'epoch_{epoch}',rows,evaluation)
            require(same_rng(rng),'Patient diagnostics changed RNG')
            require(counts=={k:d.read_calls for k,d in datasets.items()},'Patient diagnostics added image reads')
            sync(device)
            h=dict(epoch=epoch,train=tm,validation=vm,patient_metrics=evaluation['metrics'],diagnostics=diag,
                training_parameters_changed=True,validation_parameters_unchanged=True,
                diagnostic_rng_unchanged=True,diagnostic_additional_reads=0,seconds=time.perf_counter()-t0)
            result['history'].append(h)
            with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(h,allow_nan=False)+'\n')
            with (out/'training.log').open('a') as f:f.write(f"epoch={epoch} train={tm['loss']} validation={vm['loss']} patient_nll={evaluation['metrics']['nll']}\n")
            write_json(out/'training_summary.json',result)
        result['training_read_counts']={k:d.read_calls for k,d in datasets.items()}
        require(result['training_read_counts']==dict(train=epochs*lengths['train'],validation=epochs*lengths['validation'],test=0),'Training read count mismatch')
        result['training_peak_memory']=memory(device);result['training_elapsed_seconds']=time.perf_counter()-start
        del model,optimizer
        if device=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        require(inputs==dict(split_samples=sha256(manifest),split_summary=sha256(summary)),'Input changed')
        prediction_sets={k:CountedDataset(manifest,k,summary) for k in datasets}
        result['reload']={}
        from predict import predict_pass
        for rule in ('patient_nll','slice_ce'):
            passes=[];durations=[]
            for attempt in range(2):
                restored,cp=load_selected(out/'selection',rule,SimpleCNN,config,provenance)
                restored=restored.to(device);before=state_sha256(restored);sync(device);t0=time.perf_counter()
                rows=predict_pass(restored,make_loader(prediction_sets['validation'],16,seed),device)
                sync(device);durations.append(time.perf_counter()-t0)
                require(state_sha256(restored)==before,'Reload evaluation updated model')
                evaluate_patients(rows,prediction_sets['validation'].records,expected_slices)
                passes.append(rows);del restored
            verify_rows(passes[0],passes[1])
            with (out/'diagnostics'/f"epoch_{cp['epoch']}"/'slice_predictions.csv').open() as f:old=list(csv.DictReader(f))
            verify_rows(passes[0],old)
            evaluation=evaluate_patients(passes[0],prediction_sets['validation'].records,expected_slices)
            pr=save_predictions(out/'reload'/rule,passes[0],evaluation)
            pr.update(epoch=cp['epoch'],checkpoint_sha256=sha256(out/'selection'/f'best_{rule}.pt'),
                repeat_equal=True,selected_epoch_equal=True,read_count=2*lengths['validation'],full_pass_seconds=durations)
            write_json(out/'reload'/rule/'summary.json',pr);result['reload'][rule]=pr
        result['reload_read_counts']={k:d.read_calls for k,d in prediction_sets.items()}
        require(result['reload_read_counts']==dict(train=0,validation=4*lengths['validation'],test=0),'Reload read count mismatch')
        result.update(status='passed',completed_epochs=epochs,selection=selector.summary,reload_peak_memory=memory(device),
            total_read_counts={k:datasets[k].read_calls+prediction_sets[k].read_calls for k in datasets},
            timing_scope='Full pipeline I/O, training, diagnostics and reload evaluation; not isolated inference latency')
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        (out/'error.log').write_text(traceback.format_exc());raise
    finally:
        result.update(elapsed_seconds=time.perf_counter()-start,actual_training_read_counts={k:d.read_calls for k,d in datasets.items()},
                      actual_reload_read_counts={k:d.read_calls for k,d in prediction_sets.items()})
        write_json(out/'training_summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','summary','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=('A','B'),required=True)
    p.add_argument('--epochs',type=int,choices=(3,30),default=3)
    p.add_argument('--seed',type=int,choices=(42,43,44),default=42)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    a=p.parse_args();run_cnn(a.manifest,a.summary,a.output_dir,a.arm,a.epochs,a.seed,a.device)

if __name__=='__main__':main()
