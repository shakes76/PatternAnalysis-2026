"""E4 B42 fixed weighted 30-epoch run; read-only comparison with accepted E3 A42."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
from train import run_training, write_json
from predict import run_prediction
from check_dataloader import require, sha256


def load(path):
    return json.loads(Path(path).read_text())


def validate_arm(root, mode, epochs, seed=42):
    root=Path(root);t=load(root/'training/training_summary.json');p=load(root/'prediction_full/prediction_summary.json')
    require(t['status']==p['status']=='passed','Arm failed')
    require(t['completed_epochs']==len(t['history'])==epochs,'Incomplete epochs')
    for key,value in dict(epochs=epochs,seed=seed,batch_size=16,learning_rate=0.001,class_weighting=mode).items():
        require(t['config'][key]==value,f'Protocol mismatch: {key}')
    require(t['config']==load(root/'training/config.json'),'Config file mismatch')
    history=[json.loads(line) for line in (root/'training/epochs.jsonl').read_text().splitlines()]
    require(history==t['history'],'Epoch log mismatch')
    require([h['epoch'] for h in history]==list(range(1,epochs+1)),'Epoch sequence mismatch')
    for h in history:
        require(h['validation']['loss']==h['validation']['unweighted_loss'],'Weighted validation')
        require(h['training_parameters_changed'] and h['validation_parameters_unchanged'],'Parameter check failed')
        for split in ('train','validation'):require(h[split]['samples']==t['dataset_lengths'][split],'Incomplete split')
    best=min(history,key=lambda h:h['validation']['loss'])
    require(t['best_epoch']==best['epoch'] and t['best_validation_loss']==best['validation']['loss'],'Best checkpoint mismatch')
    require(t['read_counts']==dict(train=epochs*t['dataset_lengths']['train'],validation=epochs*t['dataset_lengths']['validation'],test=0),'Training reads mismatch')
    require(p['training_summary_sha256']==sha256(root/'training/training_summary.json'),'Training summary hash mismatch')
    require(p['checkpoint_sha256']==t['checkpoint_sha256'] and p['best_epoch']==t['best_epoch'],'Checkpoint source mismatch')
    require(p['input_sha256']==t['input_sha256'],'Prediction inputs mismatch')
    require(p['repeat_equal'] and p['parameters_unchanged'] and t['checkpoint_reload_equal'],'Reload/state check failed')
    require(p['read_counts']==dict(train=0,validation=2*t['dataset_lengths']['validation'],test=0),'Prediction reads mismatch')
    for name,digest in p['output_sha256'].items():require(sha256(root/'prediction_full'/name)==digest,'Prediction CSV hash mismatch')
    for k,v in p['slice_metrics'].items():require(v==best['validation'][k],'Selected slice metrics mismatch')
    return t,p


def preflight_reference(root, manifest, summary, epochs, device):
    root=Path(root);t,p=validate_arm(root,'none',epochs);e=load(root/'e3_summary.json')
    require(e['status']=='passed','E3 did not pass')
    require(e['evaluations']['full']['training_summary_sha256']==sha256(root/'training/training_summary.json'),'E3 training hash mismatch')
    require(e['evaluations']['full']['prediction_summary_sha256']==sha256(root/'prediction_full/prediction_summary.json'),'E3 prediction hash mismatch')
    require(e['evaluations']['full']['checkpoint_sha256']==t['checkpoint_sha256'],'E3 checkpoint link mismatch')
    require(sha256(root/'comparison.csv')==e['comparison_sha256'],'E3 comparison hash mismatch')
    require(t['input_sha256']==dict(split_samples=sha256(manifest),split_summary=sha256(summary)),'A42 input mismatch')
    require(t['config']['device']==device,'A/B device mismatch')
    for name,digest in {**t['code_sha256'],**p['code_sha256']}.items():
        require(sha256(Path(__file__).with_name(name))==digest,f'A42 source changed: {name}')
    return dict(reference_dir=str(root.resolve()),e3_summary_sha256=sha256(root/'e3_summary.json'),
        training_summary_sha256=sha256(root/'training/training_summary.json'),
        prediction_summary_sha256=sha256(root/'prediction_full/prediction_summary.json'),
        checkpoint_sha256=t['checkpoint_sha256'],output_sha256=p['output_sha256'])


def metric_row(t,p):
    m=p['patient_metrics']
    return dict(best_epoch=t['best_epoch'],best_validation_loss=t['best_validation_loss'],
        patient_accuracy=m['accuracy'],patient_ad_recall=m['per_class']['AD']['recall'],
        patient_nc_specificity=m['per_class']['NC']['recall'],patient_ad_precision=m['per_class']['AD']['precision'],
        patient_macro_f1=m['macro_f1'],patient_auroc=m['auroc'],
        patient_balanced_accuracy=(m['per_class']['AD']['recall']+m['per_class']['NC']['recall'])/2,
        patient_ad_true_positive=m['confusion_matrix'][1][1],patient_nc_false_positive=m['confusion_matrix'][0][1])


def compare_arms(a_dir,b_dir,epochs=30,seed=42):
    a,ap=validate_arm(a_dir,'none',epochs,seed);b,bp=validate_arm(b_dir,'balanced',epochs,seed)
    for k,v in a['config'].items():
        if k not in ('class_weighting','class_weight_details'):require(v==b['config'][k],f'A/B config mismatch: {k}')
    require(a['versions']==b['versions'] and a['gpu_name']==b['gpu_name'],'A/B runtime mismatch')
    require(a['code_sha256']==b['code_sha256'] and ap['code_sha256']==bp['code_sha256'],'A/B code mismatch')
    require(a['input_sha256']==b['input_sha256'],'A/B input mismatch')
    require(a['initial_state_sha256']==b['initial_state_sha256'],'A/B initialization mismatch')
    require([h['train']['order_sha256'] for h in a['history']]==[h['train']['order_sha256'] for h in b['history']],'A/B training order mismatch')
    require(ap['config']==bp['config'],'A/B prediction config mismatch')
    require(a['config']['class_weight_details']['weights'] is None,'A must have no weights')
    ac=a['config']['class_weight_details'];bc=b['config']['class_weight_details']
    require(ac['patient_counts']==bc['patient_counts'] and ac['slice_counts']==bc['slice_counts'],'A/B training counts mismatch')
    n=sum(bc['patient_counts'].values());require(bc['weights']==[n/(2*bc['patient_counts'][k]) for k in ('NC','AD')],'B weights mismatch')
    comparison={f'A{seed}':metric_row(a,ap),f'B{seed}':metric_row(b,bp)}
    delta={k:comparison[f'B{seed}'][k]-comparison[f'A{seed}'][k] for k in comparison[f'A{seed}'] if k.startswith('patient_') and comparison[f'A{seed}'][k] is not None and comparison[f'B{seed}'][k] is not None}
    return dict(comparison=comparison,delta_B_minus_A=delta,
                paired_checks=dict(initialization_equal=True,all_epoch_orders_equal=True,
                                   other_config_equal=True,code_inputs_runtime_equal=True,validation_unweighted=True))


def run_weighted(manifest,summary,a_dir,output_dir,device='cuda',epochs=30,
                 expected_lengths={'train':8160,'validation':2720,'test':2720},expected_slices=20,expected_patients=136):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    result=dict(status='running',generated_at=datetime.now(timezone.utc).isoformat(),runner_sha256=sha256(Path(__file__)),evaluations={})
    try:
        require(epochs>=3,'At least 3 epochs required')
        reference=preflight_reference(a_dir,manifest,summary,epochs,device)
        result['A42_reference']=reference
        t=run_training(manifest,summary,out/'training',device=device,epochs=epochs,batch_size=16,
            learning_rate=0.001,seed=42,expected_lengths=expected_lengths,class_weighting='balanced')
        prefix=load(out/'training/first3/training_summary.json')
        require(prefix['history']==t['history'][:3] and prefix['config']==t['config'],'B first3 prefix mismatch')
        require(sha256(out/'training/first3/training_summary.json')==t['first3_snapshot']['training_summary_sha256'],'B first3 hash mismatch')
        best=min(prefix['history'],key=lambda h:h['validation']['loss'])
        require(prefix['best_epoch']==best['epoch'] and prefix['best_validation_loss']==best['validation']['loss'],'B first3 best mismatch')
        predictions=[]
        for name,source in [('first3',out/'training/first3'),('full',out/'training')]:
            p=run_prediction(manifest,summary,source/'best.pt',source/'training_summary.json',out/('prediction_'+name),
                device=device,batch_size=16,expected_lengths=expected_lengths,expected_slices=expected_slices,expected_patients=expected_patients)
            predictions.append(p)
            result['evaluations'][name]=dict(training_summary_sha256=sha256(source/'training_summary.json'),
                checkpoint_sha256=sha256(source/'best.pt'),prediction_summary_sha256=sha256(out/('prediction_'+name)/'prediction_summary.json'))
        require(reference==preflight_reference(a_dir,manifest,summary,epochs,device),'A42 reference changed during E4')
        result.update(compare_arms(a_dir,out,epochs))
        with (out/'comparison_A42_B42.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['arm']+list(result['comparison']['A42']));writer.writeheader()
            for arm,row in result['comparison'].items():writer.writerow(dict(arm=arm,**row))
        result.update(status='passed',error_count=0,completed_epochs=epochs,
            total_read_counts={s:t['read_counts'][s]+sum(p['read_counts'][s] for p in predictions) for s in ('train','validation','test')},
            comparison_sha256=sha256(out/'comparison_A42_B42.csv'),
            scope='B42 versus existing A42; one paired seed, validation only; no effectiveness claim')
    except Exception as exc:
        result.update(status='failed',error_count=1,error=f'{type(exc).__name__}: {exc}')
        (out/'e4_error.log').write_text(traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;write_json(out/'e4_summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('manifest','summary','a42-dir','output-dir'):parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args();run_weighted(args.manifest,args.summary,args.a42_dir,args.output_dir,args.device)

if __name__=='__main__':main()
