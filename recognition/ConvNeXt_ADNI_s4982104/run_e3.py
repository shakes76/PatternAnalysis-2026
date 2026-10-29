"""E3 A42: fixed 30-epoch unweighted baseline and two validation evaluations."""
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


def compare_checkpoints(root, expected_epochs=30):
    root=Path(root)
    load=lambda path:json.loads(path.read_text())
    full=load(root/'training/training_summary.json')
    prefix=load(root/'training/first3/training_summary.json')
    require(full['status']==prefix['status']=='passed','Training/snapshot failed')
    require(full['completed_epochs']==len(full['history'])==expected_epochs,'Incomplete full training')
    require(prefix['completed_epochs']==len(prefix['history'])==3,'Incomplete first3')
    require(prefix['history']==full['history'][:3],'Prefix history mismatch')
    require(prefix['config']==full['config'],'Prefix config mismatch')
    require(prefix['initial_state_sha256']==full['initial_state_sha256'],'Prefix initialization mismatch')
    require(prefix['code_sha256']==full['code_sha256'] and prefix['input_sha256']==full['input_sha256'],'Prefix provenance mismatch')
    config=full['config']
    for k,v in dict(seed=42,batch_size=16,learning_rate=0.001,class_weighting='none',epochs=expected_epochs).items():
        require(config[k]==v,f'E3 config mismatch: {k}')
    require(config['class_weight_details']['weights'] is None,'E3 must be unweighted')
    require(full['first3_snapshot']['training_summary_sha256']==sha256(root/'training/first3/training_summary.json'), 'Prefix summary hash mismatch')
    require(full['first3_snapshot']['checkpoint_sha256']==prefix['checkpoint_sha256'],'Prefix checkpoint link mismatch')
    result={}
    for name,training,train_dir in [('first3',prefix,root/'training/first3'),('full',full,root/'training')]:
        require([h['epoch'] for h in training['history']]==list(range(1,len(training['history'])+1)),'Epoch sequence mismatch')
        best=min(training['history'],key=lambda h:h['validation']['loss'])
        require(training['best_epoch']==best['epoch'] and training['best_validation_loss']==best['validation']['loss'],'Best epoch mismatch')
        for row in training['history']:
            require(row['train']['samples']==training['dataset_lengths']['train'] and row['validation']['samples']==training['dataset_lengths']['validation'],'Incomplete split')
            require(row['training_parameters_changed'] and row['validation_parameters_unchanged'],'Parameter checks failed')
            require(row['validation']['loss']==row['validation']['unweighted_loss'],'Weighted validation')
        require(training['read_counts']==dict(train=len(training['history'])*training['dataset_lengths']['train'],
            validation=len(training['history'])*training['dataset_lengths']['validation'],test=0),'Training read mismatch')
        require(sha256(train_dir/'best.pt')==training['checkpoint_sha256'],'Checkpoint file hash mismatch')
        pred_dir=root/('prediction_'+name);pred=load(pred_dir/'prediction_summary.json')
        require(pred['status']=='passed' and pred['repeat_equal'] and pred['parameters_unchanged'],'Prediction failed')
        require(pred['training_summary_sha256']==sha256(train_dir/'training_summary.json'),'Prediction summary source mismatch')
        require(pred['checkpoint_sha256']==training['checkpoint_sha256'] and pred['best_epoch']==training['best_epoch'],'Prediction checkpoint mismatch')
        require(pred['input_sha256']==training['input_sha256'],'Prediction input mismatch')
        require(pred['read_counts']==dict(train=0,validation=2*training['dataset_lengths']['validation'],test=0),'Prediction read mismatch')
        for filename,digest in pred['output_sha256'].items():
            require(sha256(pred_dir/filename)==digest,'Prediction output hash mismatch')
        m=pred['patient_metrics']
        result[name]=dict(best_epoch=training['best_epoch'],best_validation_loss=training['best_validation_loss'],
            patient_accuracy=m['accuracy'],patient_ad_recall=m['per_class']['AD']['recall'],
            patient_nc_specificity=m['per_class']['NC']['recall'],patient_ad_precision=m['per_class']['AD']['precision'],
            patient_macro_f1=m['macro_f1'],patient_auroc=m['auroc'],
            patient_balanced_accuracy=(m['per_class']['AD']['recall']+m['per_class']['NC']['recall'])/2,
            patient_count=pred['patient_count'],slice_count=pred['slice_count'])
    return result


def run_baseline(manifest, summary, output_dir, device='cuda', epochs=30,
                 expected_lengths={'train':8160,'validation':2720,'test':2720},
                 expected_slices=20,expected_patients=136):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    result=dict(status='running',generated_at=datetime.now(timezone.utc).isoformat(),
                runner_sha256=sha256(Path(__file__)),evaluations={})
    try:
        require(epochs>=3,'At least 3 epochs required for prefix evaluation')
        training=run_training(manifest,summary,out/'training',device=device,epochs=epochs,
            batch_size=16,learning_rate=0.001,seed=42,expected_lengths=expected_lengths,class_weighting='none')
        predictions=[]
        for name,source in [('first3',out/'training/first3'),('full',out/'training')]:
            pred=run_prediction(manifest,summary,source/'best.pt',source/'training_summary.json',
                out/('prediction_'+name),device=device,batch_size=16,expected_lengths=expected_lengths,
                expected_slices=expected_slices,expected_patients=expected_patients)
            predictions.append(pred)
            result['evaluations'][name]=dict(training_summary_sha256=sha256(source/'training_summary.json'),
                checkpoint_sha256=sha256(source/'best.pt'),
                prediction_summary_sha256=sha256(out/('prediction_'+name)/'prediction_summary.json'))
        comparison=compare_checkpoints(out,expected_epochs=epochs)
        with (out/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['checkpoint']+list(comparison['first3']))
            writer.writeheader()
            for name,row in comparison.items():writer.writerow(dict(checkpoint=name,**row))
        # The first3 counts are a prefix of full training and must not be added twice.
        counts={s:training['read_counts'][s]+sum(p['read_counts'][s] for p in predictions)
                for s in ('train','validation','test')}
        result.update(status='passed',error_count=0,comparison=comparison,total_read_counts=counts,
            completed_epochs=epochs,comparison_sha256=sha256(out/'comparison.csv'),
            same_checkpoint=result['evaluations']['first3']['checkpoint_sha256']==result['evaluations']['full']['checkpoint_sha256'],
            scope='A42 unweighted; first3 vs full best on validation only. No performance improvement required.')
    except Exception as exc:
        result.update(status='failed',error_count=1,error=f'{type(exc).__name__}: {exc}')
        (out/'e3_error.log').write_text(traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        write_json(out/'e3_summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('manifest','summary','output-dir'):parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    # Public E3 CLI deliberately fixes the protocol to 30 epochs/A42/none.
    run_baseline(args.manifest,args.summary,args.output_dir,args.device)

if __name__=='__main__':main()
