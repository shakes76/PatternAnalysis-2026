"""Paired one-epoch compatibility check, followed by validation prediction."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from train import run_training, write_json
from predict import run_prediction
from check_dataloader import require, sha256


def compare_pair(root):
    root=Path(root)
    results={arm:json.loads((root/arm/'training/training_summary.json').read_text()) for arm in ('A','B')}
    predictions={arm:json.loads((root/arm/'prediction/prediction_summary.json').read_text()) for arm in ('A','B')}
    a,b=results['A'],results['B']
    for arm in results:
        r=results[arm];p=predictions[arm]
        require(r['status']==p['status']=='passed', 'Run failed')
        require(r['completed_epochs']==1 and len(r['history'])==1,'Not a one-epoch check')
        require(r['read_counts']==dict(train=r['dataset_lengths']['train'],validation=r['dataset_lengths']['validation'],test=0),'Training read mismatch')
        require(p['read_counts']==dict(train=0,validation=2*r['dataset_lengths']['validation'],test=0),'Prediction read mismatch')
        require(p['checkpoint_sha256']==r['checkpoint_sha256'],'Prediction checkpoint mismatch')
        require(p['training_summary_sha256']==sha256(root/arm/'training/training_summary.json'),'Training provenance mismatch')
        require(p['input_sha256']==r['input_sha256'],'Prediction inputs mismatch')
        require(p['repeat_equal'] and p['parameters_unchanged'],'Prediction checks failed')
        require(r['history'][0]['validation']['loss']==r['history'][0]['validation']['unweighted_loss'],'Weighted validation')
    require(a['config']['class_weighting']=='none' and b['config']['class_weighting']=='balanced','Wrong arms')
    for k,v in a['config'].items():
        if k not in ('class_weighting','class_weight_details'):
            require(v==b['config'][k],f'Configuration mismatch: {k}')
    require(a['initial_state_sha256']==b['initial_state_sha256'],'Initialization mismatch')
    require(a['history'][0]['train']['order_sha256']==b['history'][0]['train']['order_sha256'],'Training order mismatch')
    require(predictions['A']['code_sha256']==predictions['B']['code_sha256'],'Prediction code mismatch')
    require(a['input_sha256']==b['input_sha256'] and a['code_sha256']==b['code_sha256'],'Input/code mismatch')
    require(a['config']['class_weight_details']['weights'] is None,'A weights must be absent')
    details=b['config']['class_weight_details'];counts=details['patient_counts'];n=sum(counts.values())
    require(details['weights']==[n/(2*counts[k]) for k in ('NC','AD')],'B weights mismatch')
    return dict(initialization_equal=True,training_order_equal=True,other_config_equal=True,
                code_and_inputs_equal=True,validation_unweighted=True,no_test_reads=True,
                weights=details['weights'])


def run_pair(manifest, summary, output_dir, device='cuda',
             expected_lengths={'train':8160,'validation':2720,'test':2720},
             expected_slices=20,expected_patients=136):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',generated_at=datetime.now(timezone.utc).isoformat(),
                runner_sha256=sha256(Path(__file__)),arms={})
    try:
        for arm,mode in [('A','none'),('B','balanced')]:
            base=out/arm;base.mkdir()
            training=run_training(manifest,summary,base/'training',device=device,epochs=1,
                batch_size=16,learning_rate=0.001,seed=42,expected_lengths=expected_lengths,class_weighting=mode)
            prediction=run_prediction(manifest,summary,base/'training/best.pt',base/'training/training_summary.json',
                base/'prediction',device=device,batch_size=16,expected_lengths=expected_lengths,
                expected_slices=expected_slices,expected_patients=expected_patients)
            result['arms'][arm]=dict(training_status=training['status'],prediction_status=prediction['status'],
                training_summary_sha256=sha256(base/'training/training_summary.json'),
                prediction_summary_sha256=sha256(base/'prediction/prediction_summary.json'))
        result.update(status='passed',error_count=0,comparison=compare_pair(out),
                      scope='One-epoch compatibility only; no evidence of weighting benefit')
    except Exception as exc:
        result.update(status='failed',error_count=1,error=f'{type(exc).__name__}: {exc}')
        (out/'e2_error.log').write_text(traceback.format_exc());raise
    finally:write_json(out/'e2_summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('manifest','summary','output-dir'):parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args();run_pair(args.manifest,args.summary,args.output_dir,args.device)

if __name__=='__main__':main()
