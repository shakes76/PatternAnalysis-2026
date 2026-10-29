"""E5 fixed paired seeds 43/44 plus read-only seed42 references; validation only."""
import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import time
import traceback
from train import run_training, write_json
from predict import run_prediction
from run_e4 import load, preflight_reference, validate_arm, compare_arms, metric_row
from check_dataloader import require, sha256


def summarize_runs(rows):
    keys=[(r['seed'],r['arm']) for r in rows]
    require(len(keys)==6 and set(keys)=={(s,a) for s in (42,43,44) for a in ('A','B')},'Need exactly all six unique runs')
    by_key={(r['seed'],r['arm']):r for r in rows}
    metrics=[k for k in rows[0] if k.startswith('patient_')]
    require(all(set(k for k in r if k.startswith('patient_'))==set(metrics) for r in rows),'Inconsistent metric fields')
    deltas=[]
    for seed in (42,43,44):
        row=dict(seed=seed)
        for metric in metrics:
            a=by_key[seed,'A'][metric];b=by_key[seed,'B'][metric]
            require(all(v is None or math.isfinite(v) for v in (a,b)),'Nonfinite metric')
            row[metric]=None if a is None or b is None else b-a
        deltas.append(row)
    stats={}
    for group in ('A','B','B_minus_A'):
        selected=[by_key[s,group] for s in (42,43,44)] if group!='B_minus_A' else deltas
        stats[group]={}
        for metric in metrics:
            values=[r[metric] for r in selected]
            complete=all(v is not None for v in values)
            stats[group][metric]=dict(mean=statistics.mean(values) if complete else None,
                sample_std=statistics.stdev(values) if complete else None,
                n_available=sum(v is not None for v in values),n_required=3)
    wins=sum(d['patient_ad_recall']>0 for d in deltas)
    candidate=(wins>=2 and stats['B_minus_A']['patient_ad_recall']['mean']>0 and
               stats['B_minus_A']['patient_balanced_accuracy']['mean']>0 and
               stats['B_minus_A']['patient_macro_f1']['mean']>=0)
    return dict(statistics=stats,paired_differences=deltas,ad_recall_improved_pairs=wins,
        candidate_for_further_validation=candidate,
        decision_rule='At least 2 positive recall pairs; mean recall and balanced accuracy higher; mean macro-F1 no lower',
        interpretation='Exploratory validation decision only; seeds share patients, not independent clinical samples')


def check_references(a_dir,b_dir,manifest,summary,epochs,device):
    a_dir=Path(a_dir);b_dir=Path(b_dir)
    a=preflight_reference(a_dir,manifest,summary,epochs,device)
    paired=compare_arms(a_dir,b_dir,epochs)
    e=load(b_dir/'e4_summary.json');require(e['status']=='passed','E4 reference failed')
    for k,v in paired.items():require(e[k]==v,'Stored E4 comparison mismatch')
    for key in ('e3_summary_sha256','training_summary_sha256','prediction_summary_sha256','checkpoint_sha256','output_sha256'):
        require(e['A42_reference'][key]==a[key],'E4 A42 source mismatch')
    require(e['evaluations']['full']['training_summary_sha256']==sha256(b_dir/'training/training_summary.json'),'B42 training source mismatch')
    require(e['evaluations']['full']['prediction_summary_sha256']==sha256(b_dir/'prediction_full/prediction_summary.json'),'B42 prediction source mismatch')
    require(e['comparison_sha256']==sha256(b_dir/'comparison_A42_B42.csv'),'B42 comparison hash mismatch')
    return dict(A42=a,B42=dict(reference_dir=str(b_dir.resolve()),
        e4_summary_sha256=sha256(b_dir/'e4_summary.json'),
        training_summary_sha256=sha256(b_dir/'training/training_summary.json'),
        prediction_summary_sha256=sha256(b_dir/'prediction_full/prediction_summary.json')))


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run_repeats(manifest,summary,a42_dir,b42_dir,output_dir,device='cuda',epochs=30,
                expected_lengths={'train':8160,'validation':2720,'test':2720},expected_slices=20,expected_patients=136):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    result=dict(status='running',generated_at=datetime.now(timezone.utc).isoformat(),
        orchestrator_sha256={n:sha256(Path(__file__).with_name(n)) for n in ('run_e5.py','run_e4.py')},
        runs=[],paired_checks={},new_run_read_counts=dict(train=0,validation=0,test=0))
    try:
        require(epochs>=3,'At least 3 epochs required')
        references=check_references(a42_dir,b42_dir,manifest,summary,epochs,device)
        result['references']=references
        base_t,base_p=validate_arm(a42_dir,'none',epochs)
        def collect(root,seed,arm,new):
            t,p=validate_arm(root,'none' if arm=='A' else 'balanced',epochs,seed)
            require(t['input_sha256']==base_t['input_sha256'] and t['code_sha256']==base_t['code_sha256'],'Across-seed input/code mismatch')
            require(t['versions']==base_t['versions'] and t['gpu_name']==base_t['gpu_name'],'Across-seed runtime mismatch')
            require(p['code_sha256']==base_p['code_sha256'] and p['config']==base_p['config'],'Across-seed prediction mismatch')
            for k,v in base_t['config'].items():
                if k not in ('seed','class_weighting','class_weight_details'):require(v==t['config'][k],f'Across-seed config mismatch: {k}')
            prefix=load(root/'training/first3/training_summary.json')
            require(prefix['history']==t['history'][:3] and prefix['config']==t['config'],'Prefix mismatch')
            require(t['first3_snapshot']['training_summary_sha256']==sha256(root/'training/first3/training_summary.json'),'Prefix hash mismatch')
            require(prefix['checkpoint_sha256']==t['first3_snapshot']['checkpoint_sha256'],'Prefix checkpoint link mismatch')
            if new:
                require(sha256(root/'training/best.pt')==t['checkpoint_sha256'],'Best weight hash mismatch')
                require(sha256(root/'training/first3/best.pt')==prefix['checkpoint_sha256'],'Prefix weight hash mismatch')
                for s in result['new_run_read_counts']:result['new_run_read_counts'][s]+=t['read_counts'][s]+p['read_counts'][s]
            result['runs'].append(dict(seed=seed,arm=arm,**metric_row(t,p),
                training_seconds=t['elapsed_seconds'],prediction_seconds=p['elapsed_seconds'],
                peak_allocated_bytes=t['peak_cuda_memory']['allocated_bytes'] if t['peak_cuda_memory'] else None,
                peak_reserved_bytes=t['peak_cuda_memory']['reserved_bytes'] if t['peak_cuda_memory'] else None,
                training_summary_sha256=sha256(root/'training/training_summary.json'),
                prediction_summary_sha256=sha256(root/'prediction_full/prediction_summary.json')))
        collect(Path(a42_dir),42,'A',False);collect(Path(b42_dir),42,'B',False)
        result['paired_checks']['42']=compare_arms(a42_dir,b42_dir,epochs)['paired_checks']
        for seed in (43,44):
            for arm,mode in [('A','none'),('B','balanced')]:
                root=out/f'{arm}{seed}';root.mkdir()
                run_training(manifest,summary,root/'training',device=device,epochs=epochs,batch_size=16,
                    learning_rate=0.001,seed=seed,expected_lengths=expected_lengths,class_weighting=mode)
                run_prediction(manifest,summary,root/'training/best.pt',root/'training/training_summary.json',
                    root/'prediction_full',device=device,batch_size=16,expected_lengths=expected_lengths,
                    expected_slices=expected_slices,expected_patients=expected_patients)
                collect(root,seed,arm,True)
                write_json(out/'e5_summary.json',result) # preserve progress, status remains running
            result['paired_checks'][str(seed)]=compare_arms(out/f'A{seed}',out/f'B{seed}',epochs,seed)['paired_checks']
        require(references==check_references(a42_dir,b42_dir,manifest,summary,epochs,device),'Seed42 reference changed')
        result.update(summarize_runs(result['runs']))
        csv_write(out/'runs.csv',result['runs']);csv_write(out/'paired_differences.csv',result['paired_differences'])
        aggregates=[dict(group=g,metric=m,**v) for g,metrics in result['statistics'].items() for m,v in metrics.items()]
        csv_write(out/'aggregate.csv',aggregates)
        result.update(status='passed',error_count=0,completed_new_runs=4,
            output_sha256={n:sha256(out/n) for n in ('runs.csv','paired_differences.csv','aggregate.csv')},
            scope='Four new 30-epoch runs; existing seed42 read-only; validation only, no test evaluation')
    except Exception as exc:
        result.update(status='failed',error_count=1,error=f'{type(exc).__name__}: {exc}')
        (out/'e5_error.log').write_text(traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'e5_summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('manifest','summary','a42-dir','b42-dir','output-dir'):parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args();run_repeats(args.manifest,args.summary,args.a42_dir,args.b42_dir,args.output_dir,args.device)

if __name__=='__main__':main()
