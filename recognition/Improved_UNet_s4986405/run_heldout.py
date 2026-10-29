"""Evaluate two frozen checkpoints on the complete held-out test split, without training."""
import argparse
from collections import Counter
import csv
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import statistics
import subprocess
import sys
import time

from compare_results import inspect_run, metric_rows, require, summed_matrices
from run_comparison import ARCHITECTURES, check_sources, read_json, write_json

HELDOUT_FILES=('run_heldout.py','heldout_protocol.json','heldout_evaluate.sbatch',
               'test_heldout.py','compare_results.py','run_comparison.py')
CASE_PATTERN=re.compile(r'case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz')


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def relative_home_path(value):
    require(isinstance(value,str) and value and not Path(value).is_absolute() and
            '..' not in Path(value).parts,'Expected safe home-relative path')
    return Path.home()/value


def validate_test_protocol(p):
    keys={'schema_version','protocol_id','frozen_date','scope','source_commit','runtime_relative',
          'data_root','test_samples','test_patients','evaluation','models','reporting','resource_budget','limitations'}
    require(isinstance(p,dict) and set(p)==keys and type(p['schema_version']) is int and p['schema_version']==1,
            'Unexpected held-out protocol schema')
    require(p['protocol_id']=='hipmri-heldout-numeric-v1','Unexpected held-out protocol ID')
    date.fromisoformat(p['frozen_date'])
    require(isinstance(p['source_commit'],str) and re.fullmatch(r'[0-9a-f]{40}',p['source_commit']),
            'Expected full frozen runtime commit')
    relative_home_path(p['runtime_relative'])
    require(isinstance(p['scope'],str) and p['scope'] and (Path(p['data_root']).is_absolute() or PurePosixPath(p['data_root']).is_absolute()), 'Invalid scope/data root')
    require(type(p['test_samples']) is int and p['test_samples']==540 and p['test_patients']==[40,41,42] and all(type(v) is int for v in p['test_patients']),
            'Expected all 540 test slices and patients 40-42')
    require(all(type(p['evaluation'].get(key)) is int for key in ('batch_size','max_samples','warmup_steps','repeats')),
            'Evaluation counts must be integers')
    require(p['evaluation']=={'device':'cuda','batch_size':1,'max_samples':0,'warmup_steps':3,
                             'repeats':3,'order':list(ARCHITECTURES)}, 'Unexpected full-test evaluation settings')
    require(isinstance(p['models'],list) and [m['architecture'] for m in p['models']]==list(ARCHITECTURES),
            'Expected both fixed architectures')
    for model in p['models']:
        require(set(model)=={'architecture','run_relative','checkpoint_epoch','checkpoint_sha256'},'Unexpected model fields')
        relative_home_path(model['run_relative'])
        require(type(model['checkpoint_epoch']) is int and model['checkpoint_epoch']>=1,'Invalid checkpoint epoch')
        require(isinstance(model['checkpoint_sha256'],str) and re.fullmatch(r'[0-9a-f]{64}',model['checkpoint_sha256']),
                'Invalid checkpoint digest')
    require(all(type(v) is int for v in p['reporting']['label_ids']),'Numeric IDs must be integers')
    require(p['reporting']=={'label_ids':list(range(6)),'metrics':['dice','iou','tp','fp','fn'],
            'aggregation':['pooled_original_pixel_counts','equal_patient_mean'],
            'model_selection':'Frozen minimum-validation-loss checkpoints; test results never select weights or hyperparameters'},
            'Unexpected fixed numeric-label reporting scope')
    require(p['resource_budget']=={'gpus':1,'gpu_type':'a100','time':'00:30:00'},'Unexpected evaluation budget')
    require(isinstance(p['limitations'],list) and p['limitations'] and
            all(isinstance(x,str) and x for x in p['limitations']),'Expected explicit limitations')
    return p


def runner_hashes(source):
    return {name:sha256_file(Path(source)/name) for name in HELDOUT_FILES}


def prepare_models(p):
    runs={}
    runtime=relative_home_path(p['runtime_relative']).resolve()
    for spec in p['models']:
        architecture=spec['architecture'];directory=relative_home_path(spec['run_relative']).resolve()
        run=inspect_run(directory,architecture)
        require(run['provenance']['source_commit']==p['source_commit'],'Frozen runtime commit mismatch')
        require(Path(run['protocol']['data']['root']).resolve()==Path(p['data_root']).resolve(),'Test data root differs from training data root')
        require(Path(run['provenance']['source_directory']).resolve()==runtime,'Frozen runtime directory mismatch')
        require(run['validation']['best_epoch']==spec['checkpoint_epoch'] and
                run['validation']['checkpoint_sha256']==spec['checkpoint_sha256'],'Fixed checkpoint metadata mismatch')
        checkpoint=directory/'training/best.pt'
        require(sha256_file(checkpoint)==spec['checkpoint_sha256'],'Fixed checkpoint bytes changed')
        runs[architecture]=run
    baseline,improved=(runs[name] for name in ARCHITECTURES)
    require(baseline['protocol']==improved['protocol'] and baseline['data']==improved['data'], 'Training protocols/manifests differ')
    require(baseline['provenance']['source_sha256']==improved['provenance']['source_sha256'],'Frozen runtime hashes differ')
    check_sources(runtime,baseline['provenance']['source_sha256'])
    return runtime,runs


def audit_test_files(p,runs):
    base=Path(p['data_root']);images=base/'keras_slices_test';masks=base/'keras_slices_seg_test'
    paths=sorted(images.glob('case_*.nii.gz'))
    require(len(paths)==p['test_samples'],'Unexpected full test slice count')
    manifest=[];seen=set()
    for image in paths:
        match=CASE_PATTERN.fullmatch(image.name)
        require(match is not None,'Unexpected test filename')
        patient,week,slice_id=map(int,match.groups());key=(patient,week,slice_id)
        require(key not in seen,'Duplicate logical test slice');seen.add(key)
        mask=masks/('seg_'+image.name[5:])
        require(mask.is_file(),'Missing test mask')
        manifest.append({'image':image.name,'mask':mask.name,'image_bytes':image.stat().st_size,
                         'mask_bytes':mask.stat().st_size,'patient_id':patient,'week':week,'slice_id':slice_id})
    observed={row['patient_id'] for row in manifest}
    require(sorted(observed)==p['test_patients'],'Unexpected test patient IDs')
    for run in runs.values():
        splits=run['config']['splits']
        require(not observed.intersection(splits['train_patients']) and
                not observed.intersection(splits['validation_patients']),'Test patients overlap training or validation')
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'manifest':manifest,'filename_size_manifest_sha256':digest,'scope':'Test filenames and byte sizes only'}


def build_test_plan(p,output,runtime,python):
    result=[]
    for trial in range(1,p['evaluation']['repeats']+1):
        for spec in p['models']:
            name=spec['architecture']
            destination=Path(output)/name/f'test_{trial}'
            command=[str(python),'-u',str(Path(runtime)/'evaluate.py'),'--checkpoint',
                     str(relative_home_path(spec['run_relative'])/'training/best.pt'),
                     '--data-root',p['data_root'],'--split','test','--output-dir',str(destination),
                     '--device','cuda','--max-samples','0','--warmup-steps','3']
            result.append((f'{name}_test_{trial}',command))
    return result


def verify_integrity(p,runtime,runs,source,expected_hashes,test_data=None):
    require(runner_hashes(source)==expected_hashes,'Held-out runner source changed')
    check_sources(runtime,runs['UNet2D']['provenance']['source_sha256'])
    for spec in p['models']:
        require(sha256_file(relative_home_path(spec['run_relative'])/'training/best.pt')==spec['checkpoint_sha256'],
                'Fixed checkpoint bytes changed')
    if test_data is not None:
        require(audit_test_files(p,runs)==test_data,'Test filename/size manifest changed')


def finite_seconds(value):
    require(type(value) in (int,float) and math.isfinite(value) and value>=0,'Invalid recorded timing')
    return value


def summarize_test_reports(output,p,runs,test_data):
    model_reports={};expected_files={row['image']:row for row in test_data['manifest']}
    patient_counts=Counter(row['patient_id'] for row in test_data['manifest'])
    case_counts=Counter((row['patient_id'],row['week']) for row in test_data['manifest'])
    for spec in p['models']:
        name=spec['architecture'];reports=[]
        for trial in range(1,4):
            report=read_json(Path(output)/name/f'test_{trial}/evaluation.json')
            require(report['architecture']==name and report['split']=='test','Unexpected architecture/split')
            require(report['smoke_only'] is False and report['partial_checkpoint'] is False and
                    report['limited_evaluation'] is False and report['max_samples']==0,'Partial test evaluation rejected')
            require(report['checkpoint_sha256']==spec['checkpoint_sha256'] and
                    report['checkpoint_epoch']==spec['checkpoint_epoch'],'Test checkpoint identity mismatch')
            require(report['model_config']==runs[name]['result']['training_gpu_profile']['model_config'],
                    'Test model configuration changed')
            require(report['processed_samples']==report['split_samples']==p['test_samples'] and
                    report['split_patient_ids']==report['processed_patient_ids']==p['test_patients'], 'Incomplete test coverage')
            require(report['device']=='cuda' and report['batch_size']==1 and report['warmup_steps_per_shape']==3,
                    'Unexpected test timing configuration')
            require(report['normalization']=='per-slice z-score','Expected frozen normalization')
            require(report['checkpoint_source_sha256']==runs[name]['config']['source_sha256'], 'Checkpoint source record differs')
            require(report['evaluation_source_sha256']=={key:runs[name]['provenance']['source_sha256'][key]
                    for key in ('evaluate.py','predict.py','model_factory.py','dataset.py','metrics.py')},
                    'Evaluation source record differs')
            metric_rows(report['pooled'])
            for group in ('patients','cases','slices'):
                for entry in report[group]:metric_rows(entry)
                require(summed_matrices(report[group])==report['pooled']['confusion'], 'Test grouped counts do not sum to pooled')
            require({row['patient_id']:row['samples'] for row in report['patients']}==dict(patient_counts),'Patient sample counts differ')
            require({(row['patient_id'],row['week']):row['samples'] for row in report['cases']}==dict(case_counts),'Case sample counts differ')
            require(len(report['slices'])==p['test_samples'] and
                    {row['image'] for row in report['slices']}==set(expected_files),'Test slice identity coverage differs')
            for entry in report['slices']:
                expected=expected_files[entry['image']]
                require(all(entry[key]==expected[key] for key in ('mask','patient_id','week','slice_id')),'Test slice identity differs')
                require(len(entry['shape'])==2 and all(type(v) is int and v>0 for v in entry['shape']) and
                        entry['pixels']==entry['shape'][0]*entry['shape'][1],'Invalid test native shape')
                finite_seconds(entry['model_forward_ms'])
            for entry in report['cases']:finite_seconds(entry['processing_seconds'])
            finite_seconds(report['median_model_forward_ms'])
            require(math.isclose(report['median_model_forward_ms'],statistics.median(row['model_forward_ms'] for row in report['slices']),
                                 rel_tol=1e-10,abs_tol=1e-12),'Recorded median differs from timed forwards')
            if reports:
                first=reports[0]
                require(report['pooled']['confusion']==first['pooled']['confusion'],'Repeated test predictions differ')
                for group,keys in (('patients',('patient_id',)),('cases',('patient_id','week')),('slices',('image',))):
                    current={tuple(row[k] for k in keys):row['confusion'] for row in report[group]}
                    original={tuple(row[k] for k in keys):row['confusion'] for row in first[group]}
                    require(current==original,'Repeated grouped test predictions differ')
                require(all(report[k]==first[k] for k in ('device_name','torch_version','torch_cuda_build')),
                        'Test runtime environment changed')
            reports.append(report)
        model_reports[name]=reports
    baseline,improved=(model_reports[name][0] for name in ARCHITECTURES)
    require(all(baseline[k]==improved[k] for k in ('device_name','torch_version','torch_cuda_build')),'Models used different runtime environments')
    for a,b in zip(sorted(baseline['slices'],key=lambda row:row['image']),sorted(improved['slices'],key=lambda row:row['image'])):
        require(a['shape']==b['shape'] and [sum(row) for row in a['confusion']]==[sum(row) for row in b['confusion']],
                'Per-slice target counts or shapes differ between models')
    rows=[];patient_rows=[];model_rows=[];case_times=[]
    for a,b in zip(metric_rows(baseline['pooled']),metric_rows(improved['pooled'])):
        require(a['target_pixels']==b['target_pixels'],'Test ground-truth counts differ between models')
        rows.append({'label_id':a['label_id'],'target_pixels':a['target_pixels'],
                     'baseline_dice':a['dice'],'improved_dice':b['dice'],
                     'dice_delta_improved_minus_baseline':None if a['dice'] is None or b['dice'] is None else b['dice']-a['dice'],
                     'baseline_iou':a['iou'],'improved_iou':b['iou'],
                     'iou_delta_improved_minus_baseline':None if a['iou'] is None or b['iou'] is None else b['iou']-a['iou']})
    for patient in p['test_patients']:
        a=next(row for row in baseline['patients'] if row['patient_id']==patient)
        b=next(row for row in improved['patients'] if row['patient_id']==patient)
        for lft,rgt in zip(metric_rows(a),metric_rows(b)):
            require(lft['target_pixels']==rgt['target_pixels'],'Patient target counts differ between models')
            patient_rows.append({'patient_id':patient,'label_id':lft['label_id'],'samples':a['samples'],
                                 'target_pixels':lft['target_pixels'],'baseline_dice':lft['dice'],'improved_dice':rgt['dice'],
                                 'baseline_iou':lft['iou'],'improved_iou':rgt['iou']})
    for spec in p['models']:
        name=spec['architecture'];reports=model_reports[name]
        timings=[report['median_model_forward_ms'] for report in reports]
        model_rows.append({'architecture':name,'selected_epoch':spec['checkpoint_epoch'],
                           'checkpoint_sha256':spec['checkpoint_sha256'],'processed_samples':p['test_samples'],
                           'patients':len(p['test_patients']),'cases':len(case_counts),
                           'device_name':reports[0]['device_name'],'torch_version':reports[0]['torch_version'],
                           'torch_cuda_build':reports[0]['torch_cuda_build'],
                           'median_of_forward_trial_medians_ms':statistics.median(timings),
                           'forward_trial_medians_ms_json':json.dumps(timings)})
        for trial,report in enumerate(reports,1):
            for case in report['cases']:
                case_times.append({'architecture':name,'trial':trial,
                    **{k:case[k] for k in ('patient_id','week','samples','pixels','processing_seconds')}})
    macro=[]
    for label in range(6):
        row={'label_id':label}
        for side in ('baseline','improved'):
            values=[r[side+'_dice'] for r in patient_rows if r['label_id']==label and r[side+'_dice'] is not None]
            row[side+'_equal_patient_mean_dice']=statistics.mean(values) if values else None
            row[side+'_defined_patients']=len(values)
        macro.append(row)
    return {'scope':'Held-out test comparison of fixed checkpoints; numerical IDs only',
            'per_class':rows,'per_patient':patient_rows,'equal_patient_mean_dice':macro,
            'model_summary':model_rows,'case_times':case_times,'limitations':p['limitations'],
            'timing_scope':baseline['latency_scope'],'case_time_scope':baseline['case_time_scope']}


def export_test_results(output,summary):
    scope={'protocol_id':summary['protocol']['protocol_id'],'source_commit':summary['source_commit'],
           'runner_commit':summary['runner_commit'],'scope':'full_heldout_test_numeric_labels'}
    for name,key in [('per_class_test.csv','per_class'),('patient_test_metrics.csv','per_patient'),
                     ('model_test_summary.csv','model_summary'),('case_test_times.csv','case_times'),
                     ('equal_patient_mean_test.csv','equal_patient_mean_dice')]:
        rows=[{**scope,**row} for row in summary[key]]
        with (Path(output)/name).open('x',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
    number=lambda value:'undefined' if value is None else f'{value:.6f}'
    lines=['# Fixed-model held-out test results','',summary['scope'],'',
           'Numeric IDs only; anatomical mapping and tutor feasibility check-off remain unconfirmed.','',
           '| ID | Baseline Dice | Improved Dice | Baseline IoU | Improved IoU |',
           '|---|---:|---:|---:|---:|']
    for row in summary['per_class']:
        lines.append('| '+' | '.join([str(row['label_id'])]+[number(row[k]) for k in
            ('baseline_dice','improved_dice','baseline_iou','improved_iou')])+' |')
    lines+=['','Runtime source commit: '+summary['source_commit'],
            '','Evaluation runner commit: '+summary['runner_commit'],'','Limitations:','']
    lines+=['- '+item for item in summary['limitations']]
    (Path(output)/'TEST_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    render_test_plots(summary,output)


def render_test_plots(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import numpy as np
    output=Path(output)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    fig.suptitle('Held-out test | fixed models | anatomical mapping unverified')
    x=np.arange(6)
    for axis,metric in zip(axes,('dice','iou')):
        for offset,side,title,color in [(-0.2,'baseline','Baseline','tab:blue'),(0.2,'improved','Improved','tab:orange')]:
            values=[row[side+'_'+metric] for row in summary['per_class']]
            axis.bar(x+offset,[np.nan if v is None else v for v in values],0.38,label=title,color=color)
            for position,value in zip(x+offset,values):
                if value is None:axis.text(position,0.03,'undefined',rotation=90,ha='center',va='bottom',fontsize=7)
        axis.set_xticks(x);axis.set_xlabel('Numeric label ID');axis.set_ylim(0,1.04)
        axis.set_title('Pooled test '+metric.upper());axis.grid(axis='y',alpha=0.25);axis.legend()
    fig.savefig(output/'heldout_metrics_comparison.png',dpi=160);plt.close(fig)
    patients=summary['protocol']['test_patients']
    differences=[]
    for patient in patients:
        values=[]
        for label in range(6):
            row=next(x for x in summary['per_patient'] if x['patient_id']==patient and x['label_id']==label)
            a,b=row['baseline_dice'],row['improved_dice']
            values.append(np.nan if a is None or b is None else 100*(b-a))
        differences.append(values)
    matrix=np.asarray(differences)
    finite=np.abs(matrix[np.isfinite(matrix)])
    limit=max(0.01,float(finite.max())) if finite.size else 1
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    fig.suptitle('Held-out patient differences and repeated model-forward timing')
    plotted=axes[0].imshow(np.ma.masked_invalid(matrix),cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    axes[0].set_xticks(range(6),range(6));axes[0].set_yticks(range(len(patients)),patients)
    axes[0].set_xlabel('Numeric label ID');axes[0].set_ylabel('Patient ID')
    axes[0].set_title('Dice change: Improved minus baseline (pp)')
    for row in range(len(patients)):
        for column in range(6):
            value=matrix[row,column]
            axes[0].text(column,row,'n/a' if not np.isfinite(value) else f'{value:+.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(plotted,ax=axes[0],shrink=0.75,label='Percentage points')
    for index,model in enumerate(summary['model_summary']):
        timings=json.loads(model['forward_trial_medians_ms_json'])
        axes[1].scatter([index-0.07,index,index+0.07],timings)
        axes[1].hlines(model['median_of_forward_trial_medians_ms'],index-0.2,index+0.2,color='black')
    axes[1].set_xticks([0,1],['Baseline','Improved']);axes[1].set_ylabel('Model-forward ms/slice, batch 1')
    axes[1].set_title('Three pass medians; black line = median');axes[1].grid(axis='y',alpha=0.25)
    fig.savefig(output/'heldout_patients_and_timing.png',dpi=160);plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--runner-commit',required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    require(re.fullmatch(r'[0-9a-f]{40}',args.runner_commit) is not None,'Expected full committed runner SHA')
    source=Path(__file__).resolve().parent
    hashes=runner_hashes(source)
    p=validate_test_protocol(read_json(source/'heldout_protocol.json'))
    require(runner_hashes(source)==hashes,'Held-out runner changed while reading protocol')
    output=args.output_dir.expanduser().resolve()
    if output.exists():raise FileExistsError(f'Held-out output already exists: {output}')
    runtime=relative_home_path(p['runtime_relative']).resolve()
    plan=build_test_plan(p,output,runtime,sys.executable)
    if args.dry_run:
        print(json.dumps({'protocol_id':p['protocol_id'],'runner_commit':args.runner_commit,
                          'source_commit':p['source_commit'],'commands':plan},indent=2))
        return plan
    require(bool(os.environ.get('SLURM_JOB_ID')),'Run full held-out evaluation inside one allocated Slurm job')
    runtime,runs=prepare_models(p)
    test_data=audit_test_files(p,runs)
    verify_integrity(p,runtime,runs,source,hashes,test_data)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',p);write_json(output/'test_data_manifest.json',test_data)
    provenance={'status':'prepared','protocol_id':p['protocol_id'],'source_commit':p['source_commit'],
                'runner_commit':args.runner_commit,'runner_source_sha256':hashes,
                'runtime_source_sha256':runs['UNet2D']['provenance']['source_sha256'],
                'checkpoint_sha256':{spec['architecture']:spec['checkpoint_sha256'] for spec in p['models']},
                'test_filename_size_manifest_sha256':test_data['filename_size_manifest_sha256'],
                'node':platform.node(),'slurm_job_id':os.environ['SLURM_JOB_ID'],'commands':plan}
    write_json(output/'test_provenance.json',provenance)
    started=time.perf_counter()
    try:
        for stage,command in plan:
            verify_integrity(p,runtime,runs,source,hashes,test_data)
            provenance['status']='running_'+stage;write_json(output/'test_provenance.json',provenance)
            print('Held-out stage:',stage,flush=True)
            print('Command:',json.dumps(command),flush=True)
            subprocess.run(command,cwd=runtime,check=True)
            verify_integrity(p,runtime,runs,source,hashes,test_data)
        summary=summarize_test_reports(output,p,runs,test_data)
        summary.update({'status':'exporting','schema_version':1,'protocol':p,'source_commit':p['source_commit'],
                        'runner_commit':args.runner_commit,'runner_source_sha256':hashes,
                        'runtime_source_sha256':provenance['runtime_source_sha256'],
                        'checkpoint_sha256':provenance['checkpoint_sha256'],
                        'test_filename_size_manifest_sha256':test_data['filename_size_manifest_sha256'],
                        'node':provenance['node'],'slurm_job_id':provenance['slurm_job_id']})
        export_test_results(output,summary)
        verify_integrity(p,runtime,runs,source,hashes,test_data)
        summary['status']='completed';summary['wall_seconds_including_evaluation_and_exports']=time.perf_counter()-started
        write_json(output/'test_comparison.json',summary)
        provenance['status']='completed';write_json(output/'test_provenance.json',provenance)
    except Exception as error:
        provenance['status']='failed';provenance['error']=f'{type(error).__name__}: {error}'
        write_json(output/'test_provenance.json',provenance)
        raise
    print('Output directory:',output,flush=True)
    print('Frozen-model full held-out test comparison: PASS',flush=True)
    return summary


if __name__=='__main__':
    main()
