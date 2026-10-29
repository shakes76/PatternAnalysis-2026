"""Compare expanded-data n/s at predeclared common validation thresholds.

Read-only with respect to models and data. No inference, training, grid refinement
or reassessment scoring. AP comes from the pinned Ultralytics implementation.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT/'recognition/YOLO_s4905914'))
from compare_validation import verify_point


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(value):
    return 'undefined' if value is None else f'{value:.2%}'


def audit_model(run, config_path, model):
    cfg = read(config_path)
    training_dir = run/f'train_{model}'
    validation_dir = run/f'val_{model}'
    train = read(training_dir/'run.json')
    ev = read(validation_dir/'evaluation.json')
    lock = read(validation_dir/'selection_lock.json')
    records = read(validation_dir/'predictions.json')
    assert (run/'COMPLETED.txt').is_file()
    assert train['status']=='completed' and train['model']==model
    assert ev['split']=='val' and lock['selection_split']=='val'
    assert sha(run/'locked_config.json')==sha(config_path)
    for item in (train,ev,lock):
        assert item['config_sha256']==sha(config_path)
        assert item['manifest_sha256']==cfg['manifest_sha256']
    assert train['config']==cfg
    assert train['initial_weights_sha256']==cfg['models'][model]['sha256']
    assert sha(training_dir/'fit/weights/best.pt')==train['best_sha256']==ev['checkpoint_sha256']==lock['checkpoint_sha256']
    assert sha(training_dir/'run.json')==ev['training_run_sha256']
    assert sha(STUDY/'code/evaluation.py')==ev['evaluator_source_sha256']
    for name,digest in train['source_sha256'].items():
        assert sha(STUDY/'code'/name)==digest,name
    args=yaml.safe_load((training_dir/'fit/args.yaml').read_text())
    for key,value in cfg['train'].items():
        assert args[key]==value,(model,key,args[key],value)
    trace_path=training_dir/'fit/optimizer_steps.jsonl'
    assert sha(trace_path)==train['optimizer_trace_sha256']
    trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(trace)==train['optimizer_steps']==8520
    history_path=training_dir/'fit/checkpoint_history.jsonl'
    assert sha(history_path)==train['checkpoint_history_sha256']
    history=[json.loads(line) for line in history_path.read_text().splitlines()]
    assert len(history)==train['checkpoint_selection']['completed_epochs']==60
    best=max(history,key=lambda r:(r['validation_mAP50_95'],r['epoch']))
    assert best['epoch']==train['checkpoint_selection']['best_epoch']
    assert train['checkpoint_selection']['weight_source']=='EMA'
    curves=list(csv.DictReader((training_dir/'fit/results.csv').open()))
    assert len(curves)==60 and all(math.isfinite(float(v)) for r in curves for v in r.values())
    inputs=read(run/'inputs/input_audit.json')
    staged=read(run/'inputs/staged/audit.json')
    assert inputs['status']=='passed' and inputs['decoded_images']==2593
    assert inputs['counts']=={'train':2263,'val':165,'reassessment':165}
    assert inputs['cross_split_exact_decoded_duplicates']==0
    assert sha(run/'inputs/staged/audit.json')==inputs['staged_audit_sha256']
    assert staged['config_sha256']==sha(config_path)
    assert len(staged['files'])==2593
    content={i:v['file_sha256'] for i,v in staged['files'].items()}
    assert hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()==train['staged_image_content_sha256']
    prepared=ROOT/'data/expanded_training_v1'
    for split,digest in cfg['manifest_sha256'].items():
        assert sha(prepared/f'{split}.csv')==digest
    truth={r['image_id']:r for r in csv.DictReader((prepared/'val.csv').open())}
    assert len(records)==len(truth)==165
    assert {r['image_id'] for r in records}==set(truth)
    assert Counter(r['class_id'] for r in records)=={0:124,1:41}
    for r in records:
        row=truth[r['image_id']]
        assert r['class_id']==int(row['class_id'])
        x,y,w,h=(float(row[k]) for k in ('center_x','center_y','box_width','box_height'))
        assert all(math.isclose(a,b,rel_tol=0,abs_tol=1e-12) for a,b in zip(r['box'],[x-w/2,y-h/2,x+w/2,y+h/2]))
    grid=ev['operating_point']['grid']
    assert [p['threshold'] for p in grid]==cfg['evaluation']['threshold_grid']
    replayed={p['threshold']:verify_point(records,p,.5) for p in grid}
    feasible=[p for p in grid if p['image_alerts']['benign_false_alert_rate']<=.1]
    chosen=max(feasible,key=lambda p:(p['classes']['Malignant']['recall'],
        p['classes']['Malignant']['precision'] or 0,-p['image_alerts']['benign_fp'],p['threshold']))
    assert chosen==ev['operating_point']['selected']
    assert chosen['threshold']==lock['threshold']
    assert not any(p.name.startswith(('eval_','reassessment')) for p in run.iterdir())
    return {'train':train,'evaluation':ev,'lock':lock,'records':records,'replayed':replayed,
            'trace':trace,'content':content,'curves':curves}


def compact_point(point):
    m=point['classes']['Malignant'];alerts=point['image_alerts']
    return {'threshold':point['threshold'],
        **{f'malignant_{k}':m[k] for k in ('tp','fp','fn','precision','recall')},
        'benign_false_alert_images':alerts['benign_fp'],
        'benign_false_alert_rate':alerts['benign_false_alert_rate'],
        'malignant_image_alerts':alerts['malignant_tp'],
        'malignant_image_alert_sensitivity':alerts['sensitivity']}


def main(n_run,s_run,output):
    plan=read(STUDY/'ns_comparison_plan.json')
    paths={'n':STUDY/'experiment_expanded.json','s':STUDY/'experiment_expanded_s.json'}
    configs={m:read(p) for m,p in paths.items()}
    for m,p in paths.items(): assert sha(p)==plan[f'{m}_config_sha256']
    assert {k for k in configs['n'] if configs['n'][k]!=configs['s'][k]}==set(plan['configuration_differences_only'])
    assert sha(n_run/'train_n/fit/weights/best.pt')==plan['n_checkpoint_sha256']
    assert sha(n_run/'val_n/evaluation.json')==plan['n_evaluation_sha256']
    assert sha(n_run/'val_n/predictions.json')==plan['n_predictions_sha256']
    assert sha(n_run/'val_n/selection_lock.json')==plan['n_selection_lock_sha256']
    assert sha(s_run/'ns_comparison_plan.json')==sha(STUDY/'ns_comparison_plan.json')
    data={m:audit_model(run,paths[m],m) for m,run in [('n',n_run),('s',s_run)]}
    assert data['n']['content']==data['s']['content']
    assert [t['lr'] for t in data['n']['trace']]==[t['lr'] for t in data['s']['trace']]
    models={}
    for m,d in data.items():
        ev,train=d['evaluation'],d['train']
        models[m]={'mAP50':ev['mAP50'],'mAP50_95':ev['mAP50_95'],'ap':ev['ap'],
            'selected_threshold':d['lock']['threshold'],'selected_point':ev['operating_point']['selected'],
            'fixed_points':{str(t):next(p for p in ev['operating_point']['grid'] if p['threshold']==t)
                            for t in [*plan['primary_fixed_thresholds'],plan['additional_fixed_reference_threshold']]},
            'training_seconds':train['elapsed_seconds'],'parameters':train['parameters'],
            'peak_cuda_allocated_bytes':train['peak_cuda_allocated_bytes'],
            'best_epoch':train['checkpoint_selection']['best_epoch'],'optimizer_calls':train['optimizer_steps'],
            'checkpoint_sha256':train['best_sha256'],
            'evaluation_sha256':sha((n_run if m=='n' else s_run)/f'val_{m}/evaluation.json')}
    table=[];pairs={};case_rows=[]
    grids={m:{p['threshold']:p for p in d['evaluation']['operating_point']['grid']} for m,d in data.items()}
    for t in plan['full_threshold_grid']:
        points={m:compact_point(grid[t]) for m,grid in grids.items()}
        n,s=points['n'],points['s']
        dtp=s['malignant_tp']-n['malignant_tp']
        dfa=s['benign_false_alert_images']-n['benign_false_alert_images']
        table.append({'threshold':t,'n':n,'s':s,'s_minus_n_malignant_tp':dtp,
                      's_minus_n_benign_false_alert_images':dfa,
                      'strict_joint_improvement_by_s':dtp>0 and dfa<0,
                      's_no_worse_on_both_and_better_on_one':dtp>=0 and dfa<=0 and (dtp>0 or dfa<0)})
        maps={m:{r['image_id']:r for r in d['replayed'][t]} for m,d in data.items()}
        tp=Counter(both=0,n_only=0,s_only=0,neither=0)
        false=Counter(both=0,n_only=0,s_only=0,neither=0)
        for i,a in maps['n'].items():
            b=maps['s'][i]
            va,vb=(a['tp'],b['tp']) if a['class_id']==1 else (a['alert'],b['alert'])
            key='both' if va and vb else 'n_only' if va else 's_only' if vb else 'neither'
            (tp if a['class_id']==1 else false)[key]+=1
            if t in [*plan['primary_fixed_thresholds'],plan['additional_fixed_reference_threshold']]:
                case_rows.append({'threshold':t,'image_id':i,'class_id':a['class_id'],
                    'n_detected':a['tp'],'s_detected':b['tp'],'n_malignant_alert':a['alert'],
                    's_malignant_alert':b['alert'],'n_class_correct_iou':a['class_correct_iou'],
                    's_class_correct_iou':b['class_correct_iou']})
        assert sum(tp.values())==41 and sum(false.values())==124
        pairs[str(t)]={'malignant_detection':dict(tp),'benign_false_alert':dict(false)}
    result={'status':'passed','scope':'expanded-data n/s exploratory validation comparison',
        'plan_sha256':sha(STUDY/'ns_comparison_plan.json'),'analysis_source_sha256':sha(Path(__file__)),
        'same_training_hyperparameters':True,'same_optimizer_learning_rates_each_call':True,
        'same_all_2593_image_file_hashes':True,'same_validation_images_and_labels':True,
        'validation_images':165,'validation_malignant':41,'validation_benign':124,
        'independently_replayed_working_points':28,'AP_independently_recomputed':False,
        'models':models,'same_threshold_comparisons':table,'paired_outcomes':pairs,
        'primary_points_with_strict_joint_improvement':[r['threshold'] for r in table
            if r['threshold'] in plan['primary_fixed_thresholds'] and r['strict_joint_improvement_by_s']],
        'all_grid_points_with_strict_joint_improvement':[r['threshold'] for r in table if r['strict_joint_improvement_by_s']],
        'reassessment_inference':False,'original_study_replaced':False,
        'limitations':plan['limitations']+['Models ran in separate jobs/nodes; measured training time is descriptive, not a controlled speed benchmark.']}
    output.mkdir(parents=True,exist_ok=False)
    for p in [output/'comparison.json',STUDY/'ns_results.json']:
        p.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    with (output/'paired_cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(case_rows[0]));writer.writeheader();writer.writerows(case_rows)
    flat=[]
    for row in table:
        for m in ('n','s'): flat.append({'model':m,**row[m]})
    with (output/'thresholds.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['扩充数据 n/s 对照：同一 validation、同一阈值','',
        '训练集均为 2,263 张；validation 均为 165 张（良性124、恶性41）。',
        '从各自原始 COCO 权重开始，训练参数相同：60轮、FP32、640、batch16、AdamW、仅翻转。',
        '两模型均完成8,520次参数更新；逐次学习率与全部原图校验值一致。','',
        '指标 | n | s']
    for title,values in [('恶性 AP50',[models[m]['ap']['1']['AP50'] for m in ('n','s')]),
                         ('良性 AP50',[models[m]['ap']['0']['AP50'] for m in ('n','s')]),
                         ('宏 mAP50:95',[models[m]['mAP50_95'] for m in ('n','s')])]:
        lines.append(f'{title} | {values[0]:.4f} | {values[1]:.4f}')
    for t in [*plan['primary_fixed_thresholds'],plan['additional_fixed_reference_threshold']]:
        row=next(r for r in table if r['threshold']==t)
        lines+=['',f'相同 confidence={t:.2f}']
        for m in ('n','s'):
            p=row[m]
            lines.append(f"{m}：恶性TP/FP/FN={p['malignant_tp']}/{p['malignant_fp']}/{p['malignant_fn']}，Recall={percent(p['malignant_recall'])}，Precision={percent(p['malignant_precision'])}；良性误警报={p['benign_false_alert_images']}/124（{percent(p['benign_false_alert_rate'])}）。")
        lines.append(f"s 是否同时增加恶性TP并减少良性误警报：{'是' if row['strict_joint_improvement_by_s'] else '否'}。")
    lines+=['','原14点网格内，分别按10%良性误警报上限选择（阈值不一定相同）：']
    for m in ('n','s'):
        p=compact_point(models[m]['selected_point'])
        lines.append(f"{m}：confidence={p['threshold']:.2f}，恶性TP={p['malignant_tp']}/41、Recall={percent(p['malignant_recall'])}，良性误警报={p['benign_false_alert_images']}/124（{percent(p['benign_false_alert_rate'])}）。")
    lines+=['','全部网格结果，不只展示有利点：','阈值 | n恶性TP | s恶性TP | n良性误警报 | s良性误警报']
    for row in table:
        lines.append(f"{row['threshold']:.2f} | {row['n']['malignant_tp']} | {row['s']['malignant_tp']} | {row['n']['benign_false_alert_images']} | {row['s']['benign_false_alert_images']}")
    lines+=['','资源（各自实际运行，非受控速度实验）：']
    for m in ('n','s'):
        d=models[m]
        lines.append(f"{m}：参数{d['parameters']:,}，训练{d['training_seconds']:.1f}秒，峰值CUDA分配{d['peak_cuda_allocated_bytes']/1024**3:.2f}GiB，选中第{d['best_epoch']}轮EMA权重。")
    lines+=['','边界：28个工作点匹配已独立复算；AP沿用固定版本Ultralytics，未独立实现。',
        '未知病灶/患者联系仍可能导致偏乐观，validation已多次用于开发；只有一个训练seed。',
        '本轮没有补充集推理，也没有替换原锁定主实验。结果不证明患者独立泛化或临床安全。',
        'n/s还存在初始化参数复用范围差异，不能把结果单独归因于网络参数量。']
    (STUDY/'NS_RESULTS_zh.txt').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(7.4,5.1),layout='constrained')
    for m,style,marker in [('n','-','o'),('s','--','s')]:
        points=[compact_point(grids[m][t]) for t in plan['full_threshold_grid']]
        ax.plot([p['benign_false_alert_rate']*100 for p in points],
                [p['malignant_recall']*100 for p in points],style,marker=marker,color='black',
                markerfacecolor='white' if m=='s' else 'black',markersize=4,label=f'YOLOv8{m}')
    ax.axvline(10,color='0.5',linestyle=':',label='10% comparison budget')
    ax.set(xlabel='Benign images with malignant alert (%)',ylabel='Malignant detection recall (%)',
           title='Same validation images; 14 predeclared thresholds',xlim=(-1,101),ylim=(-1,101))
    ax.grid(alpha=.18);ax.legend(frameon=False)
    fig.savefig(output/'threshold_tradeoff_ns.png',dpi=180)
    plt.close(fig)
    print(json.dumps({'status':'passed','primary_joint_improvement_thresholds':result['primary_points_with_strict_joint_improvement'],
                      'n_AP50_malignant':models['n']['ap']['1']['AP50'],
                      's_AP50_malignant':models['s']['ap']['1']['AP50']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-run',type=Path,required=True)
    parser.add_argument('--s-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();main(a.n_run.resolve(),a.s_run.resolve(),a.output.resolve())
