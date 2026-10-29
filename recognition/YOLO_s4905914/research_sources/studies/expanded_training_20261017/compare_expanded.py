"""Audit the completed exploratory n run against the same baseline validation set.

No training, inference, new threshold grid or supplementary assessment occurs.
The primary metric and displayed working points were declared before this run.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
BASE_CODE = ROOT / 'recognition/YOLO_s4905914'
sys.path.insert(0, str(BASE_CODE))
from compare_validation import overlaps, verify_point


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(value):
    return 'undefined (zero denominator)' if value is None else f'{value:.2%}'


def audit(run, config_path, prepared, source_code, counts):
    cfg = read(config_path)
    train, evaluation, lock, records = (read(run / p) for p in (
        'train_n/run.json', 'val_n/evaluation.json', 'val_n/selection_lock.json', 'val_n/predictions.json'))
    assert train['status'] == 'completed' and train['model'] == 'n'
    assert evaluation['split'] == 'val' and lock['selection_split'] == 'val'
    for item in (train, evaluation, lock):
        assert item['config_sha256'] == sha(config_path)
        assert item['manifest_sha256'] == cfg['manifest_sha256']
    assert train['config'] == cfg
    assert sha(run/'locked_config.json') == sha(config_path)
    assert sha(run/'train_n/fit/weights/best.pt') == train['best_sha256'] == evaluation['checkpoint_sha256'] == lock['checkpoint_sha256']
    assert train['initial_weights_sha256'] == cfg['models']['n']['sha256']
    assert sha(run/'train_n/run.json') == evaluation['training_run_sha256']
    assert sha(source_code/'evaluation.py') == evaluation['evaluator_source_sha256']
    assert train['checkpoint_selection']['completed_epochs'] == cfg['train']['epochs'] == 60
    expected_steps = math.ceil(counts['train']/16)*60
    assert train['optimizer_steps'] == expected_steps
    trace = run/'train_n/fit/optimizer_steps.jsonl'
    assert sha(trace) == train['optimizer_trace_sha256']
    assert len(trace.read_text().splitlines()) == expected_steps
    history_path=run/'train_n/fit/checkpoint_history.jsonl'
    assert sha(history_path)==train['checkpoint_history_sha256']
    history=[json.loads(line) for line in history_path.read_text().splitlines()]
    assert len(history)==60
    selected_epoch=max(history,key=lambda row:(row['validation_mAP50_95'],row['epoch']))
    assert selected_epoch['epoch']==train['checkpoint_selection']['best_epoch']
    curves=list(csv.DictReader((run/'train_n/fit/results.csv').open()))
    assert len(curves)==60
    assert all(math.isfinite(float(value)) for row in curves for value in row.values())
    for split in ('train', 'val', 'reassessment'):
        assert sha(prepared/f'{split}.csv') == cfg['manifest_sha256'][split]
    truth = {r['image_id']: r for r in csv.DictReader((prepared/'val.csv').open())}
    assert len(records) == len(truth) == 165
    assert {r['image_id'] for r in records} == set(truth)
    assert Counter(r['class_id'] for r in records) == {0:124,1:41}
    for record in records:
        row=truth[record['image_id']]
        assert record['class_id'] == int(row['class_id'])
        x,y,w,h=(float(row[k]) for k in ('center_x','center_y','box_width','box_height'))
        assert all(math.isclose(a,b,abs_tol=1e-12,rel_tol=0) for a,b in zip(record['box'],[x-w/2,y-h/2,x+w/2,y+h/2]))
    points = evaluation['operating_point']['grid']
    assert [p['threshold'] for p in points] == cfg['evaluation']['threshold_grid']
    rows_by_threshold = {p['threshold']:verify_point(records,p,0.5) for p in points}
    feasible = [p for p in points if p['image_alerts']['benign_false_alert_rate'] <= .1]
    selected = max(feasible, key=lambda p:(p['classes']['Malignant']['recall'],
        p['classes']['Malignant']['precision'] or 0,-p['image_alerts']['benign_fp'],p['threshold']))
    assert selected == evaluation['operating_point']['selected']
    assert selected['threshold'] == lock['threshold']
    wrong = []
    for name,digest in train['source_sha256'].items():
        if sha(source_code/name) != digest:
            wrong.append(name)
    assert set(wrong) <= {'transfer.py'}, wrong
    return train,evaluation,lock,records,rows_by_threshold


def main(run, output):
    output.mkdir(parents=True, exist_ok=False)
    baseline = ROOT/'outputs/provisional_20261017_669143'
    cfg=read(STUDY/'experiment_expanded.json')
    base_cfg=read(BASE_CODE/'experiment_provisional.json')
    for key in ('train','evaluation','models','classes','checkpoint_selection'):
        assert cfg[key] == base_cfg[key],key
    assert (run/'COMPLETED.txt').is_file()
    assert not any(p.name.startswith(('eval_', 'reassessment')) for p in run.iterdir())
    inputs=read(run/'inputs/input_audit.json')
    assert inputs['status']=='passed' and inputs['decoded_images']==2593
    assert inputs['cross_split_exact_decoded_duplicates']==0
    assert inputs['counts']=={'train':2263,'val':165,'reassessment':165}
    assert sha(STUDY/'experiment_expanded.json')==read(STUDY/'preparation.json')['config_sha256']
    models={}
    evaluations={};paired_rows={};records_by_name={}
    for name,run_dir,config_path,prepared,source,count in (
        ('restricted',baseline,BASE_CODE/'experiment_provisional.json',ROOT/'data/known_lesion_v1',BASE_CODE,768),
        ('expanded',run,STUDY/'experiment_expanded.json',ROOT/'data/expanded_training_v1',STUDY/'code',2263)):
        train,ev,lock,records,rows=audit(run_dir,config_path,prepared,source,{'train':count})
        evaluations[name]=ev;paired_rows[name]=rows;records_by_name[name]=records
        models[name]={'training_images':count,'training_epochs':60,'optimizer_calls':train['optimizer_steps'],
            'training_seconds':train['elapsed_seconds'],'best_epoch':train['checkpoint_selection']['best_epoch'],
            'mAP50':ev['mAP50'],'mAP50_95':ev['mAP50_95'],'ap':ev['ap'],
            'selected_threshold':lock['threshold'],'selected_point':ev['operating_point']['selected'],
            'fixed_0_05':next(p for p in ev['operating_point']['grid'] if p['threshold']==.05),
            'fixed_0_50':next(p for p in ev['operating_point']['grid'] if p['threshold']==.5),
            'checkpoint_sha256':train['best_sha256'],'evaluation_sha256':sha(run_dir/'val_n/evaluation.json'),
            'predictions_sha256':sha(run_dir/'val_n/predictions.json')}
    for name in ('restricted','expanded'):
        assert len(records_by_name[name])==165
    paired={}
    for threshold in (.05,.5):
        maps={name:{r['image_id']:r for r in rs[threshold]} for name,rs in paired_rows.items()}
        result=Counter(both=0,restricted_only=0,expanded_only=0,neither=0)
        for i,a in maps['restricted'].items():
            if a['class_id']!=1: continue
            b=maps['expanded'][i]
            key='both' if a['tp'] and b['tp'] else 'restricted_only' if a['tp'] else 'expanded_only' if b['tp'] else 'neither'
            result[key]+=1
        paired[str(threshold)]=dict(result)
    delta=models['expanded']['ap']['1']['AP50']-models['restricted']['ap']['1']['AP50']
    result={'status':'passed','scope':'exploratory_validation_comparison','primary_metric':'Malignant AP50',
        'primary_metric_expanded_minus_restricted':delta,'validation_images_identical':True,
        'validation_count':165,'malignant_validation_count':41,'independently_replayed_working_points':28,
        'AP_independently_recomputed':False,'reassessment_inference':False,'models':models,
        'paired_malignant_tp':paired,'analysis_source_sha256':sha(Path(__file__)),
        'input_audit_sha256':sha(run/'inputs/input_audit.json'),
        'limitations':['Unrecorded identity overlap remains possible; no complete lesion/patient independence claim.',
            'Same epochs but 8520 versus 2880 optimizer calls; not a pure sample-size causal experiment.',
            'Training case mix and class balance changed as well as size.',
            'Previously exposed validation reused for development; one seed and only 41 malignant validation images.',
            'Original locked study/model/thresholds are not replaced; no new blind evaluation.']}
    (output/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (STUDY/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    fields=['cohort','threshold','malignant_tp','malignant_fp','malignant_fn','malignant_precision','malignant_recall','benign_false_alert_images','benign_false_alert_rate']
    with (output/'validation_thresholds.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,ev in evaluations.items():
            for p in ev['operating_point']['grid']:
                m=p['classes']['Malignant'];alerts=p['image_alerts']
                writer.writerow(dict(cohort=name,threshold=p['threshold'],malignant_tp=m['tp'],malignant_fp=m['fp'],malignant_fn=m['fn'],malignant_precision=m['precision'],malignant_recall=m['recall'],benign_false_alert_images=alerts['benign_fp'],benign_false_alert_rate=alerts['benign_false_alert_rate']))
    lines=['训练扩充探索：相同 validation 上的 n 对照','',
        '主要指标在运行前声明为恶性 AP50。以下全部为同一 165 张 validation 图片，恶性 41、良性 124。',
        '补充集未重新推理；原实验阈值锁与报告未修改。','',
        '指标 | 原 768 张训练 | 扩充 2263 张训练',
        f"恶性 AP50 | {models['restricted']['ap']['1']['AP50']:.4f} | {models['expanded']['ap']['1']['AP50']:.4f}",
        f"宏 mAP50:95 | {models['restricted']['mAP50_95']:.4f} | {models['expanded']['mAP50_95']:.4f}",
        f"主要指标差值（扩充减原）: {delta:+.4f}",'']
    for title,key in [('固定 confidence=0.05','fixed_0_05'),('固定 confidence=0.50','fixed_0_50'),('同一 10% 良性误警报预算选择规则','selected_point')]:
        lines.append(title)
        for name,label in [('restricted','原训练'),('expanded','扩充训练')]:
            p=models[name][key];m=p['classes']['Malignant'];a=p['image_alerts']
            lines.append(f"{label}：阈值 {p['threshold']:.2f}；恶性 TP/FP/FN={m['tp']}/{m['fp']}/{m['fn']}；Recall={percent(m['recall'])}，Precision={percent(m['precision'])}；良性误警报 {a['benign_fp']}/124={percent(a['benign_false_alert_rate'])}。")
        lines.append('')
    lines+=['解释边界：','未知病例关联仍可能使结果偏乐观。训练更新次数、类别比例和病例构成也改变，不能单独归因于数据量。',
        '所有阈值已在原网格内；没有在补充集上调参，也不把此探索认定为教师批准或新的盲测。',
        '匹配统计独立复算通过；AP 使用固定版本 Ultralytics，不声称独立实现。']
    (STUDY/'RESULTS_zh.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'passed','primary_metric_delta':delta,'models':{n:{k:v[k] for k in ['mAP50_95','selected_threshold']} for n,v in models.items()}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args();main(a.run.resolve(),a.output.resolve())
