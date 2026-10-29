"""Audit the fixed final candidate and compare descriptively with cached baseline n.

No inference, threshold sweep, checkpoint selection or resampling is performed.
Detection replay and primary-box classification/IoU are independently calculated.
"""
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
from compare_validation import read,sha,overlaps,verify_point,close
from complete_course_metrics import compare_numeric
from classification_metrics import summarize_predictions


def audit(root,run,output):
    code=Path(__file__).resolve().parent
    plan=read(code/'course_closeout_plan.json');chosen=plan['selected_model']
    assert (run/'COMPLETED.txt').is_file()
    ev=read(run/'evaluation/evaluation.json');started=read(run/'evaluation/started.json')
    bundle=read(run/'bundle_manifest.json')
    device=read(run/'device_decision.json')
    assert device==read(root/'studies/final_reassessment_20261022/device_decision.json')
    assert not device['inference_results_seen_before_device_decision'] and ev['device']=='cpu'
    assert bundle==read(root/'studies/final_reassessment_20261022/bundle_manifest.json')
    assert ev['reporting_plan_sha256']==sha(code/'course_closeout_plan.json')==bundle['plan_sha256']==started['plan_sha256']
    assert ev['checkpoint_sha256']==chosen['checkpoint_sha256']==sha(root/chosen['checkpoint_path'])
    assert ev['config_sha256']==sha(root/chosen['config_path'])
    lock=root/chosen['evaluation_path']/'selection_lock.json'
    assert ev['selection_lock_sha256']==sha(lock)==plan['selection_records_sha256'][str(lock.relative_to(root))]
    for rel,h in plan['selection_records_sha256'].items():assert sha(root/rel)==h
    for name,h in plan['reporting_code_sha256'].items():assert sha(code/name)==h
    for name,h in started['source_sha256'].items():assert h==bundle['files']['code/'+name]==sha(code/name)
    manifest=root/'data/known_lesion_v1/reassessment.csv'
    assert ev['manifest_sha256']==sha(manifest)==plan['manifest_sha256']['reassessment']
    truth={r['image_id']:r for r in csv.DictReader(manifest.open())}
    rows=read(run/'evaluation/predictions.json')
    assert ev['predictions_sha256']==sha(run/'evaluation/predictions.json')
    assert len(rows)==len(truth)==165 and {r['image_id'] for r in rows}==set(truth)
    source=read(root/'outputs/provisional_20261017_669143/inputs/staged/audit.json')
    assert set(started['files'])==set(truth)
    for r in rows:
        gt=truth[r['image_id']]
        assert r['class_id']==int(gt['class_id'])
        x,y,w,h=[float(gt[k]) for k in ('center_x','center_y','box_width','box_height')]
        np.testing.assert_allclose(r['box'],[x-w/2,y-h/2,x+w/2,y+h/2],rtol=0,atol=1e-12)
        assert started['files'][r['image_id']]['file_sha256']==source['files'][r['image_id']]['file_sha256']
    assert not ev['fresh_blind_test'] and not ev['selection_changed'] and not ev['threshold_search']
    assert ev['split']=='reassessment' and ev['operating_point']['threshold']==chosen['confidence']==.4
    assert Counter(r['class_id'] for r in rows)=={0:124,1:41}
    compare_numeric(summarize_predictions(rows,.4,.5),ev['metrics'])
    point=ev['metrics']['detection'];matched=verify_point(rows,point,.5)
    mapped={r['image_id']:r for r in matched}
    matrix=np.zeros((2,3),dtype=int)
    primary={k:[] for k in ('primary_box_ignore_class','primary_box_correct_class')}
    counts=Counter();cases=[];used=set()
    per_image={}
    for r in sorted(rows,key=lambda r:r['image_id']):
        ps=sorted([p for p in r['predictions'] if p['confidence']>.4],key=lambda p:(-p['confidence'],p['class_id'],*p['box']))
        ious=overlaps(r['box'],ps);c=r['class_id'];pred=ps[0]['class_id'] if ps else 2
        matrix[c,pred]+=1
        primary['primary_box_ignore_class'].append((c,float(ious[0]) if ps else 0.))
        primary['primary_box_correct_class'].append((c,float(ious[0]) if ps and pred==c else 0.))
        localized_b=any(p['class_id']==0 and o>=.5 for p,o in zip(ps,ious))
        if c==1 and not mapped[r['image_id']]['tp']:
            category='localized_benign_prediction' if localized_b else 'no_retained_prediction' if not ps else 'other_localization_or_classification_failure'
            counts[category]+=1
        per_image[r['image_id']]={'class_id':c,'gt_box':r['box'],'detection_tp':mapped[r['image_id']]['tp'],
            'primary_class':pred if ps else None,'retained':[dict(p,iou=float(o)) for p,o in zip(ps,ious)]}
    cm=ev['metrics']['classification']
    assert matrix.tolist()==cm['confusion_matrix']
    correct=int(matrix[0,0]+matrix[1,1]);covered=int(matrix[:,:2].sum())
    close(correct/165,cm['accuracy']);close(covered/165,cm['coverage'])
    f1s=[]
    for i,name in enumerate(('Benign','Malignant')):
        tp=int(matrix[i,i]);fp=int(matrix[1-i,i]);fn=int(matrix[i].sum()-tp)
        c=cm['classes'][name]
        assert (tp,fp,fn)==(c['tp'],c['fp'],c['fn'])
        close(tp/(tp+fp) if tp+fp else None,c['precision']);close(tp/(tp+fn),c['recall'])
        f1=2*tp/(2*tp+fp+fn);close(f1,c['f1']);f1s.append(f1)
        d=point['classes'][name];close(2*d['tp']/(2*d['tp']+d['fp']+d['fn']),ev['metrics']['detection_f1'][name])
    close(float(np.mean(f1s)),cm['macro_f1'])
    close((124*f1s[0]+41*f1s[1])/165,cm['support_weighted_f1'])
    for kind in ev['metrics']['localization']:
        loc=ev['metrics']['localization'][kind]
        if kind in primary:vals=primary[kind]
        else:
            sourcekey={'all_gt_class_correct':'class_correct_iou','all_gt_class_agnostic':'class_agnostic_iou','tp_only':'tp_iou'}[kind]
            vals=[(r['class_id'],r[sourcekey]) for r in matched if r[sourcekey] is not None]
        close(float(np.mean([v for _,v in vals])) if vals else None,loc['overall']['mean'])
        assert len(vals)==loc['overall']['denominator']
        for i,name in enumerate(('Benign','Malignant')):
            v=[v for c,v in vals if c==i]
            close(float(np.mean(v)) if v else None,loc['per_class'][name]['mean'])
            assert len(v)==loc['per_class'][name]['denominator']
    close(float(np.mean([v['AP50'] for v in ev['ap'].values()])),ev['mAP50'])
    close(float(np.mean([v['AP50_95'] for v in ev['ap'].values()])),ev['mAP50_95'])
    oldspec=plan['cached_runs']['baseline_n_reassessment'];oldpath=root/oldspec['evaluation_path']
    assert sha(oldpath/'predictions.json')==oldspec['predictions_sha256']
    oldev=read(oldpath/'evaluation.json')
    oldmatched={r['image_id']:r for r in verify_point(read(oldpath/'predictions.json'),oldev['operating_point'],.5)}
    paired=Counter()
    for r in matched:
        if r['class_id']==1:
            old=oldmatched[r['image_id']]['tp'];new=r['tp']
            paired['both' if old and new else 'geometry_only' if new else 'original_n_only' if old else 'neither']+=1
    # Lowest ID per diagnostic category; illustrative, not a random or causal sample.
    predicates=[('Malignant GT with localized benign prediction',lambda c:c['class_id']==1 and not c['detection_tp'] and any(p['class_id']==0 and p['iou']>=.5 for p in c['retained'])),
        ('Benign GT with malignant alert',lambda c:c['class_id']==0 and any(p['class_id']==1 for p in c['retained'])),
        ('Malignant alert with insufficient localization',lambda c:c['class_id']==1 and not c['detection_tp'] and any(p['class_id']==1 for p in c['retained'])),
        ('No retained prediction on malignant GT',lambda c:c['class_id']==1 and not c['retained'])]
    for label,predicate in predicates:
        for identifier,c in per_image.items():
            if identifier not in used and predicate(c):
                cases.append({'image_id':identifier,'category':label,**c});used.add(identifier);break
    result={'status':'passed','new_reassessment_inference':True,'new_training':False,'selection_changed':False,
        'fresh_blind_test':False,'model':'n_geometry seed 3710','threshold':.4,'source_run':str(run.relative_to(root)),
        'plan_sha256':sha(code/'course_closeout_plan.json'),'evaluation_sha256':sha(run/'evaluation/evaluation.json'),
        'predictions_sha256':ev['predictions_sha256'],'checkpoint_sha256':ev['checkpoint_sha256'],
        'selection_lock_sha256':ev['selection_lock_sha256'],'manifest_sha256':ev['manifest_sha256'],
        'metrics':ev['metrics'],'ap':ev['ap'],'mAP50':ev['mAP50'],'mAP50_95':ev['mAP50_95'],
        'malignant_fn_partition':dict(counts),'paired_malignant_vs_original_n':{k:paired[k] for k in ('both','geometry_only','original_n_only','neither')},
        'cases':cases,'case_selection':'Lowest ID per available category, unique images; illustrative only',
        'hardware':(run/'hardware.txt').read_text().strip(),'device':'cpu',
        'device_decision_sha256':sha(run/'device_decision.json'),
        'command_resource':read(run/'command_resource.json'),'AP_independently_recomputed':False,
        'fixed_threshold_metrics_independently_recomputed':True,'audit_source_sha256':sha(Path(__file__))}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','model','threshold','mAP50','mAP50_95','malignant_fn_partition','paired_malignant_vs_original_n')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('final_candidate_reassessment_20261022.json'))
    a=p.parse_args();audit(a.root.resolve(),a.run.resolve(),a.output.resolve())
