"""Audit the single locked reassessment and create report evidence, without inference."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as effects
import numpy as np
from PIL import Image

from compare_validation import read, sha, overlaps, verify_point


def analyse(root, run, image_root):
    code = root/'recognition/YOLO_s4905914'
    plan = read(code/'reassessment_plan.json')
    assert (run/'COMPLETED.txt').is_file()
    assert read(run/'reassessment_plan.json') == plan
    assert sha(code/'experiment_provisional.json') == plan['config_sha256'] == sha(run/'locked_config.json')
    cfg = read(code/'experiment_provisional.json')
    manifest = root/'data/known_lesion_v1/reassessment.csv'
    assert sha(manifest) == cfg['manifest_sha256']['reassessment']
    truth = {r['image_id']: r for r in csv.DictReader(manifest.open())}
    assert len(truth) == 165
    assets = code/'assets'; assets.mkdir(exist_ok=True)
    summary = {'status': 'passed', 'split': 'reassessment', 'fresh_blind_test': False,
               'course_acceptance_confirmed': False, 'primary_model': 'n',
               'source_run': str(run.relative_to(root)), 'analysis_source_sha256': sha(Path(__file__)),
               'plan_sha256': sha(code/'reassessment_plan.json'), 'models': {}}
    records, matched = {}, {}
    for model in ('n','s'):
        val_lock = root/f'outputs/provisional_20261017_669143/val_{model}/selection_lock.json'
        expected = plan['models'][model]
        assert sha(val_lock) == sha(run/f'selection_lock_{model}.json') == expected['selection_lock_sha256']
        e = read(run/f'eval_{model}/evaluation.json')
        assert e['split']=='reassessment' and e['fresh_blind_test'] is False
        assert e['config_sha256']==plan['config_sha256']
        assert e['manifest_sha256']==cfg['manifest_sha256']
        assert e['checkpoint_sha256']==expected['checkpoint_sha256']
        assert e['selection_lock_sha256']==expected['selection_lock_sha256']
        assert e['operating_point']['threshold']==expected['threshold']
        assert 'grid' not in e['operating_point']
        assert e['evaluator_source_sha256']==sha(code/'evaluation.py')
        training = root/f'outputs/provisional_20261017_669143/train_{model}/run.json'
        assert sha(training)==e['training_run_sha256']
        rows = read(run/f'eval_{model}/predictions.json')
        assert len(rows)==165 and {r['image_id'] for r in rows}==set(truth)
        for r in rows:
            gt = truth[r['image_id']]
            assert r['class_id']==int(gt['class_id'])
            x,y,w,h=[float(gt[k]) for k in ('center_x','center_y','box_width','box_height')]
            np.testing.assert_allclose(r['box'],[x-w/2,y-h/2,x+w/2,y+h/2],rtol=0,atol=1e-12)
        verified = verify_point(rows,e['operating_point'],cfg['evaluation']['match_iou'])
        records[model] = {r['image_id']:r for r in rows}
        matched[model] = {r['image_id']:r for r in verified}
        b = read(run/f'benchmark_{model}/benchmark.json')
        measures = read(run/f'benchmark_{model}/measurements.json')
        assert b['checkpoint_sha256']==expected['checkpoint_sha256']
        assert b['selection_lock_sha256']==expected['selection_lock_sha256']
        assert b['manifest_sha256']==cfg['manifest_sha256']['val']
        assert b['source_sha256']==sha(code/'benchmark_inference.py')
        assert len(measures)==495 and b['warmup_calls']==10 and b['timed_repeats']==3
        wall = np.asarray([m['wall_ms'] for m in measures])
        assert np.isfinite(wall).all() and (wall>0).all()
        assert math.isclose(b['wall']['mean_ms'],float(wall.mean()))
        assert math.isclose(b['serial_images_per_second'],1000/float(wall.mean()))
        summary['models'][model] = {**{k:e[k] for k in ('mAP50','mAP50_95','ap','operating_point','checkpoint_sha256','selection_lock_sha256')},
                                   'benchmark':b, 'evaluation_sha256':sha(run/f'eval_{model}/evaluation.json'),
                                   'predictions_sha256':sha(run/f'eval_{model}/predictions.json')}
    ids = sorted(truth)
    def retained(model,image_id):
        return [p for p in records[model][image_id]['predictions'] if p['confidence']>plan['models'][model]['threshold']]
    def qualified(model,image_id,cls=None,low_floor=False):
        r=records[model][image_id]; ps=r['predictions'] if low_floor else retained(model,image_id)
        return any(iou>=.5 and (cls is None or p['class_id']==cls) for p,iou in zip(ps,overlaps(r['box'],ps)))
    selections=[]; used=set()
    def choose(category,model,predicate):
        available=[i for i in ids if i not in used and predicate(i)]
        if not available: return False
        image_id=available[0];used.add(image_id)
        selections.append({'category':category,'selection_model':model,'image_id':image_id})
        return True
    choose('Localized benign prediction on malignant GT','n',lambda i:int(truth[i]['class_id'])==1 and not matched['n'][i]['tp'] and qualified('n',i,0))
    choose('Malignant false alert on benign GT','n',lambda i:int(truth[i]['class_id'])==0 and matched['n'][i]['alert'])
    choose('Malignant alert with insufficient localization','n',lambda i:int(truth[i]['class_id'])==1 and matched['n'][i]['alert'] and not matched['n'][i]['tp'])
    for model in ('n','s'):
        if choose('Additional same-class prediction after a TP',model,lambda i,m=model:matched[m][i]['tp'] and sum(p['class_id']==int(truth[i]['class_id']) for p in retained(m,i))>=2):break
    choose('No retained box; low-score malignant candidate exists','n',lambda i:int(truth[i]['class_id'])==1 and not retained('n',i) and qualified('n',i,1,True))
    if len(selections)<5:
        choose('Different detection outcomes between models','n/s',lambda i:int(truth[i]['class_id'])==1 and matched['n'][i]['tp']!=matched['s'][i]['tp'])
    assert 3<=len(selections)<=5
    for number,selection in enumerate(selections,1):
        image_id=selection['image_id']; path=image_root/f'{image_id}.jpg'
        im=Image.open(path).convert('RGB')
        fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
        selection['models']={};selection['class_id']=int(truth[image_id]['class_id'])
        selection['image_sha256']=sha(path)
        for j,model in enumerate(('n','s')):
            record=records[model][image_id]; pred=retained(model,image_id)
            ious=overlaps(record['box'],pred);ax=axes[j]
            ax.imshow(im,extent=(0,1,1,0));ax.set_aspect(im.height/im.width);ax.set_axis_off()
            for box,ls in [(record['box'],'--')]+[(p['box'],'-') for p in pred]:
                x1,y1,x2,y2=box; patch=Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='black',linewidth=1.3,linestyle=ls)
                patch.set_path_effects([effects.Stroke(linewidth=3,foreground='white'),effects.Normal()]);ax.add_patch(patch)
            lines=[f'{"M" if p["class_id"] else "B"} score={p["confidence"]:.3f}, IoU={iou:.3f}' for p,iou in zip(pred,ious)]
            ax.set_title(f'{model} at {plan["models"][model]["threshold"]:.2f}: '+('TP' if matched[model][image_id]['tp'] else 'FN')+'\n'+('\n'.join(lines) if lines else 'No retained prediction'),fontsize=8)
            selection['models'][model]={'gt_matched':matched[model][image_id]['tp'],'malignant_alert':matched[model][image_id]['alert'],
                'retained':[dict(p,IoU=float(iou)) for p,iou in zip(pred,ious)],
                'best_low_floor_malignant_score_at_IoU_0_5':max([p['confidence'] for p,iou in zip(record['predictions'],overlaps(record['box'],record['predictions'])) if p['class_id']==1 and iou>=.5],default=None),
                'pairwise_same_class_IoU':[float(overlaps(p['box'],[q])[0]) for k,p in enumerate(pred) for q in pred[k+1:] if q['class_id']==p['class_id']]}
        filename=f'reassessment_case_{number}.png';fig.savefig(assets/filename,dpi=180);plt.close(fig)
        selection['figure']='assets/'+filename
    summary['case_selection']='Lowest image ID per explicit category, without reusing an image. Categories are illustrative, not a random sample or causal experiment.'
    summary['cases']=selections
    summary['paired_malignant']={key:sum(int(truth[i]['class_id'])==1 and condition(i) for i in ids) for key,condition in {
        'both':lambda i:matched['n'][i]['tp'] and matched['s'][i]['tp'],
        'n_only':lambda i:matched['n'][i]['tp'] and not matched['s'][i]['tp'],
        's_only':lambda i:matched['s'][i]['tp'] and not matched['n'][i]['tp'],
        'neither':lambda i:not matched['n'][i]['tp'] and not matched['s'][i]['tp']}.items()}
    (code/'reassessment_results_669815.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (run/'paired_cases.csv').open('w',newline='') as stream:
        fields=['image_id','class_id','n_tp','s_tp','n_alert','s_alert'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({'image_id':i,'class_id':int(truth[i]['class_id']),**{f'{m}_{k}':matched[m][i][k] for m in ('n','s') for k in ('tp','alert')}} for i in ids)
    print(json.dumps({'status':'passed','models':{m:{'mAP50':v['mAP50'],'mAP50_95':v['mAP50_95'],'operating_point':v['operating_point'],'benchmark_wall':v['benchmark']['wall'],'fps':v['benchmark']['serial_images_per_second']} for m,v in summary['models'].items()},'cases':[{'id':c['image_id'],'category':c['category']} for c in selections]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--image-root',type=Path,required=True);a=p.parse_args();analyse(a.root.resolve(),a.run.resolve(),a.image_root)
