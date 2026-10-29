"""Evaluate exactly the predeclared policy on existing validation raw scores."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path.insert(0,str(S/'code'))
from evaluation import gt_record,score_records
from single_lesion_policy import one_prediction
from compare import qualifies
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())

def evaluate(seed):
    p=read(S/'single_lesion_plan.json');assert sha(S/'single_lesion_policy.py')==p['code_sha256']
    folder=R/'outputs'/('known_followup_diagnostics_20261022'+('' if seed==3710 else '_seed3711'))
    diagnostic=read(folder/'raw_diagnosis.json')
    assert diagnostic['cpu_GPU_cached_operating_counts_match']
    raw=np.load(folder/'raw_two_class_scores.npz')
    manifest=R/'data/known_lesion_v1/val.csv';assert sha(manifest)==p['manifest_sha256']['val']
    rows=list(csv.DictReader(manifest.open()));assert set(raw.files)=={r['image_id'] for r in rows}
    grid=[]
    for t in p['threshold_grid']:
        records=[{**gt_record(row),'predictions':one_prediction(raw[row['image_id']],t)} for row in rows]
        score=score_records(records,0.)
        score.update(ratio_threshold=t,threshold_semantics='ratio threshold; detection-score cutoff is zero after single-box decision')
        grid.append(score)
    feasible=[x for x in grid if x['image_alerts']['benign_false_alert_rate']<=p['benign_false_alert_budget']]
    def key(x):
        m=x['classes']['Malignant']
        return (m['tp'],m['precision'] or 0,-x['image_alerts']['benign_fp'],x['ratio_threshold'])
    selected=max(feasible,key=key) if feasible else None
    old=R/'studies/known_lesion_optimization_20261021'
    baseline=(read(old/'results.json')['reports']['n_geometry'] if seed==3710 else
        read(old/'confirmation_results.json')['reports']['n_geometry_seed3711'])
    result={'status':'completed','seed':seed,'scope':'validation only','new_reassessment_inference':False,
        'plan_sha256':sha(S/'single_lesion_plan.json'),'raw_archive_sha256':sha(folder/'raw_two_class_scores.npz'),
        'checkpoint_sha256':diagnostic['checkpoint_sha256'],'selected':selected,'grid':grid,
        'passes_improvement_rule':qualifies({'selected':selected},baseline),
        'ratio_is_malignancy_probability':False,'AP_not_used_to_rank_this_policy':True}
    if selected:
        records=[{**gt_record(row),'predictions':one_prediction(raw[row['image_id']],selected['ratio_threshold'])} for row in rows]
        result['strict_matching']={str(t):score_records(records,0.,t) for t in (.75,.8)}
        (S/f'policy_predictions_seed{seed}.json').write_text(json.dumps(records,indent=2)+'\n')
    (S/f'policy_results_seed{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('grid','strict_matching')},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[3710,3711],default=3710)
    evaluate(parser.parse_args().seed)
