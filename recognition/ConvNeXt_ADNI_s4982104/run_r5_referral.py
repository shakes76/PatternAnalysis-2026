"""R5d: validation batch referral simulation at fixed main threshold 0.5.

Rank raw ensemble distance from 0.5 without labels. Report observed errors in
referred and retained groups; never assume a human corrects referred cases.
"""
import argparse
from decimal import Decimal, ROUND_CEILING
import json
import math
from pathlib import Path

from run_r5_ensemble import MODELS, checked, indexed, require, sha, write_csv
from run_r5_temperature import load_inputs as load_raw, validate_fit_rows
from run_r5_threshold import load_inputs as load_calibrated

RATES = (0., .05, .1, .2)


def rank_scores(scores):
    """Only patient_id and p_ad_raw may be accessed by this selection function."""
    require(bool(scores), 'Empty referral cohort')
    seen=set();ranked=[]
    for row in scores:
        pid=row['patient_id'];p=float(row['p_ad_raw'])
        require(isinstance(pid,str) and pid and pid not in seen,'Missing/duplicate patient')
        require(math.isfinite(p) and 0<=p<=1,'Invalid raw probability')
        seen.add(pid);ranked.append((abs(p-.5),pid))
    return [pid for _,pid in sorted(ranked)]


def select_referrals(scores,rate):
    require(type(rate) in (int,float) and math.isfinite(rate) and 0<=rate<=1,'Invalid referral rate')
    ranked=rank_scores(scores)
    count=int((Decimal(str(rate))*len(ranked)).to_integral_value(rounding=ROUND_CEILING))
    return ranked[:count]


def confusion(rows):
    cm={'tn':0,'fp':0,'fn':0,'tp':0}
    for r in rows:
        y=int(r['label']);pred=int(float(r['p_ad'])>=.5)
        cm[{(0,0):'tn',(0,1):'fp',(1,0):'fn',(1,1):'tp'}[y,pred]]+=1
    return cm


def divide(numerator,denominator):
    return numerator/denominator if denominator else None


def evaluate(rows,rate):
    rows=validate_fit_rows(rows)
    scores=[{'patient_id':r['patient_id'],'p_ad_raw':r['p_ad_raw']} for r in rows]
    referred=set(select_referrals(scores,rate))
    groups={'full':rows,'referred':[r for r in rows if r['patient_id'] in referred],
            'retained':[r for r in rows if r['patient_id'] not in referred]}
    result=dict(requested_referral_rate=rate,threshold=.5)
    for name,items in groups.items():
        cm=confusion(items);n=len(items);errors=cm['fp']+cm['fn']
        values=dict(n=n,**cm,ad_n=cm['tp']+cm['fn'],nc_n=cm['tn']+cm['fp'],errors=errors,
                    error_rate=divide(errors,n),accuracy=divide(n-errors,n),
                    ad_recall=divide(cm['tp'],cm['tp']+cm['fn']),
                    nc_specificity=divide(cm['tn'],cm['tn']+cm['fp']))
        result.update({name+'_'+k:v for k,v in values.items()})
    result.update(actual_referral_rate=result['referred_n']/result['full_n'],
                  coverage=result['retained_n']/result['full_n'],
                  error_capture_rate=divide(result['referred_errors'],result['full_errors']),
                  ad_miss_capture_rate=divide(result['referred_fn'],result['full_fn']))
    for k in ('n','tp','fn','tn','fp','errors','ad_n','nc_n'):
        require(result['full_'+k]==result['referred_'+k]+result['retained_'+k],'Count conservation failed')
    return result


def match_raw(raw,calibrated):
    raw=indexed(raw,'patient_id');calibrated=indexed(calibrated,'patient_id')
    require(raw.keys()==calibrated.keys(),'Raw/calibrated patient mismatch')
    for pid,row in raw.items():
        cal=calibrated[pid]
        require(all(str(row[k])==str(cal[k]) for k in ('image_id','label','slice_count','split')),'Raw/calibrated metadata mismatch')
        require(float(row['p_ad'])==float(cal['p_ad_raw']),'Raw probability differs from accepted R5a')


def run(ensemble_source,temperature_source,output):
    output=Path(output)
    require(not output.exists(),'Output already exists; use a new directory')
    raw,raw_hashes=load_raw(ensemble_source)
    calibrated,temperatures,cal_hashes=load_calibrated(temperature_source)
    hashes={**raw_hashes,**cal_hashes};results=[];flags={}
    for model in MODELS:
        rows=calibrated[model];match_raw(raw[model],rows)
        scores=[dict(patient_id=r['patient_id'],p_ad_raw=r['p_ad_raw']) for r in rows]
        ranked=rank_scores(scores);ranks={pid:i+1 for i,pid in enumerate(ranked)}
        selected={rate:set(select_referrals(scores,rate)) for rate in RATES}
        for rate in RATES:results.append(dict(model=model,**evaluate(rows,rate)))
        flags[model]=[]
        for r in sorted(rows,key=lambda r:r['patient_id']):
            p=float(r['p_ad']);raw_p=float(r['p_ad_raw'])
            flags[model].append(dict(r,confidence=max(p,1-p),uncertainty_distance=abs(raw_p-.5),
                referral_rank=ranks[r['patient_id']],is_error=int(int(p>=.5)!=int(r['label'])),
                is_ad_miss=int(int(r['label'])==1 and p<.5),
                **{f'refer_{int(rate*100)}_percent':int(r['patient_id'] in selected[rate]) for rate in RATES}))
    for path,digest in hashes.items():checked(path,digest)
    output.mkdir(parents=True)
    write_csv(output/'referral_metrics.csv',results)
    primary=[r for r in results if r['requested_referral_rate']==.1]
    write_csv(output/'primary_10_percent.csv',primary)
    for model,rows in flags.items():write_csv(output/f'{model}_referral_flags.csv',rows)
    from plot_r5_referral import render
    render(output/'referral_metrics.csv',output)
    summary=dict(status='passed',split='validation',patient_count=136,class_counts={'AD':44,'NC':92},
        protocol=dict(main_threshold=.5,primary_referral_rate=.1,referral_rates=list(RATES),
            ranking='raw ensemble abs(p_ad-0.5) ascending, then patient_id lexicographic',
            selection_inputs=['patient_id','p_ad_raw'],rounding='ceil(rate * batch size)',
            interpretation='referral simulation only; no human correction assumed',
            scope='development validation, fixed prespecified operating points; R5c thresholds not used'),
        denominators=dict(error_capture_rate='referred_errors / full_errors',
            ad_miss_capture_rate='referred_fn / full_fn',retained_error_rate='retained_errors / retained_n',
            retained_ad_recall='retained_tp / retained_ad_n',zero_denominators='null; blank in CSV'),
        temperatures=temperatures,results=results,primary_10_percent=primary,
        input_sha256=hashes,inputs_unchanged=True,image_reads={'train':0,'validation':0,'test':0},gpu_hours=0,
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ('run_r5_referral.py','plot_r5_referral.py','run_r5_threshold.py','run_r5_temperature.py','run_r5_ensemble.py')},
        output_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'referral_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ensemble-source',required=True,type=Path)
    parser.add_argument('--temperature-source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    report=run(args.ensemble_source,args.temperature_source,args.output)
    print(json.dumps(report['primary_10_percent'],indent=2))
