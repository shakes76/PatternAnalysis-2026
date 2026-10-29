"""R5c: supplemental AD recall >=80% thresholds for calibrated patient ensembles.

Read accepted R5b validation CSVs only. Preserve main threshold 0.5, temperatures,
locked weights and historical R3f results. No training or image access.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from run_r5_ensemble import MODELS, SEEDS, checked, read_csv, require, sha, write_csv
from run_r5_temperature import scale, validate_fit_rows
from run_r3_threshold import choose_threshold

TEMPERATURE_SUMMARY_SHA = '0c13afa9f7b35b36d5fb04dacb3ff0ba4ad9e8d212fa10075c8adc51e27c7bd9'


def select(rows):
    """Reuse the accepted R3f rule after checking validation-only R5 rows."""
    rows = validate_fit_rows(rows)
    patients = [dict(r,label=int(r['label']),slice_count=int(r['slice_count']),p_ad=float(r['p_ad'])) for r in rows]
    return choose_threshold(patients, patients, target_recall=.8)


def validate_calibrated(rows, temperature):
    rows = validate_fit_rows(rows)
    require(math.isfinite(temperature) and .05 < temperature < 20, 'Unaccepted temperature')
    for row in rows:
        require(float(row['temperature']) == temperature, 'Temperature mismatch')
        raw = float(row['p_ad_raw']); calibrated = float(row['p_ad'])
        seeds = [float(row[f'p_ad_seed{s}']) for s in SEEDS]
        require(all(math.isfinite(p) and 0<=p<=1 for p in seeds), 'Invalid seed probability')
        require(math.isfinite(raw) and 0<=raw<=1 and abs(math.fsum(seeds)/3-raw)<=1e-12, 'Raw ensemble mismatch')
        require(abs(scale(raw,temperature)-calibrated)<=1e-12, 'Calibration probability mismatch')
        require(int(row['prediction_raw']) == int(raw>=.5) == int(calibrated>=.5), 'Main classification mismatch')
    return rows


def load_inputs(source):
    source = Path(source)
    path = checked(source/'temperature_summary.json',TEMPERATURE_SUMMARY_SHA)
    summary = json.loads(path.read_text())
    require(summary['status']=='passed' and summary['calibration_accepted'] is True, 'R5b not accepted')
    require(summary['split']=='validation' and summary['source_job']=='706895', 'Wrong source split/job')
    require(summary['patient_count']==136 and summary['class_counts']=={'AD':44,'NC':92}, 'Wrong cohort')
    require(set(summary['temperatures'])==set(MODELS), 'Missing/extra model temperatures')
    hashes={str(path.resolve()):TEMPERATURE_SUMMARY_SHA}
    def verified(name):
        p=checked(source/name,summary['output_sha256'][name])
        hashes[str(p.resolve())]=sha(p)
        return p
    fits=json.loads(verified('temperature_fits.json').read_text())
    require(set(fits)==set(MODELS), 'Wrong fitted model identities')
    models={};identities=None
    for model in MODELS:
        temperature=summary['temperatures'][model];fit=fits[model]
        require(fit['status']=='passed' and fit['temperature']==temperature and fit['converged'] is True
                and fit['at_boundary'] is False and not fit['review_reasons'], 'Unaccepted temperature fit')
        rows=validate_calibrated(read_csv(verified(f'{model}_patient_predictions.csv')),temperature)
        require(len(rows)==136 and Counter(int(r['label']) for r in rows)=={0:92,1:44}, 'Wrong patient/class counts')
        current={(r['patient_id'],r['image_id'],int(r['label'])) for r in rows}
        require(identities is None or current==identities, 'Model patient identities differ')
        identities=current;models[model]=rows
    return models,summary['temperatures'],hashes


def run(source,output):
    output=Path(output)
    require(not output.exists(), 'Output already exists; use a new directory')
    models,temperatures,hashes=load_inputs(source)
    details=[];comparisons=[];candidates=[];decisions={}
    for model,rows in models.items():
        result=select(rows);selected=result['selected'];baseline=result['baseline']
        require(result['required_tp']==36 and selected['tp']>=36,'Recall gate failed')
        details.append(dict(model=model,temperature=temperatures[model],**result))
        for stage,values in [('main_0.5',baseline),('supplement_ad80',selected)]:
            comparisons.append(dict(model=model,stage=stage,temperature=temperatures[model],**values))
        for candidate in result['candidates']:
            candidates.append(dict(model=model,**candidate,required_tp=36,feasible=candidate['tp']>=36,
                                   selected=candidate['threshold']==selected['threshold']))
        decisions[model]=[dict(r,prediction_main=int(float(r['p_ad'])>=.5),
                              threshold_ad80=selected['threshold'],
                              prediction_ad80=int(float(r['p_ad'])>=selected['threshold'])) for r in rows]
    for path,digest in hashes.items(): checked(path,digest)
    output.mkdir(parents=True)
    write_csv(output/'threshold_comparison.csv',comparisons)
    write_csv(output/'threshold_candidates.csv',candidates)
    costs=[]
    for result in details:
        base,chosen=result['baseline'],result['selected']
        costs.append(dict(model=result['model'],threshold=chosen['threshold'],
            fewer_ad_misses=base['fn']-chosen['fn'],additional_nc_false_alarms=chosen['fp']-base['fp'],
            baseline_fn=base['fn'],selected_fn=chosen['fn'],baseline_fp=base['fp'],selected_fp=chosen['fp'],
            accuracy_change=chosen['accuracy']-base['accuracy']))
    write_csv(output/'threshold_costs.csv',costs)
    for model,rows in decisions.items():write_csv(output/f'{model}_patient_decisions.csv',rows)
    (output/'threshold_details.json').write_text(json.dumps(details,indent=2,allow_nan=False)+'\n')
    summary=dict(status='passed',split='validation',source_job='706895',patient_count=136,
        class_counts={'AD':44,'NC':92},thresholds={r['model']:r['selected']['threshold'] for r in details},
        temperatures=temperatures,models=details,costs=costs,input_sha256=hashes,inputs_unchanged=True,
        protocol=dict(main_threshold=.5,target_recall=.8,required_ad_detections=36,
            decision='calibrated p_ad >= threshold',candidates='unique calibrated probabilities plus zero',
            selection='maximize NC specificity among feasible candidates; tie: highest threshold',
            calibration_order='20 slice means -> three-seed equal probability mean -> fixed patient T -> threshold',
            scope='development validation; test recall not guaranteed'),
        image_reads={'train':0,'validation':0,'test':0},gpu_hours=0,
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in
                     ('run_r5_threshold.py','run_r3_threshold.py','run_r5_temperature.py','run_r5_ensemble.py')},
        output_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'threshold_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=run(args.source,args.output)
    print(json.dumps(dict(status=result['status'],thresholds=result['thresholds'],costs=result['costs']),indent=2))
