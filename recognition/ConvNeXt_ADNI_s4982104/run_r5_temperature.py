"""R5b patient-level temperature calibration on accepted R5a validation ensembles.

Numerics use the standard library; reliability figures require matplotlib.
No model, image or test-set access. Fit one positive scalar per model family.
"""
import argparse
import bisect
import json
import math
import platform
from collections import Counter
from pathlib import Path

from run_r5_ensemble import (MODELS, SEEDS, checked, indexed, metrics, probability,
                            read_csv, require, sha, write_csv)

ENSEMBLE_SUMMARY_SHA = 'cd2e5f92fd07b7154b3ba04ffdfe65f3aedca02a02577e5eb5b64c0c9cf2d109'
EPS = 1e-7
T_MIN, T_MAX = .05, 20.0
LOG_T_TOL = 1e-10
MAX_ITERATIONS = 256


def logit(p):
    p = float(p)
    require(math.isfinite(p) and 0 <= p <= 1, 'Invalid probability')
    p = max(EPS, min(1-EPS, p))
    return math.log(p) - math.log1p(-p)


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def scale(p, temperature):
    require(math.isfinite(temperature) and temperature > 0, 'Invalid temperature')
    return sigmoid(logit(p) / temperature)


def objective(rows, temperature):
    """Stable, uncapped Bernoulli NLL; clip input p only, before logit."""
    require(rows and math.isfinite(temperature) and temperature > 0, 'Invalid objective input')
    terms = []
    for row in rows:
        require(str(row['label']) in ('0', '1'), 'Invalid label')
        x = logit(row['p_ad']) / temperature
        # softplus(-x) for AD and softplus(x) for NC, without cancellation.
        margin = -x if int(row['label']) else x
        terms.append(max(margin, 0) + math.log1p(math.exp(-abs(margin))))
    return math.fsum(terms) / len(terms)


def validate_fit_rows(rows):
    by_id = indexed(rows, 'patient_id')
    require(set(str(r['label']) for r in rows) == {'0','1'}, 'Fit requires both classes')
    for row in rows:
        require(row['split'] == 'validation', 'Temperature fitting requires validation split')
        require(str(row['slice_count']) == '20' and bool(row['image_id']), 'Invalid patient scan metadata')
        probability(row)
    return [by_id[k] for k in sorted(by_id)]


def fit_temperature(rows, max_iterations=MAX_ITERATIONS):
    """Deterministic golden-section minimization on log(T), with endpoint checks.

Bernoulli NLL is convex in inverse temperature: the positive-temperature
objective is unimodal. Endpoints and T=1 are explicitly compared. Boundary,
flat or non-converged fits are saved as needs_review, never auto-approved.
"""
    rows = validate_fit_rows(rows)
    require(type(max_iterations) is int and max_iterations >= 1, 'Invalid iteration budget')
    left, right = math.log(T_MIN), math.log(T_MAX)
    ratio = (math.sqrt(5)-1)/2
    c = right-ratio*(right-left); d = left+ratio*(right-left)
    trace = []
    def evaluate(t):
        loss = objective(rows,t)
        trace.append(dict(temperature=t,nll=loss))
        return loss
    baseline = evaluate(1.0)
    low, high = evaluate(T_MIN), evaluate(T_MAX)
    fc, fd = evaluate(math.exp(c)), evaluate(math.exp(d))
    iteration = 0
    while right-left > LOG_T_TOL and iteration < max_iterations:
        if fc < fd:
            right,d,fd = d,c,fc
            c = right-ratio*(right-left); fc = evaluate(math.exp(c))
        else:
            left,c,fc = c,d,fd
            d = left+ratio*(right-left); fd = evaluate(math.exp(d))
        iteration += 1
    mid_t = math.exp((left+right)/2)
    middle = evaluate(mid_t)
    # Prefer the baseline on exact loss ties, then documented endpoints.
    loss,t = min([(baseline,1.0),(low,T_MIN),(high,T_MAX),(middle,mid_t)],key=lambda pair:pair[0])
    boundary = abs(math.log(t/T_MIN)) <= 1e-6 or abs(math.log(t/T_MAX)) <= 1e-6
    flat = max(baseline,low,high)-min(baseline,low,high) <= 1e-12
    converged = right-left <= LOG_T_TOL
    gradient = math.fsum(logit(r['p_ad'])*(scale(r['p_ad'],t)-int(r['label'])) for r in rows)/len(rows)
    reasons = []
    if boundary: reasons.append('temperature_at_boundary')
    if flat: reasons.append('temperature_not_identifiable')
    if not converged: reasons.append('optimizer_not_converged')
    if not boundary and not flat and abs(gradient) > 1e-6: reasons.append('stationarity_check_failed')
    return dict(status='needs_review' if reasons else 'passed',review_reasons=reasons,
        temperature=t,objective_nll=loss,baseline_nll=baseline,converged=converged,
        at_boundary=boundary,flat_objective=flat,gradient_inverse_temperature=gradient,
        method='golden_section_on_log_temperature',bounds=[T_MIN,T_MAX],
        log_temperature_tolerance=LOG_T_TOL,max_iterations=max_iterations,
        iterations=iteration,final_log_interval=[left,right],evaluations=trace)


def reliability(rows):
    """Confidence ECE: 10 equal-width [left,right) bins; final bin includes 1."""
    require(bool(rows), 'Empty reliability input')
    groups = [[] for _ in range(10)]
    edges = [i/10 for i in range(11)]
    for row in rows:
        p = float(row['p_ad'])
        require(math.isfinite(p) and 0 <= p <= 1 and str(row['label']) in ('0','1'), 'Invalid reliability input')
        confidence = max(p,1-p)
        k = min(9,bisect.bisect_right(edges,confidence)-1)
        groups[k].append((confidence,int(int(p>=.5)==int(row['label']))))
    bins = []
    for k,items in enumerate(groups):
        count = len(items)
        conf = math.fsum(c for c,_ in items)/count if count else None
        acc = sum(a for _,a in items)/count if count else None
        gap = abs(conf-acc) if count else None
        bins.append(dict(bin=k,left=edges[k],right=edges[k+1],count=count,
                         mean_confidence=conf,accuracy=acc,absolute_gap=gap))
    ece = math.fsum(b['count']/len(rows)*b['absolute_gap'] for b in bins if b['count'])
    return bins,ece


def calibrated_rows(rows, temperature):
    result = []
    for row in rows:
        p = float(row['p_ad']); q = scale(p,temperature)
        require(int(p>=.5)==int(q>=.5), 'Numerical class change after scaling')
        result.append(dict(row,p_ad_raw=p,p_ad=q,prediction_raw=int(p>=.5),
                           prediction=int(q>=.5),confidence_raw=max(p,1-p),
                           confidence=max(q,1-q),temperature=temperature))
    return result


def load_inputs(source):
    source = Path(source)
    summary_path = checked(source/'ensemble_summary.json',ENSEMBLE_SUMMARY_SHA)
    summary = json.loads(summary_path.read_text())
    require(summary['status']=='passed' and summary['split']=='validation' and summary['source_job']=='706895', 'Unaccepted R5a source')
    require(summary['calibrated'] is False and summary['seeds']==list(SEEDS), 'Wrong ensemble configuration')
    require(summary['patient_count']==136 and summary['class_counts']=={'AD':44,'NC':92}, 'Wrong cohort size')
    hashes = {str(summary_path.resolve()):ENSEMBLE_SUMMARY_SHA}
    models = {}; identities = None
    for model in MODELS:
        filename = f'{model}_patient_predictions.csv'
        path = checked(source/filename,summary['output_sha256'][filename])
        hashes[str(path.resolve())] = sha(path)
        rows = validate_fit_rows(read_csv(path))
        require(len(rows)==136 and Counter(int(r['label']) for r in rows)=={0:92,1:44}, 'Wrong cohort')
        current = {(r['patient_id'],r['image_id'],str(r['label'])) for r in rows}
        require(identities is None or identities==current, 'Model patient mismatch')
        identities = current
        for row in rows:
            ps=[float(row[f'p_ad_seed{s}']) for s in SEEDS]
            require(all(math.isfinite(p) and 0<=p<=1 for p in ps), 'Invalid seed probability')
            require(abs(math.fsum(ps)/3-float(row['p_ad'])) <= 1e-12, 'Wrong ensemble mean')
        models[model] = rows
    return models,hashes


def run(source, output):
    output = Path(output)
    require(not output.exists(), 'Output already exists; use a new directory')
    models,hashes = load_inputs(source)
    fits = {}; tables = {}; bin_rows = []; metric_rows = []
    for model,rows in models.items():
        fit = fit_temperature(rows); fits[model]=fit
        calibrated = calibrated_rows(rows,fit['temperature']); tables[model]=calibrated
        for stage,items,t in [('raw',rows,1.0),('calibrated',calibrated,fit['temperature'])]:
            bins,ece = reliability(items)
            bin_rows.extend(dict(model=model,stage=stage,**b) for b in bins)
            measured = metrics(items)
            measured['nll_probability_clipped_1e7'] = measured['nll']
            # Fit/evaluation use stable original-logit NLL without output clipping.
            measured['nll'] = objective(rows,t)
            metric_rows.append(dict(model=model,stage=stage,temperature=t,ece=ece,**measured))
        raw_m,cal_m = metric_rows[-2:]
        require(all(raw_m[k]==cal_m[k] for k in ('tn','fp','fn','tp','accuracy','ad_recall','nc_recall')), 'Classification changed')
    for path,digest in hashes.items(): checked(path,digest)
    output.mkdir(parents=True)
    for model,rows in tables.items():write_csv(output/f'{model}_patient_predictions.csv',rows)
    write_csv(output/'calibration_metrics.csv',metric_rows)
    write_csv(output/'reliability_bins.csv',bin_rows)
    (output/'temperature_fits.json').write_text(json.dumps(fits,indent=2,allow_nan=False)+'\n')
    from plot_r5_reliability import render
    render(output/'reliability_bins.csv',output)
    status = 'passed' if all(f['status']=='passed' for f in fits.values()) else 'needs_review'
    report = dict(status=status,calibration_accepted=status=='passed',split='validation',
        scope='development validation fit; independent test not evaluated',source_job='706895',
        temperatures={m:f['temperature'] for m,f in fits.items()},metrics=metric_rows,
        input_sha256=hashes,inputs_unchanged=True,patient_count=136,class_counts={'AD':44,'NC':92},
        config=dict(input_clip_epsilon=EPS,temperature_bounds=[T_MIN,T_MAX],threshold=.5,
            optimizer='golden_section_on_log_temperature',log_t_tolerance=LOG_T_TOL,max_iterations=MAX_ITERATIONS,
            weights='equal per patient',ece_bins=10,ece_definition='max(p,1-p) versus correctness; weighted absolute bin gaps',
            bins='[k/10,(k+1)/10), final bin includes 1; empty bins null',
            nll='stable Bernoulli NLL of clipped-input logits/T, no output clipping',
            legacy_nll='nll_probability_clipped_1e7 retains R5a probability-clipped reporting'),
        image_reads={'train':0,'validation':0,'test':0},gpu_hours=0,
        python_version=platform.python_version(),
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in
                     ('run_r5_temperature.py','run_r5_ensemble.py','plot_r5_reliability.py')},
        output_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'temperature_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=run(args.source,args.output)
    print(json.dumps(dict(status=result['status'],temperatures=result['temperatures'],metrics=result['metrics']),indent=2))
    raise SystemExit(0 if result['status']=='passed' else 2)
