"""R3f: validation-only threshold analysis of the nine frozen R3e patient-NLL bests.

No model loading, training, image reads or probability calibration. Candidate
thresholds are unique patient probabilities plus zero; p >= threshold is AD.
Maximize NC specificity subject to AD recall >= 0.8, then prefer larger threshold.
"""
import argparse
import csv
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_bytes(path, hashes):
    path = Path(path)
    require(str(path) in hashes, 'Input absent from frozen R3e audit: ' + str(path))
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == hashes[str(path)], 'Frozen input changed: ' + str(path))
    return data


def metrics_at_threshold(patients, threshold):
    cm = [[0, 0], [0, 0]]
    for row in patients:
        cm[row['label']][int(row['p_ad'] >= threshold)] += 1
    tn, fp = cm[0]; fn, tp = cm[1]
    recall = tp / (tp + fn); specificity = tn / (tn + fp)
    f1_ad = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    f1_nc = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.
    return dict(threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp,
                accuracy=(tn + tp) / len(patients), ad_recall=recall,
                ad_precision=tp / (tp + fp) if tp + fp else 0.,
                nc_specificity=specificity, balanced_accuracy=(recall + specificity) / 2,
                macro_f1=(f1_ad + f1_nc) / 2)


def choose_threshold(patients, expected_patients, target_recall=.8):
    require(type(target_recall) in (int, float) and math.isfinite(target_recall)
            and 0 < target_recall <= 1, 'Invalid recall target')
    patients = list(patients); expected_patients = list(expected_patients)
    require(patients and expected_patients, 'Empty cohort')
    expected = {}
    for row in expected_patients:
        pid = row.get('patient_id')
        require(isinstance(pid, str) and pid and pid not in expected, 'Missing/duplicate expected patient')
        require(row.get('split') == 'validation', 'Threshold fitting is validation only')
        require(type(row.get('label')) is int and row['label'] in (0, 1), 'Invalid expected label')
        require(isinstance(row.get('image_id'), str) and row['image_id'], 'Invalid scan identity')
        require(type(row.get('slice_count')) is int and row['slice_count'] == 20, 'Expected 20 slices per patient')
        expected[pid] = row
    seen = set()
    for row in patients:
        pid = row.get('patient_id')
        require(isinstance(pid, str) and pid in expected and pid not in seen, 'Unexpected/duplicate patient')
        seen.add(pid)
        require(type(row.get('label')) is int and row['label'] in (0, 1), 'Invalid label')
        require(type(row.get('slice_count')) is int and row['slice_count'] == 20, 'Expected 20 slices')
        require(all(row.get(k) == expected[pid][k] for k in ('image_id', 'label', 'slice_count')), 'Cohort metadata mismatch')
        probability = row.get('p_ad')
        require(type(probability) in (int, float) and math.isfinite(probability)
                and 0 <= probability <= 1, 'Invalid probability')
    require(seen == set(expected), 'Missing patients')
    require({row['label'] for row in patients} == {0, 1}, 'Both classes required')
    positive_count = sum(row['label'] for row in patients)
    required_tp = int((Decimal(str(target_recall)) * positive_count).to_integral_value(rounding=ROUND_CEILING))
    candidates = [metrics_at_threshold(patients, t) for t in sorted({0., *(float(r['p_ad']) for r in patients)})]
    feasible = [r for r in candidates if r['tp'] >= required_tp]
    selected = max(feasible, key=lambda r: (r['tn'], r['threshold']))
    baseline = metrics_at_threshold(patients, .5)
    return dict(target_recall=target_recall, required_tp=required_tp, positive_count=positive_count,
                baseline=baseline, selected=selected,
                delta={k: selected[k] - baseline[k] for k in baseline if k != 'threshold'},
                candidates=candidates)


def analyze(main_csv, manifest, output_dir):
    """R3e audit hashes anchor all original records; main-row identities are cross-checked."""
    from patient_evaluation import evaluate_patients, POLICY
    main_csv, manifest, output_dir = map(Path, (main_csv, manifest, output_dir))
    require(not output_dir.exists(), 'Output already exists')
    audit_path = main_csv.with_name('audit_summary.json')
    audit_bytes = audit_path.read_bytes(); audit = json.loads(audit_bytes)
    require(audit.get('status') == 'passed' and audit.get('runs') == 9, 'R3e audit not accepted')
    frozen = audit['inputs_sha256']; touched = {}
    def read(path):
        data = verified_bytes(path, frozen); touched[str(path)] = hashlib.sha256(data).hexdigest()
        return data.decode()
    def load(path): return json.loads(read(path))
    def csv_rows(path): return list(csv.DictReader(read(path).splitlines()))
    for name in ('patient_evaluation.py', 'train.py'):
        read(Path(__file__).with_name(name))
    main_bytes = main_csv.read_bytes(); main = list(csv.DictReader(main_bytes.decode().splitlines()))
    identities = [(r['model'], int(r['seed'])) for r in main]
    require(len(main) == 9 and set(identities) == {(m, seed) for m in ('CNN-A', 'CNN-B', 'ConvNeXt-F') for seed in (42, 43, 44)}, 'Expected nine unique model/seed runs')
    expected = [r for r in csv_rows(manifest) if r['split'] == 'validation']
    require(len(expected) == 2720, 'Expected frozen validation cohort')
    output_rows, candidate_rows, decisions, details = [], [], [], []
    for row in main:
        source = Path(row['source']); seed = int(row['seed']); model = row['model']
        training = load(source/'training_summary.json'); selection = load(source/'selection/selection_summary.json')
        cfg = training['config']
        require(training['status'] == selection['status'] == 'passed', 'Run did not pass')
        require(selection == training['selection'] and selection['policy'] == POLICY, 'Selection policy mismatch')
        require(cfg['seed'] == seed and cfg['epochs'] == 30 and cfg['early_stopping'] is False and cfg['threshold'] == .5, 'Protocol mismatch')
        actual_model = 'ConvNeXt-F' if cfg['model'] == 'convnext_tiny' else 'CNN-' + cfg['arm']
        require(model == actual_model, 'Model identity mismatch')
        require(set(training['provenance']['input_sha256'].values()) == {frozen[str(manifest)], frozen[str(manifest.with_name('split_summary.json'))]}, 'Manifest mismatch')
        best = selection['best']['patient_nll']; history = training['history']
        require([x['epoch'] for x in history] == list(range(1, 31)), 'Incomplete history')
        earliest = min(history, key=lambda h: h['patient_metrics']['nll'])
        require(best['epoch'] == int(row['epoch']) == earliest['epoch'], 'Frozen best epoch mismatch')
        require(best['value'] == earliest['patient_metrics']['nll'], 'Selection value mismatch')
        folder = source/'reload/patient_nll'; summary = load(folder/'summary.json')
        require(summary['epoch'] == best['epoch'] and summary['checkpoint_sha256'] == best['sha256'], 'Reload checkpoint mismatch')
        require(summary['repeat_equal'] and summary['selected_epoch_equal'], 'Reload not verified')
        slices = csv_rows(folder/'slice_predictions.csv'); saved = csv_rows(folder/'patient_predictions.csv')
        for name, digest in summary['output_sha256'].items():
            require(touched.get(str(folder/name)) == digest, 'Prediction file hash mismatch')
        for r in slices: r.update(label=int(r['label']), p_ad=float(r['p_ad']))
        evaluation = evaluate_patients(slices, expected)
        patients = evaluation['patients']; metrics = evaluation['metrics']
        lookup = {r['patient_id']: r for r in saved}
        require(len(lookup) == len(saved) == 136, 'Patient CSV count mismatch')
        for patient in patients:
            saved_patient = lookup[patient['patient_id']]
            require(all(str(patient[k]) == saved_patient[k] for k in ('image_id','label','slice_count','prediction')), 'Patient CSV metadata mismatch')
            require(math.isclose(patient['p_ad'], float(saved_patient['p_ad']), abs_tol=1e-12), 'Patient probability mismatch')
        expected_patients = [dict(r, split='validation') for r in patients]
        result = choose_threshold(patients, expected_patients)
        for k, v in result['baseline'].items():
            require(math.isclose(v, float(row[k]), rel_tol=1e-10, abs_tol=1e-10), 'R3e baseline mismatch: ' + k)
        require(int(row['patients']) == len(patients), 'R3e patient count mismatch')
        for k in ('nll','brier','auroc'):
            require(math.isclose(metrics[k], float(row[k]), rel_tol=1e-10, abs_tol=1e-10), 'R3e invariant metric mismatch')
        # Independent order-statistic check: highest feasible threshold is the
        # required-th highest AD probability, including all ties at that value.
        boundary = sorted((r['p_ad'] for r in patients if r['label'] == 1), reverse=True)[result['required_tp']-1]
        require(result['selected']['threshold'] == boundary, 'Order-statistic independent check failed')
        identity = dict(model=model,seed=seed,epoch=best['epoch'],checkpoint_sha256=best['sha256'])
        output_rows.append(dict(**identity,target_recall=.8,required_tp=result['required_tp'],
            **{'baseline_'+k:v for k,v in result['baseline'].items()},
            **{'selected_'+k:v for k,v in result['selected'].items()},
            **{'delta_'+k:v for k,v in result['delta'].items()},
            **{'unchanged_'+k:metrics[k] for k in ('nll','brier','auroc')}))
        candidate_rows.extend(dict(model=model,seed=seed,**x,feasible=x['tp']>=result['required_tp']) for x in result['candidates'])
        decisions.extend(dict(model=model,seed=seed,patient_id=r['patient_id'],label=r['label'],p_ad=r['p_ad'],baseline_prediction=int(r['p_ad']>=.5),selected_prediction=int(r['p_ad']>=boundary)) for r in patients)
        details.append(dict(**identity,source=str(source),**{k:v for k,v in result.items() if k!='candidates'}))
    aggregates=[]
    for model in ('CNN-A','CNN-B','ConvNeXt-F'):
        rows=[r for r in output_rows if r['model']==model]
        fields=[k for k in rows[0] if k.startswith(('baseline_','selected_','delta_')) and not k.endswith('threshold')]
        aggregates.append(dict(model=model,n_seeds=3,**{k+suffix:fun([r[k] for r in rows]) for k in fields for suffix,fun in [('_mean',statistics.mean),('_sd',statistics.stdev)]}))
    for path,digest in touched.items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'Input changed during analysis')
    require(main_csv.read_bytes()==main_bytes and audit_path.read_bytes()==audit_bytes,'R3e index changed during analysis')
    output_dir.mkdir(parents=True,exist_ok=False)
    for name,rows in [('nine_thresholds.csv',output_rows),('model_mean_sample_sd.csv',aggregates),('all_candidate_thresholds.csv',candidate_rows),('patient_decisions.csv',decisions)]:
        with (output_dir/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(status='passed',scope='validation development analysis only',runs=details,
        rule='AD recall >= 0.8; maximize NC specificity; ties choose largest threshold; p >= threshold is AD',
        candidates='unique patient probabilities plus zero',baseline_threshold=.5,
        independent_order_statistic_check=True,probabilities_and_checkpoints_unchanged=True,
        training_runs=0,mri_reads=0,test_image_reads=0,inputs_sha256=touched,
        main_csv_sha256=hashlib.sha256(main_bytes).hexdigest(),r3e_audit_sha256=hashlib.sha256(audit_bytes).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output_dir.glob('*.csv')})
    (output_dir/'threshold_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-csv',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.main_csv,args.manifest,args.output_dir)
    print(json.dumps(dict(status=result['status'],runs=len(result['runs']),mri_reads=0),indent=2))
