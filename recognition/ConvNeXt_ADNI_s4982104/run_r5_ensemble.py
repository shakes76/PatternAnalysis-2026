"""R5a: audit job 706895 validation CSVs and average three seeds; no image IO.

Only Python's standard library is used. The accepted R4 summary and frozen
checkpoint manifest are pinned; a new source job requires a reviewed code change.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

SEEDS = (42, 43, 44)
MODELS = ('CNN-A', 'CNN-B', 'ConvNeXt-F')
LOCK_SHA = 'd781fb6bec3f9e378aaf6ddfa08229aadb2fc3c7361159f01e87b3be0106ebb4'
SUMMARY_SHA = '63c14e61db05f6927d4d14a2bb65757541210bdec9cd34cfeab908cf39f14200'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    require(sha(path) == expected, f'SHA mismatch: {path}')
    return Path(path)


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def indexed(rows, key):
    result = {r[key]: r for r in rows}
    require(bool(rows) and len(result) == len(rows), f'Empty/duplicate {key}')
    require(all(result), f'Empty {key}')
    return result


def probability(row):
    p = float(row['p_ad'])
    require(str(row['label']) in ('0', '1'), 'Invalid label')
    require(math.isfinite(p) and 0 <= p <= 1, 'Invalid probability')
    require(int(row['prediction']) == int(p >= .5), 'Inconsistent prediction')
    return p


def ensemble(seed_rows):
    require(set(seed_rows) == set(SEEDS), 'Expected seeds 42/43/44 exactly')
    maps = {s: indexed(seed_rows[s], 'patient_id') for s in SEEDS}
    require(all(m.keys() == maps[42].keys() for m in maps.values()), 'Missing patient')
    result = []
    for patient in sorted(maps[42]):
        rows = [maps[s][patient] for s in SEEDS]
        require(all(str(r['slice_count']) == '20' for r in rows), 'Expected 20 slices')
        require(all(all(str(r[k]) == str(rows[0][k]) for k in ('image_id', 'label')) for r in rows), 'Patient metadata mismatch')
        ps = [probability(r) for r in rows]
        p = math.fsum(ps) / 3
        result.append(dict(patient_id=patient, image_id=rows[0]['image_id'],
            label=int(rows[0]['label']), slice_count=20, split='validation',
            **{f'p_ad_seed{s}': v for s, v in zip(SEEDS, ps)}, p_ad=p, prediction=int(p >= .5)))
    return result


def validate_slices(patients, slices, manifest):
    """Validate exact frozen validation identities, then independently rebuild means."""
    expected = indexed(manifest, 'file_path')
    actual = indexed(slices, 'file_path')
    require(actual.keys() == expected.keys(), 'Slice identity mismatch')
    groups = defaultdict(list)
    for path, row in actual.items():
        ref = expected[path]
        require(ref['split'] == 'validation', 'Non-validation row')
        require(ref['label'] in ('NC', 'AD'), 'Invalid manifest label')
        require(all(str(row[k]) == str(ref[k]) for k in ('patient_id', 'image_id')), 'Slice metadata mismatch')
        require(int(row['label']) == int(ref['label'] == 'AD'), 'Slice label mismatch')
        groups[row['patient_id']].append(probability(row))
    pm = indexed(patients, 'patient_id')
    require(pm.keys() == groups.keys(), 'Patient identity mismatch')
    for pid, ps in groups.items():
        row = pm[pid]
        refs = [r for r in manifest if r['patient_id'] == pid]
        require(len(ps) == 20 and str(row['slice_count']) == '20', 'Slice count mismatch')
        require(len({r['image_id'] for r in refs}) == 1 and all(str(row['image_id']) == r['image_id'] and int(row['label']) == int(r['label']=='AD') for r in refs), 'Patient metadata mismatch')
        require(abs(probability(row) - math.fsum(ps)/20) <= 1e-12, 'Slice mean mismatch')


def metrics(rows):
    """Patient metrics, AD=1; AUC by pairwise comparison with half credit for ties."""
    ys = [int(r['label']) for r in rows]
    ps = [float(r['p_ad']) for r in rows]
    require(bool(rows) and set(ys) == {0, 1}, 'Metrics require both classes')
    require(all(math.isfinite(p) and 0 <= p <= 1 for p in ps), 'Invalid probability')
    counts = Counter((y, int(p >= .5)) for y,p in zip(ys,ps))
    tn,fp,fn,tp = (counts[k] for k in ((0,0),(0,1),(1,0),(1,1)))
    pos = [p for y,p in zip(ys,ps) if y]; neg = [p for y,p in zip(ys,ps) if not y]
    clipped = [max(1e-7,min(1-1e-7,p)) for p in ps]
    return dict(n=len(rows), tn=tn, fp=fp, fn=fn, tp=tp, accuracy=(tn+tp)/len(rows),
        ad_recall=tp/(tp+fn), nc_recall=tn/(tn+fp),
        macro_f1=(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))/2,
        auroc=sum((p>q)+.5*(p==q) for p in pos for q in neg)/(len(pos)*len(neg)),
        nll=math.fsum(-math.log(p if y else 1-p) for y,p in zip(ys,clipped))/len(rows),
        brier=math.fsum((p-y)**2 for y,p in zip(ys,ps))/len(rows))


def validate_summary(summary, entry, lock_sha):
    require(summary['status'] == 'passed' and summary['entry'] == entry, 'Wrong model/seed/epoch/checkpoint entry')
    require(summary['lock_sha256'] == lock_sha, 'Wrong lock')
    require(summary['environment']['slurm_job_id'] == '706895', 'Wrong job')
    expected = dict(split='validation',batch_size=4,precision='float32',tf32=False,amp=False)
    require(all(summary['protocol'].get(k) == v for k,v in expected.items()), 'Wrong unified protocol')
    require(all(summary.get(k) is True for k in ('historical_predictions_match_R3','unified_repeats_match_reference','parameters_unchanged')), 'Unaccepted R4 run')
    require(summary['read_counts']['train'] == summary['read_counts']['test'] == 0, 'Unexpected image reads')


def run(source, split_file, output):
    source,split_file,output = map(Path,(source,split_file,output))
    require(not output.exists(), 'Output already exists; use a new directory')
    inputs = {}
    def track(path, expected):
        path = checked(path,expected)
        inputs[str(path.resolve())] = expected
        return path
    lock_path = Path(__file__).with_name('r4_frozen_bests.json')
    lock = json.loads(track(lock_path,LOCK_SHA).read_text())
    top = json.loads(track(source/'r4_summary.json',SUMMARY_SHA).read_text())
    expected_runs = {f'{m}_{s}' for m in MODELS for s in SEEDS}
    require(top['status']=='passed' and top['lock_sha256']==LOCK_SHA and top['test_image_reads']==0, 'Invalid R4 summary')
    require(len(top['completed'])==9 and set(top['completed'])==expected_runs, 'Incomplete nine runs')
    manifest = read_csv(track(split_file,lock['data_files']['split_seed42_60_20_20/split_samples.csv']))
    val = [r for r in manifest if r['split']=='validation']
    require(len(val)==2720 and len({r['patient_id'] for r in val})==136, 'Wrong validation cohort')
    entries = {(e['model'],e['seed']): e for e in lock['runs']}
    require(len(lock['runs'])==9 and set(entries)=={(m,s) for m in MODELS for s in SEEDS}, 'Invalid lock runs')
    predictions = {m:{} for m in MODELS}; provenance=[]; single=[]
    for model in MODELS:
        for seed in SEEDS:
            name=f'{model}_{seed}'
            def load_file(filename):
                rel=f'{name}/{filename}'
                return track(source/rel,top['output_sha256'][rel])
            summary=json.loads(load_file('efficiency_summary.json').read_text())
            entry=entries[model,seed]
            validate_summary(summary,entry,LOCK_SHA)
            patients=read_csv(load_file('unified_reference_patient_predictions.csv'))
            slices=read_csv(load_file('unified_reference_slice_predictions.csv'))
            validate_slices(patients,slices,val)
            require(Counter(int(r['label']) for r in patients)=={0:92,1:44}, 'Wrong class counts')
            predictions[model][seed]=patients
            provenance.append(dict(model=model,seed=seed,epoch=entry['epoch'],checkpoint_sha256=entry['checkpoint_sha256']))
            single.append(dict(model=model,seed=seed,**metrics(patients)))
    ensembles={m:ensemble(predictions[m]) for m in MODELS}
    main=[dict(model=m,threshold=.5,**metrics(ensembles[m])) for m in MODELS]
    aggregates=[]
    for model in MODELS:
        group=[r for r in single if r['model']==model]
        aggregates.append(dict(model=model,**{f'{k}_{stat}':func(r[k] for r in group)
            for k in ('accuracy','ad_recall','nc_recall','macro_f1','auroc','nll','brier')
            for stat,func in (('mean',statistics.mean),('sample_sd',statistics.stdev))}))
    for path,digest in inputs.items(): checked(path,digest)
    output.mkdir(parents=True)
    for model,rows in ensembles.items():write_csv(output/f'{model}_patient_predictions.csv',rows)
    write_csv(output/'ensemble_metrics.csv',main)
    write_csv(output/'nine_unified_metrics.csv',single)
    write_csv(output/'seed_metrics_mean_sd.csv',aggregates)
    report=dict(status='passed',source_job='706895',split='validation',seeds=list(SEEDS),
        aggregation='mean of three patient probabilities; each patient is mean of 20 slices',
        threshold=.5,calibrated=False,patient_count=136,class_counts={'AD':44,'NC':92},
        source_runs=provenance,input_sha256=inputs,inputs_unchanged=True,
        image_reads={'train':0,'validation':0,'test':0},gpu_hours=0,
        code_sha256=sha(__file__),metrics=main,
        output_sha256={p.name:sha(p) for p in sorted(output.glob('*.csv'))})
    (output/'ensemble_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--split-file',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.source,args.split_file,args.output)['metrics'],indent=2))
