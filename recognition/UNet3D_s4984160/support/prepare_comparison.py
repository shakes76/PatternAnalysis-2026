"""Verify archived scores and prepare comparison data without editing the report.

Only reads local, archived test results. Missing 2D results stay null. This
does not load a checkpoint, submit a job, or edit README.md.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'support/saved_results/test_results'
OUTPUT = ROOT / 'support/compare_results'
ARCHIVE_2D = ROOT.parent / 'UNet2D_s4984160/support/saved_results/test_results'
RESOURCE = ROOT / 'support/saved_results/resource_results'


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def score_rows(rows):
    """Recompute case/patient/pooled scores from archived integer counts."""
    result = []
    for label in range(6):
        group = [r for r in rows if int(r['class_id']) == label]
        keys = [r['case'] for r in group]
        if not group or len(set(keys)) != len(keys):
            raise ValueError(f'Missing/duplicate class {label} rows')
        totals = {k: sum(int(r[k]) for r in group)
                  for k in ('tp', 'fp', 'fn', 'gt_voxels', 'pred_voxels')}
        dice, iou, by_patient = [], [], {}
        for row in group:
            tp, fp, fn = (int(row[k]) for k in ('tp', 'fp', 'fn'))
            if int(row['gt_voxels']) != tp + fn or int(row['pred_voxels']) != tp + fp:
                raise ValueError('Inconsistent archived counts')
            d = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
            j = tp / (tp + fp + fn) if tp + fp + fn else None
            for name, value in [('dice', d), ('iou', j)]:
                stored = float(row[name]) if row[name] else None
                if (stored is None) != (value is None) or (
                        value is not None and not math.isclose(value, stored, abs_tol=1e-12)):
                    raise ValueError(f'{row["case"]}: {name} disagrees with counts')
            if d is not None:
                dice.append(d)
                by_patient.setdefault(row['patient_id'], []).append(d)
            if j is not None:
                iou.append(j)
        tp, fp, fn = (totals[k] for k in ('tp', 'fp', 'fn'))
        result.append(dict(class_id=label, class_name=group[0]['class_name'],
                           **totals, mean_case_dice=mean(dice),
                           std_case_dice=statistics.pstdev(dice) if dice else None,
                           mean_case_iou=mean(iou), valid_dice_cases=len(dice),
                           mean_patient_dice=mean([mean(v) for v in by_patient.values()]),
                           pooled_dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                           pooled_precision=tp/(tp+fp) if tp+fp else None,
                           pooled_recall=tp/(tp+fn) if tp+fn else None,
                           pooled_volume_ratio=(tp+fp)/(tp+fn) if tp+fn else None))
    return result


def verify_summary(observed, expected):
    if len(observed) != 6 or len(expected) != 6:
        raise ValueError('Expected all six classes')
    for actual, frozen in zip(observed, expected):
        for key, value in actual.items():
            if isinstance(value, float):
                if not math.isclose(value, frozen[key], abs_tol=1e-12):
                    raise ValueError(f'Summary mismatch: class {actual["class_id"]}, {key}')
            elif value != frozen[key]:
                raise ValueError(f'Summary mismatch: {key}')


def load_completed_2d(reference, reference_full_rows, reference_shared_rows, entries, report):
    """Verify the completed paired experiment independently from its CSV counts."""
    if not (ARCHIVE_2D/'summary.json').exists():
        return None
    summary = json.loads((ARCHIVE_2D/'summary.json').read_text(encoding='utf-8'))
    if not (summary['test_evaluated'] and summary['shared_gt_all_voxels_equal']
            and summary['reference_counts_match_archive']):
        raise ValueError('Completed 2D test/alignment evidence missing')
    if summary['reference_checkpoint_sha256'] != reference['checkpoint_sha256']:
        raise ValueError('Paired experiment used a different 3D reference')
    if summary['checkpoint_epoch'] != 5 or summary['checkpoint_sha256'] != (
            '3a666e057827897d294d9df9604ac325617fa874bf541402011537847024174b'):
        raise ValueError('Unexpected selected 2D checkpoint')
    if summary['shared_grid'] != [64,128,30] or summary['shared_W_indices'] != list(range(49,79)):
        raise ValueError('Unexpected paired grid')
    expected = {(r['case'], r['patient_id'], r['class_id']) for r in reference_full_rows}
    names = dict(native_2d='native_2d_per_case_metrics.csv',
                 shared_2d='shared_2d_per_case_metrics.csv',
                 shared_3d='reference_3d_shared_metrics.csv')
    scopes = {}
    for scope, name in names.items():
        rows = read_csv(ARCHIVE_2D/name)
        if {(r['case'], r['patient_id'], r['class_id']) for r in rows} != expected:
            raise ValueError(f'{scope}: cohort differs from original 3D test')
        metrics = score_rows(rows)
        verify_summary(metrics, summary[scope]['class_metrics'])
        prostate = [r for r in rows if r['class_id'] == '5']
        distances = [float(r['hd95_mm']) for r in prostate if r['hd95_status']=='ok']
        frozen = summary[scope]
        if (frozen['scans'], frozen['patients']) != (27,6):
            raise ValueError('Unexpected scope cohort')
        if len(distances) != frozen['prostate_hd95_valid_cases'] or not math.isclose(
                mean(distances), frozen['prostate_mean_hd95_mm'], abs_tol=1e-12):
            raise ValueError(f'{scope}: HD95 disagrees')
        if len(prostate)-len(distances) != frozen['prostate_hd95_invalid_cases']:
            raise ValueError('HD95 validity count differs')
        for key in ('prostate_fp_in_bladder','prostate_fp_in_rectum'):
            if sum(int(r[key]) for r in prostate) != frozen[key]:
                raise ValueError(f'{scope}: invasion count differs')
        scopes[scope] = dict(metrics=metrics, rows=rows,
                             cases={r['case']:r for r in prostate})
    prior = {(r['case'],r['class_id']):r for r in reference_shared_rows}
    for r in scopes['shared_3d']['rows']:
        original = prior[(r['case'],r['class_id'])]
        if any(int(r[k]) != int(original[k]) for k in ('tp','fp','fn')):
            raise ValueError('Reinferred shared 3D counts differ from original archive')
    gt = {(r['case'],r['class_id']):r['gt_voxels'] for r in scopes['shared_3d']['rows']}
    if any(r['gt_voxels'] != gt[(r['case'],r['class_id'])] for r in scopes['shared_2d']['rows']):
        raise ValueError('2D/3D shared GT count mismatch')
    paired = read_csv(ARCHIVE_2D/'paired_cases.csv')
    if len(paired)!=27 or len({r['case'] for r in paired})!=27:
        raise ValueError('Missing or duplicate paired cases')
    for r in paired:
        if int(r['gt_mismatch_voxels'])!=0:
            raise ValueError('Shared GT mismatch')
        for key,scope in [('native_2d_dice','native_2d'),('shared_2d_dice','shared_2d'),('shared_3d_dice','shared_3d')]:
            if not math.isclose(float(r[key]),float(scopes[scope]['cases'][r['case']]['dice']),abs_tol=1e-12):
                raise ValueError('Paired Dice differs from scope rows')
        if not math.isclose(float(r['delta_2d_minus_3d']),float(r['shared_2d_dice'])-float(r['shared_3d_dice']),abs_tol=1e-12):
            raise ValueError('Paired delta differs')
    for key,field in [('gt_scans','gt_touches_shared_W_boundary'),
                      ('prediction_2d_scans','prediction_2d_touches_shared_W_boundary'),
                      ('prediction_3d_scans','prediction_3d_touches_shared_W_boundary')]:
        if sum(r[field]=='True' for r in paired)!=summary['shared_boundary_touches'][key]:
            raise ValueError('Boundary contact count differs')
    for item in entries:
        if item['model']=='improved_2d_unet':continue
        if item['model']=='standard_3d_unet' and item['region']=='full_volume':continue
        scope = 'shared_3d' if item['model']=='standard_3d_unet' else (
            'shared_2d' if item['region']=='shared_band' else 'native_2d')
        data=summary[scope]; metrics=scopes[scope]['metrics'][5]
        item.update(status='measured', **{k:metrics[k] for k in metrics if k in item},
                    hd95_mm=data['prostate_mean_hd95_mm'],
                    fp_in_bladder=data['prostate_fp_in_bladder'],fp_in_rectum=data['prostate_fp_in_rectum'])
    patient_rows=[]
    for patient in sorted({r['patient_id'] for r in paired}):
        group=[r for r in paired if r['patient_id']==patient]
        a=mean([float(r['shared_2d_dice']) for r in group]);b=mean([float(r['shared_3d_dice']) for r in group])
        patient_rows.append(dict(patient_id=patient,scans=len(group),shared_2d_dice=a,shared_3d_dice=b,delta_2d_minus_3d=a-b))
    deltas=[float(r['delta_2d_minus_3d']) for r in paired]
    result=dict(mean_scan_delta=mean(deltas),median_scan_delta=statistics.median(deltas),
                mean_patient_delta=mean([r['delta_2d_minus_3d'] for r in patient_rows]),
                scans_2d_higher=sum(d>1e-8 for d in deltas),scans_3d_higher=sum(d< -1e-8 for d in deltas),
                patients_2d_higher=sum(r['delta_2d_minus_3d']>1e-8 for r in patient_rows),
                native_2d_scans_below_075=sum(float(r['native_2d_dice'])<.75 for r in paired))
    native=scopes['native_2d']['metrics'][5]['mean_case_dice']
    if summary['native_2d_target_met'] != (native>=.75):raise ValueError('Target status differs')
    report.update(checkpoint_2d_sha256=summary['checkpoint_sha256'],
                  sources_2d_sha256={name:hashlib.sha256((ARCHIVE_2D/name).read_bytes()).hexdigest()
                    for name in ['summary.json',*names.values(),'paired_cases.csv']},
                  shared_hd95_and_organ_confusion='Independently measured in frozen paired job 716656',
                  shared_2d_alignment_status='Exact GT equality for all 27 scans; reference counts match archive',
                  shared_boundary_touches=summary['shared_boundary_touches'],
                  paired_descriptive_results=result,
                  ranking_status='Descriptive comparison of these pipelines, not an isolated dimensionality ablation')
    return dict(summary=summary,scopes=scopes,paired=paired,patients=patient_rows,descriptive=result)


def verified_resources(reference, full_rows):
    """Verify whole-case aggregates and the direct prediction against raw records."""
    profile = json.loads((RESOURCE/'summary.json').read_text())
    paired = json.loads((ARCHIVE_2D/'summary.json').read_text())
    if (profile['checkpoint_3d_sha256'] != reference['checkpoint_sha256'] or
            profile['checkpoint_2d_sha256'] != paired['checkpoint_sha256']):
        raise ValueError('Resource checkpoint mismatch')
    if not profile['models_measured_separately'] or not profile['no_training'] or profile['tf32']:
        raise ValueError('Unexpected resource protocol')
    rows = read_csv(RESOURCE/'per_case_profile.csv')
    modes = {(p['model'], p['precision']) for p in profile['profiles']}
    if len(rows) != 81 or modes != {('standard_2d_unet','fp32'), ('standard_3d_unet','fp32'), ('standard_3d_unet','bf16')}:
        raise ValueError('Incomplete resource profiles')
    expected_cases = {r['case'] for r in full_rows}
    for p in profile['profiles']:
        group = [r for r in rows if (r['model'], r['precision']) == (p['model'], p['precision'])]
        if len(group) != 27 or {r['case'] for r in group} != expected_cases or p['scans'] != 27:
            raise ValueError('Resource cohort mismatch')
        if p['non_model_allocated_bytes_before_load'] != 0:
            raise ValueError('Unexpected resident GPU allocation')
        expected_parameters = paired['parameters_2d'] if p['model']=='standard_2d_unet' else reference['parameters']
        if p['parameters'] != expected_parameters:
            raise ValueError('Parameter count mismatch')
        for field, aggregate in [('forward_case_ms_mean', statistics.mean),
                                 ('forward_case_ms_median', statistics.median),
                                 ('prediction_case_ms_mean', statistics.mean),
                                 ('prediction_case_ms_median', statistics.median),
                                 ('peak_allocated_gib', max), ('peak_reserved_gib', max)]:
            source = field.replace('_mean','').replace('_median','')
            value = aggregate(float(r[source]) for r in group)
            if not math.isclose(value, p[field], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'Resource aggregate differs: {field}')
    demo = json.loads((RESOURCE/'predict_demo/summary.json').read_text())
    if demo['checkpoint_sha256'] != reference['checkpoint_sha256'] or demo['precision'] != 'bf16':
        raise ValueError('Direct prediction checkpoint/precision mismatch')
    for metric in demo['class_metrics']:
        original = next(r for r in full_rows if r['case']==demo['case'] and int(r['class_id'])==metric['class_id'])
        for key in ('tp','fp','fn','gt_voxels','pred_voxels'):
            if int(original[key]) != metric[key]:
                raise ValueError(f'Direct prediction count differs: {key}')
        if metric['class_id']==5 and not math.isclose(float(original['hd95_mm']), demo['prostate_hd95_mm'], abs_tol=1e-10):
            raise ValueError('Direct prediction HD95 differs')
    for marker in ('profile_complete.txt','predict_demo/prediction_complete.txt'):
        if not (RESOURCE/marker).read_text().strip():
            raise ValueError('Missing resource completion marker')
    if (RESOURCE/'resource_717015.err').stat().st_size:
        raise ValueError('Resource error log is not empty')
    sources = {str(path.relative_to(RESOURCE)).replace('\\','/'): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(RESOURCE.rglob('*')) if path.is_file() and path.name!='README.md'}
    return dict(job_id=717015, measurement=profile, direct_prediction=demo, sources_sha256=sources)


def main():
    summary = json.loads((ARCHIVE / 'summary.json').read_text(encoding='utf-8'))
    full_rows = read_csv(ARCHIVE / 'per_case_metrics.csv')
    shared_rows = read_csv(ARCHIVE / 'shared_2d_region_metrics.csv')
    full, shared = score_rows(full_rows), score_rows(shared_rows)
    verify_summary(full, summary['class_metrics'])
    verify_summary(shared, summary['shared_2d_region']['class_metrics'])
    keys = {(r['case'], r['patient_id'], r['class_id']) for r in full_rows}
    if keys != {(r['case'], r['patient_id'], r['class_id']) for r in shared_rows}:
        raise ValueError('Full/shared scan identities differ')
    if len(keys) != 27 * 6 or len({r['patient_id'] for r in full_rows}) != 6:
        raise ValueError('Unexpected test cohort')
    # On the recorded 256-wide original axis, nearest mask resize uses every
    # second source index: W=49..78 -> original Y=98,100,...,156.
    if summary['target_shape'] != [64, 128, 128]:
        raise ValueError('The documented shared-grid protocol needs this target shape')
    if summary['shared_2d_region']['original_indices_inclusive'] != [98, 157]:
        raise ValueError('Unexpected supplied 2D band')
    for key in ('prostate_fp_in_bladder', 'prostate_fp_in_rectum'):
        if sum(int(r[key]) for r in full_rows if r['class_id'] == '5') != summary[key]:
            raise ValueError(f'Confusion count mismatch: {key}')
    distances = [float(r['hd95_mm']) for r in full_rows
                 if r['class_id'] == '5' and r['hd95_status'] == 'ok']
    if len(distances) != summary['prostate_hd95_valid_cases'] or not math.isclose(
            mean(distances), summary['prostate_mean_hd95_mm'], abs_tol=1e-12):
        raise ValueError('Full-volume HD95 mismatch')

    common = dict(region='shared_band', grid='D64_H128_W30; full-grid W49..78',
                  patients=6, scans=27, source_indices='98,100,...,156',
                  aggregation='mean_scan; pooled metrics separately labelled')
    metric_keys = ['mean_case_dice', 'std_case_dice', 'mean_patient_dice',
                   'mean_case_iou', 'pooled_dice', 'pooled_precision',
                   'pooled_recall', 'pooled_volume_ratio']
    entries = []
    for region, metrics in [('full_volume', full[5]), ('shared_band', shared[5])]:
        item = dict(common, model='standard_3d_unet', region=region,
                    status='measured', **{k: metrics[k] for k in metric_keys},
                    hd95_mm=None, fp_in_bladder=None, fp_in_rectum=None)
        if region == 'full_volume':
            item.update(grid='D64_H128_W128', source_indices='full-grid W indices 0..127',
                        hd95_mm=summary['prostate_mean_hd95_mm'],
                        fp_in_bladder=summary['prostate_fp_in_bladder'],
                        fp_in_rectum=summary['prostate_fp_in_rectum'])
        entries.append(item)
    for model, status in [('standard_2d_unet', 'awaiting_frozen_test_evaluation'),
                          ('improved_2d_unet', 'optional_not_scheduled')]:
        entries.append(dict(common, model=model, status=status,
                            **{k: None for k in metric_keys},
                            hd95_mm=None, fp_in_bladder=None, fp_in_rectum=None))
        entries.append(dict(common, model=model, status=status, region='native_2d_band',
                            grid='60 slices; original unpadded 256x128 or 256x144',
                            source_indices='98..157 inclusive; all 60 planes',
                            **{k: None for k in metric_keys},
                            hd95_mm=None, fp_in_bladder=None, fp_in_rectum=None))
    sources = {name: hashlib.sha256((ARCHIVE/name).read_bytes()).hexdigest()
               for name in ['summary.json', 'per_case_metrics.csv',
                            'shared_2d_region_metrics.csv', 'split.json']}
    report = dict(checkpoint_sha256=summary['checkpoint_sha256'],
                  test_case_ids=sorted({r['case'] for r in full_rows}),
                  test_patient_ids=sorted({r['patient_id'] for r in full_rows}),
                  sources_sha256=sources, entries=entries,
                  shared_hd95_and_organ_confusion='not measured in shared-band CSV',
                  shared_2d_alignment_status='pending actual 2D predictions and exact GT grid check',
                  ranking_status='no paired ranking until aligned 2D test evaluation')
    completed = load_completed_2d(summary,full_rows,shared_rows,entries,report)
    OUTPUT.mkdir(exist_ok=True)
    resources = verified_resources(summary, full_rows) if completed and RESOURCE.exists() else None
    if resources:
        report['resource_profile'] = resources
        with (OUTPUT/'resources.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(resources['measurement']['profiles'][0]))
            writer.writeheader();writer.writerows(resources['measurement']['profiles'])
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/'metrics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with (OUTPUT/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader(); writer.writerows(entries)
    if completed:
        with (OUTPUT/'patient_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(completed['patients'][0]))
            writer.writeheader();writer.writerows(completed['patients'])

    print('Verified archived 3D, paired 2D and resource records; wrote comparison data only. README was not modified.')


if __name__ == '__main__':
    main()
