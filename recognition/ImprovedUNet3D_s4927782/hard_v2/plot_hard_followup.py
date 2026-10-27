"""Plot the complete three-pipeline Hard follow-up from its audited JSON.

No pixel arrays, model inference or model selection are involved. Every declared
case and all six patients must be present. The unsuccessful new 2D QA model is
not relabelled as a trained candidate or inserted as a fourth comparison arm.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

ORDER = ('v1_unet2d', 'v1_improved3d', 'hard_improved3d')
LABELS = ('Original 2D', 'Original 3D', 'New Hard 3D')
COLORS = ('#738297', '#AE8197', '#DC714B')
RESOURCE_FIELDS = (
    ('training_loop_seconds', 'Training loop time (seconds)', 1),
    ('mean_whole_case_inference_ms', 'Whole-case inference (ms)', 1),
    ('peak_training_allocated_cuda_bytes', 'Training peak allocated\nGPU memory (GiB)', 2**30),
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def mean_defined(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def equal(value, expected, message):
    if value is None or expected is None:
        if value is not None or expected is not None:
            raise ValueError(message)
    elif not math.isfinite(float(value)) or abs(value - expected) > 1e-12:
        raise ValueError(message)


def dice_value(value):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('Every plotted Dice must be finite in [0,1] or explicitly undefined.')


def validate(report):
    """Require complete, identical populations and patient-balanced summaries."""
    if report.get('status') != 'completed' or report.get('split') not in ('validation', 'test'):
        raise ValueError('Only completed validation or retrospective test reports may be plotted.')
    if not isinstance(report.get('scientific_status'), str) or not report['scientific_status'].strip():
        raise ValueError('Report must state its exploratory scientific status.')
    if not re.fullmatch('[0-9a-f]{64}', report.get('scope_plan_sha256', '')):
        raise ValueError('The report must identify its fixed scope plan.')
    if set(report.get('models', {})) != set(ORDER):
        raise ValueError('Exactly the two frozen references and one new Hard model are required.')
    expected_cases = 30 if report['split'] == 'test' else 38
    population, patient_visits = None, None
    for name in ORDER:
        model = report['models'][name]
        cases, aggregate = model['cases'], model['aggregate']
        if len(cases) != expected_cases:
            raise ValueError('Every model must retain the complete declared split.')
        keys = [(c['patient'], c['visit']) for c in cases]
        if len(set(keys)) != len(keys) or any(not p or not v for p, v in keys):
            raise ValueError('Cases must have unique nonempty patient/visit identities.')
        if population is None:
            population = set(keys)
        elif population != set(keys):
            raise ValueError('The three displayed case populations differ.')
        grouped = defaultdict(list)
        for case in cases:
            if [row['class_index'] for row in case['per_class']] != list(range(6)):
                raise ValueError('All six classes must be retained in their fixed order.')
            for row in case['per_class']:
                dice_value(row['dice'])
            fg = mean_defined(row['dice'] for row in case['per_class'][1:])
            equal(case['foreground_mean_dice'], fg, 'Case foreground Dice is inconsistent.')
            grouped[case['patient']].append(case)
        if len(grouped) != 6:
            raise ValueError('All six evaluation patients must be retained.')
        current_visits = {p: len(rows) for p, rows in grouped.items()}
        if patient_visits is None:
            patient_visits = current_visits
        elif patient_visits != current_visits:
            raise ValueError('Patient visit populations differ between models.')
        if aggregate['num_cases'] != expected_cases or aggregate['num_patients'] != 6:
            raise ValueError('Aggregate population differs from the complete case list.')
        if [row['class_index'] for row in aggregate['per_class']] != list(range(6)):
            raise ValueError('Aggregate class ordering differs.')
        patients = aggregate['patients']
        by_patient = {p['patient']: p for p in patients}
        if len(patients) != 6 or set(by_patient) != set(grouped):
            raise ValueError('Aggregate patient summaries are incomplete or duplicated.')
        for patient, rows in grouped.items():
            item = by_patient[patient]
            if item['num_cases'] != len(rows):
                raise ValueError('Aggregate patient visit count differs.')
            if [row['class_index'] for row in item['per_class']] != list(range(6)):
                raise ValueError('Patient summary class ordering differs.')
            for ci in range(6):
                expected = mean_defined(c['per_class'][ci]['dice'] for c in rows)
                equal(item['per_class'][ci]['dice'], expected, 'Patient class Dice is inconsistent.')
            equal(item['foreground_mean_dice'], mean_defined(c['foreground_mean_dice'] for c in rows),
                  'Patient foreground Dice is inconsistent.')
        for ci in range(6):
            expected = mean_defined(p['per_class'][ci]['dice'] for p in patients)
            equal(aggregate['per_class'][ci]['dice'], expected, 'Aggregate Dice is not patient balanced.')
            dice_value(aggregate['per_class'][ci]['dice'])
        equal(aggregate['foreground_mean_dice'], mean_defined(p['foreground_mean_dice'] for p in patients),
              'Aggregate foreground Dice is not patient balanced.')
        resource = model['resources']
        for field in ('training_loop_seconds', 'mean_whole_case_inference_ms',
                      'peak_training_allocated_cuda_bytes', 'peak_inference_allocated_cuda_bytes'):
            value = resource[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Resource plots require actual positive finite measurements.')
        for field in ('successful_updates', 'parameters'):
            if type(resource[field]) is not int or resource[field] <= 0:
                raise ValueError('Successful updates and parameters must be positive integers.')
        if type(resource['amp_skipped_updates']) is not int or resource['amp_skipped_updates'] < 0:
            raise ValueError('AMP skipped updates must be a nonnegative integer.')
    return dict(sorted(patient_visits.items()))


def cohort_title(split):
    return ('Previously inspected TEST cohort: exploratory comparison' if split == 'test'
            else 'Development VAL cohort: exploratory comparison')


def watermark(figure, report):
    if report.get('synthetic_fixture') is True:
        figure.text(.5, .5, 'SYNTHETIC FIXTURE - NOT ACTUAL MODEL RESULTS',
                    ha='center', va='center', fontsize=21, color='#A83232', alpha=.35, rotation=12)


def plot_report(report_path, output_dir):
    report_path, output = Path(report_path), Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Use a fresh plot directory; previous evidence is preserved.')
    input_digest = sha(report_path)
    report = json.loads(report_path.read_text())
    patient_visits = validate(report)
    patients = list(patient_visits)
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    style = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
             'axes.spines.right': False, 'axes.labelcolor': '#263345', 'text.color': '#263345',
             'xtick.color': '#263345', 'ytick.color': '#263345'}
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.15]})
        width = .24
        for i, (key, label, color) in enumerate(zip(ORDER, LABELS, COLORS)):
            model = report['models'][key]
            aggregate = model['aggregate']
            dice = [x['dice'] for x in aggregate['per_class'][1:]] + [aggregate['foreground_mean_dice']]
            axes[0].bar(np.arange(6) + (i - 1) * width,
                        [np.nan if x is None else x for x in dice], width=width,
                        label=label, color=color, zorder=3)
            lookup = {p['patient']: p['per_class'][5]['dice'] for p in aggregate['patients']}
            values = [lookup[p] for p in patients]
            bars = axes[1].bar(np.arange(6) + (i - 1) * width,
                               [np.nan if x is None else x for x in values], width=width,
                               label=label, color=color, zorder=3)
            for bar, value in zip(bars, values):
                if value is not None:
                    axes[1].text(bar.get_x() + bar.get_width()/2, value + .018, f'{value:.2f}',
                                 ha='center', va='bottom', fontsize=7.2, rotation=90)
                else:
                    axes[1].text(bar.get_x() + bar.get_width()/2, .02, 'N/A',
                                 ha='center', va='bottom', fontsize=7.2, rotation=90)
        axes[0].set_xticks(range(6), ['Body', 'Bones', 'Bladder', 'Rectum', 'Prostate', 'FG mean'])
        axes[0].set_title('All foreground classes and foreground macro Dice', fontsize=11, pad=12)
        axes[1].set_xticks(range(6), [f'{p}\n(n={patient_visits[p]})' for p in patients])
        axes[1].set_title('Prostate Dice for every evaluation patient', fontsize=11, pad=12)
        for ax in axes:
            ax.set_ylim(0, 1.08)
            ax.set_yticks(np.arange(0, 1.01, .2))
            ax.set_ylabel('Patient-balanced Dice')
            ax.grid(axis='y', color='#E5EAF0', linewidth=.8, zorder=0)
            ax.axhline(.70, color='#8C704D', linestyle='--', linewidth=.8, zorder=2)
        axes[1].set_ylabel('Mean visit Dice within patient')
        figure.suptitle(cohort_title(report['split']), fontsize=15, y=.98)
        figure.text(.5, .91, 'All predictions use the same original 64 x 128 x 128 grid and ground truth.',
                    ha='center', fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .085), ncol=3, frameon=False)
        figure.text(.5, .045, 'All six patients and all cases retained. Dashed .70 is a numerical project objective, not a safety threshold.',
                    ha='center', fontsize=9, color='#5A6678')
        figure.text(.5, .018, 'No new independent test claim; complete pipelines differ in resolution, sampling, seed, context and checkpoint selector.',
                    ha='center', fontsize=8.8, color='#5A6678')
        watermark(figure, report)
        figure.subplots_adjust(top=.80, bottom=.27, left=.06, right=.99, wspace=.24)
        destination = output / 'common_grid_dice.png'
        figure.savefig(destination, dpi=180)
        plt.close(figure)
        paths.append(destination)

        figure, axes = plt.subplots(1, 3, figsize=(12, 5.1))
        for ax, (field, title, scale) in zip(axes, RESOURCE_FIELDS):
            values = [report['models'][key]['resources'][field] / scale for key in ORDER]
            bars = ax.bar(range(3), values, color=COLORS, zorder=3)
            ax.set_xticks(range(3), ['Original 2D', 'Original 3D', 'New Hard 3D'], rotation=22, ha='right')
            ax.set_title(title, fontsize=10.5, pad=13)
            ax.set_ylim(0, max(values) * 1.25)
            ax.grid(axis='y', color='#E5EAF0', linewidth=.8, zorder=0)
            for bar, value in zip(bars, values):
                text = f'{value:.2f}' if value >= .01 else f'{value:.2g}'
                ax.text(bar.get_x() + bar.get_width()/2, value + max(values)*.025,
                        text, ha='center', fontsize=9)
        figure.suptitle('Measured costs of the three complete pipelines', fontsize=14, y=.98)
        figure.text(.5, .065, 'Training loop excludes validation, saving, setup and queue waits. Inference uses each recorded prediction path.',
                    ha='center', fontsize=9)
        figure.text(.5, .025, 'Allocated GPU memory is not total device use. Sampling, selector and possibly hardware differ; no isolated 3D effect.',
                    ha='center', fontsize=9)
        watermark(figure, report)
        figure.subplots_adjust(top=.78, bottom=.27, left=.06, right=.99, wspace=.34)
        destination = output / 'resource_costs.png'
        figure.savefig(destination, dpi=180)
        plt.close(figure)
        paths.append(destination)
    if sha(report_path) != input_digest:
        raise RuntimeError('Input report changed during plotting; figures are not certified.')
    receipt = {
        'status': 'completed', 'input_report_sha256': input_digest, 'plot_script_sha256': sha(__file__),
        'scope_plan_sha256': report['scope_plan_sha256'], 'split': report['split'],
        'model_order': list(ORDER), 'patient_visit_counts': patient_visits,
        'cases_per_model': 30 if report['split'] == 'test' else 38,
        'synthetic_fixture': report.get('synthetic_fixture') is True,
        'new_independent_test_claimed': False,
        'cpu_seconds': time.process_time() - start_cpu,
        'wall_seconds': time.perf_counter() - start_wall,
        'figures': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths},
    }
    with (output / 'plot_receipt.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = plot_report(args.report, args.output_dir)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
