"""Fixed negative-side referral comparison on saved validation predictions.

No training, inference, temperature fitting, cutoff search or final-test access.
The 0.05 cap is a declared research setting, not a clinical safety standard.
NC is the dataset/code label; CN is the assessment terminology. Both mean the
control category in this project. Clinical diagnoses are not made by this code.

Assistance: ChatGPT drafted this extension and tests; review, verify and disclose.
Reuses selected_a1.py and confidence_analysis.py without modifying them.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

POLICIES = ('reference', 'strict_cn', 'no_auto_cn')
CN_AD_PROBABILITY_CAP = 0.05
CODE_FILES = ('negative_policy.py', 'NEGATIVE_POLICY_METHOD.md',
              'tests_negative_policy/test_negative_policy.py')
PLAN_FILE = 'NEGATIVE_POLICY_PLAN.json'
OUTPUT_FOLDER = 'negative-policy-comparison'
MARKER = '<!-- COMP3710-NEGATIVE-POLICY-COMPARISON -->'
PUBLIC_FILES = ('negative_policy_summary.json', 'negative_policy_rules.json',
                'README_negative_policy.md', PLAN_FILE)
BUNDLE = 'COMP3710_Negative_Policy_Results.zip'
CAUTION = (
    'Research-only secondary analysis. No model, primary classification cutoff or '
    'existing operating rule is changed. The 0.05 cap is a fixed experimental '
    'choice, not proof of 95% correctness. These validation patients have already '
    'been used repeatedly for model and policy development; cross-fitting does '
    'not undo that reuse. Referrals are not diagnoses. A no-auto-CN policy has '
    'zero automated negative errors by construction, not demonstrated clinical '
    'safety or 100% diagnostic sensitivity. Version 1 has already exposed the '
    'test cohort. This extension performs no new test evaluation.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def unique(pairs):
        value = {}
        for k, v in pairs:
            require(k not in value, 'Duplicate JSON key.')
            value[k] = v
        return value
    result = json.loads(Path(path).read_text(encoding='utf-8-sig'),
                        object_pairs_hook=unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON.')))
    require(isinstance(result, dict), 'Expected a JSON object.')
    return result


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
    temp.replace(path)


def probabilities(values):
    result = [float(p) for p in values]
    require(result and all(math.isfinite(p) and 0 <= p <= 1 for p in result),
            'Probabilities must be finite, in [0,1], and nonempty.')
    return result


def actions(probability, threshold, policy):
    """Fixed three-way rule. None disables the original automatic rule entirely.

    The AD side is identical to the reference. strict_cn can only remove CN
    acceptance; no_auto_cn removes it even at p(AD)=0. Exactly 0.5 is AD under
    the primary classifier; it is never automatically CN here.
    """
    p = probabilities(probability)
    require(policy in POLICIES, 'Unknown policy.')
    require(threshold is None or (not isinstance(threshold, bool) and
            math.isfinite(threshold) and 0.5 <= threshold <= 1), 'Invalid original threshold.')
    out = []
    for q in p:
        accepted = threshold is not None and max(q, 1 - q) >= threshold
        if accepted and q >= 0.5:
            out.append('AD_flag')
        elif accepted and q < 0.5 and policy != 'no_auto_cn' and (
                policy == 'reference' or q <= CN_AD_PROBABILITY_CAP):
            out.append('auto_CN')
        else:
            out.append('refer')
    return out


def interval(count, n):
    require(0 <= count <= n, 'Invalid interval counts.')
    if n == 0:
        return {'count': count, 'denominator': n, 'estimate': None, 'lower': None, 'upper': None}
    z = 1.959963984540054
    p = count / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {'count': count, 'denominator': n, 'estimate': p,
            'lower': max(0., centre - half), 'upper': min(1., centre + half)}


def metrics(labels, decisions):
    y = list(labels)
    require(y and len(y) == len(decisions) and
            all(type(v) is int and v in (0, 1) for v in y), 'Invalid labels/actions.')
    require(all(a in ('auto_CN', 'AD_flag', 'refer') for a in decisions), 'Unknown action.')
    cells = [[sum(label == c and action == a for label, action in zip(y, decisions))
              for a in ('auto_CN', 'AD_flag', 'refer')] for c in (0, 1)]
    tn, fp, rn = cells[0]
    fn, tp, ra = cells[1]
    n = len(y)
    negatives, positives, referred = tn + fn, fp + tp, rn + ra
    accepted = negatives + positives
    ads = sum(y)
    error = interval(fn + fp, accepted)
    return {
        'n_patients': n, 'n_true_AD': ads, 'n_true_CN': n - ads,
        'action_table_rows_true_CN_AD_columns_auto_CN_AD_flag_review': cells,
        'n_auto_CN': negatives, 'n_AD_flags': positives, 'n_unresolved_referrals': referred,
        'n_automated_research_labels': accepted, 'research_label_coverage': accepted / n,
        'n_AD_auto_CN': fn, 'n_CN_AD_flag': fp, 'n_AD_AD_flag': tp,
        'n_AD_referred': ra, 'n_CN_referred': rn,
        'automated_research_label_accuracy': 1 - error['estimate'] if accepted else None,
        'automated_research_label_error': error,
        'AD_auto_CN_fraction_of_all_AD': interval(fn, ads),
        'AD_fraction_among_auto_CN': interval(fn, negatives),
        'negative_predictive_value': tn / negatives if negatives else None,
        'AD_flag_precision': tp / positives if positives else None,
        'AD_flag_fraction_of_all_AD': tp / ads if ads else None,
        'n_clinical_reviews_if_all_AD_flags_also_reviewed': referred + positives,
        'clinical_review_fraction_under_that_assumption': (referred + positives) / n,
        'clinical_review_workload_is_an_assumption_not_measured': True,
        'interval_note': 'Descriptive Wilson 95% intervals; do not account for model/policy selection or cross-fit dependence.',
        'no_human_review_accuracy_assumed': True}


def compare_actions(labels, p, thresholds):
    """Allow a single threshold or one already-fitted threshold per OOF patient."""
    values = probabilities(p)
    tau = list(thresholds)
    require(len(tau) == len(values), 'Threshold count mismatch.')
    choices = {name: [actions([q], t, name)[0] for q, t in zip(values, tau)] for name in POLICIES}
    ref = choices['reference']
    for name in POLICIES[1:]:
        other = choices[name]
        require(all((a == 'AD_flag') == (b == 'AD_flag') for a, b in zip(ref, other)), 'AD flags changed.')
        require(all(b != 'auto_CN' or a == 'auto_CN' for a, b in zip(ref, other)), 'CN acceptance expanded.')
    result = {name: metrics(labels, a) for name, a in choices.items()}
    for name in POLICIES:
        changes = [(y, a, b) for y, a, b in zip(labels, ref, choices[name]) if a != b]
        require(all(a == 'auto_CN' and b == 'refer' for _, a, b in changes), 'Unexpected policy transition.')
        result[name]['relative_to_reference'] = {
            'additional_referrals': len(changes),
            'additional_AD_referrals_previously_auto_CN': sum(y == 1 for y, _, _ in changes),
            'additional_CN_referrals_previously_auto_CN': sum(y == 0 for y, _, _ in changes),
            'AD_flags_unchanged': True,
            'all_patient_binary_accuracy_unchanged': True,
            'accepted_error_rate_not_guaranteed_to_improve': True}
    return result, choices


def check_reference(actual, expected):
    matrix = actual['action_table_rows_true_CN_AD_columns_auto_CN_AD_flag_review']
    expected_matrix = expected['accepted_confusion_matrix_rows_true_columns_predicted_NC_AD']
    require([row[:2] for row in matrix] == expected_matrix and
            actual['n_automated_research_labels'] == expected['n_accepted'] and
            actual['n_unresolved_referrals'] == expected['n_referred'] and
            actual['n_AD_referred'] == expected['n_ad_referred'],
            'Original reference policy does not reproduce; preserve the evidence.')


def inspect_patients(candidate, rows, model_report, rule):
    """Check private diagnostic rows against verified original validation outputs.

    Fold memberships are rebuilt using the original deterministic patient rule.
    Existing fold temperatures/thresholds are reused, never refitted here.
    """
    from confidence_analysis import fold_assignments, scaled_probabilities
    from analyse_confidence import same_numbers
    same_numbers(rule['temperature'], model_report['full_validation_temperature_fit']['temperature'], 'fixed temperature')
    same_numbers(rule['confidence_threshold'], model_report['operating_rule']['confidence_threshold'], 'fixed referral cutoff')
    require(isinstance(rows, list) and len(rows) == len(candidate.subjects), 'Incomplete private diagnostics.')
    indexed = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('subject_id'), str), 'Malformed private record.')
        sid = row['subject_id']
        require(sid not in indexed, 'Duplicate private record.')
        indexed[sid] = row
    require(set(indexed) == set(candidate.subjects), 'Unexpected patient cohort.')
    ordered = [indexed[s] for s in candidate.subjects]
    for row, label, p in zip(ordered, candidate.labels, candidate.p):
        require(type(row.get('label')) is int and row['label'] == label, 'Private labels differ.')
        same_numbers(row.get('raw_p_ad'), p, 'raw validation probabilities')
    folds = fold_assignments(candidate.subjects, candidate.labels)
    full = scaled_probabilities(candidate.p, rule['temperature']).tolist()
    require([row.get('fold_index') for row in ordered] == folds.tolist(), 'Fold assignments differ.')
    same_numbers([row.get('full_fit_calibrated_p_ad') for row in ordered], full, 'calibrated probabilities')
    fold_rows = model_report['crossfit_diagnostic']['fold_results']
    n_folds = model_report['crossfit_diagnostic']['n_folds']
    require(n_folds == 5 and len(fold_rows) == 5 and
            {f['fold'] for f in fold_rows} == set(range(5)), 'Unexpected cross-fitting structure.')
    oof = [None] * len(full)
    taus = [None] * len(full)
    for f in fold_rows:
        ii = [i for i, k in enumerate(folds) if k == f['fold']]
        require(len(ii) == f['held_n'], 'Held fold count differs.')
        fitted_p = scaled_probabilities([candidate.p[i] for i in ii], f['temperature']).tolist()
        for i, p in zip(ii, fitted_p):
            oof[i] = p
            taus[i] = f['confidence_threshold']
    same_numbers([row.get('oof_p_ad') for row in ordered], oof, 'cross-fitted probabilities')
    require(all((p >= .5) == (q >= .5) == (r >= .5)
                for p, q, r in zip(candidate.p, full, oof)), 'Classification changed after scaling.')
    fit_result, fit_actions = compare_actions(candidate.labels, full, [rule['confidence_threshold']] * len(full))
    oof_result, oof_actions = compare_actions(candidate.labels, oof, taus)
    for name, key in ((fit_actions, 'full_fit_accepted'), (oof_actions, 'oof_accepted')):
        flags = [r.get(key) for r in ordered]
        require(all(type(v) is bool for v in flags) and
                flags == [a != 'refer' for a in name['reference']], 'Saved acceptance decisions differ.')
    check_reference(fit_result['reference'], model_report['operating_rule']['development_fit_triage'])
    check_reference(oof_result['reference'], model_report['crossfit_diagnostic']['pooled_triage'])
    per_fold = []
    for f in fold_rows:
        ii = [i for i, k in enumerate(folds) if k == f['fold']]
        report, _ = compare_actions([candidate.labels[i] for i in ii], [oof[i] for i in ii], [taus[i] for i in ii])
        check_reference(report['reference'], f['held_triage'])
        per_fold.append({'fold': f['fold'], 'temperature': f['temperature'],
                         'original_confidence_threshold': f['confidence_threshold'], 'policies': report})
    return {'name': candidate.name, 'selected_epoch': candidate.report['selected_epoch'],
            'primary_classifier_metrics_unchanged': model_report['original_validation_metrics']['patient'],
            'calibration_and_reference_rules_unchanged': True,
            'full_validation_apparent': fit_result,
            'crossfit_diagnostic': {'pooled': oof_result, 'fold_results': per_fold,
                'note': 'Existing fold-specific temperatures and reference cutoffs reused. Not an independent evaluation of the single full-fit rule; model selection already used this cohort.'}}


def load_inputs(private):
    import selected_a1 as sa
    from analyse_confidence import same_numbers
    private = Path(private).resolve()
    ctx = sa.context(private)
    require(ctx['project'] == Path(__file__).resolve().parent, 'Use the installed project script, not the tools copy.')
    folder = sa.pointer(private, sa.POINTER)
    plan = sa.read_json(ctx['project'] / sa.PLAN_FILE)
    require(plan == sa.selection(ctx), 'The recorded A1 selection has changed.')
    summary = sa.validate_completed(folder, plan)
    rules = sa.read_json(folder / sa.RULES)
    require(summary['source_code_sha256'] == ctx['sources'], 'Confidence sources differ.')
    require(rules['manifest_sha256'] == ctx['bundle'].manifest_hashes and
            rules['selection_plan_sha256'] == sa.sha256(ctx['project'] / sa.PLAN_FILE), 'Rules use different inputs.')
    private_path = folder / 'PRIVATE_patient_diagnostics.json'
    private_rows = sa.read_json(private_path)
    require(set(private_rows) == {'cnn', 'a1'} and set(rules['models']) == {'cnn', 'a1'}, 'Unexpected model set.')
    for alias, candidate in ctx['candidates'].items():
        m, r = summary['models'][alias], rules['models'][alias]
        require(m['checkpoint_sha256'] == r['checkpoint_sha256'] == candidate.hashes['checkpoint_sha256'] and
                m['selected_epoch'] == r['selected_epoch'] == candidate.report['selected_epoch'] and
                r['diagnostic_threshold'] == .5, 'Selected model identity differs.')
        same_numbers(m['original_validation_metrics'], candidate.report['selected_validation_metrics'], 'original metrics')
        same_numbers(r['temperature'], m['full_validation_temperature_fit']['temperature'], 'fixed temperature')
        same_numbers(r['confidence_threshold'], m['operating_rule']['confidence_threshold'], 'original referral')
    inputs = {'confidence_summary_sha256': sha256(folder / sa.SUMMARY),
              'operating_rules_sha256': sha256(folder / sa.RULES),
              'selection_sha256': sha256(ctx['project'] / sa.PLAN_FILE),
              'private_diagnostics_sha256': sha256(private_path),
              'validation_manifest_sha256': ctx['bundle'].manifest_hashes['validation.csv']}
    return ctx, summary, rules, private_rows, inputs


def source_hashes(project):
    return {name: sha256(project / name) for name in CODE_FILES}


def make_plan(ctx, summary, rules, inputs):
    return {'format_version': 1, 'status': 'FIXED_SECONDARY_VALIDATION_POLICY_COMPARISON',
        'primary_classification_cutoff': .5, 'primary_model_selection_changes': False,
        'strict_CN_maximum_calibrated_p_AD': CN_AD_PROBABILITY_CAP,
        'policy_ids': list(POLICIES), 'no_threshold_search': True,
        'no_policy_winner_selected_from_validation': True,
        'primary_reference_policy_preserved': True,
        'setting_rationale': 'A fixed stricter negative-side confidence illustration (0.95), chosen before this comparison; not an empirical clinical risk bound or a course-imposed value.',
        'input_hashes': inputs, 'source_hashes': source_hashes(ctx['project']),
        'models': {a: {'name': r['name'], 'selected_epoch': r['selected_epoch'],
                       'checkpoint_sha256': r['checkpoint_sha256'], 'temperature': r['temperature'],
                       'original_confidence_threshold': r['confidence_threshold']}
                   for a, r in rules['models'].items()},
        'test_evaluation_authorized_by_this_plan': False,
        'next_evaluation_intent': 'Preserve the primary reference analysis and report BOTH predefined secondary variants alongside it; no best-on-test policy selection.',
        'caution': CAUTION}


def git(project, *args):
    r = subprocess.run(['git', '-C', str(project), *args], capture_output=True)
    require(r.returncode == 0, 'Git check failed; inspect without discarding changes.')
    return r.stdout


def require_committed(project):
    prefix = git(project, 'rev-parse', '--show-prefix').decode().strip()
    require(prefix.startswith('recognition/') and prefix != 'recognition/' and
            git(project, 'branch', '--show-current').decode().strip() == 'topic-recognition', 'Wrong project/branch.')
    for name in (*CODE_FILES, PLAN_FILE):
        expected = git(project, 'show', f'HEAD:{prefix}{name}')
        require(expected == (project / name).read_bytes(), 'Commit the unchanged new code and plan before analysis.')


def prepare(private):
    ctx, summary, rules, rows, inputs = load_inputs(private)
    plan = make_plan(ctx, summary, rules, inputs)
    path = ctx['project'] / PLAN_FILE
    if path.exists():
        require(read_json(path) == plan, 'An existing policy plan differs; do not overwrite it.')
        print('NEGATIVE_POLICY_PLAN_ALREADY_PRESENT')
    else:
        write_json(path, plan)
        print('NEGATIVE_POLICY_PLAN_CREATED')


def export_rules(rules, inputs):
    out = {'format_version': 1, 'primary_classification_cutoff': .5,
        'models': {}, 'policies_all_retained_no_winner_selected': True,
        'new_temperature_or_threshold_fit_performed': False,
        'input_hashes': inputs, 'caution': CAUTION}
    for alias, r in rules['models'].items():
        tau = r['confidence_threshold']
        out['models'][alias] = {'name': r['name'], 'checkpoint_sha256': r['checkpoint_sha256'],
            'selected_epoch': r['selected_epoch'], 'temperature': r['temperature'],
            'diagnostic_threshold': .5, 'policies': {
                'reference': {'original_confidence_threshold': tau, 'additional_CN_probability_cap': None,
                              'automatic_CN_disabled': False},
                'strict_cn': {'original_confidence_threshold': tau, 'additional_CN_probability_cap': .05,
                              'automatic_CN_disabled': False},
                'no_auto_cn': {'original_confidence_threshold': tau, 'additional_CN_probability_cap': None,
                              'automatic_CN_disabled': True}},
            'application': 'First apply original reference rule; remove auto_CN if cap is exceeded or automatic_CN_disabled. AD flags are unchanged. If original threshold is null, refer everyone.'}
    return out


def fmt(v, percent=False):
    if v is None:
        return 'Not estimable'
    return f'{100*v:.2f}%' if percent else str(v)


def markdown(report):
    lines = ['## Negative-side caution: fixed referral-policy comparison', '', CAUTION, '',
        'The selected neural networks, ordinary 0.5 class decisions and saved temperatures are unchanged. '
        'The original symmetric policy is retained. Two secondary policies are declared before this analysis: '
        '(1) accept an original CN proposal only if calibrated P(AD) <= 0.05; '
        '(2) accept no CN proposals. The original AD-flag branch is unchanged in both.', '',
        '**0.95 model confidence is not a demonstrated 95% chance of being correct.** '
        'No cutoffs were searched or fitted, and no policy is selected as a winner from this comparison.', '',
        'For each scope below, counts are shown as research actions. An AD flag is not a confirmed diagnosis. '
        'If AD flags also require a clinician, the relevant workload is flags plus unresolved referrals.', '']
    for scope, title in [('full_validation_apparent','Apparent full-validation results'),
                         ('crossfit_diagnostic','Existing cross-fitted development diagnostic')]:
        lines += [f'### {title}', '',
                  '| Model / policy | Auto CN | AD flags | Unresolved review | AD wrongly auto CN / all AD | AD among auto CN | Research-label coverage | Accepted error |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for a, m in report['models'].items():
            policies = m[scope] if scope != 'crossfit_diagnostic' else m[scope]['pooled']
            for name, r in policies.items():
                lines.append(f'| {m["name"]} / {name} | {r["n_auto_CN"]} | {r["n_AD_flags"]} | '
                    f'{r["n_unresolved_referrals"]} | {r["n_AD_auto_CN"]}/{r["n_true_AD"]} | '
                    f'{fmt(r["AD_fraction_among_auto_CN"]["estimate"],True)} | '
                    f'{fmt(r["research_label_coverage"],True)} | '
                    f'{fmt(r["automated_research_label_error"]["estimate"],True)} |')
        lines += ['']
    lines += ['### What may and may not be concluded', '',
        'Stricter negative acceptance can only remove automatic CN outputs; it cannot add them or '
        'increase the count of AD patients assigned automatic CN for the same input probabilities. '
        'This is a logical property, not evidence of improved disease recognition. It also refers more '
        'true CN patients. The error fraction among remaining accepted outputs need not improve.', '',
        'The JSON includes per-class action tables, additional AD/CN referrals relative to the reference, '
        'both false-negative denominators and descriptive Wilson intervals. Zero observed events with '
        'a small denominator is not proof of zero future risk. If no CN output is accepted, negative '
        'predictive value is undefined. No-auto-CN implies every patient requires a clinician if AD flags '
        'also require confirmation; zero automatic negatives is not 100% diagnostic recall.', '',
        'Cross-fitted results reuse the original fold-specific temperatures and confidence cutoffs. '
        'They diagnose the development procedure, not the single full-fit rule on an independent cohort. '
        'All candidate variants remain secondary; the primary accuracy result is unchanged and the '
        'assessment\'s 0.80 test-accuracy requirement is not established by referral coverage or accepted accuracy.', '',
        '### Recommendation and next stage', '',
        'Do not use this research prototype for autonomous clearance of real patients. Preserve the '
        'original model and policies and carry the declared secondary variants to a fixed, transparently '
        'reported revised evaluation. That evaluation must acknowledge prior test exposure and must not '
        'select a policy using the new test results. The historical predict.py interface is not changed '
        'by this extension.', '',
        '### Assistance and verification', '',
        'ChatGPT assisted with implementation, synthetic tests and this draft. Record the source review '
        'and local checks actually performed in the Artificial Intelligence Usage Disclosure. '
        'No clinical validation or measured human-review outcomes are implied.', '']
    return '\n'.join(lines)


def validate_output(folder, plan):
    value = read_json(folder / 'negative_policy_summary.json')
    require(value.get('comparison_completed') is True and read_json(folder / PLAN_FILE) == plan,
            'Saved comparison is incomplete or uses a different plan.')
    require(set(value.get('output_hashes', {})) == set(PUBLIC_FILES) - {'negative_policy_summary.json'}, 'Incomplete output hashes.')
    for name, digest in value['output_hashes'].items():
        require(sha256(folder / name) == digest, 'Saved output changed; do not overwrite.')
    return value


def bundle(folder):
    temp = folder / (BUNDLE + '.tmp')
    with ZipFile(temp, 'w', ZIP_DEFLATED) as z:
        for name in PUBLIC_FILES:
            z.write(folder / name, name)
    temp.replace(folder / BUNDLE)


def run(private, *, enforce_git=True):
    ctx, old_summary, old_rules, rows, inputs = load_inputs(private)
    plan = make_plan(ctx, old_summary, old_rules, inputs)
    require(read_json(ctx['project'] / PLAN_FILE) == plan, 'Run prepare, review and commit the plan first.')
    if enforce_git:
        require_committed(ctx['project'])
    folder = ctx['private'] / OUTPUT_FOLDER
    folder.mkdir(exist_ok=True)
    # Reuse the tested cross-platform lock; the original training functions are never called.
    from train_convnext_v2 import run_lock
    with run_lock(folder):
        if (folder / 'negative_policy_summary.json').exists():
            validate_output(folder, plan)
            bundle(folder)
            print('NEGATIVE_POLICY_COMPARISON_ALREADY_COMPLETED')
        else:
            report = {'format_version': 1, 'comparison_completed': False,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'new_training_performed': False, 'new_inference_performed': False,
                'new_calibration_fitted': False, 'new_threshold_search_performed': False,
                'test_images_opened': 0, 'new_test_model_evaluation_performed': False,
                'test_manifest_read_for_existing_integrity_checks_only': True,
                'historical_outputs_modified': False, 'primary_classification_unchanged': True,
                'models': {}, 'input_hashes': inputs, 'plan_sha256': sha256(ctx['project'] / PLAN_FILE),
                'caution': CAUTION}
            for alias, c in ctx['candidates'].items():
                report['models'][alias] = inspect_patients(c, rows[alias]['patients'],
                    old_summary['models'][alias], old_rules['models'][alias])
            # Re-read input bindings to catch concurrent evidence changes before publication.
            _, _, _, _, after = load_inputs(private)
            require(after == inputs and source_hashes(ctx['project']) == plan['source_hashes'], 'Inputs/code changed during analysis.')
            write_json(folder / PLAN_FILE, plan)
            write_json(folder / 'negative_policy_rules.json', export_rules(old_rules, inputs))
            (folder / 'README_negative_policy.md').write_text(markdown(report), encoding='utf-8')
            report['comparison_completed'] = True
            report['output_hashes'] = {n:sha256(folder/n) for n in PUBLIC_FILES if n != 'negative_policy_summary.json'}
            write_json(folder / 'negative_policy_summary.json', report)
            bundle(folder)
            print('NEGATIVE_POLICY_COMPARISON_COMPLETED')
    pointer = ctx['private'] / 'latest_negative_policy.txt'
    temp = pointer.with_suffix('.tmp')
    temp.write_text(str(folder.resolve()) + '\n', encoding='utf-8')
    temp.replace(pointer)
    print('VALIDATION_ONLY_PRIMARY_ACCURACY_UNCHANGED')
    print(folder / BUNDLE)
    return folder


def publish(private):
    ctx, summary, rules, rows, inputs = load_inputs(private)
    plan = make_plan(ctx, summary, rules, inputs)
    folder = ctx['private'] / OUTPUT_FOLDER
    validate_output(folder, plan)
    readme = ctx['project'] / 'README.md'
    # Read bytes to avoid changing historical CRLF/CR sequences.
    old = readme.read_bytes()
    if MARKER.encode() in old:
        print('NEGATIVE_POLICY_REPORT_ALREADY_PRESENT')
        return
    backup = ctx['private'] / ('negative-policy-readme-backup-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    backup.mkdir()
    shutil.copy2(readme, backup / 'README.md')
    section = (folder / 'README_negative_policy.md').read_bytes()
    temp = readme.with_suffix('.md.tmp')
    temp.write_bytes(old + b'\n\n' + MARKER.encode() + b'\n' + section)
    temp.replace(readme)
    print('NEGATIVE_POLICY_REPORT_APPENDED')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'run', 'publish'))
    p.add_argument('--private-root', type=Path, required=True)
    args = p.parse_args()
    {'prepare': prepare, 'run': run, 'publish': publish}[args.command](args.private_root)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f'NEGATIVE_POLICY_STOPPED: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('Preserve current data and previous results; do not retrain or alter saved rules.', file=sys.stderr)
        raise SystemExit(1)
