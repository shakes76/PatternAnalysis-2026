"""Report the completed exploratory runs without changing selection locks."""
import hashlib
import json


def add_expanded_section(code, add, page, table, pct):
    provenance = json.loads((code / 'expanded_evidence_provenance.json').read_text())
    for name, record in provenance.items():
        assert hashlib.sha256((code / name).read_bytes()).hexdigest() == record['sha256'], name
    comparison = json.loads((code / 'expanded_ns_results_670000.json').read_text())
    expansion = json.loads((code / 'expanded_n_results_669929.json').read_text())
    plan = json.loads((code / 'expanded_ns_comparison_plan.json').read_text())
    decision = json.loads((code / 'coursework_model_decision.json').read_text())
    assert comparison['status'] == expansion['status'] == 'passed'
    assert comparison['plan_sha256'] == hashlib.sha256((code / 'expanded_ns_comparison_plan.json').read_bytes()).hexdigest()
    assert plan['primary_fixed_thresholds'] == [0.05, 0.5]
    assert comparison['primary_points_with_strict_joint_improvement'] == []
    assert comparison['all_grid_points_with_strict_joint_improvement'] == [0.7]
    assert not comparison['reassessment_inference'] and not comparison['original_study_replaced']
    assert [(d['role'], d['model'], d['confidence']) for d in decision['choices']] == [
        ('coursework_primary', 'YOLOv8n', 0.5), ('expanded_development_candidate', 'YOLOv8s', 0.5)]
    models = comparison['models']
    rows = {r['threshold']: r for r in comparison['same_threshold_comparisons']}
    assert models['n']['checkpoint_sha256'] != models['s']['checkpoint_sha256']

    page()
    add(r'''## Additional experiment: expanded training data

After completing the main experiment, the training set was expanded to test whether more data would help. The original 768 training images were kept and 1,495 images with benign/malignant labels but missing lesion IDs were added. The indeterminate image remained excluded. Validation and reassessment images and labels were unchanged. The new models were evaluated on validation only, and the main experiment was kept as originally reported.

The added images also lack patient IDs. Checks found no recorded links or exact decoded-image duplicates crossing the splits, but unknown lesion or patient links may still connect them. No IDs were invented, and image deduplication does not establish independence.''')
    table(['Design feature', 'Main study', 'Expanded exploration'], [
        ['Training images (Benign / Malignant)', '768 (576 / 192)', '2,263 (1,826 / 437)'],
        ['Validation images (Benign / Malignant)', '165 (124 / 41)', 'Same images and labels'],
        ['Epochs / optimizer calls, each model', '60 / 2,880', '60 / 8,520'],
        ['Missing-lesion training images', '0', '1,495'],
    ])
    add(r'''Expanded n and s started from their respective original COCO weights. Both used 60 epochs, seed 3710, FP32, 640-pixel input, batch and nominal batch 16, AdamW, the same learning-rate schedule and flips only. Each completed 8,520 optimizer calls. Image hashes and per-call learning rates matched between the runs. Checkpoints were selected by validation macro mAP50:95 using EMA weights.

Compared with main-study n, expanded n increased malignant AP50 from 0.3591 to 0.5311 on the same validation images. At confidence 0.05, malignant TP rose from 25 to 33, alongside an increase in benign false-alert images from 38 to 55. The larger training set also changed class balance and increased optimizer calls, so the result cannot be attributed to sample count alone.''')
    table(['Expanded-model validation AP', 'YOLOv8n', 'YOLOv8s'], [
        ['Macro mAP50', *[f'{models[m]["mAP50"]:.4f}' for m in ('n', 's')]],
        ['Macro mAP50:95', *[f'{models[m]["mAP50_95"]:.4f}' for m in ('n', 's')]],
        *[[f'{label} {metric.replace("_", ":")}', *[f'{models[m]["ap"][cls][metric]:.4f}' for m in ('n', 's')]]
          for cls, label in [('0', 'Benign'), ('1', 'Malignant')]
          for metric in ('AP50', 'AP50_95')],
    ])
    add('s has higher macro mAP but lower malignant AP50 and AP50:95. The higher overall average therefore does not show better malignant classification and detection.')
    add('Training took %.2f / %.2f minutes for n/s, with peak allocated CUDA memory of %.2f / %.2f GiB. Selected EMA epochs were %d / %d, and parameter counts were 3,011,238 / 11,136,374. Jobs 669929 and 670000 ran on separate A100 nodes, so these times are not a controlled speed comparison.' % (
        models['n']['training_seconds'] / 60, models['s']['training_seconds'] / 60,
        models['n']['peak_cuda_allocated_bytes'] / 2**30, models['s']['peak_cuda_allocated_bytes'] / 2**30,
        models['n']['best_epoch'], models['s']['best_epoch']))

    page()
    add(r'''## Expanded n/s: comparison at the same thresholds

Before training expanded s, confidence 0.05 and 0.50 were chosen as the main comparison points. Confidence 0.70 was an additional reference because expanded n had already selected it. Improvement in both measures required more malignant TP and fewer benign false-alert images. The original 14-threshold grid was kept. Both models used class-aware NMS at 0.7 and correct-class IoU >= 0.5 matching.''')
    common_rows = []
    for t in [*plan['primary_fixed_thresholds'], plan['additional_fixed_reference_threshold']]:
        for title, render in [
            ('Malignant TP / FP / FN', lambda p: f'{p["malignant_tp"]} / {p["malignant_fp"]} / {p["malignant_fn"]}'),
            ('Malignant recall / precision', lambda p: f'{pct(p["malignant_recall"])} / {pct(p["malignant_precision"])}'),
            ('Benign images with malignant alert', lambda p: f'{p["benign_false_alert_images"]}/124 ({pct(p["benign_false_alert_rate"])})'),
        ]:
            common_rows.append([f'{t:.2f}: {title}', *[render(rows[t][m]) for m in ('n', 's')]])
    table(['Confidence and measure', 'YOLOv8n', 'YOLOv8s'], common_rows)
    add(r'''At both main thresholds, s produces fewer benign false alerts but also detects fewer malignant lesions. At 0.70, s detects one more malignant lesion and alerts on two fewer benign images; this is the only grid point that improves both counts. At that threshold, both models detect 11 malignant cases, only n detects one, only s detects two, and both miss 27. The five n false alerts and three s false alerts occur on different benign images. The net improvement is small and does not show that s is consistently better across cases.

Malignant FP counts refer to prediction boxes, while benign false alerts count images. For example, n at 0.05 produces 85 malignant FP boxes and alerts on 55 benign images. These quantities should not be treated as interchangeable, and an alert does not require a correctly localized box.''')
    add('### Thresholds selected under the same false-alert limit')
    table(['Measure', 'Expanded n', 'Expanded s'], [
        ['Existing validation-selected confidence', *[f'{models[m]["selected_threshold"]:.2f}' for m in ('n', 's')]],
        ['Malignant detections / recall', *[
            f'{models[m]["selected_point"]["classes"]["Malignant"]["tp"]}/41 ({pct(models[m]["selected_point"]["classes"]["Malignant"]["recall"])})'
            for m in ('n', 's')]],
        ['Benign images with malignant alert', *[
            f'{models[m]["selected_point"]["image_alerts"]["benign_fp"]}/124 ({pct(models[m]["selected_point"]["image_alerts"]["benign_false_alert_rate"])})'
            for m in ('n', 's')]],
    ])
    add('Both selected thresholds meet the limit of at most 12 benign false-alert images on validation. s has higher malignant recall within that limit, but also more false alerts. Expanded s at 0.50 was therefore considered a development candidate at that stage. Following the later advice to use only known-lesion-ID images, both expanded models are retained as historical exploratory results and are excluded from the current model choice. It compares separately selected thresholds; it does not show that s improves both measures at the same threshold. Main-study n at 0.50 remains the original baseline.')

    page()
    add(r'''## Expanded n/s: all threshold results

The table includes all 14 planned thresholds. Malignant TP is out of 41 lesions, and benign alerts are out of 124 images. All results are from validation.''')
    table(['Confidence', 'n M TP', 's M TP', 'n B alerts', 's B alerts'], [
        [f'{t:.2f}', rows[t]['n']['malignant_tp'], rows[t]['s']['malignant_tp'],
         rows[t]['n']['benign_false_alert_images'], rows[t]['s']['benign_false_alert_images']]
        for t in plan['full_threshold_grid']])
    add(r'''![Expanded n/s validation trade-off](assets/expanded_threshold_tradeoff.png)

Markers show the tested thresholds; joining lines do not represent additional tests. All 28 n/s threshold results were independently recalculated from saved predictions. AP was computed by Ultralytics and was not independently recalculated.

Only one seed and 41 malignant validation images were used. Validation had already informed development decisions, and unknown identity links may make results optimistic. Differences in architecture and pretrained-weight loading also remain. Neither expanded model was evaluated on reassessment, and the original models and thresholds were unchanged.''')
