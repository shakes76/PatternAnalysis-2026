"""Build course-required reporting and the bounded follow-up analysis."""
import json
from report_metric_rationale import add_metric_rationale


def add_course_closeout(code, add, page, table, pct):
    metrics = json.loads((code/'course_metrics_20261022.json').read_text())
    follow = json.loads((code/'followup_results_20261022.json').read_text())
    policy = json.loads((code/'followup_policy_decision_20261022.json').read_text())
    assert metrics['status'] == 'passed' and not metrics['selection_changed']
    reports = metrics['reports']
    current = reports['n_geometry_validation']['metrics']
    f = lambda x: 'N/A' if x is None else f'{x:.4f}'
    pc = lambda x: 'N/A' if x is None else pct(x)
    keys = ['baseline_n_validation', 'baseline_s_validation', 'n_cls1_validation',
            'n_long120_validation', 'n_geometry_validation', 'n_resolution896_validation',
            'n_positive2_validation', 's_geometry_validation']

    page()
    add('## Course clarification and reporting definitions')
    add('The student supplied a reply from Zhao on 22 October. It asks for both overall and per-class localization IoU, and classification accuracy with additional metrics such as F1 or recall for imbalanced data. IoU >= 0.8 remains mandatory, but the reply says that an unmet target will not lose marks if reasonable trials and best effort are supported by sound analysis. This is not a guarantee of the overall grade. The supplied reply is preserved in `teacher_response_20261022.txt`; the original email or Ed thread was not independently fetched.')
    add('In a follow-up Ed reply supplied on 23 October, Zhao recommends reading the paper and choosing metrics that support the claims, treating the project as open research. This does not specify one IoU denominator or withdraw the earlier target. I therefore explain the choices and their limits below; I do not treat a missing universal formula as a reason to delay reporting. The screenshot transcription is in `teacher_response_20261023.txt`.')
    table(['Measure', 'Definition'], [
        ['Image classification', 'The class of the highest-confidence retained box. The same fixed score threshold as detection is used. Ties use class ID then box coordinates. GT is not used to choose the box, and no localization test is applied.'],
        ['Accuracy and no prediction', 'Correct image labels / all images. No retained box is an abstention and counts as incorrect, including on benign images. Coverage is reported separately.'],
        ['Classification F1 and recall', 'For each class, TP/FP/FN come from one image decision. Abstentions add FN for the true class. F1 = 2TP/(2TP+FP+FN). Macro F1 averages the two class F1 values.'],
        ['Detection F1 and recall', 'Use all boxes and the original one-to-one, correct-class IoU >= 0.5 matcher. Duplicate or unmatched boxes remain FP. These counts can differ from image classification.'],
        ['Overall IoU', 'Sum the individual IoUs and divide by their count. All-GT measures use 165 images, with 124 benign and 41 malignant. TP-only measures use the matched TP count. The equal-weight mean of class means is a separate macro statistic.'],
    ])
    add('The all-GT best-box diagnostics use the greatest GT overlap among retained boxes, either requiring the correct class or ignoring class. If no eligible box exists, the value is zero; a same-class box with IoU below 0.5 still contributes its actual overlap. These are oracle diagnostics and do not penalize extra boxes. TP-only IoU uses matched correct-class predictions and excludes missed GTs. Primary-box IoU instead uses the same GT-free highest-score box as classification, with zero for no prediction; the correct-class version also assigns zero to a wrong-class primary box.')
    add('The 22 October additions are fixed in `course_closeout_plan.json`; the 23 October revision adds literature checks and interpretation. Neither changes the checkpoint, threshold or comparison budget. The 10% benign-image alert limit is my experimental policy, not a course or clinical standard. Confidence is a model score, not a calibrated probability of cancer.')

    add_metric_rationale(add, page, table)

    page()
    add('## Added classification metrics: validation')
    add('All rows use the same 165 validation images and the operating thresholds already selected in their original studies. This is descriptive reporting on a repeatedly used development set, not an independent comparison or a fresh test.')
    table(['Model', 'Conf.', 'Accuracy', 'Macro F1', 'M recall'], [
        [reports[k]['label'], f"{reports[k]['threshold']:.2f}",
         pc(reports[k]['metrics']['classification']['accuracy']),
         f(reports[k]['metrics']['classification']['macro_f1']),
         pc(reports[k]['metrics']['classification']['classes']['Malignant']['recall'])] for k in keys])
    c = current['classification']
    add('### Current candidate: n with light geometry, confidence 0.40')
    table(['Ground truth', 'Predicted benign', 'Predicted malignant', 'No prediction'], [
        ['Benign', *c['confusion_matrix'][0]], ['Malignant', *c['confusion_matrix'][1]]])
    table(['Class', 'Precision', 'Recall', 'F1'], [[name, pc(v['precision']), pc(v['recall']), f(v['f1'])]
                                               for name, v in c['classes'].items()])
    add(f"Accuracy is {c['correct']}/165 ({pc(c['accuracy'])}), with predictions on {c['covered']}/165 images ({pc(c['coverage'])} coverage). Balanced accuracy is {pc(c['balanced_accuracy'])}. Accuracy among only the covered images is {pc(c['conditional_accuracy_covered_only'])}; it excludes abstentions and is not the headline accuracy. Always predicting benign would achieve 124/165 = 75.15% accuracy and zero malignant recall. This is a class-prior reference, not a detector, and shows why accuracy alone is insufficient.")
    add('Malignant classification recall is 16/41, while malignant detection recall is 18/41. Two images have a valid malignant detection but a higher-confidence benign box, so the single image decision is wrong. Benign and malignant detection F1 values are '+f(current['detection_f1']['Benign'])+' and '+f(current['detection_f1']['Malignant'])+'. The 12/124 benign image alerts count any malignant box; only seven benign images have a malignant primary box. These are different definitions, not inconsistent counts.')

    page()
    add('## Overall and per-class IoU: validation')
    add('The current candidate is n with light geometry at confidence 0.40. Each cell shows mean IoU and its denominator. Overall is pooled over observations, not the unweighted average of the two class values.')
    kinds = [('all_gt_class_correct', 'Best correct-class box, all GT'),
             ('all_gt_class_agnostic', 'Best box ignoring class, all GT'),
             ('tp_only', 'Matched correct-class TP only'),
             ('primary_box_ignore_class', 'Highest-score box, ignore class'),
             ('primary_box_correct_class', 'Highest-score box, correct class')]
    def iou_row(label, values):
        return [label, *[f"{f(v['mean'])} (N={v['denominator']})"
                         for v in [values['overall'], values['per_class']['Benign'], values['per_class']['Malignant']]]]
    table(['IoU definition', 'Overall', 'Benign', 'Malignant'],
          [iou_row(label, current['localization'][kind]) for kind, label in kinds])
    add('The TP-only means exceed 0.8, but omit 45 unmatched GTs, including 23 malignant lesions. They do not demonstrate that the full pipeline meets the target across all lesions. The all-GT correct-class mean is 0.6238 overall; its class-macro counterpart is 0.5408. The two differ because benign images are more numerous.')
    add('### Same all-GT correct-class diagnostic across validation runs')
    table(['Model at its selected threshold', 'Overall / 165', 'Benign / 124', 'Malignant / 41'], [
        [reports[k]['label'], *[f(v['mean']) for v in [
            reports[k]['metrics']['localization']['all_gt_class_correct']['overall'],
            reports[k]['metrics']['localization']['all_gt_class_correct']['per_class']['Benign'],
            reports[k]['metrics']['localization']['all_gt_class_correct']['per_class']['Malignant']]]]
        for k in keys])
    add('These all-GT averages remain paired with detection precision, recall and F1, because best-box IoU alone does not penalize duplicate predictions. No class-agnostic or TP-only value is substituted for the correct-class all-GT metric used in the original selection rule.')

    page()
    add('## Added metrics: original supplementary reassessment')
    add('The following additions reuse the original saved reassessment predictions. No new threshold scan or inference was required. These are the original n and s models, not the later geometry candidate. The 165 images had prior exposure and this evaluation remains retrospective.')
    rr = [reports['baseline_'+s+'_reassessment'] for s in 'ns']
    table(['Measure', 'Original n, confidence 0.50', 'Original s, confidence 0.80'], [
        ['Image accuracy', *[pc(r['metrics']['classification']['accuracy']) for r in rr]],
        ['Classification macro F1', *[f(r['metrics']['classification']['macro_f1']) for r in rr]],
        ['Classified images / 165', *[r['metrics']['classification']['covered'] for r in rr]],
        *[[name+' classification recall / F1', *[pc(r['metrics']['classification']['classes'][name]['recall'])+' / '+f(r['metrics']['classification']['classes'][name]['f1']) for r in rr]] for name in ('Benign','Malignant')],
        *[[name+' detection F1', *[f(r['metrics']['detection_f1'][name]) for r in rr]] for name in ('Benign','Malignant')],
    ])
    for r in rr:
        add('### '+r['label']+': overall and classwise localization')
        table(['IoU definition', 'Overall', 'Benign', 'Malignant'],
              [iou_row(label, r['metrics']['localization'][kind]) for kind, label in kinds[:3]])
    add('These are the original baseline results. The later geometry checkpoint has now completed its own fixed retrospective evaluation, reported in the final sections. The earlier ten-cache metrics file is preserved as a historical reporting snapshot; its pending field records the state before that final run.')

    page()
    add('## Bounded follow-up experiments: 22 October')
    add('I tested three further changes against the geometry reference, using the same 768 training and 165 validation images, COCO initialization, 60 epochs, AdamW, FP32, batch 16 and seed 3710. Mosaic, mixed-image augmentation and oversampling remained off. The 896-input run changed only resolution. The loss-weight run doubled only the malignant positive BCE term during training; it kept soft targets, assignment, box loss and DFL unchanged, and used ordinary validation loss. The size comparison changed n to s with the same geometry preset.')
    labels = {'reference_geometry': 'n, geometry reference', 'n_resolution896': 'n, input 896',
              'n_positive2': 'n, positive weight 2', 's_geometry': 's, geometry'}
    order = list(labels)
    def point(k): return follow['reports'][k]['selected']
    table(['Model', 'Conf.', 'M TP / 41', 'B alerts / 124', 'M all-GT IoU'], [
        [labels[k], f"{point(k)['threshold']:.2f}", point(k)['classes']['Malignant']['tp'],
         point(k)['image_alerts']['benign_fp'], f(point(k)['classes']['Malignant']['iou']['all_gt_class_correct']['mean'])] for k in order])
    add('The planned replacement rule required at least three extra malignant TP, a higher malignant all-GT correct-class IoU, and no more than a 0.03 drop in benign all-GT correct-class IoU, while meeting the same alert budget. None qualified. The next seed-confirmation stage was therefore not triggered, and the prior reference was retained.')
    table(['Model', 'At 0.40: M TP / B alerts', 'M AP50 / AP50:95'], [
        [labels[k], str(follow['reports'][k]['fixed_thresholds']['0.4']['classes']['Malignant']['tp'])+' / '+str(follow['reports'][k]['fixed_thresholds']['0.4']['image_alerts']['benign_fp']),
         f(follow['reports'][k]['ap']['1']['AP50'])+' / '+f(follow['reports'][k]['ap']['1']['AP50_95'])] for k in order])
    add('Positive weighting improved malignant AP50 from 0.4646 to 0.5052. At confidence 0.40 it also raised malignant TP from 18 to 23, but benign alerts rose from 12 to 24. Under the fixed alert budget its selected threshold was 0.70, yielding 17 malignant TP. Weighting was therefore not uniformly worse; it failed the specific replacement rule. A larger input or larger model likewise did not improve the selected operating point in this run. One seed does not establish a universal ranking.')
    table(['Model', 'Best epoch / 60', 'Training-call time', 'Peak CUDA allocation'], [
        [labels[k], follow['reports'][k]['resources']['checkpoint_selection']['best_epoch'],
         f"{follow['reports'][k]['resources']['elapsed_seconds']/60:.2f} min",
         f"{follow['reports'][k]['resources']['peak_cuda_allocated_bytes']/1024**3:.2f} GiB"] for k in order])
    add('All runs completed 2,880 optimizer calls on an A100. Time includes the training call and its internal validation, but excludes queueing, installation and the separate evaluation. Nodes differed. These are resource observations, not phone measurements.')

    page()
    add('## Error evidence, stopping decision and remaining work')
    add('For the retained geometry model, the 23 malignant validation FN comprised 17 lesions with a sufficiently localized benign prediction and six images without a retained prediction. Raw-score inspection at confidence 0.40 found no well-localized malignant candidate above the threshold for those 23 lesions. At the diagnostic floor of 0.001, 30 lesions had a retained valid malignant candidate; the other 11 had a localized malignant score above the floor but lost the class argmax to benign. At these two checkpoints, none was lost solely in the subsequent NMS stage. GT was used only to diagnose this chain, not to choose operational predictions. CPU diagnostic TP/FP/FN matched the GPU cache; minor IoU differences are not mixed into the main tables.')
    add('A separate GT-free single-box experiment chose the candidate with the largest class score and classified it using M/(B+M). Its plan was recorded before scoring the policy. The selected ratio cutoff of 0.60 gave 17/41 malignant TP and 12/124 benign alerts, failing the improvement rule. The ratio is not a calibrated cancer probability and is not the ordinary confidence threshold. This policy was rejected and did not replace the current detector.')
    add('The evidence supports a classification and score-selection weakness, but not a single cause. The loss-weight experiment tested one imbalance intervention and revealed an operating-point trade-off. Extra training also changed learning-rate decay; geometry was tested as a combined preset; size and resolution changes did not isolate diagnosis-label quality. Hair, rulers, reflections, lesion size and metadata label quality were not controlled interventions. They remain possible factors, not established explanations for the failures.')
    add('My final development choice remains YOLOv8n with light geometry, seed 3710, confidence 0.40. The earlier second-seed check supported its gain over original n on the same validation images. This bounded series covers training duration, loss weighting, augmentation, resolution, capacity and prediction policy. Continuing to search this small validation set would increase selection bias. I therefore stop model tuning and report the shortfall and trade-offs under the teacher\'s best-effort provision, without claiming that this automatically satisfies every marking criterion.')
    add('### Reproducing and checking the added metrics')
    add('`classification_metrics.py` defines image decisions, classification Accuracy/F1/Recall, detection F1 and pooled/classwise IoU. `complete_course_metrics.py` checks cached image membership, GT, checkpoints, thresholds and file hashes, then reproduces all original detection counts before adding the new metrics. Seven targeted tests cover abstentions, wrong location, conflicting classes, duplicate boxes, pooled denominators, ties and invalid records. The existing 40 tests also passed. An independent NumPy implementation reproduced classification counts, F1 and the all-GT IoU calculations across all ten caches. No training or threshold selection occurs in this reporting step.')
    add('''```bash
python recognition/YOLO_s4905914/complete_course_metrics.py
python recognition/YOLO_s4905914/build_final_report.py
python recognition/YOLO_s4905914/export_readme_pdf.py \\
  --output output/pdf/COMP3710_Project3_Report.pdf
```''')
    add('Cached predictions and source manifests must be restored for the first command; they are not training inputs supplied through Git. The second command uses the included summary JSON files. The third exports that same README to PDF, with a source-hash record. The current inference command and exact checkpoint/config/threshold hashes are recorded in the decision files; original examples remain explicitly labelled as historical.')
    add('The final evaluation uses the separate `evaluate_locked_candidate.py` command and `audit_final_reassessment.py` audit. Its launch package and the pre-inference CPU decision are in `studies/final_reassessment_20261022/`. The earlier train/validation-only experiment loaders and their frozen selection rules remain unchanged.')
    add('The source code and results remain available for review. The student must verify personal check-off completion, the applicable deadline and acceptance of imported Ultralytics components. The teacher\'s metric clarification does not answer those points. A course PR and the README PDF submission remain separate required actions; no new PR or Turnitin submission was made during this update.')
