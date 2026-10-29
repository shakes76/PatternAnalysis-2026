"""Final fixed retrospective candidate results; no decision or threshold changes."""
import json


def add_final_candidate(code,add,page,table,pct):
    path=code/'final_candidate_reassessment_20261022.json'
    if not path.exists():return
    final=json.loads(path.read_text())
    assert final['status']=='passed' and not final['selection_changed'] and not final['fresh_blind_test']
    old=json.loads((code/'course_metrics_20261022.json').read_text())['reports']['baseline_n_reassessment']
    m=final['metrics'];c=m['classification'];d=m['detection'];loc=m['localization']
    f=lambda x:'N/A' if x is None else f'{x:.4f}'
    pc=lambda x:'N/A' if x is None else pct(x)
    page()
    add('## Final candidate: fixed retrospective evaluation')
    add('After the 22 October clarification, I evaluated the retained YOLOv8n geometry checkpoint (seed 3710) on the same 165 supplementary reassessment images used in the earlier baseline study. The checkpoint and confidence > 0.40 were fixed before this run. I did not scan thresholds or change the model after seeing these results. The 23 October revision only clarifies reporting. Prior image exposure remains, so this is a retrospective evaluation, not a new blind test.')
    add('The queued A100 job was cancelled before it started because its estimated start was the next morning. This final run used the local CPU with FP32 and the pinned Ultralytics/PyTorch versions. The photo hashes, single-letterbox preprocessing, class-aware single-label NMS at 0.70 and IoU >= 0.50 detection matching were unchanged. Device choice was recorded before inference. Small numerical differences across devices remain possible; this run supplies no A100 speed measurement.')
    table(['Measure','Original n, 0.50','Final geometry n, 0.40'],[
        ['Image classification accuracy',pc(old['metrics']['classification']['accuracy']),pc(c['accuracy'])],
        ['Classification macro F1',f(old['metrics']['classification']['macro_f1']),f(c['macro_f1'])],
        ['Malignant classification recall',pc(old['metrics']['classification']['classes']['Malignant']['recall']),pc(c['classes']['Malignant']['recall'])],
        ['Malignant detection TP / 41',old['metrics']['detection']['classes']['Malignant']['tp'],d['classes']['Malignant']['tp']],
        ['Malignant detection precision',pc(old['metrics']['detection']['classes']['Malignant']['precision']),pc(d['classes']['Malignant']['precision'])],
        ['Malignant detection F1',f(old['metrics']['detection_f1']['Malignant']),f(m['detection_f1']['Malignant'])],
        ['Benign images with malignant alert / 124',old['metrics']['detection']['image_alerts']['benign_fp'],d['image_alerts']['benign_fp']],
        ['mAP50 / mAP50:95',f(old['mAP50'])+' / '+f(old['mAP50_95']),f(final['mAP50'])+' / '+f(final['mAP50_95'])],
    ])
    add('Each model keeps its own validation-selected threshold. This compares the two fixed operating policies; it does not isolate architecture or augmentation at a common score cutoff. The validation alert budget is not a guarantee on another set. AP is calculated across confidence ranks using the fixed low-score floor; it is not AP at confidence 0.40.')
    table(['Final model class','AP50','AP50:95','Detection recall'],[[name,f(final['ap'][str(i)]['AP50']),f(final['ap'][str(i)]['AP50_95']),pc(d['classes'][name]['recall'])] for i,name in enumerate(('Benign','Malignant'))])
    paired=final['paired_malignant_vs_original_n']
    add(f"Among the 41 malignant images, both models detect {paired['both']}, only geometry n detects {paired['geometry_only']}, only original n detects {paired['original_n_only']}, and both miss {paired['neither']}. These paired counts describe this sample; they do not establish a population-level improvement. The selected model remains the validation-selected geometry checkpoint.")

    page()
    add('## Final candidate: classification and localization')
    table(['Ground truth','Predicted benign','Predicted malignant','No prediction'],[
        ['Benign',*c['confusion_matrix'][0]],['Malignant',*c['confusion_matrix'][1]]])
    table(['Image class','Precision','Recall','F1'],[[name,pc(v['precision']),pc(v['recall']),f(v['f1'])] for name,v in c['classes'].items()])
    add(f"Full-cohort accuracy is {c['correct']}/165 ({pc(c['accuracy'])}); coverage is {c['covered']}/165 ({pc(c['coverage'])}). Abstentions count as incorrect. Balanced accuracy is {pc(c['balanced_accuracy'])}, and macro F1 is {f(c['macro_f1'])}. This image-class decision uses the highest-confidence retained box without using GT. All boxes still enter detection matching and false-positive counts.")
    add('Always predicting benign would give 75.15% accuracy and zero malignant recall on this class mix, so accuracy alone would be misleading. The final model classifies 15 malignant images correctly, but two of their boxes have IoU below 0.5, leaving 13 malignant detection TP. Three benign images have a correct benign primary box plus an extra malignant alert; this explains 13 benign alerts versus 10 malignant image-class decisions on benign GT.')
    kinds=[('all_gt_class_correct','Best correct-class box, all GT'),('all_gt_class_agnostic','Best box ignoring class, all GT'),('tp_only','Matched correct-class TP only'),('primary_box_ignore_class','Highest-score box, ignore class'),('primary_box_correct_class','Highest-score box, correct class')]
    table(['IoU definition','Overall','Benign','Malignant'],[[label,*[f(v['mean'])+' (N='+str(v['denominator'])+')' for v in [loc[k]['overall'],loc[k]['per_class']['Benign'],loc[k]['per_class']['Malignant']]]] for k,label in kinds])
    a=loc['all_gt_class_correct'];tp=loc['tp_only']
    add(f"The all-GT correct-class mean is {f(a['overall']['mean'])} overall, {f(a['per_class']['Benign']['mean'])} for benign and {f(a['per_class']['Malignant']['mean'])} for malignant. The TP-only table excludes {165-tp['overall']['denominator']} unmatched GTs, including {41-tp['per_class']['Malignant']['denominator']} malignant lesions. It cannot replace the all-GT result or establish successful detection of those missed lesions.")
    add('The matched subset exceeds 0.8 on average, overall and for each class; it does not show that every TP has IoU >= 0.8. Both all-GT and primary-box means remain below 0.8. The evidence therefore supports accurate boxes among many successful detections, but not an unqualified claim that localization and recognition meet the target across the cohort. I report this shortfall with the controlled trials under the earlier best-effort provision; the latest reply does not guarantee a grade.')
    add('`final_candidate_reassessment_20261022.json` contains the verified summary. `audit_final_reassessment.py` checks the image membership and labels, original photo hashes, checkpoint, lock, reporting code and prediction hashes. It independently replays TP/FP/FN, alerts, classification counts/F1 and every reported IoU denominator. AP values remain the pinned Ultralytics implementation. The original ten-cache reporting record is preserved as an earlier snapshot; this separate record supplies the newly completed evaluation.')

    page()
    add('## Final candidate: descriptive failure checks')
    counts=final['malignant_fn_partition'];fn=d['classes']['Malignant']['fn']
    add(f"There are {fn}/41 malignant detection FN at the fixed operating point: {counts.get('localized_benign_prediction',0)} have a retained benign box with IoU >= 0.5, {counts.get('no_retained_prediction',0)} have no retained box, and {counts.get('other_localization_or_classification_failure',0)} have other class/location failures. These categories describe the output; they do not prove why the network failed.")
    table(['Case and GT','Observed output at confidence > 0.40'],[
        [case['image_id']+'; '+('Malignant' if case['class_id'] else 'Benign')+'; '+case['category'],
         '; '.join(('M' if p['class_id'] else 'B')+f" score={p['confidence']:.3f}, IoU={p['iou']:.3f}" for p in case['retained']) or 'No retained box'] for case in final['cases']])
    add('Examples use the lowest available image ID in each specified failure category, without reusing an image. They are illustrative, not a random sample. A well-localized benign prediction on a malignant GT is a classification failure; an absent retained box does not by itself prove that the image was too small, low contrast or obscured. No parameter changes followed this inspection.')
    add('The earlier five illustrated cases remain labelled as original baseline examples. Their boxes are not presented as predictions from the final geometry model. The final candidate\'s quantitative errors above come from its own saved prediction cache.')
    add('### Conclusions supported by the evidence')
    add('At the locked operating point, malignant detection recall is 13/41 (31.71%) while benign recall is 104/124 (83.87%). Malignant classification recall is also low at 15/41 (36.59%). Together with the 16 localized benign predictions on malignant GTs, this identifies class discrimination as a major observed weakness. The ten absent retained predictions and two other failures show that the weakness is not exclusively classification. The outputs do not establish whether imbalance, label definitions, augmentation or visual artefacts caused these errors.')
    add('The final model has more malignant TP and fewer benign alerts than the original n policy on this sample, but the comparison includes different validation-selected thresholds and only 41 malignant images. One malignant case changes recall by 2.44 percentage points. These descriptive differences do not establish statistical significance or performance on new patients. The benign alert rate is 13/124 (10.48%), so the validation budget of at most 10% does not transfer exactly to reassessment.')
    add('Known lesion links are separated, but 708 cohort images lack patient IDs. Repeated validation use and historical exposure of reassessment images further limit generalization claims. The completed trials justify retaining the selected model for this coursework analysis, not claiming that its performance ceiling has been reached. A stronger future test would require newly held-out groups and a decision rule fixed before viewing them. No such test, clinical validation or mobile-device evaluation is claimed here.')
