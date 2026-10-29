"""Plain monochrome report supplement from completed, verified results only."""
import json
from pathlib import Path
from html import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
S=Path(__file__).resolve().parent;R=S.parents[1]
read=lambda p:json.loads(p.read_text())
x=read(S/'results.json');d=read(S/'decision.json');policy=read(S/'policy_results_seed3710.json')
assert d['status']=='completed' and x['status']=='completed_initial'
diag=read(R/'outputs/known_followup_diagnostics_20261022/raw_diagnosis.json')
unc=read(S/'conditional_uncertainty.json')
labels={'reference_geometry':'Previous n + geometry','n_resolution896':'n + geometry, 896 px',
        'n_positive2':'n + geometry, M weight 2','s_geometry':'s + geometry',
        'single_lesion_policy':'Single-box score ratio'}
body=ParagraphStyle('body',fontName='Times-Roman',fontSize=10.5,leading=14,spaceAfter=8)
title=ParagraphStyle('title',parent=body,fontName='Times-Bold',fontSize=16,leading=19,spaceAfter=12)
heading=ParagraphStyle('heading',parent=body,fontName='Times-Bold',fontSize=12,leading=15,spaceBefore=8)
small=ParagraphStyle('small',parent=body,fontSize=8.5,leading=10.5,spaceAfter=4)
cell=ParagraphStyle('cell',parent=small,spaceAfter=0)
story=[]
def p(t,style=body):story.append(Paragraph(t,style))
def table(rows,widths):
    data=[[Paragraph(escape(str(v)),cell) for v in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),
        ('LINEBELOW',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,-1),(-1,-1),.4,colors.black),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
        ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.extend([t,Spacer(1,9)])
fmt=lambda n:'N/A' if n is None else f'{n:.3f}'
reports=x['reports']
p('COMP3710 Project 3: Further Validation Experiments',title)
p('22 October 2026. Supplement to the existing project report. These are development results on the same validation set, not a new test result.')
p('Purpose and fixed data',heading)
p('I tested whether the previous YOLOv8n model could detect more malignant lesions without increasing the allowed number of benign-image false alerts. The cohort still contains 1,098 images with known lesion IDs: 768 train, 165 validation and 165 excluded retrospective reassessment images. Only train and validation were used in this round. Validation contains 124 benign and 41 malignant images.')
p('All three training comparisons used fresh original COCO weights, 60 epochs, FP32, AdamW, batch 16 and seed 3710. Light geometry augmentation was kept in all runs. I changed one factor per comparison: image size from 640 to 896; malignant positive-term BCE weight from 1 to 2; or model size from n to s. Weighting is applied to soft assignment targets and is not oversampling. Validation loss remains unweighted.')
p('Main comparison',heading)
rows=[['Model / rule','Cutoff','M TP / 41','B alerts / 124','M IoU, all GT','M AP50','Macro AP50:95']]
for name,r in reports.items():
    a=r['selected'];m=a['classes']['Malignant']
    rows.append([labels[name],f"{a['threshold']:.2f}",m['tp'],a['image_alerts']['benign_fp'],
        fmt(m['iou']['all_gt_class_correct']['mean']),fmt(r['ap']['1']['AP50']),fmt(r['mAP50_95'])])
a=policy['selected'];m=a['classes']['Malignant']
rows.append([labels['single_lesion_policy'],f"{a['ratio_threshold']:.2f} ratio",m['tp'],a['image_alerts']['benign_fp'],fmt(m['iou']['all_gt_class_correct']['mean']),'N/A','N/A'])
table(rows,[140,52,51,60,62,59,87])
p('Each cutoff was selected from the same predeclared 14-point grid, allowing at most 12 benign-image alerts. The last row uses M/(B+M) on one selected raw candidate, so its cutoff is not the ordinary YOLO confidence threshold. It was a separately recorded, diagnosis-driven extension. No GT was used to select its box.',small)
p('Decision',heading)
sel=reports[d['selected']]['selected'];m=sel['classes']['Malignant']
p(f"The selected development model is <b>{escape(labels[d['selected']])}</b>, at confidence <b>{d['confidence']:.2f}</b>. It has {m['tp']}/41 malignant TP, {sel['image_alerts']['benign_fp']}/124 benign-image alerts and malignant all-GT correct-class IoU {m['iou']['all_gt_class_correct']['mean']:.3f}. {escape(d['reason'])}.")
p('The replacement rule required at least 3 additional malignant TP, higher malignant all-GT IoU and no decrease greater than 0.03 in benign all-GT IoU. A qualifying training candidate had to pass the same rule at seed 3711. The reporting seed stays 3710.')
p('A useful change would not prove a single cause of the earlier failures. For example, positive weighting changes the training objective and the resulting scores; it does not establish that class imbalance alone caused the errors.',small)

story.append(PageBreak())
p('Localization and error diagnosis',title)
chosen=reports[d['selected']]['selected']['classes']
p('For the selected model, benign precision/recall are '
    f"{chosen['Benign']['precision']:.3f}/{chosen['Benign']['recall']:.3f}; malignant precision/recall are "
    f"{chosen['Malignant']['precision']:.3f}/{chosen['Malignant']['recall']:.3f}. Matching uses the correct class and IoU >= 0.5.")
rows=[['Model','B IoU, all GT (124)','M IoU, all GT (41)','B IoU, TP only (n)','M IoU, TP only (n)']]
for name,r in reports.items():
    c=r['selected']['classes'];b=c['Benign'];m=c['Malignant']
    rows.append([labels[name],fmt(b['iou']['all_gt_class_correct']['mean']),fmt(m['iou']['all_gt_class_correct']['mean']),
        f"{fmt(b['iou']['tp_only']['mean'])} ({b['tp']})",f"{fmt(m['iou']['tp_only']['mean'])} ({m['tp']})"])
table(rows,[151,90,90,90,90])
p('All-GT correct-class IoU averages over every real lesion in the class. If there is no retained prediction of the correct class, that image contributes zero. TP-only IoU uses only class-correct detections matched at IoU >= 0.5. These denominators answer different questions; a high TP-only value does not account for missed lesions.')
rows=[['Model','M TP at IoU >= 0.5','M TP at IoU >= 0.75','M TP at IoU >= 0.8']]
for name,r in reports.items():
    rows.append([labels[name],r['selected']['classes']['Malignant']['tp'],
        r['strict_matching']['0.75']['classes']['Malignant']['tp'],r['strict_matching']['0.8']['classes']['Malignant']['tp']])
table(rows,[196,105,105,105])
p('Raw scores and NMS',heading)
p('I repeated the previous geometry model on the 165 validation images and recorded both raw class scores. CPU and the original GPU cache gave the same TP, FP and FN counts at confidence 0.40. At that cutoff, 18 malignant images had a qualifying post-NMS box; the other 23 had no well-localized raw malignant score above 0.40. At score floor 0.001, 30 had a qualifying malignant prediction and 11 lost the class argmax to benign. No case had a qualifying malignant candidate lost solely between argmax and NMS in these checks.')
p('The GT boxes were used only to diagnose which candidates were well localized. These oracle checks cannot be used as an inference rule or as evidence that detection is complete. Lowering NMS alone is not supported by this diagnosis.')
p('The additional single-box rule selected the candidate with highest max(B,M), then used its score ratio to choose the class. It produced 17 malignant TP and 12 benign alerts, compared with 18 and 12 for the original rule. It was rejected, with no second-seed trial.')
p('Course IoU target',heading)
p('This round does not establish the course target of test IoU >= 0.8. These are validation results, the course denominator is still unclear, and no new independent test was used. Class-agnostic or TP-only means must not be substituted for all-GT correct-class results without saying so.')

story.append(PageBreak())
p('Confirmation, uncertainty and resources',title)
if d['confirmation_triggered']:
    cf=read(S/'confirmation_results.json');rows=[['Seed 3711 comparison','Cutoff','M TP / 41','B alerts / 124','M all-GT IoU']]
    for label,r in [('Previous n + geometry',cf['reference']),('Candidate',cf['candidate'])]:
        a=r['selected'];m=a['classes']['Malignant'];rows.append([label,a['threshold'],m['tp'],a['image_alerts']['benign_fp'],fmt(m['iou']['all_gt_class_correct']['mean'])])
    table(rows,[196,65,75,85,90])
    p('The confirmation '+('passed' if cf['passed'] else 'did not pass')+' the predeclared rule. This uses the same validation images with another training seed; it is a repeatability check, not independent evaluation.')
else:
    p('No initial training candidate met the replacement rule, so no additional confirmation training was triggered. The previous geometry model already has the earlier seed 3711 check. This does not remove uncertainty from repeated validation use.')
rows=[['Candidate minus previous geometry','M recall change (pp)','Conditional 95% interval (pp)']]
for name,u in unc.items():
    z=u['differences']['malignant_recall'];lo,hi=z['paired_percentile_95_interval']
    rows.append([labels[name],f"{100*z['candidate_minus_baseline_mean']:+.1f}",f'{100*lo:+.1f} to {100*hi:+.1f}'])
table(rows,[250,110,151])
p('Intervals use 5,000 paired bootstrap draws of recorded lesion/patient components. They condition on the fitted models and selected cutoffs. They do not correct for choosing models or cutoffs on validation, and they cannot account for missing patient links.',small)
rows=[['Model','10% budget: M TP / B alerts','20% budget: M TP / B alerts','30% budget: M TP / B alerts']]
for name,r in reports.items():
    points=[r['selected'],r['relaxed_budgets_diagnostic_only']['0.2'],r['relaxed_budgets_diagnostic_only']['0.3']]
    rows.append([labels[name]]+[f"{a['classes']['Malignant']['tp']} / {a['image_alerts']['benign_fp']}" for a in points])
table(rows,[181,110,110,110])
p('The 20% and 30% budgets were predeclared descriptive checks, not criteria for changing the main model. The maximum benign counts are 24 and 37 respectively. More alerts are not evidence of safer deployment.',small)
rows=[['Model','Training minutes','Peak allocated GPU GiB','Parameters','Selected epoch / 60']]
for name,r in reports.items():
    z=r['resources'];rows.append([labels[name],f"{z['elapsed_seconds']/60:.1f}",f"{z['peak_cuda_allocated_bytes']/2**30:.2f}",f"{z['parameters']:,}",z['checkpoint_selection']['best_epoch']])
table(rows,[166,76,94,86,89])
p('Training times exclude queueing, environment setup and input audits. All new training used the school A100. These measurements do not establish phone speed or an end-to-end clinical workflow.',small)
story.append(PageBreak())
p('Interpretation and reproducibility',title)
p('Fixed-threshold comparisons',heading)
rows=[['Model','0.05: M TP / B alerts','0.40: M TP / B alerts','0.40: M all-GT IoU']]
for name,r in reports.items():
    a=r['fixed_thresholds']['0.05'];b=r['fixed_thresholds']['0.4']
    rows.append([labels[name],f"{a['classes']['Malignant']['tp']} / {a['image_alerts']['benign_fp']}",
        f"{b['classes']['Malignant']['tp']} / {b['image_alerts']['benign_fp']}",fmt(b['classes']['Malignant']['iou']['all_gt_class_correct']['mean'])])
table(rows,[181,110,110,110])
p('A common threshold is useful for comparison, but differently trained models can have different score distributions. The main decision therefore uses the common false-alert budget. Neither comparison converts a confidence score into a probability of cancer.')
p('Limits and next technical option',heading)
p('The main remaining problem is class discrimination at an acceptable false-alert count. The known-ID subset and its fixed splits were preserved; full patient independence still cannot be proved where patient IDs are missing. Four additional comparisons were made in this round, including the rejected postprocessing extension, and all are reported.')
p('A more substantial future option is to keep YOLO localization and train a separate classifier using training-image lesion crops. An end-to-end validation comparison would have to classify crops from predicted boxes, not GT boxes. GT-crop validation could only diagnose the classifier in isolation. That pipeline has not been trained or validated here; its fit to the course component requirements should be made explicit.')
p('The implementation uses Ultralytics 8.3.221 detection loss and PyTorch BCEWithLogitsLoss positive weighting. It does not implement a new YOLO architecture. See the locked plans and runnable code under studies/known_lesion_followup_20261022.',small)
p('Implementation references: <link href="https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/utils/loss.py">Ultralytics v8.3.221 detection loss</link>; <link href="https://docs.pytorch.org/docs/2.8/generated/torch.nn.BCEWithLogitsLoss.html">PyTorch 2.8 BCEWithLogitsLoss</link>.',small)
output=R/'output/pdf/COMP3710_Validation_Followup_20261022.pdf'
SimpleDocTemplate(str(output),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=40,
    title='COMP3710 Project 3: Further Validation Experiments',author='COMP3710 Project 3').build(story)
print(output)
