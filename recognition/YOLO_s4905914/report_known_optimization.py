"""Append the known-lesion-only optimization evidence to the original report."""
import json

def add_known_optimization(code,add,page,table,pct):
    r=json.loads((code/'known_optimization_results_20261021.json').read_text())
    assert r['status']=='completed' and r['scope']=='validation_only' and not r['new_reassessment_inference']
    names={'baseline_n':'Original n','baseline_s':'Original s','n_cls1':'n, cls=1.0','n_geometry':'n, light geometry','n_long120':'n, 120 epochs'}
    order=list(names)
    def p(n):return r['reports'][n]['selected']
    def m(n):return p(n)['classes']['Malignant']
    def f(x):return 'N/A' if x is None else f'{x:.4f}'
    page()
    add('## Known-lesion-only optimization: 21 October')
    add('After the earlier experiments, I was advised to use the 1,098 images with known lesion IDs. This advice was reported in the project conversation; it does not confirm full patient independence or the other course requirements. I kept the original 768 training and 165 validation images. The 165 reassessment images were excluded from this optimization. No new reassessment inference was run and its results were not used to select these settings.')
    add('The original n model missed 28/41 malignant validation cases at confidence 0.50. Of these, 20 had a box with IoU >= 0.5 but the wrong class, two had a malignant box below IoU 0.5, and six had no prediction above the threshold. This points to classification as an important weakness, but does not establish its cause.')
    table(['Variant','Change from original n'],[
      ['n, cls=1.0','Classification loss coefficient 0.5 to 1.0; 60 epochs'],
      ['n, light geometry','Rotation +/-15 degrees, translation 0.05, scale 0.15; 60 epochs'],
      ['n, 120 epochs','120 epochs instead of 60, with the same linear LR rule']])
    add('All variants started from original COCO n weights with seed 3710, FP32, 640-pixel input, batch 16 and AdamW. There was no oversampling, Mosaic, mixed-image or colour augmentation. The cls change affects both classes. The geometry experiment tests a combined preset. The longer run also changes LR decay and offers more chances to select a checkpoint, so it is not a pure test of extra updates at an unchanged schedule.')
    add('The original 14-threshold grid, class-aware single-label NMS at 0.7, matching IoU >= 0.5 and 10% benign-alert limit were retained. Each model selected its threshold on validation. Candidate ranking used malignant recall, then malignant all-GT correct-class IoU, malignant AP50:95 and fewer benign alerts. A gain needed at least three additional malignant TP and higher all-GT IoU to trigger a second-seed comparison.')
    table(['Model','Conf.','M TP / 41','B alerts / 124','M all-GT IoU'],[
      [names[n],f"{p(n)['threshold']:.2f}",m(n)['tp'],p(n)['image_alerts']['benign_fp'],f(m(n)['iou']['all_gt_class_correct']['mean'])] for n in order])
    add('M denotes malignant and B benign. All selected points meet the validation alert budget. The IoU column covers all 41 malignant GTs; missing correct-class predictions contribute zero. Counts refer to image annotations, not unique patients. These are development results on a repeatedly used validation set, not fresh test results.')
    page()
    add('## Optimization: AP and threshold trade-offs')
    table(['Model','M AP50','M AP50:95','B AP50:95','Macro mAP50:95'],[
        [names[n],f(r['reports'][n]['ap']['1']['AP50']),f(r['reports'][n]['ap']['1']['AP50_95']),f(r['reports'][n]['ap']['0']['AP50_95']),f(r['reports'][n]['mAP50_95'])] for n in order])
    table(['Model','0.05: M TP / B alerts','0.50: M TP / B alerts'],[
        [names[n],*[f"{r['reports'][n]['fixed_thresholds'][t]['classes']['Malignant']['tp']} / {r['reports'][n]['fixed_thresholds'][t]['image_alerts']['benign_fp']}" for t in ('0.05','0.5')]] for n in order])
    add('The fixed-threshold table uses the same score cutoff for every model. The preceding table compares separately selected cutoffs under a common alert budget. Both are useful: a different score scale can change results at 0.50 without improving ranking across thresholds.')
    add('The 120-epoch run has higher malignant AP50 than original n, but at confidence 0.80 it produces 13 malignant TP and 13 benign alerts, just outside the budget. The next tested cutoff, 0.90, retains nine TP and one benign alert. This result depends on the operating region and the fixed grid; it does not establish that extra training is always harmful.')
    add('![Known-lesion validation threshold trade-off](assets/known_optimization_threshold_tradeoff.png)')
    add('Each marker is a tested threshold, and the larger outlined marker is its selected point. The vertical line marks 10% of 124 benign images; at most 12 alerts qualify. Connecting lines are visual guides, not additional tested thresholds.')
    page()
    add('## Optimization: localization, resources and decision')
    table(['Model','B all-GT IoU','M TP-only IoU (N)','M TP at IoU >= 0.75'],[
        [names[n],f(p(n)['classes']['Benign']['iou']['all_gt_class_correct']['mean']),f"{f(m(n)['iou']['tp_only']['mean'])} ({m(n)['tp']})",f"{r['reports'][n]['strict_IoU_075']['classes']['Malignant']['tp']}/41"] for n in order])
    add('The stricter IoU comparison keeps each model\'s already-selected confidence. TP-only means exclude misses and use different denominators, so a high value cannot demonstrate good detection across all malignant cases.')
    table(['New n run','Best epoch / budget','Training-call time','Peak allocated GPU memory'],[
        [names[n],f"{r['reports'][n]['resources']['checkpoint_selection']['best_epoch']} / {r['reports'][n]['resources']['checkpoint_selection']['completed_epochs']}",f"{r['reports'][n]['resources']['elapsed_seconds']/60:.2f} min",f"{r['reports'][n]['resources']['peak_cuda_allocated_bytes']/1024**3:.2f} GiB"] for n in order[2:]])
    add('Times include the model training call, including its validation and finalization, but exclude environment installation, input auditing and the separate final validation command. GPU memory is peak PyTorch allocated memory, not total device use. All new runs used the same pinned Linux packages and an A100; the recorded optimizer calls matched their planned budgets. Nodes differed, so timings are not a controlled comparison of training settings.')
    decision=r['final_decision']
    add(decision['report_summary'])
    if decision['status']=='completed':
        add('The earlier prediction example reproduces the original main study. For the current choice, `known_optimization_decision_20261021.json` records the exact config, checkpoint and validation-lock paths to pass to `predict.py`. These paths and their hashes were checked together; the selected confidence is read from the lock.')
    best=r['decision']['highest_ranked']
    if best!='baseline_n':
        ci=r['conditional_bootstrap'][best]
        delta=ci['differences']['malignant_recall'];lo,hi=delta['paired_percentile_95_interval']
        add(f"For the highest-ranked initial candidate, the malignant-recall difference from original n was {100*delta['candidate_minus_baseline_mean']:.1f} percentage points. A paired bootstrap over {ci['known_identity_components']} known-identity groups gave a conditional 95% interval of {100*lo:.1f} to {100*hi:.1f} points (5,000 draws). Models and selected thresholds were held fixed. This interval does not correct for model or threshold selection, and missing patient links may leave additional dependence.")
    add('Only train/val manifests were available to the isolated study loader. It rejected reassessment evaluation before opening inputs. Each input audit decoded 933 images, checked dimensions against the original box source and found no cross-split exact duplicate. The original training configs, results and selection locks remain unchanged. The reproducible study, plan, configs, run map and comparison script are in `studies/known_lesion_optimization_20261021/`; the report contains copied summary JSON and figures. No further search is planned within this small validation set.')
    if 'confirmation' in r:
        page();add('## Optimization: second-seed confirmation')
        c=r['confirmation']
        add('The initial candidate passed the prespecified improvement rule, so original n and that candidate were each retrained from COCO at seed 3711. Each selected a threshold using the same validation grid and alert budget. The second seed was a consistency check on the same images, not independent validation. Neither run was chosen simply for having the better seed.')
        table(['Run / seed','Conf.','M TP / 41','B alerts / 124','M all-GT IoU'],c['report_rows'])
        add(c['report_interpretation'])
        second_seed=[('Original n / 3711','n_baseline_seed3711'),
                     ('Candidate / 3711',c['candidate']+'_seed3711')]
        table(['Run / seed','M AP50','M AP50:95','M TP-only IoU (N)'],[
            [label,f(c['reports'][key]['ap']['1']['AP50']),
             f(c['reports'][key]['ap']['1']['AP50_95']),
             f"{f(c['reports'][key]['selected']['classes']['Malignant']['iou']['tp_only']['mean'])} ({c['reports'][key]['selected']['classes']['Malignant']['tp']})"]
            for label,key in second_seed])
        add('These secondary measures do not change the prespecified decision rule. TP-only IoU still excludes missed lesions, while AP uses the low-score sweep rather than just the selected operating point.')
        table(['Run / seed','Best epoch / budget','Training-call time','Peak allocated GPU memory'],[
            [label,f"{c['reports'][key]['resources']['checkpoint_selection']['best_epoch']} / {c['reports'][key]['resources']['checkpoint_selection']['completed_epochs']}",
             f"{c['reports'][key]['resources']['elapsed_seconds']/60:.2f} min",
             f"{c['reports'][key]['resources']['peak_cuda_allocated_bytes']/1024**3:.2f} GiB"]
            for label,key in second_seed])
        add('Both confirmation runs used the same 60-epoch budget and 2,880 optimizer calls. Timing and memory use the same definitions as the preceding resource table; queueing and environment installation are excluded.')
