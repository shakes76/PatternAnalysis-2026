"""Separate Hard-only follow-up after the original paired-v2 TRAIN QA stopped.

This is an explicitly new, adaptive scope decision, not a successful continuation
of the failed all-or-none pair and not a claim of a reproduced software bug.
The only new candidate is the Hard3D model that passed the unchanged TRAIN gates.
The failed new2D run remains failed; original frozen v1 2D/3D are references.

train_hard_body below is a static, reviewable copy of frozen v2.train. Its only
body change is the named entry QA validator. All optimisation/sampling/model/
loss/checkpoint-selection statements are AST-identical to the frozen source.
No module globals are monkeypatched and no QA passing record is manufactured.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import os
import platform
import re
import time
import numpy as np
import v2_pipeline as v2
from v2_pipeline import (
    torch, DataLoader, CLASS_NAMES, load_new_manifest, _identity, seed_everything,
    SampledTrainingDataset, create_model, training_class_weights, _restore_rng,
    _fresh_directory, utc_now, sha256, write_json, _sync, _optimizer_step,
    evaluate, _rng_state, _save_checkpoint, _plot_history, read_json,
)
from evaluate_v2_deterministic import collect_runtime, require_matching_flags

HERE=Path(__file__).resolve().parent
HARD='improved3d'
SCOPE_ID='hard_only_after_stopped_v2_pair'


def require(condition,message):
    if not condition: raise ValueError(message)


def load_scope(path, base_path, manifest_path):
    scope=read_json(path); base=v2.load_protocol(base_path)
    require(scope.get('experiment_id')==SCOPE_ID,'Unexpected Hard-only scope identity')
    require(str(scope.get('status','')).startswith('frozen'),'Hard-only scope must be frozen before execution')
    require(scope.get('base_protocol_sha256')==base['_sha256'],'Base scientific protocol changed')
    require(scope.get('manifest_sha256')==sha256(manifest_path),'Hard-only cache identity changed')
    require(scope.get('stopped_pair_status')=='stopped_after_QA_failure','Original pair must remain stopped')
    require(scope.get('configuration_exactly_inherited') is True and scope.get('configuration')==base['configuration'],'Hard configuration may not change')
    require(scope.get('formal_models')==[HARD] and scope.get('seed')==3711 and scope.get('epochs')==30,'Exactly one unchanged Hard candidate is allowed')
    require(scope.get('no_new_parameters_or_candidates') is True,'Scope must prohibit additional scientific variants')
    require(scope.get('max_gpu_walltime_minutes')==60,'Hard-only GPU budget is fixed at60minutes')
    require(scope.get('reused_reference_models')==['v1_unet2d','v1_improved3d'],'Only original frozen references are reused')
    v2.verify_review_sources(base,Path(base_path).resolve().parent)
    require({'train_hard_followup.py','hard_followup_gpu.slurm','evaluate_v2_deterministic.py'}.issubset(scope.get('required_scope_sha256',{})),'Bind the actual executor, driver and runtime helper')
    for filename,wanted in scope.get('required_scope_sha256',{}).items():
        require(sha256(v2._safe_relative(Path(path).resolve().parent,filename))==wanted,'Hard-only execution source changed: '+filename)
    require(scope.get('required_scope_sha256',{}).get('train_hard_followup.py')==sha256(__file__),'Scope must bind this actual executor')
    scope['_sha256']=sha256(path)
    return scope,base


def validate_hard_qa_receipt(qa_path,manifest_path,protocol):
    """Accept only the unchanged, passed Hard part of the genuinely failed pair.

    Does not change the failed combined status or inspect VAL/TEST performance.
    """
    qa_path=Path(qa_path).resolve(strict=True); qa=read_json(qa_path)
    identity=dict(manifest_sha256=sha256(manifest_path),protocol_sha256=protocol['_sha256'])
    require(qa.get('status')=='failed_fit_gate' and qa.get('generalisation_evidence') is False,'Original paired QA failure must be preserved')
    require(all(qa.get(k)==v for k,v in identity.items()),'QA identity changed')
    require(qa.get('models',{}).get('unet2d',{}).get('passed') is False,'Do not relabel failed new2D as passed')
    hard=qa.get('models',{}).get(HARD,{})
    flags=['passed','full_image_only_inference_passed','native_geometry_serialisation_passed','independent_metric_check_passed','sampler_batch_backward_passed']
    require(all(hard.get(k) is True for k in flags),'Hard requires every original real-data QA gate')
    require(hard.get('real_sampler_batch',{}).get('test_or_validation_pixels_read')==0,'Sampler QA must be TRAIN-only')
    batch=hard.get('real_sampler_batch',{})
    require(batch.get('passed') is True and batch.get('all_parameter_gradients_present_and_finite') is True and batch.get('unscaled_backward') is True,'Actual Hard sampler/backward gate incomplete')
    auxiliary=batch.get('auxiliary_weight_gradient_absolute_sums',[])
    require(len(auxiliary)==2 and all(type(x) in [int,float] and np.isfinite(x) and x>0 for x in auxiliary),'Both actual deep-supervision gradients must be nonzero')
    fit_path=qa_path.parent/'fit/fit_receipt.json';fit=read_json(fit_path)
    require(sha256(fit_path)==qa.get('fit_receipt_sha256'),'Fit receipt bytes changed')
    gates=protocol['qa']
    require(fit.get('status')=='failed_fit_gate' and fit.get('amp')==gates['fit_amp'] and fit.get('seed')==gates['fit_seed'] and fit.get('steps')==gates['fit_steps'] and fit.get('case')==gates['fit_case'],'Original fixed400-step fit evidence changed')
    require(all(fit.get(k)==v for k,v in identity.items()),'Fit identity changed')
    require(fit.get('models',{}).get('unet2d',{}).get('passed') is False,'Failed new2D fit must stay failed')
    result=fit.get('models',{}).get(HARD,{});history=result.get('history',[])
    require(result.get('passed') is True and history and history[0].get('step')==0 and history[-1].get('step')==400,'Hard did not pass at the original final fixed step')
    first,last=history[0],history[-1]
    for value in [first.get('loss'),last.get('loss'),last.get('prostate_dice'),last.get('foreground_dice')]:
        require(type(value) in [int,float] and np.isfinite(value),'Invalid Hard final metrics')
    require(first['loss']>0 and last['loss']/first['loss']<=gates['maximum_final_initial_loss_ratio'] and last['prostate_dice']>=gates['minimum_fit_prostate_dice'] and last['foreground_dice']>=gates['minimum_fit_foreground_dice'],'Hard failed an unchanged numeric gate')
    require(hard.get('final_fit_prostate_dice')==last['prostate_dice'] and hard.get('final_fit_foreground_dice')==last['foreground_dice'],'Hard combined receipt differs from actual fit')
    for row in history:
        values=row.get('per_class',[])
        require([x.get('class_index') for x in values]==list(range(6)),'Hard fit must retain every class')
        for item in values:
            tp,fp,fn=item['tp'],item['fp'],item['fn'];den=2*tp+fp+fn
            expected=2*tp/den if den else None
            require(item['dice']==expected,'Hard fit confusion arithmetic differs')
    through_path=qa_path.parent.parent/'through_704494/pipeline_receipt.json';old=read_json(through_path)
    require(sha256(through_path)==qa.get('original_through_receipt_sha256'),'Original image-only through receipt changed')
    require(old.get('status')=='passed' and old.get('selected_case_keys')==gates['through_cases'],'Original TRAIN image-only gate failed')
    require(old.get('manifest_sha256')==protocol['required_original_sha256']['artifacts/cache64/manifest.json'] and old.get('scope',{}).get('test_arrays_read')==0,'Original image-only scope/identity changed')
    require(old.get('original_modules_sha256')==protocol['required_original_sha256']['modules.py'] and old.get('original_utils_sha256')==protocol['required_original_sha256']['utils.py'] and old.get('adapter_sha256')==protocol['required_review_sha256']['infer_native.py'] and old.get('verifier_sha256')==protocol['required_review_sha256']['real_pipeline_check.py'],'Original through-source identity changed')
    require(len(old.get('cases',[]))==2,'Original through cases incomplete')
    for case in old['cases']:
        require(case.get('preprocessing_elementwise_equal') is True and case.get('split')=='train','Original through preprocessing gate missing')
        for name in ['unet2d',HARD]:
            item=case.get('models',{}).get(name,{})
            require(item.get('checkpoint_sha256')==protocol['required_original_sha256'][f'artifacts/full_seed42/{name}/best.pt'] and item.get('cache_prediction_elementwise_equal') is True and item.get('native_affine_matches_image') is True,'Original reference-model forward/geometry gate changed')
    require(len(qa.get('cases',[]))==2 and [[r['patient'],r['visit']] for r in qa['cases']]==gates['through_cases'] and all(r.get('image_elementwise_equal') is True for r in qa['cases']),'New image-only preprocessing check incomplete')
    return qa


def train_hard_body(manifest_path, model_name, output_dir, protocol, dependencies, qa_receipt,
          device, resume=None):
    manifest = load_new_manifest(manifest_path, protocol, dependencies)
    identity = _identity(manifest_path, protocol, model_name, dependencies)
    validate_hard_qa_receipt(qa_receipt, manifest_path, protocol)
    config = protocol["configuration"]
    if config["amp"] and device.type != "cuda":
        raise ValueError("Frozen formal training requires CUDA AMP. Use the Python API for CPU fixtures/QA.")
    seed_everything(config["seed"])
    dataset = SampledTrainingDataset(manifest_path, model_name, config, dependencies, manifest)
    batch_size = config["train_batch_size_2d"] if model_name == "unet2d" else config["train_batch_size_3d"]
    expected_steps = (len(dataset) + batch_size - 1) // batch_size
    model = create_model(model_name, config, dependencies).to(device)
    weights, counts = training_class_weights(manifest_path, manifest)
    weights = weights.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    history, best_score, best_epoch, start_epoch = [], None, None, 1
    if resume:
        output = Path(output_dir).resolve(strict=True)
        if Path(resume).resolve().parent != output:
            raise ValueError("Resume only in the original checkpoint directory.")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["identity"] != identity:
            raise ValueError("Resume identity differs: model/config/data/protocol/source.")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        history, best_score, best_epoch = checkpoint["history"], checkpoint["best_score"], checkpoint["best_epoch"]
        start_epoch = checkpoint["epoch"] + 1
        if start_epoch > config["epochs"]:
            raise ValueError("The fixed 30-epoch budget is already exhausted.")
        _restore_rng(checkpoint["rng_state"])
    else:
        output = _fresh_directory(output_dir)
    run = {"status": "running", "started_at": utc_now(), "identity": identity,
           "qa_receipt_sha256": sha256(qa_receipt), "class_names": CLASS_NAMES,
           "class_weights": weights.detach().cpu().tolist(), "training_voxel_counts": counts,
           "class_weight_formula": "TRAIN full-new-grid counts only; inverse sqrt frequency; normalize class mean; clamp [.05,10]; normalize class mean again; CE only",
           "requested_epochs": config["epochs"], "expected_samples_per_epoch": len(dataset),
           "expected_batches_per_epoch": expected_steps, "validation_epochs": config["validation_epochs"],
           "selection": config["selection"], "parameters": sum(p.numel() for p in model.parameters()),
           "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
           "python_version": platform.python_version(), "torch_version": str(torch.__version__),
           "numpy_version": np.__version__, "cuda_version": torch.version.cuda,
           "metrics_grid": "v2_downsampled_resampled_grid_96x192x192", "native_resolution_accuracy_claimed": False,
           "test_pixels_loaded_during_training": False,
           "completed_epochs": start_epoch - 1}
    write_json(output / "run.json", run)
    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            if sha256(manifest_path) != identity["manifest_sha256"]:
                raise RuntimeError("Immutable manifest changed during training.")
            dataset.set_epoch(epoch)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                num_workers=config["workers"], pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(config["seed"] + epoch), drop_last=False)
            model.train()
            _sync(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            examples, batches, skipped, loss_sum = 0, 0, 0, 0.0
            sampling = {"foreground_requested": 0, "foreground_missing_fallback": 0,
                        "selected_class_counts": [0] * 6, "hw_flip_count": 0}
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["mask"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=config["amp"]):
                    outputs = model(images)
                    loss = dependencies.modules.segmentation_loss(outputs, targets, class_weights=weights)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite training loss; checkpoint not advanced.")
                scaler.scale(loss).backward()
                gradient_norm, overflow = _optimizer_step(model, optimizer, scaler, config["clip_grad"], config["amp"])
                skipped += int(overflow)
                count = int(images.shape[0])
                examples += count
                batches += 1
                loss_sum += float(loss.detach()) * count
                sampling["foreground_requested"] += int(batch["foreground_requested"].sum())
                sampling["foreground_missing_fallback"] += int(batch["foreground_missing_fallback"].sum())
                sampling["hw_flip_count"] += int(batch["hw_flip_count"].sum())
                for chosen in batch["selected_class"].tolist():
                    if chosen > 0:
                        sampling["selected_class_counts"][chosen] += 1
            _sync(device)
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device)
            if batches != expected_steps or examples != len(dataset):
                raise RuntimeError("Actual sampling schedule did not complete its fixed per-epoch budget.")
            optimizer.zero_grad(set_to_none=True)
            del outputs, loss, images, targets
            evaluation = evaluate(model, manifest_path, manifest, "validation", model_name, device,
                config, dependencies) if epoch in config["validation_epochs"] else None
            score = evaluation["patient_balanced_prostate_dice"] if evaluation else None
            if evaluation and score is None:
                raise RuntimeError("Validation prostate score is undefined; no model selection is possible.")
            improved = score is not None and (best_score is None or score > best_score)
            if improved:
                best_score, best_epoch = score, epoch
            history.append({"epoch": epoch, "train_loss": loss_sum / examples,
                "training_examples": examples, "training_batches": batches,
                "optimizer_steps": batches - skipped, "amp_overflow_skipped_optimizer_steps": skipped,
                "last_gradient_norm_before_clipping": gradient_norm,
                "train_seconds": seconds, "peak_allocated_cuda_bytes_training": peak,
                "sampling": sampling, "validation_prostate_dice": score, "validation": evaluation})
            payload = {"schema_version": 2, "identity": identity, "config": {**config, "model": model_name,
                "label_ids": manifest["label_ids"]}, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                "best_score": best_score, "best_epoch": best_epoch, "history": history, "rng_state": _rng_state()}
            _save_checkpoint(output / "latest.pt", payload)
            if improved:
                _save_checkpoint(output / "best.pt", payload)
            if epoch == config["epochs"]:
                _save_checkpoint(output / "final.pt", payload)
            write_json(output / "history.json", history)
            _plot_history(history, output / "curves.png")
            _restore_rng(payload["rng_state"])
            run.update(completed_epochs=epoch, best_epoch=best_epoch, best_validation_prostate_dice=best_score,
                       actual_optimizer_steps=sum(r["optimizer_steps"] for r in history),
                       actual_training_examples=sum(r["training_examples"] for r in history))
            write_json(output / "run.json", run)
            print(f"{model_name} epoch {epoch}/30 loss={history[-1]['train_loss']:.5f} "
                  f"VAL prostate={score if score is not None else 'not scheduled'} steps={batches-skipped}/{batches}", flush=True)
        run.update(status="completed", completed_at=utc_now(),
                   best_checkpoint_sha256=sha256(output / "best.pt"),
                   final_checkpoint_sha256=sha256(output / "final.pt"))
        write_json(output / "run.json", run)
    except BaseException as error:
        run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", run)
        raise
    return output


def run_training(manifest_path,output_dir,protocol,dependencies,qa_path,scope,scope_path,device):
    require(device.type=='cuda','The one formal Hard run requires CUDA AMP')
    qa=validate_hard_qa_receipt(qa_path,manifest_path,protocol)
    require(sha256(qa_path)==scope.get('qa_receipt_sha256'),'Scope QA digest differs')
    require(sha256(Path(qa_path).parent/'fit/fit_receipt.json')==scope.get('fit_receipt_sha256'),'Scope fit digest differs')
    output=Path(output_dir).absolute()
    require(not output.exists() and not output.is_symlink(),'Use a fresh Hard-only run directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    receipt_path=output.parent/'scope_execution.json'
    require(not receipt_path.exists(),'This Hard-only scope was already attempted; inspect and preserve it')
    receipt=dict(status='running',started_at=utc_now(),experiment_id=scope['experiment_id'],
        scope_plan_sha256=scope['_sha256'],base_protocol_sha256=protocol['_sha256'],
        manifest_sha256=sha256(manifest_path),training_executor_sha256=sha256(__file__),
        scope_driver_sha256=scope['required_scope_sha256']['hard_followup_gpu.slurm'],
        original_pair_status='stopped_after_QA_failure',original_combined_qa_status=qa['status'],
        failed_new2d_preserved=True,qa_receipt_sha256=sha256(qa_path),
        fit_receipt_sha256=scope['fit_receipt_sha256'],formal_models=[HARD],
        optimisation_body='Static frozen-v2.train body; only the entry QA validator is named differently',
        warmstart=False,resume=False,new_validation_or_test_performance_used_for_scope_choice=False)
    write_json(receipt_path,receipt)
    try:
        # No QA weights are saved or loaded. The copied body constructs a fresh
        # model and optimizer, with the unchanged base seed and no resume.
        result=train_hard_body(manifest_path,HARD,output,protocol,dependencies,qa_path,device,resume=None)
        run=read_json(output/'run.json')
        require(run.get('status')=='completed' and run.get('completed_epochs')==30,'Hard run did not finish its fixed budget')
        receipt.update(status='completed',completed_at=utc_now(),
            completed_epochs=30,best_checkpoint_sha256=sha256(output/'best.pt'),
            final_checkpoint_sha256=sha256(output/'final.pt'),
            run_sha256=sha256(output/'run.json'),history_sha256=sha256(output/'history.json'))
        write_json(receipt_path,receipt)
        return result
    except BaseException as error:
        receipt.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=f'{type(error).__name__}: {error}')
        write_json(receipt_path,receipt)
        raise


def freeze_selection(run_dir,manifest_path,protocol,scope,output_path):
    require(not Path(output_path).exists(),'Hard selection was already frozen')
    run_dir=Path(run_dir);run=read_json(run_dir/'run.json');history=read_json(run_dir/'history.json')
    execution=read_json(run_dir.parent/'scope_execution.json')
    require(execution.get('status')=='completed' and execution.get('scope_plan_sha256')==scope['_sha256'] and execution.get('training_executor_sha256')==sha256(__file__),'Completed Hard execution provenance differs')
    require(run.get('status')=='completed' and run.get('completed_epochs')==30 and [r['epoch'] for r in history]==list(range(1,31)),'Exactly30 Hard epochs must finish before TEST')
    scheduled=[r for r in history if r['validation_prostate_dice'] is not None]
    require([r['epoch'] for r in scheduled]==protocol['configuration']['validation_epochs'],'Validation schedule changed')
    best=max(scheduled,key=lambda r:(r['validation_prostate_dice'],-r['epoch']))
    require(run['best_epoch']==best['epoch'],'Run selection was not earliest strict maximum')
    digest=sha256(run_dir/'best.pt')
    require(run['best_checkpoint_sha256']==digest==execution['best_checkpoint_sha256'],'Selected Hard weight bytes changed')
    checkpoint=torch.load(run_dir/'best.pt',map_location='cpu',weights_only=True)
    require(checkpoint['epoch']==best['epoch'] and checkpoint['best_epoch']==best['epoch'],'Actual checkpoint epoch differs')
    require(checkpoint['identity']==run['identity'] and checkpoint['identity']['model']==HARD and checkpoint['identity']['protocol_sha256']==protocol['_sha256'] and checkpoint['identity']['manifest_sha256']==sha256(manifest_path) and checkpoint['identity']['configuration']==protocol['configuration'],'Selected checkpoint scientific identity changed')
    result=dict(status='frozen',created_at_utc=utc_now(),experiment_id=scope['experiment_id'],
        original_pair_status='stopped_after_QA_failure',scope_plan_sha256=scope['_sha256'],
        base_protocol_sha256=protocol['_sha256'],protocol_sha256=protocol['_sha256'],
        manifest_sha256=sha256(manifest_path),training_executor_sha256=sha256(__file__),
        scope_driver_sha256=scope['required_scope_sha256']['hard_followup_gpu.slurm'],
        scope_execution_sha256=sha256(run_dir.parent/'scope_execution.json'),
        models={HARD:dict(completed_epochs=30,checkpoint_sha256=digest,checkpoint_epoch=best['epoch'],
            validation_prostate_dice=best['validation_prostate_dice'],run_sha256=sha256(run_dir/'run.json'),history_sha256=sha256(run_dir/'history.json'))},
        test_status='Previously inspected cohort; next evaluation is retrospective, not a new blind test')
    write_json(output_path,result)
    return output_path


def require_hard_selection_freeze(path,manifest_path,protocol,scope,checkpoint_digest=None):
    require(path is not None,'Hard TEST requires the actual selected-checkpoint freeze')
    selected=read_json(path)
    require(selected.get('status')=='frozen' and selected.get('experiment_id')==scope['experiment_id'] and selected.get('scope_plan_sha256')==scope['_sha256'],'Hard selection scope changed')
    require(selected.get('manifest_sha256')==sha256(manifest_path) and selected.get('base_protocol_sha256')==protocol['_sha256'] and selected.get('protocol_sha256')==protocol['_sha256'],'Hard selection data/base-protocol changed')
    require(set(selected.get('models',{}))=={HARD},'Never fabricate a missing new2D selection')
    hard=selected['models'][HARD]
    require(hard.get('completed_epochs')==30 and re.fullmatch('[a-f0-9]{64}',hard.get('checkpoint_sha256','')) is not None,'Hard must complete30epochs and freeze its selected checkpoint')
    require(type(hard.get('checkpoint_epoch')) is int and 1<=hard['checkpoint_epoch']<=30,'Hard selected epoch must be within its fixed budget')
    if checkpoint_digest is not None:require(hard['checkpoint_sha256']==checkpoint_digest,'TEST checkpoint differs from frozen Hard selection')
    return sha256(path)


def prepare_test_hard(manifest_path,raw_root,protocol,dependencies,scope,selection_freeze):
    """Same frozen preprocessor and immutable TEST paths; one genuine Hard freeze."""
    path=Path(manifest_path).resolve(strict=True)
    freeze_digest=require_hard_selection_freeze(selection_freeze,path,protocol,scope)
    digest=sha256(path);manifest=load_new_manifest(path,protocol,dependencies)
    destination=path.parent/'test_completion_receipt.json'
    require(not destination.exists(),'TEST cache completion was already attempted; preserve its receipt')
    receipt=dict(status='preparing_test',manifest_sha256=digest,protocol_sha256=protocol['_sha256'],
        scope_plan_sha256=scope['_sha256'],selection_freeze_sha256=freeze_digest,
        actual_scope='oneHard_after_stopped_pair',started_at=utc_now(),records=[])
    write_json(destination,receipt)
    try:
        for record in manifest['records']:
            if record['split']!='test':continue
            image,mask=v2._source_paths(raw_root,record)
            arrays,metadata=dependencies.prepare.preprocess_pair(image,mask,manifest['target_shape_dhw'],manifest['label_ids'])
            v2._check_source_geometry(record,metadata)
            cache_digest=v2._write_cache(v2._safe_relative(path.parent,record['cache_path']),arrays)
            receipt['records'].append(dict(patient=record['patient'],visit=record['visit'],cache_path=record['cache_path'],
                cache_sha256=cache_digest,source_image_sha256=sha256(image),source_mask_sha256=sha256(mask),class_voxel_counts=metadata['class_voxel_counts']))
            write_json(destination,receipt)
            print(f"Hard cache stage2: {len(receipt['records'])}/30",flush=True)
        require(len(receipt['records'])==30 and sha256(path)==digest,'Hard TEST scope or immutable manifest changed')
        receipt.update(status='completed',completed_at=utc_now(),manifest_bytes_unchanged=True)
        write_json(destination,receipt)
        return destination
    except BaseException as error:
        receipt.update(status='failed',error=f'{type(error).__name__}: {error}');write_json(destination,receipt);raise


def evaluate_hard(manifest_path,checkpoint_path,split,output_dir,protocol,dependencies,scope,scope_path,device,selection_freeze=None):
    manifest=load_new_manifest(manifest_path,protocol,dependencies)
    identity=_identity(manifest_path,protocol,HARD,dependencies)
    checkpoint_digest=sha256(checkpoint_path)
    freeze_digest=require_hard_selection_freeze(selection_freeze,manifest_path,protocol,scope,checkpoint_digest) if split=='test' else None
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    require(checkpoint.get('identity')==identity,'Hard checkpoint data/model/scientific identity changed')
    require(not Path(output_dir).exists(),'Use a fresh Hard evaluation directory')
    v2.seed_everything(protocol['configuration']['seed'])
    before=collect_runtime(torch,device,protocol['configuration']);require_matching_flags(before)
    model=create_model(HARD,protocol['configuration'],dependencies).to(device)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    output=_fresh_directory(output_dir)
    result=evaluate(model,manifest_path,manifest,split,HARD,device,protocol['configuration'],dependencies,output)
    after=collect_runtime(torch,device,protocol['configuration']);require_matching_flags(after)
    require(before==after,'Hard evaluation execution flags/environment changed')
    require(sha256(checkpoint_path)==checkpoint_digest and sha256(manifest_path)==identity['manifest_sha256'] and sha256(scope_path)==scope['_sha256'],'Hard evaluation inputs changed')
    result.update(identity=identity,checkpoint_sha256=checkpoint_digest,checkpoint_epoch=checkpoint['epoch'],split=split,
        chosen_by='Unchanged predeclared validation class5 strict maximum; never TEST',class_names=CLASS_NAMES,
        scope_plan_sha256=scope['_sha256'],actual_scope_id=scope['experiment_id'],completed_at=utc_now())
    if freeze_digest:result['selection_freeze_sha256']=freeze_digest
    write_json(output/'metrics.json',result);write_json(output/'evaluation.json',result)
    runtime=dict(status='completed',created_at_utc=utc_now(),evaluation_sha256=sha256(output/'evaluation.json'),metrics_sha256=sha256(output/'metrics.json'),
        checkpoint_sha256=checkpoint_digest,manifest_sha256=identity['manifest_sha256'],protocol_sha256=protocol['_sha256'],base_protocol_sha256=protocol['_sha256'],
        scope_plan_sha256=scope['_sha256'],adapter_file='train_hard_followup.py',adapter_sha256=sha256(__file__),
        scope_driver_sha256=scope['required_scope_sha256']['hard_followup_gpu.slurm'],**after)
    write_json(output/'runtime_receipt.json',runtime)
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['train','freeze','prepare-test','evaluate'],required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--scope-plan',type=Path,required=True)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--qa-receipt',type=Path)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--selection-freeze',type=Path)
    p.add_argument('--raw-root',type=Path)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args(argv)
    scope,base=load_scope(a.scope_plan,a.protocol,a.manifest)
    deps=v2.load_dependencies(a.project_root,base)
    if a.mode=='prepare-test':
        if a.raw_root is None:p.error('prepare-test requires --raw-root')
        out=prepare_test_hard(a.manifest,a.raw_root,base,deps,scope,a.selection_freeze)
    elif a.mode=='freeze':
        if a.run_dir is None or a.selection_freeze is None:p.error('freeze requires --run-dir and --selection-freeze')
        out=freeze_selection(a.run_dir,a.manifest,base,scope,a.selection_freeze)
    else:
        if torch is None:p.error('Measured Torch environment is required')
        device=torch.device(a.device)
        if device.type=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
        if a.output_dir is None:p.error('train/evaluate requires fresh --output-dir')
        if a.mode=='train':
            if a.qa_receipt is None:p.error('train requires original failed-pair --qa-receipt')
            out=run_training(a.manifest,a.output_dir,base,deps,a.qa_receipt,scope,a.scope_plan,device)
        else:
            if a.checkpoint is None:p.error('evaluate requires --checkpoint')
            out=evaluate_hard(a.manifest,a.checkpoint,a.split,a.output_dir,base,deps,scope,a.scope_plan,device,a.selection_freeze)
    print(f'Hard-only {a.mode} completed: {out}',flush=True)
    return 0


if __name__=='__main__':main()
