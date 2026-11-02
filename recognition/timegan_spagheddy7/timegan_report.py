"""Reconstruction diagnostics and reproducible conditioning reports; no model training."""

import json
from pathlib import Path

import numpy as np
import torch

from dataset import PRICE_COLUMNS, VOLUME_COLUMNS, PROJECT_DIR, prepare_datasets, sha256_file
from evaluation import decode_batch, validation_windows, validity_rates, ValidationEvaluator, depth_images
from predict import load_timegan
from protocol import anchor_context, fit_anchor_context
from train import timegan_code_hashes, write_json


def shifted_anchors(anchors, seed):
    """Move every window index by a seeded nonzero shift; prices can still repeat."""
    if len(anchors) < 2:
        raise ValueError("Anchor diagnostic requires at least two windows.")
    shift = int(np.random.default_rng(seed).integers(1, len(anchors)))
    return np.roll(anchors, shift, axis=0), shift


@torch.inference_mode()
def diagnose(data_dir, checkpoint_path, destination, train_count=512, prepared=None):
    torch.set_num_threads(1)
    model, scaler, checkpoint = load_timegan(checkpoint_path)
    datasets, fitted_scaler, audit = prepare_datasets(data_dir, 32, 32) if prepared is None else prepared
    np.testing.assert_array_equal(scaler.minimum, fitted_scaler.minimum)
    np.testing.assert_array_equal(scaler.maximum, fitted_scaler.maximum)
    result = dict(scope="Observed-input reconstruction and anchor controls, not noise-only generation scores.",
                  checkpoint_sha256=sha256_file(Path(checkpoint_path)), architecture=checkpoint["architecture"],
                  test_evaluated=False, source_files=timegan_code_hashes(), splits={})
    for split in ("train", "validation"):
        dataset = datasets[split]
        starts, anchors, real = validation_windows(dataset, 4)
        if split == "train":
            take = np.linspace(0, len(starts) - 1, min(train_count, len(starts)), dtype=int)
            starts, anchors, real = starts[take], anchors[take], real[take]
        features = torch.from_numpy(np.stack([dataset.features[s:s + 32] for s in starts]))
        context = anchor_context(anchors, checkpoint.get("anchor_context_scaler"), checkpoint["config"].get("anchor_conditioning", "none"))
        context = None if context is None else torch.from_numpy(context)
        recovered = torch.cat([model.reconstruct(features[i:i+128], None if context is None else context[i:i+128]) for i in range(0, len(features), 128)])
        decoded = decode_batch(recovered, scaler, anchors)
        roundtrip = decode_batch(features, scaler, anchors)
        mse = (features - recovered).square().mean((0, 1)).numpy()
        variance = features.var((0, 1), unbiased=False).numpy()
        groups = {name: dict(scaled_mse=float(mse[columns].mean()), scaled_variance=float(variance[columns].mean()),
                            mse_over_variance=float(mse[columns].mean() / variance[columns].mean()))
                  for name, columns in (("price", PRICE_COLUMNS), ("volume", VOLUME_COLUMNS))}
        controls = []
        for seed in (42, 43, 44):
            moved, shift = shifted_anchors(anchors, seed)
            controls.append(dict(seed=seed, cyclic_shift=shift, **validity_rates(decode_batch(features, scaler, moved))))
        result["splits"][split] = dict(windows=len(starts), raw_start_indices=(starts + dataset.raw_offset).tolist(),
            real=validity_rates(real), roundtrip=validity_rates(roundtrip), reconstruction=validity_rates(decoded),
            feature_groups=groups, feature_mse=mse.tolist(), feature_variance=variance.tolist(),
            price_rmse_usd=float(np.sqrt(np.square(decoded[..., PRICE_COLUMNS] - real[..., PRICE_COLUMNS]).mean())),
            roundtrip_max_price_error_usd=float(np.abs(roundtrip[..., PRICE_COLUMNS] - real[..., PRICE_COLUMNS]).max()),
            reconstruction_invalid_by_update=[validity_rates(decoded[:, i:i + 1])["invalid_rate"] for i in range(32)],
            shuffled_anchor_runs=controls,
            shuffled_anchor_mean={key: float(np.mean([c[key] for c in controls])) for key in validity_rates(real)})
    write_json(destination, result)
    return result


def check_conditioning_pair(runs):
    """Reject comparisons that change more than the supplied context values."""
    settings = []
    for mode in ("zero", "matched"):
        run = runs[mode]
        if run["config"].get("anchor_conditioning") != mode or run.get("test_evaluated") is not False:
            raise ValueError("Use completed matched/zero validation runs, with no test scoring.")
        settings.append({k: v for k, v in run["config"].items() if k not in ("anchor_conditioning", "objective")})
    if settings[0] != settings[1]:
        raise ValueError("Paired runs must share architecture, seed, losses and stage budgets.")
    for key in ("comparison_signature", "metric_specification", "source_files", "parameter_count"):
        if runs["zero"][key] != runs["matched"][key]:
            raise ValueError(f"Paired runs differ in {key}.")


def conditioning_screen(matched, zero, noise_rows):
    """The predeclared pilot screen; it does not certify simulator quality."""
    gain = 100 * (zero["invalid_rate_mean"] - matched["invalid_rate_mean"])
    criteria = [("Invalidity gain at least 5 percentage points", gain >= 5, gain)]
    for metric in ("kl_spread", "kl_mid_return", "abs_return_acf_mae"):
        difference = matched[metric + "_mean"] - zero[metric + "_mean"]
        criteria.append((metric + " increase at most 5%", matched[metric + "_mean"] <= 1.05 * zero[metric + "_mean"], difference))
    difference = matched["depth_ssim_mean"] - zero["depth_ssim_mean"]
    criteria.append(("Depth SSIM drop at most 0.01", difference >= -.01, difference))
    if len(noise_rows) != 3:
        raise ValueError("The noise screen needs all three predetermined cases.")
    ratios = [r[key] for r in noise_rows for key in ("Price SD / real", "Volume SD / real")]
    if not np.isfinite(ratios).all():
        raise ValueError("Noise ratios must be finite.")
    criteria.append(("Fixed-context noise SD ratios at least 0.1 in every case", min(ratios) >= .1, min(ratios)))
    return [dict(Criterion=label, Passed=bool(passed), Value=float(value)) for label, passed, value in criteria]


def analyse_conditioning(data_dir, matched_run, zero_run, destination, baseline_path=None):
    """Replay saved fits and regenerate validation diagnostics in a new folder.

    Read each CSV pair once and reuse the prepared data for reconstruction.
    Saved training timings remain historical; this function never calls fit().
    """
    baseline_path = baseline_path or PROJECT_DIR / "reports/ar/anchor_conditioned.json"
    torch.set_num_threads(1)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Report already exists; choose a new report directory to preserve it.")
    folders = {"zero": Path(zero_run), "matched": Path(matched_run)}
    runs = {mode: json.loads((folder / "results.json").read_text(encoding="utf-8")) for mode, folder in folders.items()}
    check_conditioning_pair(runs)
    if runs["zero"]["source_files"] != timegan_code_hashes():
        raise ValueError("Model sources differ from these runs; use their archived source or new fits.")
    previous_path = PROJECT_DIR / "reports/timegan/conditioning.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
    plan = dict(protocol="anchor_conditioned_v1", training_seeds=[runs["zero"]["config"]["training_seed"]],
                generation_seeds=runs["zero"]["config"]["generation_seeds"], settings=runs["zero"]["config"],
                decision="Screen invalidity, both KL scores, ACF, SSIM and fixed-context noise together; one fitted seed is exploratory.")
    prepared = prepare_datasets(data_dir, 32, 32)
    datasets, scaler, audit = prepared
    training, validation = datasets["train"], datasets["validation"]
    starts, anchors, real = validation_windows(validation, 4)
    actual_features = np.stack([validation.features[s:s + 32] for s in starts])
    real_sd = actual_features.reshape(-1, 40).std(0)
    case_indices = np.array([0, (len(starts) - 1) // 2, len(starts) - 1])
    ar = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    result = dict(scope=f"Pilot: one training seed ({plan['training_seeds'][0]}) per mode; generation-seed variation is not retraining uncertainty.",
        plan=plan, comparison=[], final_comparison=[], reconstruction_rows=[], noise_rows=[], shifted_context_rows=[], cases=[], resources=[], figures={},
        verification=dict(checkpoint_replays={}, final_checkpoints_replayed=True, finite_outputs=True, test_evaluated=False),
        source_files=timegan_code_hashes(), analysis_sha256=sha256_file(Path(__file__)), models={})
    all_books = {}
    for mode in ('zero','matched'):
        folder = folders[mode]
        name, run = folder.name, runs[mode]
        assert run['comparison_signature']==ar['comparison_signature']
        assert run['source_files'] == timegan_code_hashes()
        assert run['reload_max_abs_difference'] == 0
        assert sha256_file(folder/'selected.pt')==run['selected_checkpoint_sha256']
        model,loaded_scaler,checkpoint=load_timegan(folder/'selected.pt')
        np.testing.assert_array_equal(loaded_scaler.minimum,scaler.minimum)
        np.testing.assert_array_equal(loaded_scaler.maximum,scaler.maximum)
        assert checkpoint['anchor_context_scaler']==fit_anchor_context(training.book[training.starts][:,PRICE_COLUMNS]/10000)
        assert checkpoint['anchor_context_scaler']==ar['context_scaler']
        assert checkpoint['source_files']==run['source_files'] and checkpoint['config']==run['config']
        result['verification']['context_statistics_match_training_and_ar']=True
        context=anchor_context(anchors,checkpoint['anchor_context_scaler'],mode)
        features=model.generate(len(starts),32,plan['generation_seeds'][0],128,context=context)
        assert torch.isfinite(features).all()
        books=decode_batch(features,scaler,anchors);all_books[mode]=books
        sample=np.load(folder/'validation_examples.npz')
        assert np.array_equal(books[:3],sample['generated'])
        reference=ValidationEvaluator(real,anchors,run['metric_specification'])
        scores,details=reference.evaluate(books)
        for key,value in scores.items():
            assert np.isclose(value,run['selected']['runs'][0][key],rtol=1e-12,atol=1e-12),(name,key)
        label='TimeGAN: '+mode+' context'
        result['comparison'].append(dict(label=label,score=run['selected']))
        result['final_comparison'].append(dict(label=label,score=run['validation_history'][-1]))
        result['verification']['checkpoint_replays'][name]=0.
        diagnostic=diagnose(data_dir,folder/'selected.pt',destination/f'{mode}_reconstruction.json',prepared=prepared)
        for split,diag in diagnostic['splits'].items():
            for group,row in diag['feature_groups'].items():
                result['reconstruction_rows'].append({'Model':label,'Split':split,'Feature group':group,'MSE / variance':row['mse_over_variance'],
                    'Reconstructed invalid (%)':100*diag['reconstruction']['invalid_rate']})
        noise_details=[]
        for index in case_indices:
            constant_context=np.repeat(context[index:index+1],128,axis=0)
            checks=[]
            for seed in plan['generation_seeds']:
                generated=model.generate(128,32,seed,128,context=constant_context).numpy()
                noise_sd=generated.std(axis=0).mean(axis=0)
                checks.append(dict(seed=seed,price_sd=float(noise_sd[::2].mean()),volume_sd=float(noise_sd[1::2].mean()),
                    price_ratio=float(noise_sd[::2].mean()/real_sd[::2].mean()),volume_ratio=float(noise_sd[1::2].mean()/real_sd[1::2].mean()),
                    unique_sequences=int(len(np.unique(generated.reshape(128,-1),axis=0)))))
            noise_details.append(dict(validation_window=int(index+1),runs=checks))
            result['noise_rows'].append({'Model':label,'Validation window':int(index+1),'Price noise SD':np.mean([r['price_sd'] for r in checks]),
                'Volume noise SD':np.mean([r['volume_sd'] for r in checks]),'Price SD / real':np.mean([r['price_ratio'] for r in checks]),
                'Volume SD / real':np.mean([r['volume_ratio'] for r in checks])})
        shifted=[]
        for seed in plan['generation_seeds']:
            moved,shift=shifted_anchors(anchors,seed)
            moved_context=anchor_context(moved,checkpoint['anchor_context_scaler'],mode)
            sampled=model.generate(len(starts),32,seed,128,context=moved_context)
            moved_books=decode_batch(sampled,scaler,anchors)
            moved_scores,_=reference.evaluate(moved_books)
            original=model.generate(len(starts),32,seed,128,context=context)
            original_scores,_=reference.evaluate(decode_batch(original,scaler,anchors))
            recorded=next(r for r in run['selected']['runs'] if r['seed']==seed)
            for key,value in original_scores.items():
                assert np.isclose(value,recorded[key],rtol=1e-12,atol=1e-12),(name,seed,key)
            delta=float((sampled-original).abs().max())
            if mode=='zero': assert delta==0
            shifted.append(dict(seed=seed,shift=shift,max_feature_difference=delta,**moved_scores))
        result['shifted_context_rows'].append({'Model':label,'Correct context invalid (%)':100*run['selected']['invalid_rate_mean'],
            'Shifted context invalid (%)':100*np.mean([r['invalid_rate'] for r in shifted]),
            'Max feature difference':max(r['max_feature_difference'] for r in shifted)})
        for index in case_indices:
            row=validity_rates(books[index:index+1]);book=books[index]
            gaps=np.c_[np.diff(book[:,0::4],axis=1),-np.diff(book[:,2::4],axis=1),book[:,0]-book[:,2]]
            result['cases'].append({'Validation window':int(index+1),'Model':mode,'Raw anchor row':int(starts[index]+validation.raw_offset),
                'Invalid (%)':100*row['invalid_rate'],'Crossed (%)':100*row['crossed_rate'],
                'Depth SSIM':float(details['depth_ssim_per_sequence'][index]),'Minimum gap (USD)':float(gaps.min()),
                'Generated volume changes':int(np.count_nonzero(np.diff(book[:,VOLUME_COLUMNS],axis=0))),
                'Real volume changes':int(np.count_nonzero(np.diff(real[index][:,VOLUME_COLUMNS],axis=0)))})
        result['resources'].append({'Model':label,'Parameters':run['parameter_count'],'Fit minutes':run['total_seconds']/60,
            'Generation ms / 1080 windows':1000*run['selected']['generation_seconds_mean'],'Device':run['device'],'Selected joint step':run['selected']['step']})
        final_model,final_scaler,final_checkpoint=load_timegan(folder/'joint_last.pt')
        assert final_checkpoint['step']==run['config']['joint_steps'] and final_checkpoint['source_files']==run['source_files']
        assert final_checkpoint['config']==run['config'] and final_checkpoint['anchor_context_scaler']==checkpoint['anchor_context_scaler']
        np.testing.assert_array_equal(final_scaler.minimum,scaler.minimum)
        np.testing.assert_array_equal(final_scaler.maximum,scaler.maximum)
        for recorded in run['validation_history'][-1]['runs']:
            final_features=final_model.generate(len(starts),32,recorded['seed'],128,context=context)
            assert torch.isfinite(final_features).all()
            final_scores,_=reference.evaluate(decode_batch(final_features,scaler,anchors))
            for key,value in final_scores.items():
                assert np.isclose(value,recorded[key],rtol=1e-12,atol=1e-12),(name,'final',key)
        del final_model
        result['models'][mode]=dict(name=name,checkpoint_sha256=run['selected_checkpoint_sha256'],diagnostics=diagnostic,
            training_history=run['history'],validation_history=run['validation_history'],
            fixed_context_noise=noise_details,shifted_context=shifted,selected=run['selected'],final=run['validation_history'][-1])
    for row in ar['candidates']:
        result['comparison'].append(dict(label='VARX(2): '+row['anchor_conditioning']+' context',score=row))
        result['resources'].append({'Model':'VARX(2): '+row['anchor_conditioning']+' context','Parameters':row['coefficient_count'],
            'Fit minutes':row['fit_seconds']/60,'Generation ms / 1080 windows':1000*row['generation_seconds_mean'],'Device':'cpu','Selected joint step':None})

    matched,zero=(result['models'][mode]['selected'] for mode in ('matched','zero'))
    gain=100*(zero['invalid_rate_mean']-matched['invalid_rate_mean'])
    noise_rows=[r for r in result['noise_rows'] if r['Model']=='TimeGAN: matched context']
    result['pilot_screen']=conditioning_screen(matched,zero,noise_rows)
    passed=all(row['Passed'] for row in result['pilot_screen'])
    result['pilot_screen_passed']=passed
    result['conclusion']=(f'Matched context gives {100*matched["invalid_rate_mean"]:.2f}% invalid snapshots versus {100*zero["invalid_rate_mean"]:.2f}% for the same-size zero-context control (gain {gain:.2f} percentage points). '
        f'Return KL is {matched["kl_mid_return_mean"]:.3f} versus {zero["kl_mid_return_mean"]:.3f}; ACF error {matched["abs_return_acf_mae_mean"]:.3f} versus {zero["abs_return_acf_mae_mean"]:.3f}; depth SSIM {matched["depth_ssim_mean"]:.3f} versus {zero["depth_ssim_mean"]:.3f}. '
        'This is a single training-seed pilot, not confirmation across retraining seeds.')
    result['next_step']='Pilot screen met: repeat both modes at matched training seeds before making a reliability claim.' if passed else 'Pilot screen not met: retain the negative/trade-off evidence; do not expand this configuration into a blind grid or call it a validated simulator.'

    result['performance_check'] = dict(scope="No training or timing benchmark is run during report generation.", rows=[])
    if all(previous.get('models', {}).get(mode, {}).get('checkpoint_sha256') == runs[mode]['selected_checkpoint_sha256'] for mode in folders):
        result['performance_check'] = previous['performance_check']
        result['performance_check_origin'] = "Historical timing measurement retained for the identical checkpoint pair; not remeasured."
    save_conditioning_figures(result, real, all_books, case_indices, ar['metric_specification']['depth_ssim'], destination)
    write_json(destination / 'conditioning.json', result)
    return result


def save_conditioning_figures(result, real, all_books, case_indices, depth_spec, folder):
    """Use fixed cases and common scales; never repair or select generated examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    def save(fig,key):
        path=folder/f'conditioning_{key}.png';fig.savefig(path,dpi=145);plt.close(fig)
        result['figures'][key]=dict(file=path.name,sha256=sha256_file(path))

    metrics=[('invalid_rate','Invalid (%)',100),('kl_spread','Spread KL',1),('kl_mid_return','Return KL',1),('abs_return_acf_mae','ACF error',1),('depth_ssim','Depth SSIM',1)]
    fig,axes=plt.subplots(1,5,figsize=(15,4),layout='constrained')
    for ax,(key,label,scale) in zip(axes,metrics):
        rows=result['comparison']; ax.bar(range(len(rows)),[r['score'][key+'_mean']*scale for r in rows],yerr=[r['score'][key+'_sd']*scale for r in rows],capsize=3,color=['#b1bac5','#2c7fb8','#d2d5b1','#6e8b3d'])
        ax.set(title=label,xticks=range(len(rows)),xticklabels=['TG zero','TG matched','AR zero','AR matched']);ax.tick_params(axis='x',rotation=65);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Starting-book conditioning pilot; error bars = generation-seed SD, one fitted model per bar')
    save(fig,'metrics')
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    for mode,model in result['models'].items():
        for ax,stage,key,title in zip(axes.ravel()[:4],['embedding','supervised','joint','joint'],
                ['reconstruction_mse','supervised_mse','generator_loss','discriminator_loss'],
                ['Reconstruction warm-up MSE','Supervisor warm-up MSE','Joint generator objective','Joint discriminator objective']):
            rows=[r for r in model['training_history'] if r['stage']==stage]
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],label=mode)
            ax.set(title=title,xlabel='Stage update');ax.grid(alpha=.2)
        rows=model['validation_history']
        for ax,key,title,scale in [(axes[1,1],'invalid_rate_mean','Validation invalid (%)',100),(axes[1,2],'kl_mid_return_mean','Validation return KL',1)]:
            ax.plot([r['step'] for r in rows],[r[key]*scale for r in rows],marker='o',label=mode)
            ax.set(title=title,xlabel='Joint update');ax.grid(alpha=.2)
    axes[0,0].legend();save(fig,'training')
    images={k:depth_images(v[case_indices],depth_spec) for k,v in dict(real=real,**all_books).items()}
    fig,axes=plt.subplots(3,3,figsize=(12,8),layout='constrained')
    for i,index in enumerate(case_indices):
        for j,mode in enumerate(('real','zero','matched')):
            picture=np.concatenate([images[mode][i,0,::-1],images[mode][i,1]],axis=0)
            heat=axes[i,j].imshow(picture,aspect='auto',vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
            axes[i,j].set(title=f'Window {index+1}: {mode}',xlabel='Update',yticks=[0,9.5,19],yticklabels=['Ask 10','Best levels','Bid 10'])
    fig.colorbar(heat,ax=axes.ravel().tolist(),shrink=.8,label='log(1 + volume) / training maximum')
    save(fig,'depth')
    fig,axes=plt.subplots(3,2,figsize=(11,7),layout='constrained')
    for i,index in enumerate(case_indices):
        for mode,books in dict(real=real,**all_books).items():
            book=books[index]
            gap=np.c_[np.diff(book[:,0::4],axis=1),-np.diff(book[:,2::4],axis=1),book[:,0]-book[:,2]]
            axes[i,0].plot(np.arange(1,33),gap.min(1),label=mode)
            axes[i,1].plot(np.arange(1,33),book[:,0]-book[:,2],label=mode)
        for ax in axes[i]: ax.axhline(0,color='black',linewidth=.7);ax.set_xlabel('Update');ax.grid(alpha=.2)
        axes[i,0].set(title=f'Window {index+1}: minimum of 19 gaps',ylabel='USD')
        axes[i,1].set(title=f'Window {index+1}: best spread',ylabel='USD')
    axes[0,0].legend();save(fig,'gaps')
