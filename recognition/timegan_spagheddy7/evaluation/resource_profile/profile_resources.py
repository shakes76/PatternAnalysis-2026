"""Finite, disposable operation profile. Never writes project files or model weights."""
import argparse, hashlib, json, os, shutil, statistics, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import psutil

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PROJECT = Path('C:/Users/franz/OneDrive/Documents/FinMath/Sem4/COMP3710/Project/2.7/PatternAnalysis-2026/recognition/timegan_spagheddy7')
DATA = PROJECT.parents[2] / 'LOBSTER_SampleFile_AMZN_2012-06-21_10'
CHECKPOINT = PROJECT / 'outputs/timegan/experiments/volume_recovery_matched_s2026/joint_last.pt'
DEADLINE = datetime(2026, 11, 5, 10, 50, tzinfo=timezone.utc).timestamp()
SOURCES = ['modules.py', 'train.py', 'dataset.py', 'evaluation.py', 'protocol.py', 'predict.py']

def utc(): return datetime.now(timezone.utc).isoformat()
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def write(name, value):
    p = ROOT / name
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, p)
def identity(p):
    return {'pid':p.pid, 'create_time':p.create_time(), 'script':str(Path(__file__).resolve()), 'cmdline':p.cmdline()}
def remaining():
    result = DEADLINE-time.time()
    if result <= 0: raise TimeoutError('Profile deadline reached')
    return result
def provenance():
    return {'source_sha256':{n:digest(PROJECT/n) for n in SOURCES}, 'checkpoint_sha256':digest(CHECKPOINT)}

def prepare():
    import numpy as np, torch
    sys.path.insert(0, str(PROJECT))
    from dataset import prepare_datasets, PRICE_COLUMNS, VOLUME_COLUMNS, PRICE_DIVISOR
    from protocol import anchor_context, fit_anchor_context
    torch.set_num_threads(1)
    c = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    ds, scaler, audit = prepare_datasets(DATA,32,32)
    training, validation = ds['train'], ds['validation']
    assert np.array_equal(scaler.minimum,c['scaler']['minimum'].numpy())
    assert np.array_equal(scaler.maximum,c['scaler']['maximum'].numpy())
    train_anchors = training.book[training.starts][:,PRICE_COLUMNS].astype(np.float64)/PRICE_DIVISOR
    train_volumes = training.book[training.starts][:,VOLUME_COLUMNS].astype(np.float64)
    assert fit_anchor_context(train_anchors,train_volumes) == c['anchor_context_scaler']
    contexts = anchor_context(train_anchors,c['anchor_context_scaler'],'matched',volumes=train_volumes,volume_mode='matched')
    values = np.stack([training.features[s:s+32] for s in training.starts])
    starts = validation.starts[validation.starts >= 4]
    anchors = validation.book[starts][:,PRICE_COLUMNS].astype(np.float64)/PRICE_DIVISOR
    volumes = validation.book[starts][:,VOLUME_COLUMNS]
    valcontexts = anchor_context(anchors,c['anchor_context_scaler'],'matched',volumes=volumes,volume_mode='matched')
    assert len(starts)==1080 and values.shape[1:]==(32,40)
    np.savez(ROOT/'training_inputs.npz', values=values, anchors=train_anchors, contexts=contexts)
    np.savez(ROOT/'generation_inputs.npz', anchors=anchors, contexts=valcontexts, raw_anchor_indices=starts+validation.raw_offset)
    # Both device workers use identical CPU-generated batch indices and noise.
    sampler = torch.Generator(device='cpu').manual_seed(2126)
    noisegen = torch.Generator(device='cpu').manual_seed(2226)
    indices = torch.stack([torch.randperm(len(values),generator=sampler)[:128] for _ in range(35)]).numpy()
    noises = torch.rand(21,128,32,40,generator=noisegen).numpy()
    np.savez(ROOT/'training_draws.npz', indices=indices, noises=noises)
    out = {'state':'complete','completed_at':utc(),'identity':identity(psutil.Process()),'training_shape':list(values.shape),'validation_windows':len(starts),
           'input_hashes':{n:digest(ROOT/n) for n in ['training_inputs.npz','generation_inputs.npz','training_draws.npz']},
           'configuration':c['config'],'source_files':provenance(), 'scope':'Training and validation contexts only; no model quality scoring or test generation. CSV preprocessing is outside profiling-worker memory/timing.'}
    write('preparation.json',out)
    print(json.dumps({'prepared':out['training_shape'],'validation_windows':len(starts)}),flush=True)

def worker(mode,device):
    started = time.perf_counter()
    name = mode+'_'+device
    write(name+'_identity.json', {'identity':identity(psutil.Process()),'started_at':utc()})
    import numpy as np, torch
    sys.path.insert(0,str(PROJECT))
    from modules import TimeGAN
    from train import TimeGANTrainer
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if device=='cuda':
        assert torch.cuda.is_available()
        torch.cuda.init()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    p = psutil.Process()
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    def memory():
        sync()
        m=p.memory_info()
        result={'rss_bytes':m.rss,'lifetime_peak_working_set_bytes':m.peak_wset,'private_commit_bytes':m.private,'lifetime_peak_pagefile_bytes':m.peak_pagefile}
        if device=='cuda':result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        return result
    points={'after_imports':memory()}
    c=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    assert c['step']==4000 and c['architecture']['hidden']==48
    model=TimeGAN(**c['architecture']).to(device)
    model.load_state_dict(c['state_dict'])
    config=c['config']
    assert sum(x.numel() for x in model.parameters())==230393
    out={'state':'running','mode':mode,'device':device,'identity':identity(p),'started_at':utc(),'architecture':c['architecture'], 'parameters':230393,
         'batch_size':128,'sequence_length':32,'cpu_threads':1,'torch_version':str(torch.__version__),'numpy_version':np.__version__,
         'device_name':torch.cuda.get_device_name(0) if device=='cuda' else '13th Gen Intel Core i7-13650HX',
         'source_files':provenance(),'memory_points':points,
         'memory_scope':'Windows lifetime per-process peak working set includes Python/imports, prepared input loading, checkpoint/model and profiling operations; excludes raw CSV preprocessing, other processes, and a full production fit. CUDA counters cover this worker and exclude driver/context memory; reserved includes allocated.'}
    if mode=='training':
        with np.load(ROOT/'training_inputs.npz') as z:
            values=torch.from_numpy(z['values'].copy()).to(device)
            anchors=torch.from_numpy(z['anchors'].copy()).to(device)
            contexts=torch.from_numpy(z['contexts'].copy()).to(device)
        with np.load(ROOT/'training_draws.npz') as z:
            indices=torch.from_numpy(z['indices'].copy()).to(device)
            noises=torch.from_numpy(z['noises'].copy()).to(device)
        trainer=TimeGANTrainer(model,learning_rate=config['learning_rate'],supervised_weight=config['supervised_weight'],moment_weight=config['moment_weight'],gamma=config['gamma'],
             structure_weight=config['structure_weight'],minimum=model.price_ladder.minimum,span=model.price_ladder.span,margin=config['structure_margin_usd'],price_reconstruction_weight=config['price_reconstruction_weight'],no_supervision=False)
        out.update(workload='Disposable final-checkpoint weights with fresh Adam states; short operation profile, not resumed production training.',training_windows=len(values),warmup_updates_per_stage=2,measured_updates_per_stage=5,
                   stages={},diagnostic_gradient_norms=False,discriminator_threshold=.15,full_cycle='Two generator/supervisor plus two joint embedder/recovery updates, then one discriminator opportunity.',common_draws_sha256=digest(ROOT/'training_draws.npz'))
        out['prepared_draw_cache_bytes']={'noise':noises.numel()*noises.element_size(),'indices':indices.numel()*indices.element_size()}
        out['training_timing_scope']='Prepared batch indexing and exact training operations; excludes RNG generation, CSV preprocessing, checkpoint loading/saving and validation scoring. A preloaded common noise/index cache remains resident, so memory is the profiled workload, not an original production-run peak.'
        bi=ni=0
        def batch():
            nonlocal bi
            ix=indices[bi];bi+=1
            return values[ix],anchors[ix],contexts[ix]
        def noise():
            nonlocal ni
            n=noises[ni];ni+=1
            return n
        def adam_steps():
            return {k:sum(float(st['step']) for st in op.state.values()) for k,op in trainer.optimizers.items()}
        points['after_model_data_optimizer_construction']=memory()
        for stage in ('embedding','supervised','joint'):
            durations=[]
            states_before=adam_steps()
            phase_started=time.perf_counter()
            for i in range(7):
                remaining();sync();t=time.perf_counter()
                if stage=='embedding':
                    real,a,ctx=batch();trainer.embedding_step(real,collect=False,anchors=a,context=ctx)
                elif stage=='supervised':
                    real,a,ctx=batch();trainer.supervisor_step(real,collect=False,context=ctx)
                else:
                    for _ in range(2):
                        real,a,ctx=batch();trainer.generator_step(real,noise(),collect=False,anchors=a,context=ctx)
                        trainer.embedding_step(real,joint=True,collect=False,anchors=a,context=ctx)
                    real,a,ctx=batch();trainer.discriminator_step(real,noise(),threshold=.15,collect=False,context=ctx)
                sync();durations.append(time.perf_counter()-t)
                print(json.dumps({'worker':name,'stage':stage,'iteration':i+1,'seconds':durations[-1]}),flush=True)
            states_after=adam_steps()
            out['stages'][stage]={'warmup_seconds':durations[:2],'measured_seconds':durations[2:],'mean_seconds':statistics.mean(durations[2:]),'sample_sd_seconds':statistics.stdev(durations[2:]),
                 'optimizer_step_sums_before':states_before,'optimizer_step_sums_after':states_after,'wall_seconds':time.perf_counter()-phase_started}
            points['after_'+stage]=memory()
            write(name+'.json',out)
        out['optimizer_state_entry_counts']={k:len(op.state) for k,op in trainer.optimizers.items()}
        assert all(out['optimizer_state_entry_counts'].values()),'Not all five Adam states were materialised; cannot label profile complete'
        out['draws_used']={'batches':bi,'noise_batches':ni}
        assert bi==35 and ni==21
        out['finite_parameters']=all(bool(torch.isfinite(x).all()) for x in model.parameters())
        assert out['finite_parameters']
    else:
        with np.load(ROOT/'generation_inputs.npz') as z:
            anchors=z['anchors'].copy();contexts=z['contexts'].copy()
        model.eval()
        points['after_model_context_loading']=memory()
        out.update(workload='Saved final checkpoint generation; no parameter updates.',count=1080,generation_batch_size=128,warmup_calls=2,warmup_sequences_per_call=1080,generation_seeds=[42,43,44],runs=[],
           timing_scope='model.generate includes common CPU noise creation, batches transferred to device and output returned to CPU; excludes model/context loading and market-metric computation.')
        for seed in (10,11):
            remaining();model.generate(1080,32,seed,128,context=contexts,anchors=anchors);sync()
        points['after_generation_warmup']=memory()
        for seed in (42,43,44):
            remaining();sync();t=time.perf_counter()
            generated=model.generate(1080,32,seed,128,context=contexts,anchors=anchors)
            sync();elapsed=time.perf_counter()-t
            assert generated.shape==(1080,32,40) and torch.isfinite(generated).all()
            out['runs'].append({'seed':seed,'seconds':elapsed,'sequences_per_second':1080/elapsed,'generated_sha256':hashlib.sha256(generated.numpy().tobytes()).hexdigest()})
            del generated
            print(json.dumps({'worker':name,'seed':seed,'seconds':elapsed}),flush=True)
        out['mean_seconds']=statistics.mean(r['seconds'] for r in out['runs'])
        out['sample_sd_seconds']=statistics.stdev(r['seconds'] for r in out['runs'])
        out['sequences_per_second']=1080/out['mean_seconds']
    points['finished']=memory()
    out.update(state='complete',finished_at=utc(),worker_wall_seconds=time.perf_counter()-started,original_checkpoint_unchanged=digest(CHECKPOINT)==out['source_files']['checkpoint_sha256'])
    assert out['original_checkpoint_unchanged']
    write(name+'.json',out)
    print(json.dumps({'worker':name,'state':'complete','memory':points['finished']}),flush=True)

def controller(inference_only=False):
    before=provenance()
    status={'started_at':utc(),'deadline_utc':datetime.fromtimestamp(DEADLINE,timezone.utc).isoformat(),'state':'running','controller':identity(psutil.Process()),'source_before':before,'jobs':[]}
    write('launcher.json',status)
    stages=[('generation','cpu'),('generation','cuda')] if inference_only else [('prepare',None),('training','cpu'),('training','cuda'),('generation','cpu'),('generation','cuda')]
    for mode,device in stages:
        remaining()
        name=mode if device is None else mode+'_'+device
        args=[sys.executable,str(Path(__file__).resolve()),'--mode',mode]
        if device:args+=['--device',device]
        with open(ROOT/(name+'.log'),'w',encoding='utf-8') as log:
            proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
            owned=identity(psutil.Process(proc.pid))
            job={'name':name,'owned_parent':owned,'started_at':utc(),'state':'running'}
            status['jobs'].append(job);write('launcher.json',status)
            try:
                proc.wait(timeout=min(480 if mode=='training' else 180,remaining()))
            except subprocess.TimeoutExpired:
                root=psutil.Process(proc.pid)
                verified=[]
                if abs(root.create_time()-owned['create_time'])<.001 and str(Path(__file__).resolve()).lower() in [s.lower() for s in root.cmdline()]:
                    verified=[identity(x) for x in root.children(recursive=True) if str(Path(__file__).resolve()).lower() in [s.lower() for s in x.cmdline()]]+[owned]
                for ident in verified:
                    try:
                        p=psutil.Process(ident['pid'])
                        if abs(p.create_time()-ident['create_time'])<.001 and str(Path(__file__).resolve()).lower() in [s.lower() for s in p.cmdline()]:p.kill()
                    except psutil.NoSuchProcess:pass
                proc.wait(timeout=10)
                job.update(state='timed_out',terminated_verified_identities=verified,finished_at=utc());status['state']='incomplete';write('launcher.json',status)
                raise TimeoutError(name+' profile exceeded finite budget')
            job.update(exit_code=proc.returncode,state='complete' if proc.returncode==0 else 'failed',finished_at=utc());write('launcher.json',status)
            if proc.returncode:raise RuntimeError(name+' failed; inspect its log')
    after=provenance();assert after==before
    results={m+'_'+d:json.loads((ROOT/(m+'_'+d+'.json')).read_text()) for m in ('training','generation') for d in ('cpu','cuda')}
    summary={'state':'complete','completed_at':utc(),'scope':'Supplementary short operation and generation profile; not original full-run peak memory or scientific retraining.',
             'sources_unchanged':True,'source_before':before,'source_after':after,'results':results}
    if inference_only:summary['generation_warmup_correction']=json.loads((ROOT/'generation_correction.json').read_text())
    write('resource_profile_results.json',summary)
    status.update(state='complete',finished_at=utc(),sources_unchanged=True);write('launcher.json',status)
    print(json.dumps({'state':'complete','summary':str(ROOT/'resource_profile_results.json')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['controller','generation_correction_controller','prepare','training','generation'],default='controller');parser.add_argument('--device',choices=['cpu','cuda'])
    args=parser.parse_args()
    try:
        remaining()
        if args.mode=='controller':controller()
        elif args.mode=='generation_correction_controller':
            previous=json.loads((ROOT/'launcher.json').read_text())
            assert previous['state']=='complete'
            archive=(ROOT/'initial_generation_warmup_diagnostic').resolve()
            assert archive.is_relative_to(ROOT.resolve())
            archive.mkdir(exist_ok=False)
            names=['launcher.json','resource_profile_results.json','plan.json']+[m+'_'+d+s for m in ('generation',) for d in ('cpu','cuda') for s in ('.json','.log','_identity.json')]
            for n in names:shutil.copy2(ROOT/n,archive/n)
            reason={'recorded_at':utc(),'scope':'Inference only, no training repetition','reason':'Original two 128-sequence warmups did not cover the 56-sequence tail of the 1080-sequence workload. Preserve original results as diagnostic, then use two full1080 warmups on both devices before the same three timed seeds.',
                    'archive':str(archive),'archived_result_sha256':digest(archive/'resource_profile_results.json'),'new_warmup_calls':2,'new_warmup_count':1080}
            write('generation_correction.json',reason)
            controller(inference_only=True)
        elif args.mode=='prepare':prepare()
        else:worker(args.mode,args.device)
    except Exception:
        print(traceback.format_exc(),flush=True)
        if args.mode in ('controller','generation_correction_controller') and (ROOT/'launcher.json').exists():
            status=json.loads((ROOT/'launcher.json').read_text())
            if status.get('state')=='running':status.update(state='failed',finished_at=utc(),error=traceback.format_exc());write('launcher.json',status)
        raise
