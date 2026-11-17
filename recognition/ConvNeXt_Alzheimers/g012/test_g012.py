"""CPU synthetic tests only: artificial rows/tensors, never MRI files."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import dataset
import train_convnext_v2 as v2
from train_schedule_only import LeaseBudget, WindowsJob
from g012 import core, engine, storage as st, results

torch.set_num_threads(1)


def rows(split='train',n=3,k=3):
    return tuple(dataset.SliceRecord(f'synthetic/{s}_{i}.jpeg',f'person{s}',f'scan{s}',i,
        'test','AD' if s%2 else 'NC',2*(s%2),s%2,split) for s in range(n) for i in range(k))


class Toy(nn.Module):
    def __init__(self,stochastic=False):
        super().__init__();self.head=nn.Linear(3,2);self.scalar=nn.Parameter(torch.tensor(1.));self.stochastic=stochastic
    def forward(self,x):
        out=self.head(x)*self.scalar
        return F.dropout(out,p=.2,training=self.training) if self.stochastic else out


def fetch(bag,angles=None):
    values=torch.arange(len(bag.rows)*3,dtype=torch.float32).reshape(-1,3)/9
    if angles is not None:values=values+torch.tensor(angles)[:,None]/100
    return values+bag.label/7


def same(test,a,b):
    if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(a.keys(),b.keys())
        for k in a:same(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):same(test,x,y)
    else:test.assertEqual(a,b)


class MathTests(unittest.TestCase):
    def test_independent_references_and_distinct_losses(self):
        z=torch.tensor([[8.,-3.],[-1.,2.],[.2,.4]],dtype=torch.float64)
        expected=[F.cross_entropy(z,torch.ones(3,dtype=torch.long)),
                  -torch.log(torch.softmax(z,1).mean(0)[1]),
                  F.cross_entropy(z.mean(0)[None],torch.tensor([1]))]
        self.assertGreater(abs(float(expected[0]-expected[1])),.1)
        for v,e in zip(core.VARIANTS,expected):torch.testing.assert_close(core.scan_loss(z,1,v),e)
        torch.testing.assert_close(core.scan_probs(z,'G0'),torch.softmax(z,1).mean(0))
        torch.testing.assert_close(core.scan_probs(z,'G1'),torch.softmax(z,1).mean(0))
        torch.testing.assert_close(core.scan_probs(z,'G2'),torch.softmax(z.mean(0),0))
        self.assertFalse(torch.allclose(core.scan_probs(z,'G1'),core.scan_probs(z,'G2')))

    def test_k_one_equality(self):
        z=torch.tensor([[.25,-.2]],dtype=torch.float64)
        for v in core.VARIANTS:
            torch.testing.assert_close(core.scan_loss(z,0,v),core.scan_loss(z,0,'G0'))
            torch.testing.assert_close(core.scan_probs(z,v),core.scan_probs(z,'G0'))

    def test_extreme_finite_logits_and_gradients(self):
        for v in core.VARIANTS:
            z=torch.tensor([[10000.,-10000.],[-10000.,10000.],[10000.,9999.]],requires_grad=True)
            loss=core.scan_loss(z,1,v);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(z.grad).all())

    def test_permutation_and_gradcheck(self):
        g=torch.Generator().manual_seed(4)
        for v in core.VARIANTS:
            z=torch.randn(4,2,dtype=torch.float64,generator=g,requires_grad=True)
            torch.testing.assert_close(core.scan_loss(z,1,v),core.scan_loss(z[[2,0,3,1]],1,v))
            torch.testing.assert_close(core.scan_probs(z,v),core.scan_probs(z[[2,0,3,1]],v))
            self.assertTrue(torch.autograd.gradcheck(lambda x:core.scan_loss(x,1,v),(z,),atol=1e-6))

    def test_analytical_gradients(self):
        for v in core.VARIANTS:
            z=torch.tensor([[.2,.4],[-.7,.3],[1.1,-.2]],dtype=torch.float64,requires_grad=True)
            p=torch.softmax(z.detach(),1);one=F.one_hot(torch.tensor(1),2)
            if v=='G0':expected=(p-one)/3
            elif v=='G1':expected=p[:,1,None]/p[:,1].sum()*(p-one)
            else:expected=(torch.softmax(z.detach().mean(0),0)-one)[None].expand_as(z)/3
            core.scan_loss(z,1,v).backward();torch.testing.assert_close(z.grad,expected)

    def test_bad_math_inputs(self):
        for z in (torch.empty(0,2),torch.ones(2,3),torch.tensor([[float('nan'),0.]])):
            with self.assertRaises(dataset.DataError):core.scan_loss(z,0,'G1')


class GroupAndRandomnessTests(unittest.TestCase):
    def test_group_complete_order_and_variable_k(self):
        source=rows(k=5);bags=core.group_records(tuple(reversed(source)),'train',5)
        self.assertEqual(sum(len(b.rows) for b in bags),15)
        self.assertEqual([r.slice_index for r in bags[0].rows],list(range(5)))
        self.assertEqual(len(core.group_records(source[:-1],'train')),3)
        with self.assertRaises(dataset.DataError):core.group_records(source[:-1],'train',5)

    def test_bad_grouping_and_partition_rejected(self):
        original=rows()
        variants=[original+(original[0],),
                  (replace(original[0],subject_id='other'),)+original[1:],
                  (replace(original[0],label=1),)+original[1:],
                  (replace(original[0],slice_index=1),)+original[1:],
                  (replace(original[0],split='validation'),)+original[1:]]
        for value in variants:
            with self.assertRaises(dataset.DataError):core.group_records(value,'train',3)
        with self.assertRaises(dataset.DataError):core.group_records(rows('test'),'test')

    def test_training_rejects_validation_before_fetch(self):
        bags=core.group_records(rows('validation'),'validation');model=Toy()
        with self.assertRaises(dataset.DataError):
            core.train_epoch(model,bags,lambda *x:self.fail('Must not fetch held-out pixels'),
                torch.optim.SGD(model.parameters(),lr=.01),torch.device('cpu'),'G0',{'order':[0,1,2]})

    def test_manifest_count_mismatch_fails(self):
        bundle=SimpleNamespace(manifest_hashes={'x':'y'},records={'train':rows(),'validation':rows('validation')})
        approved={'manifest_sha256':{'x':'y'},'counts':{'train':{},'validation':{}}}
        with self.assertRaises(dataset.DataError):core.checked_groups(bundle,approved)

    def test_native_g2_patient_probability_not_patient_logits(self):
        source=(replace(rows('validation',2,1)[0],subject_id='same'),replace(rows('validation',2,1)[1],subject_id='same',label=0))
        bags=core.group_records(source,'validation')
        logits=[torch.tensor([[0.,10.]]),torch.tensor([[0.,-1.]])]
        metrics,p=core.aggregate(bags,logits,'G2')
        expected=sum(float(torch.softmax(z.mean(0),0)[1]) for z in logits)/2
        self.assertEqual(p['patients'][0]['p_ad'],expected)
        self.assertNotAlmostEqual(expected,float(torch.softmax(torch.cat(logits).mean(0),0)[1]))

    def test_order_angles_independent_and_rng_isolated(self):
        bags=core.group_records(rows(k=20),'train')
        a=core.schedule(bags,7);torch.rand(50);b=core.schedule(bags,7)
        same(self,a,b)
        self.assertEqual(sorted(a['order']),[0,1,2])
        self.assertGreater(len(set(a['angles'][0])),1)
        self.assertTrue(all(-5<=angle<=5 for scan in a['angles'] for angle in scan))
        self.assertNotEqual(a['augmentation_digest'],core.schedule(bags,8)['augmentation_digest'])
        state=v2.rng_state()
        with core.isolated_rng():
            torch.rand(25);np.random.random(7);random.random()
        same(self,state,v2.rng_state())

    def test_full_convnext_initial_tensor_identity(self):
        values=[]
        for variant in core.VARIANTS:
            v2.seed_everything(42);model=v2.make_model(core.base_config('cpu'))
            values.append(core.tensor_digest(model.state_dict()));del model
        self.assertEqual(len(set(values)),1)


class TrainingAndRecoveryTests(unittest.TestCase):
    def test_two_scans_and_one_scan_tail_vs_explicit_reference(self):
        bags=core.group_records(rows(),'train');matched=core.schedule(bags,1)
        for variant in core.VARIANTS:
            torch.manual_seed(2);actual=Toy();reference=deepcopy(actual)
            a=torch.optim.AdamW(actual.parameters(),lr=.001,weight_decay=.05)
            b=torch.optim.AdamW(reference.parameters(),lr=.001,weight_decay=.05)
            result=core.train_epoch(actual,bags,fetch,a,torch.device('cpu'),variant,matched)
            for start in (0,2):
                indices=matched['order'][start:start+2];b.zero_grad(set_to_none=True)
                losses=[]
                for i in indices:
                    z=reference(fetch(bags[i],matched['angles'][i]));y=bags[i].label
                    if variant=='G0':loss=F.cross_entropy(z,torch.full((len(z),),y,dtype=torch.long))
                    elif variant=='G1':loss=-torch.log(torch.softmax(z,1).mean(0)[y])
                    else:loss=F.cross_entropy(z.mean(0)[None],torch.tensor([y]))
                    losses.append(loss)
                torch.stack(losses).mean().backward();nn.utils.clip_grad_norm_(reference.parameters(),5.);b.step()
            self.assertEqual(result['optimizer_updates'],2)
            for x,y in zip(actual.parameters(),reference.parameters()):torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)

    def test_adamw_groups_defaults_and_original_schedule(self):
        model=Toy();optimizer=v2.make_optimizer(model,core.base_config('cpu'))
        self.assertEqual(optimizer.defaults['betas'],(.9,.999));self.assertEqual(optimizer.defaults['eps'],1e-8)
        for group in optimizer.param_groups:
            for p in group['params']:self.assertEqual(group['weight_decay'],.05 if p.ndim>1 else 0.)
        for epoch,expected in ((1,1e-4),(3,3e-4),(4,3e-4),(13,.00021997611422),(30,1e-6)):
            v2.set_epoch(model,optimizer,core.base_config('cpu'),epoch)
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'],expected,places=13)
        self.assertGreater(optimizer.param_groups[0]['lr'],0)

    def check_recovery(self,variant):
        train=core.group_records(rows(),'train');val=core.group_records(rows('validation'),'validation')
        device=torch.device('cpu');env={'synthetic':True};expected={'synthetic_config':variant,'aggregation':core.NATIVE[variant]}
        def fresh():
            v2.seed_everything(42);m=Toy(stochastic=True);o=v2.make_optimizer(m,core.base_config('cpu'))
            return m,o,engine.initial_state(m,o,expected,env,core.tensor_digest(m.state_dict()))
        m,o,whole=fresh()
        for epoch in range(1,6):engine.advance(whole,m,o,train,val,fetch,fetch,device,variant,epoch)
        expected_rng=v2.rng_state()
        with tempfile.TemporaryDirectory(prefix='g012 recovery with spaces ') as directory:
            folder=Path(directory);m,o,partial=fresh()
            for epoch in range(1,3):
                engine.advance(partial,m,o,train,val,fetch,fetch,device,variant,epoch)
                v2.save_state(folder/'PRIVATE_resume.pt',partial)
            now=[0.];clock=lambda:now[0]
            budget=LeaseBudget(folder/'active_budget.json',cap=100.,clock=clock);now[0]=7.;budget.reserve()
            charged=st.read(folder/'active_budget.json')['charged_seconds']
            # Repeated partial work changes model/global RNG; never commit it.
            loss=core.scan_loss(m(fetch(train[0])),train[0].label,variant);loss.backward();o.step()
            resumed_model,resumed_optimizer,_=fresh()
            resumed=st.recover_state(folder,expected,resumed_model,resumed_optimizer,env)
            again=LeaseBudget(folder/'active_budget.json',cap=100.,clock=clock)
            self.assertGreaterEqual(again.base,charged)
            for epoch in range(3,6):engine.advance(resumed,resumed_model,resumed_optimizer,train,val,fetch,fetch,device,variant,epoch)
            same(self,whole,resumed);same(self,expected_rng,v2.rng_state())
            again.finish('SYNTHETIC_FINISHED')
            self.assertGreaterEqual(st.read(folder/'active_budget.json')['charged_seconds'],charged)

    def test_G0_epoch_recovery_model_optimizer_rng_history_selection_and_partial_charge(self):self.check_recovery('G0')
    def test_G1_epoch_recovery_model_optimizer_rng_history_selection_and_partial_charge(self):self.check_recovery('G1')
    def test_G2_epoch_recovery_model_optimizer_rng_history_selection_and_partial_charge(self):self.check_recovery('G2')

    def test_gpu_warning_and_headroom_guard_with_mock_device_only(self):
        device=torch.device('cuda')
        memory={'peak_allocated_gib':4.6,'peak_reserved_gib':4.8}
        with patch.object(v2,'reset_memory_peak'),patch.object(v2,'memory_info',return_value=memory),patch.object(torch.cuda,'mem_get_info',return_value=(5*2**30,12*2**30)):
            result=engine.Resources(device)
            self.assertTrue(result.value['warning_allocated_above_4_5_gib'])
        with patch.object(v2,'reset_memory_peak'),patch.object(v2,'memory_info',return_value=memory),patch.object(torch.cuda,'mem_get_info',return_value=(1*2**30,12*2**30)):
            with self.assertRaisesRegex(dataset.DataError,'HEADROOM_BLOCKER'):engine.Resources(device)

    def test_strict_scalar_shapes_keys_dtype_and_aggregation_config(self):
        model=Toy();state=v2.tensor_state(model)
        for kind in ('scalar','dtype','missing','extra','nan'):
            bad=deepcopy(state)
            if kind=='scalar':bad['scalar']=torch.ones(1)
            elif kind=='dtype':bad['scalar']=bad['scalar'].double()
            elif kind=='missing':bad.pop('scalar')
            elif kind=='extra':bad['extra']=torch.zeros(1)
            else:bad['scalar']=torch.tensor(float('nan'))
            with self.assertRaises(dataset.DataError):core.strict_state(model,bad)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);o=v2.make_optimizer(model,core.base_config('cpu'))
            saved=engine.initial_state(model,o,{'mode':'G2','native':core.NATIVE['G2']},{'cpu':True},'x')
            v2.save_state(folder/'PRIVATE_resume.pt',saved)
            with self.assertRaises(dataset.DataError):st.recover_state(folder,{'mode':'G2','native':core.NATIVE['G1']},model,o,{'cpu':True})

    def test_previous_state_fallback_and_no_silent_fresh_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);m=Toy();o=v2.make_optimizer(m,core.base_config('cpu'))
            state=engine.initial_state(m,o,{'x':1},{'cpu':True},'x')
            v2.save_state(folder/'PRIVATE_resume.pt',state);v2.save_state(folder/'PRIVATE_resume.pt',state)
            (folder/'PRIVATE_resume.pt').write_bytes(b'corrupt retained evidence')
            st.recover_state(folder,{'x':1},m,o,{'cpu':True})
            self.assertEqual(len(list(folder.glob('PRIVATE_unreadable-*.pt'))),1)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):st.recover_state(Path(tmp),{},m,o,{'cpu':True})

    def test_earliest_tie_and_minimum_run_length(self):
        train=core.group_records(rows(),'train');val=core.group_records(rows('validation'),'validation')
        m=Toy();o=v2.make_optimizer(m,core.base_config('cpu'));s=engine.initial_state(m,o,{}, {},'x')
        metrics,pred=core.aggregate(val,[m(fetch(b)).detach() for b in val],'G0')
        with patch.object(core,'train_epoch',return_value={'training_objective':.5,'optimizer_updates':2,'training_scans':3,'max_preclip_gradient_norm':1.}),patch.object(core,'predict',return_value=(metrics,pred)):
            for epoch in range(1,13):
                engine.advance(s,m,o,train,val,fetch,fetch,torch.device('cpu'),'G0',epoch)
                if epoch<12:self.assertFalse(s['completed'])
        self.assertTrue(s['completed']);self.assertEqual(s['best']['epoch'],1)

    def test_replay_tolerance_and_unchanged_classes(self):
        bags=core.group_records(rows('validation'),'validation');m=Toy()
        _,a=core.aggregate(bags,[m(fetch(b)).detach() for b in bags],'G2');b=deepcopy(a)
        self.assertTrue(core.replay_difference(a,b)['patients']['classes_unchanged'])
        b['slices'][0]['p_ad']+=3e-5
        with self.assertRaises(dataset.DataError):core.replay_difference(a,b)
        one={'slices':[{'image_path':'x','label':0,'p_ad':.499999}], 'scans':[{'scan_id':'x','label':0,'p_ad':.499999}], 'patients':[{'subject_id':'x','label':0,'p_ad':.499999}]}
        two=deepcopy(one);two['patients'][0]['p_ad']=.500001
        with self.assertRaises(dataset.DataError):core.replay_difference(one,two)


class CommandAndExportTests(unittest.TestCase):
    def test_preflight_git_scope_uses_repository_root(self):
        # Read-only source check; no training/data calls or Git mutations.
        repository=st.BASE.parents[1]
        name='recognition/ConvNeXt_Alzheimers/modules.py'
        process=subprocess.run(['git','-C',str(repository),'ls-files','--error-unmatch','--',name],capture_output=True,text=True)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertEqual(process.stdout.strip(),name)

    def supervised_fixture(self,root):
        plan={'sources':{}};bundle=SimpleNamespace(manifest_hashes={})
        for variant in core.VARIANTS:
            folder=root/('smoke-'+variant);folder.mkdir()
            (folder/'PRIVATE_smoke.pt').write_bytes(b'synthetic_not_model_data')
            st.immutable_json(folder/'smoke.json',{'passed':True,'identity':engine.identity(plan,variant,{}),
                'checkpoint_sha256':st.sha(folder/'PRIVATE_smoke.pt')})
        st.immutable_json(root/'a1_historical_regression.json',{'passed':True,'plan_sha256':core.digest(plan)})
        return plan,bundle,SimpleNamespace(study_root=root,data_root=root/'fake data',splits_dir=root/'fake splits',variant='G0')

    def test_completed_supervisor_skips_without_new_child_or_budget(self):
        with tempfile.TemporaryDirectory(prefix='g012 completed spaces ') as tmp:
            root=Path(tmp);plan,bundle,args=self.supervised_fixture(root);folder=root/'G0';folder.mkdir()
            st.immutable_json(folder/'attempt.json',{'identity':engine.identity(plan,'G0',{})})
            st.immutable_json(folder/'completion.json',{})
            st.immutable_json(folder/'active_budget.json',{'charged_seconds':123.})
            before=(folder/'active_budget.json').read_bytes()
            with patch.object(st,'context',return_value=(plan,root,{},bundle,{})),patch.object(st,'verify_completion'),patch.object(engine.subprocess,'Popen') as child:
                engine.supervised(args);self.assertFalse(child.called)
            self.assertEqual(before,(folder/'active_budget.json').read_bytes())

    def test_supervisor_classifies_completion_after_deadline_inconclusive(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,bundle,args=self.supervised_fixture(root)
            ledger=SimpleNamespace(deadline=10.,reserve=lambda:True,finish=Mock())
            process=SimpleNamespace(poll=Mock(side_effect=[None,0,0]),returncode=0,wait=Mock(),kill=Mock())
            job=SimpleNamespace(close=Mock())
            with patch.object(st,'context',return_value=(plan,root,{},bundle,{})),patch.object(st,'check_disk'),patch.object(st,'verify_completion'),patch.object(engine,'Budget',return_value=ledger),patch.object(engine,'WindowsJob',return_value=job),patch.object(engine.subprocess,'Popen',return_value=process),patch.object(engine.time,'monotonic',return_value=11.),patch.object(engine.time,'sleep'):
                with self.assertRaises(dataset.DataError):engine.supervised(args)
            self.assertEqual(st.read(root/'G0'/'run_status.json')['status'],'WALL_LIMIT_INCONCLUSIVE')
            ledger.finish.assert_called_once_with('WALL_LIMIT_INCONCLUSIVE')

    @unittest.skipUnless(sys.platform=='win32','Windows kill-on-close helper')
    def test_existing_windows_job_kills_synthetic_child(self):
        process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            job=WindowsJob(process);self.assertIsNone(process.poll())
            job.close();process.wait(timeout=10)
            # Windows kill-on-close may report exit code zero. Termination
            # within ten seconds of a thirty-second child is the assertion.
            self.assertIsNotNone(process.returncode)
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=10)

    def test_parser_help_and_fixed_variants_no_extra_seeds(self):
        from g012.run import parser
        args=parser().parse_args(['smoke','--data-root','C:/fake path/data','--splits-dir','C:/fake path/splits','--study-root','C:/fake path/study','--variant','G2'])
        self.assertEqual(str(args.data_root),'C:\\fake path\\data' if sys.platform=='win32' else 'C:/fake path/data')
        result=subprocess.run([sys.executable,'-B',str(st.BASE/'g012/run.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0);self.assertIn('smoke',result.stdout)
        self.assertNotIn('--seed',parser().format_help())

    def test_export_privacy_guard(self):
        for value in ({'scan_id':'secret'},{'x':'C:/private/folder'},{'x':'001_S_1234'},{'model_state':{}},{'predictions':[]}):
            with self.assertRaises(dataset.DataError):results.validate_public(value)
        results.validate_public({'metrics':{'accuracy':.5},'sources':{'g012/core.py':'hash'}})

    def test_screen_g0_eligible_no_two_patient_margin_and_nll_not_gate(self):
        row={'selected_score':{'correct':86,'AD_detections':22,'macro_f1':.81,'log_loss':100.},
             'protocol_complete':True,'integrity_and_replay_passed':True}
        self.assertTrue(results.screen(row,.8012151782643586)['passes_development_screen'])
        row['protocol_complete']=False
        self.assertFalse(results.screen(row,.8012151782643586)['passes_development_screen'])

    def test_incomplete_compare_and_bundle_allowlist_with_space_paths(self):
        bags=core.group_records(rows('validation'),'validation');m=Toy()
        metrics,pred=core.aggregate(bags,[m(fetch(b)).detach() for b in bags],'G0')
        history=[{'epoch':1.,'patient_accuracy':metrics['patient']['accuracy'],'patient_macro_f1':metrics['patient']['macro_f1'],'patient_log_loss':metrics['patient']['log_loss']}]
        plan={'a1_metrics':metrics,'configurations':{v:core.config(v) for v in core.VARIANTS}}
        with tempfile.TemporaryDirectory(prefix='g012 export spaces ') as tmp:
            root=Path(tmp);args=SimpleNamespace(study_root=root)
            st.immutable_json(root/'PRIVATE_a1_reference.json',{'predictions':pred,'history':history,'metrics':metrics,'completed_epoch':1,'parameters':9,'recorded_resources':{}})
            st.immutable_json(root/'PRIVATE_matched_schedule.json',{'initial_tensor_sha256':'x','epochs':[]})
            with patch.object(st,'context',return_value=(plan,root,{},None,None)):
                path=results.bundle(args);again=results.bundle(args)
                self.assertEqual(path,again)
                import zipfile
                with zipfile.ZipFile(path) as archive:
                    self.assertEqual(tuple(archive.namelist()),results.ALLOWLIST)
                    result=json.loads(archive.read('G012_STUDY_SUMMARY.json'))
                    self.assertFalse(result['all_three_protocols_completed'])
                    self.assertEqual([r['status'] for r in result['variants']],['NOT_STARTED']*3)
                    results.validate_public(result)

    def test_completed_all_variant_exports_native_replay_and_no_overwrite(self):
        train=core.group_records(rows(),'train');val=core.group_records(rows('validation'),'validation')
        v2.seed_everything(42);baseline=Toy();metrics,pred=core.aggregate(val,[baseline(fetch(b)).detach() for b in val],'G0')
        initial=core.tensor_digest(baseline.state_dict())
        plan={'sources':{},'a1_metrics':metrics,'configurations':{v:core.config(v) for v in core.VARIANTS}}
        history=[{'epoch':1.,'patient_accuracy':metrics['patient']['accuracy'],'patient_macro_f1':metrics['patient']['macro_f1'],'patient_log_loss':metrics['patient']['log_loss']}]
        with tempfile.TemporaryDirectory(prefix='g012 completed export spaces ') as tmp:
            root=Path(tmp);args=SimpleNamespace(study_root=root);bundle=SimpleNamespace(manifest_hashes={})
            st.immutable_json(root/'PRIVATE_a1_reference.json',{'predictions':pred,'history':history,'metrics':metrics,'completed_epoch':1,'parameters':9,'recorded_resources':{}})
            matched=core.schedule(train,1)
            st.immutable_json(root/'PRIVATE_matched_schedule.json',{'initial_tensor_sha256':initial,'epochs':[{'order_digest':matched['order_digest'],'augmentation_digest':matched['augmentation_digest']}]})
            st.immutable_json(root/'a1_historical_regression.json',{'passed':True,'plan_sha256':core.digest(plan)})
            for variant in core.VARIANTS:
                folder=root/variant;folder.mkdir();v2.seed_everything(42);model=Toy();optimizer=v2.make_optimizer(model,core.base_config('cpu'))
                expected=engine.identity(plan,variant,{})
                state=engine.initial_state(model,optimizer,expected,{'synthetic':True},initial)
                engine.advance(state,model,optimizer,train,val,fetch,fetch,torch.device('cpu'),variant,1)
                state.update(completed=True,stop_reason='SYNTHETIC_REPORT_FIXTURE',epoch_loop_seconds=[.01])
                engine.finish(folder,state)
                actual=state['best']['predictions']
                st.immutable_json(folder/'PRIVATE_replay_predictions.json',actual)
                st.immutable_json(folder/'checkpoint_replay.json',{'identity':expected,'passed':True,
                    'checkpoint_sha256':st.sha(folder/'PRIVATE_selected.pt'),'predictions_sha256':st.sha(folder/'PRIVATE_replay_predictions.json'),
                    'absolute_probability_tolerance':2e-5,'native_aggregation':core.NATIVE[variant],
                    'checks':core.replay_difference(actual,actual),'metrics':state['best']['metrics'],'resources':{},'inference_seconds':.1})
                st.immutable_json(folder/'run_status.json',{'status':'COMPLETED_TRAINING'})
                st.immutable_json(folder/'active_budget.json',{'charged_seconds':10.,'sessions':[{'status':'SYNTHETIC'}]})
            with patch.object(st,'context',return_value=(plan,root,{},bundle,{})):
                path=results.bundle(args);self.assertEqual(path,results.bundle(args))
                import zipfile
                with zipfile.ZipFile(path) as archive:
                    result=json.loads(archive.read('G012_STUDY_SUMMARY.json'))
                    self.assertTrue(result['all_three_protocols_completed'])
                    self.assertEqual(result['variants'][2]['replay']['native_aggregation'],core.NATIVE['G2'])
                    self.assertEqual(sum(result['variants'][1]['versus_G0']['paired_correctness_counts'].values()),3)
                    results.validate_public(result)
                with zipfile.ZipFile(path,'a') as archive:archive.writestr('unapproved.txt','tamper')
                with self.assertRaises(dataset.DataError):results.bundle(args)

    def test_completed_receipt_skip_verification_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);p=folder/'history.csv';p.write_text('synthetic')
            value={'identity':{'x':1},'hashes':{'history.csv':st.sha(p)}}
            st.immutable_json(folder/'completion.json',value);st.verify_completion(folder,{'x':1})
            p.write_text('changed')
            with self.assertRaises(dataset.DataError):st.verify_completion(folder,{'x':1})

    def test_concurrent_budget_reads_serialize_existing_atomic_writer(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'live.json';ledger=engine.Budget(path);errors=[]
            def writer():
                try:
                    for i in range(60):ledger.reserve()
                except BaseException as exc:errors.append(exc)
            thread=threading.Thread(target=writer);thread.start()
            for i in range(200):self.assertIn('charged_seconds',st.read_budget(path))
            thread.join();self.assertFalse(errors)

    def test_budget_cap_and_crash_reservation_no_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'budget.json';now=[0.];clock=lambda:now[0]
            first=LeaseBudget(path,cap=20.,clock=clock);now[0]=8.;first.reserve()
            charged=st.read(path)['charged_seconds'];second=LeaseBudget(path,cap=20.,clock=clock)
            self.assertEqual(second.base,charged);now[0]=30.
            self.assertFalse(second.reserve());second.finish('WALL_LIMIT_INCONCLUSIVE')
            self.assertEqual(st.read(path)['charged_seconds'],20.)
            with self.assertRaises(dataset.DataError):LeaseBudget(path,cap=20.,clock=clock)


if __name__=='__main__':unittest.main(verbosity=2)
