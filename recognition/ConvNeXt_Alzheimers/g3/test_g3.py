"""Synthetic only. No dataset pixels, trained weights, or study execution.

Declared before execution: float64 algebra/gradients atol=rtol=1e-10;
float32 algebra/forward/loss atol=2e-7, rtol=2e-6; full shared gradients
atol=2e-6, rtol=2e-5. Exact equality for initialization/RNG/recovery/updates.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
import dataset
import train_convnext_v2 as v2
from train_schedule_only import LeaseBudget
from g012 import core,engine
from g012.test_g012 import rows,fetch,same
from g3 import model as gm,operations as ops,storage as st,execution as ex,results
torch.set_num_threads(2)


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.feature=nn.Linear(3,768);self.head=nn.Linear(768,2)
        self.scalar=nn.Parameter(torch.tensor(1.));self.attention=gm.Attention()
    def forward(self,x):
        h=self.feature(x)*self.scalar;z=self.head(h);a=self.attention(h)
        return z,(a[:,None]*z).sum(0),a


class ModelTests(unittest.TestCase):
    def test_full_initial_state_count_rng_and_exact_feature_head_stochastic_path(self):
        v2.seed_everything(42);original=v2.make_model(core.base_config('cpu'))
        self.assertEqual(core.tensor_digest(original.state_dict()),st.INITIAL)
        before=v2.rng_state();candidate=gm.GatedScan(deepcopy(original));same(self,before,v2.rng_state())
        self.assertEqual(sum(p.numel() for p in original.parameters()),27818594)
        self.assertEqual(sum(p.numel() for p in candidate.parameters()),28015586)
        for name,p in original.state_dict().items():self.assertTrue(torch.equal(p,candidate.state_dict()[name]))
        captured=[];handle=original.head.register_forward_hook(lambda m,i,o:captured.append(i[0]))
        x=torch.randn(2,1,224,224);original.train();candidate.train()
        torch.manual_seed(909);expected=original(x);rng=v2.rng_state()
        torch.manual_seed(909);actual,scan,a=candidate(x);same(self,rng,v2.rng_state())
        handle.remove();self.assertEqual(captured[0].shape,(2,768));self.assertTrue(torch.equal(expected,actual))
        torch.testing.assert_close(scan,original.head(captured[0].mean(0)),atol=2e-7,rtol=2e-6)
        left=core.scan_loss(expected,1,'G2');right=gm.loss(scan,1)
        torch.testing.assert_close(left,right,atol=2e-7,rtol=2e-6)
        left.backward();right.backward()
        for name,p in original.named_parameters():
            torch.testing.assert_close(p.grad,dict(candidate.named_parameters())[name].grad,atol=2e-6,rtol=2e-5)
        self.assertGreater(float(candidate.attention.w.weight.grad.abs().sum()),0.)

    def test_float64_uniform_feature_logit_algebra_and_gradients(self):
        v2.seed_everything(123);h=torch.randn(5,768,dtype=torch.float64,requires_grad=True)
        head=nn.Linear(768,2).double();z=head(h)
        a=gm.Attention().double()(h);self.assertTrue(torch.equal(a,torch.full_like(a,.2)))
        lhs=(a[:,None]*z).sum(0);rhs=head(h.mean(0))
        torch.testing.assert_close(lhs,rhs,atol=1e-10,rtol=1e-10)
        gl=torch.autograd.grad(gm.loss(lhs,0),(h,*head.parameters()),retain_graph=True)
        gr=torch.autograd.grad(gm.loss(rhs,0),(h,*head.parameters()))
        for a,b in zip(gl,gr):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)

    def test_uniform_reference_shared_gradients_clipping_and_AdamW_update(self):
        v2.seed_everything(57)
        original=nn.ModuleDict({'feature':nn.Linear(3,768),'head':nn.Linear(768,2)}).double()
        reference=deepcopy(original);config=core.base_config('cpu')
        a=v2.make_optimizer(original,config);b=v2.make_optimizer(reference,config)
        x=torch.randn(5,3,dtype=torch.float64)
        # Production historical G2 path versus synthetic uniform feature/head
        # algebra. No attention constructor or parameters in either optimiser.
        z=original['head'](original['feature'](x))
        uniform=reference['head'](reference['feature'](x).mean(0))
        left=core.scan_loss(z,1,'G2');right=gm.loss(uniform,1)
        torch.testing.assert_close(left,right,atol=1e-10,rtol=1e-10)
        left.backward();right.backward()
        for p,q in zip(original.parameters(),reference.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-10)
        for net,opt in ((original,a),(reference,b)):
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
        for p,q in zip(original.parameters(),reference.parameters()):
            torch.testing.assert_close(p,q,atol=1e-10,rtol=1e-10)

    def test_attention_nonuniform_fixture_and_permutation_and_k1(self):
        a=gm.Attention().double();h=torch.randn(7,768,dtype=torch.float64)
        self.assertAlmostEqual(float(a(h).sum()),1.)
        with torch.no_grad():a.w.weight.fill_(2.)
        z=torch.arange(14,dtype=torch.float64).reshape(7,2).pow(2)
        weights=a(h);pooled=(weights[:,None]*z).sum(0)
        self.assertGreater(float((pooled-z.mean(0)).abs().max()),1e-6)
        order=torch.tensor([2,6,0,5,1,3,4])
        torch.testing.assert_close(pooled,(a(h[order])[:,None]*z[order]).sum(0),atol=1e-10,rtol=1e-10)
        self.assertTrue(torch.equal(a(h[:1]),torch.ones(1,dtype=torch.float64)))
        d=ops.diagnostics(torch.ones(1));self.assertEqual(d['normalised_entropy'],1.);self.assertEqual(d['effective_slices'],1.)
        with self.assertRaises(dataset.DataError):a(h[:0])

    def test_extreme_finite_attention_and_NC_AD_loss(self):
        a=gm.Attention();h=(torch.randn(4,768)*10000).requires_grad_()
        with torch.no_grad():a.w.weight.fill_(10000)
        weights=a(h);z=torch.tensor([[10000.,-10000.],[-10000.,10000.],[10000.,9999.],[-5.,7.]])
        self.assertTrue(torch.isfinite(weights).all());self.assertAlmostEqual(float(weights.sum()),1.,places=6)
        for label in (0,1):
            value=gm.loss((weights[:,None]*z).sum(0),label)
            torch.testing.assert_close(value,F.cross_entropy(((weights[:,None]*z).sum(0))[None],torch.tensor([label])))
            value.backward(retain_graph=True);self.assertTrue(torch.isfinite(h.grad).all())

    def test_zero_w_then_VU_gradient_progression(self):
        v2.seed_everything(12);m=Toy();x=torch.randn(4,3);o=torch.optim.SGD(m.parameters(),lr=.01)
        _,z,a=m(x);gm.loss(z,1).backward()
        for layer in (m.attention.V,m.attention.U):self.assertEqual(float(layer.weight.grad.abs().sum()),0.)
        self.assertGreater(float(m.attention.w.weight.grad.abs().sum()),0.)
        self.assertGreater(float(m.feature.weight.grad.abs().sum()),0.)
        self.assertGreater(float(m.head.weight.grad.abs().sum()),0.)
        o.step();o.zero_grad();_,z,_=m(x);gm.loss(z,1).backward()
        for layer in (m.attention.V,m.attention.U):self.assertGreater(float(layer.weight.grad.abs().sum()),0.)


class WorkflowTests(unittest.TestCase):
    def test_pair_tail_optimizer_groups_and_global_clip_reference(self):
        v2.seed_everything(18);m=Toy();reference=deepcopy(m);c=core.base_config('cpu')
        o=v2.make_optimizer(m,c);r=v2.make_optimizer(reference,c)
        bags=core.group_records(rows(),'train');matched=core.schedule(bags,1)
        result=ops.train_epoch(m,bags,lambda b,a:fetch(b,a)*100,o,torch.device('cpu'),'G3',matched)
        r.zero_grad(set_to_none=True);norms=[]
        for pos,index in enumerate(matched['order']):
            b=bags[index];_,z,_=reference(fetch(b,matched['angles'][index])*100)
            (F.cross_entropy(z[None],torch.tensor([b.label]))/min(2,len(bags)-(pos//2)*2)).backward()
            if (pos+1)%2==0 or pos==len(bags)-1:
                norms.append(float(torch.nn.utils.clip_grad_norm_(reference.parameters(),5.,error_if_nonfinite=True)));r.step();r.zero_grad(set_to_none=True)
        same(self,m.state_dict(),reference.state_dict());same(self,o.state_dict(),r.state_dict())
        self.assertEqual(result['optimizer_updates'],2);self.assertEqual(result['clipped_update_fraction'],sum(n>5 for n in norms)/2)
        groups={id(p):g for g in o.param_groups for p in g['params']}
        for name,p in m.named_parameters():
            self.assertEqual(groups[id(p)]['weight_decay'],.05 if p.ndim>1 else 0.)
            self.assertEqual(groups[id(p)]['betas'],(.9,.999));self.assertEqual(groups[id(p)]['eps'],1e-8)

    def test_validation_rng_and_native_weights_replay(self):
        v2.seed_everything(42);m=Toy();bags=core.group_records(rows('validation'),'validation');before=v2.rng_state()
        metrics,pred=ops.predict(m,bags,fetch,torch.device('cpu'),'G3');same(self,before,v2.rng_state())
        ops.verify_native(pred)
        self.assertTrue(ops.replay_difference(pred,deepcopy(pred))['attention']['slice_order_unchanged'])
        bad=deepcopy(pred);bad['scans'][0]['attention_weights'][0]+=3e-5
        with self.assertRaises(dataset.DataError):ops.replay_difference(pred,bad)
        bad=deepcopy(pred);bad['scans'][0]['slice_order'].reverse()
        with self.assertRaises(dataset.DataError):ops.replay_difference(pred,bad)

    def test_strict_schema_scalar_dtype_source_feature_keys_and_wrong_G2(self):
        m=Toy();identity=ex.identity({'sources':{}},'G3',{})
        value={'format_version':'G3-1','identity':identity,'epoch':1,'initial_tensor_sha256':'x','model_state':v2.tensor_state(m)}
        gm.strict_checkpoint(m,value,identity)
        for kind in ('schema','format','source','feature','scalar','dtype','nan','missing','extra'):
            bad=deepcopy(value)
            if kind=='schema':bad['extra']=1
            elif kind=='format':bad['format_version']='G012-1'
            elif kind=='source':bad['identity']['source_sha256']={'changed':'x'}
            elif kind=='feature':bad['identity']['configuration']['feature']='wrong'
            elif kind=='scalar':bad['model_state']['scalar']=torch.ones(1)
            elif kind=='dtype':bad['model_state']['scalar']=torch.tensor(1.,dtype=torch.float64)
            elif kind=='nan':bad['model_state']['scalar']=torch.tensor(float('nan'))
            elif kind=='missing':bad['model_state'].pop('attention.w.weight')
            else:bad['model_state']['surprise']=torch.ones(1)
            with self.assertRaises(dataset.DataError):gm.strict_checkpoint(m,bad,identity)
        uniform=nn.Linear(3,2)
        with self.assertRaises(dataset.DataError):core.strict_state(uniform,m.state_dict())

    def test_G3_atomic_epoch_recovery_rng_history_selection_and_partial_budget(self):
        train=core.group_records(rows(),'train');val=core.group_records(rows('validation'),'validation');device=torch.device('cpu')
        def fresh():
            v2.seed_everything(42);m=Toy();o=v2.make_optimizer(m,core.base_config('cpu'))
            return m,o,engine.initial_state(m,o,{'fixed':'G3'},{'cpu':True},'x')
        m,o,whole=fresh()
        for epoch in range(1,5):engine.advance(whole,m,o,train,val,fetch,fetch,device,'G3',epoch,operations=ops)
        final_rng=v2.rng_state()
        with tempfile.TemporaryDirectory(prefix='g3 recovery spaces ') as tmp:
            folder=Path(tmp);m,o,state=fresh()
            for epoch in (1,2):engine.advance(state,m,o,train,val,fetch,fetch,device,'G3',epoch,operations=ops)
            v2.save_state(folder/'PRIVATE_resume.pt',state)
            now=[0.];ledger=LeaseBudget(folder/'active_budget.json',cap=100.,clock=lambda:now[0]);now[0]=7.;ledger.reserve()
            charged=st.read(folder/'active_budget.json')['charged_seconds']
            _,z,_=m(fetch(train[0]));gm.loss(z,0).backward();o.step()
            m,o,_=fresh();state=st.recover_state(folder,{'fixed':'G3'},m,o,{'cpu':True})
            again=LeaseBudget(folder/'active_budget.json',cap=100.,clock=lambda:now[0]);self.assertGreaterEqual(again.base,charged)
            for epoch in (3,4):engine.advance(state,m,o,train,val,fetch,fetch,device,'G3',epoch,operations=ops)
            same(self,whole,state);same(self,final_rng,v2.rng_state())

    def test_export_privacy_and_fixed_no_U3_or_extra_seed_parser(self):
        from g3.run import parser
        for value in ({'scan_id':'x'},{'path':'C:/Users/private'},{'attention_weights':[1.]}):
            with self.assertRaises(dataset.DataError):results.privacy(value)
        for variant in ('G2','U3','G4'):
            with self.assertRaises(SystemExit):parser().parse_args(['run','--variant',variant])
        self.assertNotIn('PRIVATE_selected.pt',results.ALLOWLIST)

    def test_runtime_supervisor_completed_skip_no_child_no_new_budget(self):
        with tempfile.TemporaryDirectory(prefix='g3 completed spaces ') as tmp:
            root=Path(tmp);folder=root/'G3';folder.mkdir();plan={'sources':{}};bundle=SimpleNamespace(manifest_hashes={})
            expected=ex.identity(plan,'G3',{})
            st.immutable_json(folder/'attempt.json',{'identity':expected});st.immutable_json(folder/'completion.json',{})
            st.immutable_json(folder/'active_budget.json',{'charged_seconds':123.})
            args=SimpleNamespace(variant='G3',study_root=root,data_root=root/'data',splits_dir=root/'splits')
            before=(folder/'active_budget.json').read_bytes()
            with patch.object(st,'context',return_value=(plan,root,{},bundle,{})),patch.object(st,'verify_completion'),patch.object(ex,'ready'),patch.object(engine.subprocess,'Popen') as child:
                ex.supervised(args);self.assertFalse(child.called)
            self.assertEqual(before,(folder/'active_budget.json').read_bytes())

    def test_training_split_rejection_before_fetch_and_conflicting_immutable_evidence(self):
        m=Toy();o=v2.make_optimizer(m,core.base_config('cpu'))
        bags=core.group_records(rows('validation'),'validation')
        with self.assertRaises(dataset.DataError):
            ops.train_epoch(m,bags,lambda *a:self.fail('must not fetch'),o,torch.device('cpu'),'G3',core.schedule(bags,1))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'receipt.json';st.immutable_json(p,{'fixed':1})
            with self.assertRaises(dataset.DataError):st.immutable_json(p,{'fixed':2})

    def test_supervisor_reserves_replay_and_passes_space_paths_to_native_child(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory(prefix='g3 parent spaces ') as tmp:
            root=Path(tmp);plan={'sources':{}};bundle=SimpleNamespace(manifest_hashes={})
            args=SimpleNamespace(variant='G3',study_root=root,data_root=root/'fake data',splits_dir=root/'fake splits',
                                 historical_root=root/'historical evidence',historical_source=root/'historical code')
            ledger=SimpleNamespace(deadline=400.,reserve=lambda:True,finish=Mock())
            child=SimpleNamespace(poll=Mock(side_effect=[None,0]),wait=Mock(),kill=Mock(),returncode=0)
            with patch.object(st,'context',return_value=(plan,root,{},bundle,{})),patch.object(st,'check_disk'),patch.object(ex,'ready'),\
                 patch.object(engine,'Budget',return_value=ledger),patch.object(engine,'WindowsJob'),\
                 patch.object(engine.time,'monotonic',return_value=101.),patch.object(engine.subprocess,'Popen',return_value=child) as native:
                with self.assertRaisesRegex(dataset.DataError,'Stopped at a blocker'):ex.supervised(args)
                command=native.call_args.args[0]
                self.assertEqual(command[2],str(st.BASE/'g3'/'run.py'))
                self.assertEqual(command[command.index('--historical-source')+1],str(args.historical_source))
                self.assertEqual(st.read(root/'G3'/'run_status.json')['status'],'WALL_LIMIT_INCONCLUSIVE')
                ledger.finish.assert_called_once_with('WALL_LIMIT_INCONCLUSIVE')

    def test_G3_complete_and_incomplete_aggregate_bundle_allowlist_with_space_paths(self):
        import zipfile
        from g012.engine import identity as old_identity
        v2.seed_everything(42);m=Toy();bags=core.group_records(rows('validation'),'validation')
        metrics,pred=ops.predict(m,bags,fetch,torch.device('cpu'),'G3')
        base={k:deepcopy(v) for k,v in metrics.items() if k in ('slice','scan','patient')}
        # This artificial A1 reference exercises the exact screen comparison;
        # real execution binds the actual 102-patient historical file by hash.
        a1=deepcopy(base);a1['patient']['macro_f1']=.8012151782643586
        with tempfile.TemporaryDirectory(prefix='g3 aggregate spaces ') as tmp:
            root=Path(tmp);histroot=root/'historical evidence';histroot.mkdir()
            plan={'sources':{},'configuration':gm.configuration(),'control_mode':'REUSE_G2','a1_configuration':{'synthetic':True}}
            bundle=SimpleNamespace(manifest_hashes={})
            args=SimpleNamespace(study_root=root,historical_root=histroot)
            history=[{'epoch':e,'patient_accuracy':base['patient']['accuracy'],'patient_macro_f1':base['patient']['macro_f1'],
                'patient_log_loss':base['patient']['log_loss'],'patient_brier':base['patient']['brier'],
                'ad_recall':base['patient']['per_class']['AD']['recall'],'nc_recall':base['patient']['per_class']['NC']['recall'],
                'selected_as_best':e==1} for e in range(1,13)]
            schedules=[{'order_digest':'synthetic','augmentation_digest':'synthetic','epoch':e} for e in range(1,13)]
            st.immutable_json(root/'PRIVATE_matched_schedule.json',{'epochs':schedules})
            st.immutable_json(root/'PRIVATE_a1_reference.json',{'predictions':pred,'metrics':a1,'history':history,
                'completed_epoch':21,'parameters':27818594,'recorded_resources':{}})
            st.immutable_json(root/'preparation.json',{'execution_seconds':1.,'images_decoded':0})
            st.immutable_json(root/'initialization.json',{'shared_digest':'synthetic','checkpoint_sha256':'synthetic'})
            replay={'passed':True,'checks':{'synthetic':True},'absolute_probability_tolerance':2e-5,'inference_seconds':1.,'resources':{}}
            for name,folder in [('G2',histroot/'G2'),('G3',root/'G3')]:
                folder.mkdir();expected=old_identity(plan,'G2',{}) if name=='G2' else ex.identity(plan,'G3',{})
                st.immutable_json(folder/'PRIVATE_selected_predictions.json',pred)
                st.immutable_json(folder/'training_summary.json',{'identity':expected,'metrics':metrics,'selected_epoch':1,
                    'completed_epoch':12,'stop_reason':'PATIENCE_AFTER_MINIMUM_12','committed_epoch_loop_seconds':10.})
                st.write_text_atomic(folder/'history.csv',st.history_csv(history))
                st.immutable_json(folder/'PRIVATE_order_augmentation_digests.json',schedules)
                st.immutable_json(folder/'completion.json',{'identity':expected,'hashes':{}})
                st.immutable_json(folder/'active_budget.json',{'charged_seconds':11.,'sessions':[]})
                st.immutable_json(folder/'run_status.json',{'status':'COMPLETED_TRAINING'})
                st.immutable_json(folder/'checkpoint_replay.json',replay)
            with patch.object(st,'context',return_value=(plan,root,{},bundle,{})),patch('g012.engine.verify_replay',return_value=replay),patch.object(ex,'verify_replay',return_value=replay),\
                 patch.object(st,'read',wraps=st.read) as reader:
                # The committed decision may not yet exist during first source
                # creation; patch only this public output input in the fixture.
                real=reader._mock_wraps
                reader.side_effect=lambda p: {'mode':'REUSE_G2'} if Path(p)==st.BASE/'g3/G3_CONTROL_COMPARABILITY.json' else real(p)
                results.compare(args);results.bundle(args);results.bundle(args)
                export=root/'aggregate_exports'/real(root/'latest_aggregate.json')['directory']
                value=real(export/'comparison.json');self.assertTrue(value['rows'][-1]['protocol_complete'])
                self.assertEqual(value['rows'][-1]['screen']['gates']['at_least_86_correct'],False)
                with zipfile.ZipFile(export/'COMP3710_G3_Seed42_Results.zip') as z:
                    self.assertTrue(set(z.namelist())<=set(results.ALLOWLIST))
                    for n in z.namelist():self.assertNotIn(b'subject_id',z.read(n))
                # A wall-limited run remains exported but cannot pass integrity.
                v2.atomic_json(root/'G3'/'run_status.json',{'status':'WALL_LIMIT_INCONCLUSIVE'})
                results.compare(args)
                export=root/'aggregate_exports'/real(root/'latest_aggregate.json')['directory']
                self.assertFalse(real(export/'comparison.json')['rows'][-1]['screen']['passes_development_screen'])


if __name__=='__main__':unittest.main(verbosity=2)
