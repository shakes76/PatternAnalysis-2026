"""Synthetic tests for the bounded schedule experiment; no real ADNI/test inference."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader
import train_schedule_only as s
import train_convnext_v2 as v2
from tests_v2.test_v2 import fixture, small_model
from dataset import DataError

torch.set_num_threads(1)

class ProtocolTests(unittest.TestCase):
    def test_only_endpoint_is_changed(self):
        for seed in s.SEEDS:
            config=asdict(s.configuration(seed));self.assertEqual(config.pop('decay_end_epoch'),12)
            self.assertEqual(config,asdict(v2.Config('scratch_augmented',seed=seed)))
        with self.assertRaises(DataError):s.configuration(45)

    def test_schedule_boundaries_and_identical_first_four(self):
        config=s.configuration(42);model=nn.Linear(2,2);optimizer=v2.make_optimizer(model,config)
        for e in range(1,31):
            actual=s.set_epoch(model,optimizer,config,e)
            self.assertTrue(all(g['lr']==actual for g in optimizer.param_groups))
            if e<=4:
                v2.set_epoch(model,optimizer,config,e)
                self.assertEqual(actual,optimizer.param_groups[0]['lr'])
        self.assertAlmostEqual(s.learning_rate(1,config),1e-4,places=15)
        self.assertEqual(s.learning_rate(3,config),3e-4)
        self.assertEqual(s.learning_rate(4,config),3e-4)
        for e in (12,13,21,30):self.assertEqual(s.learning_rate(e,config),1e-6)
        self.assertGreater(s.learning_rate(11,config),1e-6)

    def test_optimizer_and_weighting_unchanged(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.head=nn.Linear(2,2);self.norm=nn.LayerNorm(2)
        model=Model();optimizer=v2.make_optimizer(model,s.configuration(42))
        self.assertEqual(optimizer.defaults['betas'],(.9,.999));self.assertEqual(optimizer.defaults['eps'],1e-8)
        for group in optimizer.param_groups:
            for p in group['params']:self.assertEqual(group['weight_decay'],.05 if p.ndim>1 else 0.)
        with tempfile.TemporaryDirectory() as temp:
            bundle=fixture(Path(temp));data=v2.WeightedDataset(bundle,s.configuration(42))
            self.assertTrue(torch.equal(data.weights,torch.ones(len(data))))

    def test_comparability_stops_on_one_patient_or_material_loss(self):
        row={k:.5 for k in (*s.LOSS_FIELDS,*s.SCORE_FIELDS)};row.update(epoch=1,optimizer_updates=664)
        self.assertTrue(s.comparability(row,row)['passed'])
        other=dict(row,patient_accuracy=.5+1/102)
        self.assertFalse(s.comparability(other,row)['passed'])
        self.assertTrue(s.comparability(dict(row,train_loss=.5000005),row)['passed'])
        self.assertFalse(s.comparability(dict(row,train_loss=.500002),row)['passed'])

    def test_tail_gradient_matches_unweighted_hard_ce(self):
        class Data(torch.utils.data.Dataset):
            split='train'
            def __len__(self):return 10
            def __getitem__(self,i):return {'image':torch.tensor([i/10.]),'label':i%2,
                                            'loss_weight':torch.tensor(1.),'split':'train'}
        a=nn.Linear(1,2);b=nn.Linear(1,2);b.load_state_dict(a.state_dict())
        oa=torch.optim.SGD(a.parameters(),lr=.01);ob=torch.optim.SGD(b.parameters(),lr=.01)
        config=replace(s.configuration(42),batch_size=4,accumulation_steps=2,device='cpu',gradient_clip=1000.)
        result=v2.train_epoch(a,DataLoader(Data(),batch_size=4),oa,torch.device('cpu'),config)
        for batch in DataLoader(Data(),batch_size=8):
            ob.zero_grad();nn.functional.cross_entropy(b(batch['image']),batch['label']).backward();ob.step()
        self.assertEqual(result['updates'],2)
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=1e-7,rtol=0)

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.bundle=fixture(self.root)
        self.config=replace(s.configuration(42),epochs=6,min_epochs=6,patience=9,batch_size=2,
                            accumulation_steps=2,device='cpu')
        self.plan={'sources':{'synthetic':'fixed'},'manifest_sha256':self.bundle.manifest_hashes}
    def run_core(self,folder,**kwargs):
        return s.train_core(folder,self.bundle,self.config,self.plan,[],model_factory=small_model,
                            enforce_comparability=False,**kwargs)
    def test_complete_resume_preserves_weights_optimizer_rng_and_best(self):
        whole=self.root/'whole';whole.mkdir();resume=self.root/'resume';resume.mkdir()
        a=self.run_core(whole);part=self.run_core(resume,max_new_epochs=2)
        self.assertFalse(part['completed']);b=self.run_core(resume)
        self.assertEqual(a['best']['metrics'],b['best']['metrics']);self.assertEqual(a['best']['epoch'],b['best']['epoch'])
        for k in a['model_state']:self.assertTrue(torch.equal(a['model_state'][k],b['model_state'][k]),k)
        self.assertTrue(torch.equal(a['rng']['torch'],b['rng']['torch']))
        for k in a['optimizer_state']['state']:
            for name,x in a['optimizer_state']['state'][k].items():
                y=b['optimizer_state']['state'][k][name]
                if isinstance(x,torch.Tensor):self.assertTrue(torch.equal(x,y))
                else:self.assertEqual(x,y)
        s.finish(resume,self.bundle,self.config,self.plan,b,model_factory=small_model)
        self.assertTrue(s.read_json(resume/'checkpoint_replay.json')['passed'])
        s.valid_completed(resume,self.plan)
        value=torch.load(resume/'PRIVATE_best_model.pt',weights_only=True)
        key=next(iter(value['model_state']));value['model_state'][key]=value['model_state'][key].reshape(-1)
        torch.save(value,resume/'bad.pt')
        with self.assertRaises(DataError):s.load_best(resume/'bad.pt',self.config,self.plan,small_model)
    def test_previous_complete_state_recovery_and_identity_refusal(self):
        run=self.root/'run';run.mkdir();self.run_core(run,max_new_epochs=2)
        (run/'PRIVATE_resume.pt').write_bytes(b'interrupted write')
        result=self.run_core(run,max_new_epochs=1);self.assertEqual(result['epoch'],2)
        self.assertEqual(len(list(run.glob('PRIVATE_unreadable-*.pt'))),1)
        with self.assertRaises(DataError):
            s.load_state(run,replace(self.config,seed=43),self.plan,v2.environment(torch.device('cpu')))
        with self.assertRaises(DataError):s.load_state(run,self.config,{**self.plan,'sources':{'changed':'x'}},v2.environment(torch.device('cpu')))
    def test_missing_states_never_create_replacement(self):
        run=self.root/'missing';run.mkdir();s.atomic(run/'initialized.json',{'seed':42})
        with self.assertRaises(DataError):self.run_core(run)
    def test_material_comparability_failure_never_commits_bad_epoch(self):
        run=self.root/'badfirst';run.mkdir()
        reference=[{k:99 for k in (*s.LOSS_FIELDS,*s.SCORE_FIELDS,'optimizer_updates')}]
        with self.assertRaises(RuntimeError):
            s.train_core(run,self.bundle,self.config,self.plan,reference,model_factory=small_model)
        state=torch.load(run/'PRIVATE_resume.pt',weights_only=True)
        self.assertEqual(state['epoch'],0)
        self.assertEqual(s.read_json(run/'run_status.json')['status'],'MATERIAL_COMPARABILITY_FAILURE')

class BudgetTests(unittest.TestCase):
    def test_crash_reservation_resume_and_partial_work_are_charged(self):
        with tempfile.TemporaryDirectory() as temp:
            clock=[100.];path=Path(temp)/'budget.json'
            first=s.LeaseBudget(path,cap=20,quantum=5,clock=lambda:clock[0])
            clock[0]=103.;first.reserve();self.assertEqual(s.read_json(path)['charged_seconds'],8.)
            # Parent crashes with a partial epoch. Uncertain five seconds stay charged.
            clock[0]=1000.;resumed=s.LeaseBudget(path,cap=20,quantum=5,clock=lambda:clock[0])
            self.assertEqual(resumed.base,8.);self.assertEqual(resumed.deadline,1012.)
            clock[0]=1004.;resumed.finish('INTERRUPTED')
            self.assertEqual(s.read_json(path)['charged_seconds'],12.)
            clock[0]=2000.;last=s.LeaseBudget(path,cap=20,quantum=5,clock=lambda:clock[0])
            clock[0]=2008.;self.assertFalse(last.reserve());last.finish('WALL_LIMIT_INCONCLUSIVE')
            self.assertEqual(s.read_json(path)['charged_seconds'],20.)
            with self.assertRaises(DataError):s.LeaseBudget(path,cap=20,clock=lambda:clock[0])
    @unittest.skipUnless(sys.platform=='win32','Windows process supervision')
    def test_job_closure_kills_worker(self):
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            job=s.WindowsJob(child);job.close();child.wait(timeout=5)
            self.assertIsNotNone(child.returncode)
        finally:
            if child.poll() is None:child.kill();child.wait()

class ReportingTests(unittest.TestCase):
    def test_acceptance_uses_fixed_controls_and_all_integrity_checks(self):
        rows=[]
        for seed,n in zip(s.SEEDS,(85,83,83)):
            rows.append({'status':'COMPLETED','integrity_passed':True,'candidate':{'correct':n,'AD':22,'log_loss':.5},
                         'control':{'correct':85 if seed==42 else 80}})
        self.assertTrue(s.acceptance(rows,.5422)['passed'])
        rows[0]['candidate']['AD']=21;self.assertFalse(s.acceptance(rows,.5422)['gates']['criterion2'])
        rows[0]['status']='WALL_LIMIT_INCONCLUSIVE';self.assertFalse(s.acceptance(rows,.5422)['assessable'])
    def test_last_five_are_fixed_tail_not_best_five(self):
        rows=[{'epoch':i,'patient_accuracy':i/10,'patient_log_loss':1/i} for i in range(1,8)]
        self.assertEqual(s.trajectory(rows)['epochs'],[3,4,5,6,7])

if __name__=='__main__':unittest.main(verbosity=2)
