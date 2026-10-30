"""Synthetic-only tests for Step 8; no private ADNI files, network or GPU used."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import DataError
from modules import (BaselineCNN, ChannelLayerNorm, ConvNeXtBlock, ConvNeXtTiny,
                     StochasticDepth, parameter_count)
from train_experiment import (ExperimentConfig, build_model, epoch_learning_rate,
    experiment_sources, load_model, optimizer_for, run_experiment, save_model,
    train_epoch)
from check_experiment import run_smoke
from tests_baseline.test_baseline import Fixture
from training_utils import seed_everything

torch.set_num_threads(1)


def small_builder(name):
    return BaselineCNN() if name == "baseline" else ConvNeXtTiny(
        depths=(1, 1, 1, 1), dims=(8, 16, 32, 64))


class ModelTests(unittest.TestCase):
    def test_full_tiny_depth_width_and_parameter_count(self):
        model = ConvNeXtTiny()
        self.assertEqual(model.depths, (3, 3, 9, 3))
        self.assertEqual(model.dims, (96, 192, 384, 768))
        self.assertEqual(parameter_count(model), 27818594)
        self.assertEqual([len(s) for s in model.stages], [3, 3, 9, 3])

    def test_full_tiny_forward_backward(self):
        model = ConvNeXtTiny().train()
        out = model(torch.randn(1, 1, 224, 224))
        self.assertEqual(out.shape, (1, 2))
        torch.nn.functional.cross_entropy(out, torch.tensor([1])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_stage_spatial_shapes(self):
        model = small_builder("convnext").eval()
        x = torch.zeros(1, 1, 224, 224)
        shapes = []
        for down, stage in zip(model.downsample, model.stages):
            x = stage(down(x))
            shapes.append(tuple(x.shape[2:]))
        self.assertEqual(shapes, [(56, 56), (28, 28), (14, 14), (7, 7)])

    def test_baseline_still_unchanged_parameter_count(self):
        self.assertEqual(parameter_count(BaselineCNN()), 359922)

    def test_wrong_inputs_rejected(self):
        model = small_builder("convnext")
        for shape in ((1,3,224,224), (1,1,240,256), (1,224,224)):
            with self.assertRaises(ValueError): model(torch.zeros(shape))

    def test_invalid_architecture_rejected(self):
        for kwargs in ({"depths": (1,1)}, {"dims": (0,1,2,3)}, {"drop_path_rate": 1.0}):
            with self.assertRaises(ValueError): ConvNeXtTiny(**kwargs)

    def test_channel_layernorm_reference(self):
        layer = ChannelLayerNorm(5)
        x = torch.randn(2,5,3,4)
        reference = nn.functional.layer_norm(x.permute(0,2,3,1), (5,),
            layer.norm.weight, layer.norm.bias, eps=1e-6).permute(0,3,1,2)
        self.assertTrue(torch.equal(layer(x), reference))

    def test_stochastic_depth_eval_identity(self):
        x = torch.randn(3,4,2,2)
        self.assertTrue(torch.equal(StochasticDepth(.5).eval()(x), x))

    def test_stochastic_depth_zero_identity(self):
        x = torch.randn(3,4,2,2)
        self.assertTrue(torch.equal(StochasticDepth(0)(x), x))

    def test_stochastic_depth_bad_probability(self):
        for p in (-.1,1,2):
            with self.assertRaises(ValueError): StochasticDepth(p)

    def test_stochastic_depth_per_sample_and_scale(self):
        torch.manual_seed(4)
        y = StochasticDepth(.5)(torch.ones(2000,3,2,2))
        self.assertTrue(set(y.flatten().tolist()) <= {0.0,2.0})
        self.assertTrue(torch.equal(y, y[:, :1, :1, :1].expand_as(y)))
        self.assertAlmostEqual(y.mean().item(), 1., delta=.07)

    def test_block_zero_layerscale_is_identity(self):
        block = ConvNeXtBlock(8, layer_scale=0)
        x = torch.randn(2,8,7,7)
        self.assertTrue(torch.equal(block(x), x))

    def test_block_structure(self):
        block = ConvNeXtBlock(8)
        self.assertEqual(block.depthwise.groups, 8)
        self.assertEqual(block.depthwise.kernel_size, (7,7))
        self.assertEqual(block.expand.out_features, 32)
        self.assertEqual(block.contract.out_features, 8)
        self.assertTrue(torch.allclose(block.gamma, torch.full((8,),1e-6)))

    def test_eval_repeatable(self):
        model = small_builder("convnext").eval()
        x = torch.randn(2,1,224,224)
        self.assertTrue(torch.equal(model(x), model(x)))

    def test_factory_rejects_other_architecture(self):
        with self.assertRaises(DataError): build_model("made-up")


class ConfigTests(unittest.TestCase):
    def test_defaults_valid(self): ExperimentConfig().validate()

    def test_bad_configurations(self):
        base = ExperimentConfig()
        for kwargs in ({"batch_size":0}, {"accumulation_steps":0}, {"warmup_epochs":30},
                       {"min_epochs":31}, {"model":"x"}, {"budget_seconds":0},
                       {"min_learning_rate":1}, {"device":"mps"}):
            with self.assertRaises(DataError): replace(base, **kwargs).validate()

    def test_warmup(self):
        c = ExperimentConfig()
        self.assertAlmostEqual(epoch_learning_rate(1,c),1e-4)
        self.assertAlmostEqual(epoch_learning_rate(2,c),2e-4)
        self.assertAlmostEqual(epoch_learning_rate(3,c),3e-4)

    def test_cosine_endpoints_and_monotonicity(self):
        c = ExperimentConfig()
        rates = [epoch_learning_rate(e,c) for e in range(4,31)]
        self.assertAlmostEqual(rates[0],3e-4)
        self.assertAlmostEqual(rates[-1],1e-6)
        self.assertTrue(all(a >= b for a,b in zip(rates,rates[1:])))

    def test_schedule_epoch_bounds(self):
        for e in (0,31):
            with self.assertRaises(DataError): epoch_learning_rate(e,ExperimentConfig())

    def test_weight_decay_exclusions_and_complete_coverage(self):
        model = small_builder("convnext")
        opt = optimizer_for(model,ExperimentConfig())
        covered = [id(p) for g in opt.param_groups for p in g["params"]]
        self.assertEqual(set(covered), {id(p) for p in model.parameters()})
        self.assertEqual(len(covered),len(set(covered)))
        for group in opt.param_groups:
            for p in group["params"]:
                self.assertEqual(group["weight_decay"], .05 if p.ndim > 1 else 0.)


class ToyDataset(Dataset):
    split = "train"
    def __init__(self,n=11):
        gen = torch.Generator().manual_seed(4)
        self.x = torch.randn(n,1,2,2,generator=gen)
        self.y = torch.arange(n) % 2
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return {"image":self.x[i], "label":self.y[i], "split":self.split}


class AccumulationTests(unittest.TestCase):
    def test_equivalent_to_full_windows_including_partial_tail(self):
        seed_everything(42)
        data = ToyDataset(11)
        a = nn.Sequential(nn.Flatten(),nn.Linear(4,2))
        b = nn.Sequential(nn.Flatten(),nn.Linear(4,2)); b.load_state_dict(a.state_dict())
        oa,ob = torch.optim.SGD(a.parameters(),lr=.1),torch.optim.SGD(b.parameters(),lr=.1)
        c = replace(ExperimentConfig(),batch_size=2,accumulation_steps=3,device="cpu")
        result = train_epoch(a,DataLoader(data,batch_size=2),oa,torch.device("cpu"),c)
        for start in (0,6):
            ob.zero_grad()
            nn.functional.cross_entropy(b(data.x[start:start+6]),data.y[start:start+6]).backward()
            nn.utils.clip_grad_norm_(b.parameters(),5.)
            ob.step()
        for p,q in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.allclose(p,q,atol=1e-7,rtol=1e-6))
        self.assertEqual(result["images"],11)
        self.assertEqual(result["optimizer_updates"],2)

    def test_validation_cannot_train(self):
        data=ToyDataset(); data.split="validation"
        model=nn.Sequential(nn.Flatten(),nn.Linear(4,2))
        with self.assertRaises(DataError):
            train_epoch(model,DataLoader(data,batch_size=8),torch.optim.SGD(model.parameters(),lr=.1),
                        torch.device("cpu"),ExperimentConfig())

    def test_incorrect_batch_size_rejected(self):
        model=nn.Sequential(nn.Flatten(),nn.Linear(4,2))
        with self.assertRaises(DataError):
            train_epoch(model,DataLoader(ToyDataset(),batch_size=2),torch.optim.SGD(model.parameters(),lr=.1),
                        torch.device("cpu"),ExperimentConfig())

    def test_nonfinite_loss_stops(self):
        data=ToyDataset(); data.x.fill_(float("nan"))
        model=nn.Sequential(nn.Flatten(),nn.Linear(4,2))
        with self.assertRaises(DataError):
            train_epoch(model,DataLoader(data,batch_size=8),torch.optim.SGD(model.parameters(),lr=.1),
                        torch.device("cpu"),ExperimentConfig())


class IntegrationTests(unittest.TestCase):
    def make_smoke(self,path,fixture,**overrides):
        value={"step8_smoke_passed":True,"manifest_sha256":fixture.bundle().manifest_hashes,
               "source_code_sha256":experiment_sources(),"batch_size":8,
               "accumulation_steps":4,"device":"cpu","seed":42}
        value.update(overrides); path.write_text(json.dumps(value)); return path

    def test_checkpoint_roundtrip_and_labels(self):
        with tempfile.TemporaryDirectory() as t, patch('train_experiment.build_model',side_effect=small_builder):
            p=Path(t)/"state.pt"; m=small_builder("convnext").eval(); h={"train.csv":"x"}
            x=torch.zeros(1,1,224,224); expected=m(x)
            save_model(p,m,replace(ExperimentConfig(),device="cpu"),h,2)
            restored,ck=load_model(p,torch.device("cpu"),h,"convnext")
            self.assertTrue(torch.equal(restored(x),expected));self.assertEqual(ck["epoch"],2)
            with self.assertRaises(DataError): load_model(p,torch.device("cpu"),{},"convnext")
            with self.assertRaises(DataError): load_model(p,torch.device("cpu"),h,"baseline")

    def test_smoke_runs_both_models_without_test_or_validation_images(self):
        with tempfile.TemporaryDirectory() as t, patch('train_experiment.build_model',side_effect=small_builder), \
             patch('check_experiment.build_model',side_effect=small_builder):
            base=Path(t); f=Fixture(base)
            for split in ("validation","test"):
                for row in f.rows[split]: (f.root/row.image_path).unlink()
            r=run_smoke(f.root,f.splits,f.pipeline,base/"smoke",device_name="cpu")
            self.assertTrue(r["step8_smoke_passed"])
            self.assertEqual(set(r["models"]),{"baseline","convnext"})
            self.assertEqual(r["final_test_images_decoded"],0)

    def test_training_and_reload_both_models_without_test_images(self):
        with tempfile.TemporaryDirectory() as t, patch('train_experiment.build_model',side_effect=small_builder):
            base=Path(t); f=Fixture(base)
            for row in f.rows["test"]: (f.root/row.image_path).unlink()
            smoke=self.make_smoke(base/"smoke.json",f)
            for name in ("baseline","convnext"):
                c=replace(ExperimentConfig(),model=name,device="cpu",epochs=2,min_epochs=1,warmup_epochs=0)
                out=base/name
                r=run_experiment(f.root,f.splits,f.pipeline,smoke,out,c)
                self.assertTrue(r["experiment_completed"])
                self.assertEqual(r["epochs_completed"],2)
                self.assertTrue(r["selected_checkpoint_reloaded_and_checked"])
                self.assertEqual(r["final_test_images_decoded"],0)
                self.assertTrue((out/"README_experiment_results.md").exists())
                self.assertTrue((out/"PRIVATE_validation_predictions.json").exists())
                json.loads((out/"experiment_summary.json").read_text())

    def test_mismatched_smoke_rejected_and_failure_report_written(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);f=Fixture(base);smoke=self.make_smoke(base/"smoke.json",f,source_code_sha256={})
            with self.assertRaises(DataError):
                run_experiment(f.root,f.splits,f.pipeline,smoke,base/"out",replace(ExperimentConfig(),device="cpu"))
            report=json.loads((base/"out"/"experiment_summary.json").read_text())
            self.assertEqual(report["status"],"FAILED")
            self.assertFalse(report["experiment_completed"])

    def test_device_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);f=Fixture(base);smoke=self.make_smoke(base/"smoke.json",f,device="cuda")
            with self.assertRaises(DataError):
                run_experiment(f.root,f.splits,f.pipeline,smoke,base/"out",replace(ExperimentConfig(),device="cpu"))

    def test_soft_time_budget_stops_at_epoch_boundary(self):
        with tempfile.TemporaryDirectory() as t, patch('train_experiment.build_model',side_effect=small_builder):
            base=Path(t);f=Fixture(base);smoke=self.make_smoke(base/"smoke.json",f)
            c=replace(ExperimentConfig(),device="cpu",epochs=2,min_epochs=2,warmup_epochs=0,budget_seconds=1e-9)
            r=run_experiment(f.root,f.splits,f.pipeline,smoke,base/"out",c)
            self.assertEqual(r["epochs_completed"],1)
            self.assertEqual(r["stopping_reason"],"soft_time_budget")

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);f=Fixture(base);smoke=self.make_smoke(base/"smoke.json",f);(base/"out").mkdir()
            with self.assertRaises(DataError):
                run_experiment(f.root,f.splits,f.pipeline,smoke,base/"out",replace(ExperimentConfig(),device="cpu"))

    def test_source_hashes_include_model_and_main_entry(self):
        hashes=experiment_sources()
        for key in ("dataset.py","modules.py","train.py","train_experiment.py","check_experiment.py"):
            self.assertEqual(len(hashes[key]),64)


class ComparisonTests(unittest.TestCase):
    def reports(self):
        import copy
        initial = {"baseline_training_completed": True,
            "final_test_model_evaluation_performed": False, "final_test_images_decoded": 0,
            "manifest_sha256": {"x":"abc"}, "preprocessing": {"channels":1},
            "split_counts": {"train":40,"validation":40}}
        baseline=copy.deepcopy(initial); baseline.update(experiment_completed=True,model_name="baseline",
            configuration={"model":"baseline","batch_size":8},source_code_sha256={"model":"abc"})
        convnext=copy.deepcopy(baseline);convnext["model_name"]="convnext";convnext["configuration"]["model"]="convnext"
        return initial,baseline,convnext

    def test_matched_reports_accepted(self):
        from compare_step8 import validate_reports
        validate_reports(*self.reports())

    def test_test_access_rejected(self):
        from compare_step8 import validate_reports
        a,b,c=self.reports();c["final_test_images_decoded"]=1
        with self.assertRaises(DataError):validate_reports(a,b,c)

    def test_mismatched_recipe_rejected(self):
        from compare_step8 import validate_reports
        a,b,c=self.reports();c["configuration"]["batch_size"]=16
        with self.assertRaises(DataError):validate_reports(a,b,c)

    def test_different_source_and_splits_rejected(self):
        from compare_step8 import validate_reports
        for key in ("source_code_sha256", "manifest_sha256", "preprocessing", "split_counts"):
            a,b,c=self.reports();c[key]={"different":True}
            with self.assertRaises(DataError):validate_reports(a,b,c)


if __name__ == '__main__':
    unittest.main()
