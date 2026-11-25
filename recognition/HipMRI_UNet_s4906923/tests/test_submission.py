"""Small synthetic checks; no private MRI, research directory or CUDA required."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import dataset as data
import utils
import train


class SubmissionTests(unittest.TestCase):
    def test_saved_record_export_recalculates_without_mri_or_weights(self):
        from contextlib import redirect_stdout
        from io import StringIO
        matrix = np.eye(6, dtype=int) * 10
        matrix[5, 0] = 5
        row = dict(case="example", confusion=matrix.tolist(), mri_sha256="mri",
                   mask_sha256="mask", shape_xyz=[2, 3, 4], spacing_mm=[1, 2, 3], surfaces={})
        bundle = dict(class_order=list(utils.CLASS_NAMES), models={"v3": dict(
            best_epoch=50, selected={"checkpoint_sha256": "weights"},
            splits={"validation": [row]})})
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/"records.json"; source.write_text(json.dumps(bundle))
            args = SimpleNamespace(records=source, model="v3", split="validation",
                                   case="example", output=Path(folder)/"export.json")
            with patch.object(train, "load_case", side_effect=AssertionError("MRI read")), \
                 patch.object(torch, "load", side_effect=AssertionError("weight read")), \
                 redirect_stdout(StringIO()):
                result = train.inspect_record(args)
                self.assertEqual(result["per_class"]["Prostate"]["dice"], .8)
                self.assertEqual(result["selected_epoch"], 50)
                self.assertEqual(json.loads(args.output.read_text()), result)
                with self.assertRaises(FileExistsError): train.inspect_record(args)

    def test_frozen_patient_split(self):
        record = data.read_split()
        self.assertEqual([len(record[k]) for k in ("train", "validation", "test")], [26, 6, 6])
        with tempfile.TemporaryDirectory() as folder:
            record["test"][0] = record["train"][0]
            path = Path(folder)/"split.json"; path.write_text(json.dumps(record))
            with self.assertRaises(ValueError): data.read_split(path)

    def test_real_nifti_pair_geometry_and_labels(self):
        import nibabel as nib
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            array = np.arange(16**3, dtype=np.float32).reshape(16,16,16)
            mask = np.zeros(array.shape, dtype=np.uint8); mask[4:8,4:8,4:8]=5
            affine = np.diag([1.,2.,3.,1.])
            for name, values in (("mri",array),("mask",mask)):
                img=nib.Nifti1Image(values,affine); img.header.set_xyzt_units("mm")
                nib.save(img,folder/(name+".nii.gz"))
            case=data.Case("B006",0,"train",folder/"mri.nii.gz",folder/"mask.nii.gz")
            vol=data.downsample(data.load_case(case))
            self.assertEqual(vol.image.shape,(8,8,8))
            np.testing.assert_array_equal(vol.affine[:3,3],[.5,1.,1.5])
            self.assertEqual(vol.spacing_mm,(2.,4.,6.))
            norm=data.zscore_nonzero(vol)
            self.assertAlmostEqual(float(norm.image.mean()),0.,places=5)
            bad=nib.Nifti1Image(mask,np.eye(4));bad.header.set_xyzt_units("mm")
            nib.save(bad,case.mask)
            with self.assertRaises(ValueError):data.load_case(case)

    def test_padding_roundtrip_and_pair_crop(self):
        case=data.Case("B006",0,"train",Path("x"),Path("y"))
        array=np.arange(17*18*19,dtype=np.float32).reshape(17,18,19)
        mask=(array%6).astype(np.uint8)
        vol=data.Volume(case,array,mask,np.eye(4),(1.,1.,1.))
        image,labels,shape=data.to_3d_model_arrays(vol)
        self.assertEqual(image.shape,(32,32,32))
        np.testing.assert_array_equal(data.from_3d_model_prediction(labels,shape),mask)
        x,y=data.crop_for_3d_model(vol,(1,2,3),(8,8,8))
        np.testing.assert_array_equal((x[0]%6).astype(int),y)

    def test_metric_aggregation_is_patient_equal(self):
        a=np.zeros((6,6),dtype=int);a[3,3]=1000
        b=np.zeros((6,6),dtype=int);b[3,0]=1
        rows=[{"patient":"A","confusion":a.tolist()}, {"patient":"B","confusion":b.tolist()}]
        self.assertEqual(utils.aggregate_cases(rows)["selection_score"],.5)
        self.assertIsNone(utils.overlap(a)["dice"][5])
        self.assertAlmostEqual(utils.overlap(a+b)["dice"][3],2000/2001)

    def test_anisotropic_surface_and_empty_status(self):
        a=np.zeros((8,8,8),dtype=np.uint8);a[2,2,2]=5
        b=np.zeros_like(a);b[2,2,3]=5
        self.assertEqual(utils.symmetric_surface_distances(a,b,(1.,1.,4.))["hd95_mm"],4.)
        self.assertEqual(utils.symmetric_surface_distances(a,np.zeros_like(a),(1.,1.,1.))["status"],"one_empty")

    def test_models_loss_and_trusted_checkpoint(self):
        torch.set_num_threads(1)
        for name in ("standard","v3"):
            model=utils.build_model(name).eval()
            x=torch.randn(1,1,16,16,16);y=torch.randint(0,6,(1,16,16,16))
            out=model(x);self.assertEqual(out.shape,(1,6,16,16,16))
            loss=utils.ce_dice_loss(out,y);loss.backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/"weights.pt"
                torch.save({"model":name,"model_state":model.state_dict()},path)
                restored=utils.load_model(name,path,"cpu")
                with torch.no_grad():self.assertTrue(torch.equal(out,restored(x)))

    def test_trainer_direct_imports_preserve_models_and_random_state(self):
        from modules import StandardUNet3D, ImprovedUNet3DV3
        self.assertIs(train.StandardUNet3D, StandardUNet3D)
        self.assertIs(train.ImprovedUNet3DV3, ImprovedUNet3DV3)
        for name in ("standard", "v3"):
            torch.manual_seed(371017)
            previous = utils.build_model(name)
            previous_rng = torch.get_rng_state()
            torch.manual_seed(371017)
            current = train.build_model(name)
            self.assertTrue(torch.equal(previous_rng, torch.get_rng_state()))
            for key, value in previous.state_dict().items():
                self.assertTrue(torch.equal(value, current.state_dict()[key]), key)

    def test_midbatch_resume_matches_uninterrupted_optimizer(self):
        # Tiny model/data isolate the recovery engine from full MRI/GPU cost.
        torch.set_num_threads(1)
        config=json.loads((train.ROOT/"config/training.json").read_text());config["epochs"]=1
        case=data.Case("B006",0,"train",Path("x"),Path("y"))
        def crop(*args,**kwargs):
            x=np.arange(64,dtype=np.float32).reshape(1,4,4,4)/64
            return x,(x[0]*64%6).astype(np.int64)
        def model(name):return torch.nn.Sequential(torch.nn.Conv3d(1,6,1),torch.nn.Dropout3d(.2))
        groups={"train":(case,),"validation":()}
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            def args(name,resume=False):
                path=folder/name;path.mkdir(exist_ok=True)
                return SimpleNamespace(model="standard",device="cpu",run_dir=path,
                    config=train.ROOT/"config/training.json",dataset_root=folder,
                    resume=resume,keep_epoch_weights=False)
            original=utils.atomic_checkpoint
            calls=[0]
            def interrupted(path,state):
                original(path,state);calls[0]+=1
                if calls[0]==3:raise RuntimeError("simulated interruption after durable batch")
            with patch.object(train,"build_model",model), patch.object(train,"load_case",lambda c:c), \
                 patch.object(train,"downsample",lambda c:c), patch.object(train,"zscore_nonzero",lambda c:SimpleNamespace(mask=np.zeros((64,64,32),dtype=np.uint8))), \
                 patch.object(train,"crop_for_3d_model",crop), patch.object(train,"plot_history",lambda *a:None):
                train._train_locked(args("full"),config,groups)
                with patch.object(train,"atomic_checkpoint",interrupted):
                    with self.assertRaisesRegex(RuntimeError,"simulated"):train._train_locked(args("resume"),config,groups)
                train._train_locked(args("resume",True),config,groups)
            a=torch.load(folder/"full/latest.pt",weights_only=False)
            b=torch.load(folder/"resume/latest.pt",weights_only=False)
            for k in a["model_state"]:self.assertTrue(torch.equal(a["model_state"][k],b["model_state"][k]))
            self.assertEqual(a["history"][0]["loss"],b["history"][0]["loss"])
            for k in a["optimizer_state"]["state"]:
                for v in a["optimizer_state"]["state"][k]:
                    self.assertTrue(torch.equal(a["optimizer_state"]["state"][k][v],b["optimizer_state"]["state"][k][v]))


if __name__=="__main__":unittest.main()
