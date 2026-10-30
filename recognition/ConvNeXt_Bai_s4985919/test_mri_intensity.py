"""Synthetic tensors only; no torchvision, ADNI, training, or inference."""
import ast
import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import mri_intensity as mi


ROOT = Path(__file__).parent


class SyntheticSet:
    def __init__(self):
        self.image = torch.tensor([[[-1., -.5, 0., .5, 1.]]]).repeat(3, 2, 1)
    def __len__(self):
        return 2
    def __getitem__(self, index):
        return self.image, index % 2


class IntensityTests(unittest.TestCase):
    def test_range_shape_channels_and_background(self):
        image = SyntheticSet().image
        for scale in (.9, 1., 1.1):
            for gamma in (.9, 1., 1.1):
                result = mi.apply_intensity(image, scale, gamma)
                self.assertEqual(result.shape, image.shape)
                self.assertEqual(result.dtype, image.dtype)
                self.assertTrue(torch.isfinite(result).all())
                self.assertGreaterEqual(result.min().item(), -1)
                self.assertLessEqual(result.max().item(), 1)
                self.assertTrue(torch.equal(result[image == -1], image[image == -1]))
                torch.testing.assert_close(result[0], result[1])
        torch.testing.assert_close(mi.apply_intensity(image, 1., 1.), image)
        expected = (((image + 1) / 2).pow(.9) * 1.1).clamp(0, 1) * 2 - 1
        torch.testing.assert_close(mi.apply_intensity(image, 1.1, .9), expected)

    def test_isolation_order_and_disabled_legacy_behavior(self):
        training, validation = SyntheticSet(), SyntheticSet()
        class Legacy:
            def __init__(self, original):
                self.original = original
            def __getitem__(self, i):
                image, label = self.original[i]
                # Stand-in verifies ordering; actual affine/noise source checked below.
                return image + .001, label
        before = validation[0][0].clone()
        wrapped = mi.make_training_input(training, True, True, Legacy)
        self.assertIs(wrapped.original.original, training)
        with mock.patch.object(random, 'uniform', side_effect=[.9, 1.1]):
            result, label = wrapped[1]
        torch.testing.assert_close(result, mi.apply_intensity(training.image, .9, 1.1) + .001)
        self.assertEqual(label, 1)
        self.assertEqual(len(wrapped.original), len(training))
        self.assertTrue(torch.equal(validation[0][0], before))
        self.assertTrue(torch.equal(training.image, before))
        disabled = mi.make_training_input(training, True, False, Legacy)
        self.assertIs(disabled.original, training)
        self.assertIs(mi.make_training_input(training, False, False, Legacy), training)
        with self.assertRaises(ValueError):
            mi.make_training_input(training, False, True, Legacy)

    def test_rng_reproducibility_resume_and_torch_rng_unchanged(self):
        wrapped = mi.MRIIntensityTrainingSet(SyntheticSet())
        random.seed(42)
        state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        first = wrapped[0][0]
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        random.setstate(state)
        self.assertTrue(torch.equal(first, wrapped[0][0]))
        resume_state = random.getstate()
        second = wrapped[0][0]
        random.setstate(resume_state)
        self.assertTrue(torch.equal(second, wrapped[0][0]))
        self.assertFalse(torch.equal(first, second))

    def test_invalid_inputs(self):
        image = SyntheticSet().image
        for invalid in (image[0], image[:1], image + 3, image * float('nan'), image.long()):
            with self.assertRaises(ValueError):
                mi.apply_intensity(invalid, 1., 1.)

    def test_optional_historical_geometry_comparison(self):
        historical = ROOT / 'evidence/comp3710_convnext/train.py'
        if not historical.is_file():
            self.skipTest('Historical geometry comparison only: evidence training source unavailable')
        def legacy_class(path):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            return ast.dump(next(n for n in tree.body if isinstance(n, ast.ClassDef)
                                 and n.name == 'MRIAugmentedTrainingSet'))
        self.assertEqual(legacy_class(ROOT / 'dataset.py'),
                         legacy_class(historical))

    def test_optional_historical_config(self):
        relative = Path('checkpoints/convnext/convnext_aug_adamw_run01/config.json')
        # Prefer the actual server checkpoint config. An invalid existing file
        # must fail, rather than silently falling back to the evidence copy.
        candidates = [ROOT / relative, ROOT / 'evidence/comp3710_convnext' / relative]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            self.skipTest('Historical config unavailable in server checkpoints or local evidence')
        config = json.loads(path.read_text(encoding='utf-8'))
        self.assertNotIn('intensity_augmentation', config)
        self.assertTrue(config['training_augmentation'])
        self.assertEqual(config['learning_rate'], .0003)
        self.assertEqual(config['model'], 'convnext')
        self.assertEqual(config['optimizer_name'], 'adamw')
        self.assertEqual(config['weight_decay'], .03)
        self.assertEqual(config['batch_size'], 8)
        self.assertEqual(config['class_to_idx'], {'AD': 0, 'NC': 1})
        self.assertIsNone(mi.intensity_protocol(False))
        # Legacy missing metadata compares equal to disabled, never to enabled.
        self.assertEqual(config.get('intensity_augmentation'), mi.intensity_protocol(False))
        self.assertNotEqual(config.get('intensity_augmentation'), mi.intensity_protocol(True))

    def test_current_geometry_and_noise_without_optional_evidence(self):
        # Execute ONLY the actual dataset wrapper class with a recording affine
        # stand-in. No dataset imports, torchvision, images, or inference.
        tree = ast.parse((ROOT / 'dataset.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == 'MRIAugmentedTrainingSet')
        calls = []
        class RecordingAffine:
            def __init__(self, **kwargs):
                calls.append(kwargs)
            def __call__(self, image):
                return image * .8
        namespace = dict(Dataset=object, RandomAffine=RecordingAffine,
                         InterpolationMode=SimpleNamespace(BILINEAR='bilinear'), torch=torch)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'dataset-wrapper', 'exec'), namespace)
        source = SyntheticSet()
        wrapper = namespace['MRIAugmentedTrainingSet'](source)
        self.assertEqual(calls, [dict(degrees=6, translate=(.03, .03), scale=(.97, 1.03),
                                     interpolation='bilinear', fill=-1.)])
        torch.manual_seed(42)
        expected = (source.image * .8 + torch.randn_like(source.image) * .01).clamp(-1., 1.)
        torch.manual_seed(42)
        image, label = wrapper[1]
        torch.testing.assert_close(image, expected)
        self.assertEqual(label, 1)
        self.assertEqual(len(wrapper), len(source))

    def test_legacy_geometry_config_and_training_integration(self):
        # Mandatory current-code checks are independent of historical artifacts.
        tree = ast.parse((ROOT / 'train.py').read_text(encoding='utf-8'))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'make_training_input']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0].id, 'train_dataset')
        self.assertEqual([ast.dump(a) for a in calls[0].args[1:]],
                         [ast.dump(ast.parse(s, mode='eval').body) for s in
                          ('args.augment', 'args.intensity_augment', 'MRIAugmentedTrainingSet')])
        flags = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                 and n.args and isinstance(n.args[0], ast.Constant)
                 and n.args[0].value == '--intensity-augment']
        self.assertEqual(len(flags), 1)
        self.assertEqual(next(k.value.value for k in flags[0].keywords if k.arg == 'action'),
                         'store_true')
        # Four metadata sites: last, resume comparison, config, best metadata.
        sites = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)
                 and any(isinstance(k, ast.Constant) and k.value == 'intensity_augmentation'
                         for k in n.keys)]
        self.assertEqual(len(sites), 4)
        for site in sites:
            value = next(v for k, v in zip(site.keys, site.values)
                         if isinstance(k, ast.Constant) and k.value == 'intensity_augmentation')
            self.assertEqual(ast.dump(value), ast.dump(ast.parse(
                'intensity_protocol(args.intensity_augment)', mode='eval').body))

    def test_actual_resume_guard_enabled_disabled_and_legacy(self):
        tree = ast.parse((ROOT / 'train.py').read_text(encoding='utf-8'))
        # Execute the actual comparison loop, without checkpoint I/O or training.
        loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
                 and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Attribute)
                 and isinstance(n.iter.func.value, ast.Dict)
                 and any(isinstance(k, ast.Constant) and k.value == 'intensity_augmentation'
                         for k in n.iter.func.value.keys)]
        self.assertEqual(len(loops), 1)
        loop = loops[0]
        args = SimpleNamespace(model='convnext', batch_size=8, learning_rate=.0003, seed=42,
                               optimizer='adamw', weight_decay=.03, augment=True,
                               intensity_augment=False, max_train_batches=None,
                               max_val_batches=None, selection_level='slice', selection_threshold=.5,
                               split_mode='strict', input_mode='single', scheduler='none',
                               warmup_epochs=1.0, ema_decay=0.0, label_smoothing=0.0)
        namespace = dict(args=args, fingerprint={'train': 'synthetic'},
                         intensity_protocol=mi.intensity_protocol)
        expected = eval(compile(ast.Expression(loop.iter.func.value), 'resume-settings', 'eval'), namespace)
        code = compile(ast.Module(body=[loop], type_ignores=[]), 'resume-guard', 'exec')
        for saved_enabled, current_enabled in ((False, False), (True, True),
                                                (False, True), (True, False), (None, False), (None, True)):
            checkpoint = dict(expected)
            if saved_enabled is None:
                checkpoint.pop('intensity_augmentation')
            else:
                checkpoint['intensity_augmentation'] = mi.intensity_protocol(saved_enabled)
            args.intensity_augment = current_enabled
            namespace['checkpoint'] = checkpoint
            with self.subTest(saved=saved_enabled, current=current_enabled):
                if bool(saved_enabled) != current_enabled:
                    with self.assertRaisesRegex(ValueError, 'intensity_augmentation'):
                        exec(code, namespace)
                else:
                    exec(code, namespace)


if __name__ == '__main__':
    unittest.main()
