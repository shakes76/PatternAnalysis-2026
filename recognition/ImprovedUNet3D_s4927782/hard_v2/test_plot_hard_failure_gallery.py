"""Synthetic-only image/GT/prediction files; no actual model results are read."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import compare_common_grid as c
import plot_hard_failure_gallery as g


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bound(path):
    return {'path':str(path),'sha256':c.file_sha256(path)}


def synthetic_study(root):
    """Thirty artificial scans with physical1.5x grids and six synthetic patients."""
    project, review = root/'original', root/'review'
    artifacts = review/'artifacts'
    oldroot, newroot = project/'artifacts/cache64', artifacts/'cache_v2'
    keys = [(f'SYN{pi}',f'Week{visit}') for pi,n in enumerate([8,6,5,4,4,3]) for visit in range(n)]
    oldshape, newshape = (6,6,6),(9,9,9)
    oldaffine = np.diag([1.5,1.5,1.5,1.]); oldaffine[:3,3]=.25
    newaffine = np.eye(4)
    target = np.zeros(oldshape,dtype=np.uint8)
    target[0] = 1; target[5,1,1] = 2; target[2,0,0] = 3; target[3,0,0] = 4
    target[1:5,2:4,2:4] = 5; target[2,1:5,1:5] = 5
    image = np.linspace(-2,2,np.prod(oldshape),dtype=np.float32).reshape(oldshape)
    image[0,0,0] = 0
    spacing = np.asarray([1.5]*3,dtype=np.float32)
    high_target = c.categorical_to_grid(target,oldaffine,newaffine,newshape)[0]
    manifests, indexed, identity_records = {},{},{}
    for version,shape,affine in [('v1',oldshape,oldaffine),('new',newshape,newaffine)]:
        records=[]
        for patient,visit in keys:
            record=dict(patient=patient,visit=visit,split='test',cache_path=f'volumes/{patient}_{visit}.npz',
                spacing_dhw_mm=[1.5]*3 if version=='v1' else [1.]*3,resampled_affine_xyz=affine.tolist())
            if version=='v1':
                cache=oldroot/record['cache_path']; cache.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(cache,image=image,mask=target,spacing=spacing)
                record['class_voxel_counts']=np.bincount(target.ravel(),minlength=6).tolist()
                identity_records[record['cache_path']]={'arrays':{name:{'sha256':c.decoded_array_sha256(array)} for name,array in
                    [('image',image),('mask',target),('spacing',spacing)]}}
            records.append(record)
        manifest=dict(num_classes=6,label_ids=list(range(6)),target_shape_dhw=list(shape),records=records,
            patient_splits={'train':[f'T{i:03}' for i in range(26)],'validation':[f'V{i:03}' for i in range(6)],
                            'test':sorted({p for p,v in keys})})
        path=(oldroot if version=='v1' else newroot)/'manifest.json';write(path,manifest)
        manifests[version]=path;indexed[version]={(r['patient'],r['visit']):r for r in records}
    book=oldroot/'cache_identity.json'
    write(book,dict(manifest_sha256=c.file_sha256(manifests['v1']),records=identity_records))
    scope=review/'hard_followup_protocol.json';write(scope,{'synthetic_fixture':True,'experiment_id':'SYNTHETIC hard-only'})
    selection=artifacts/'hard_selection.json'
    write(selection,dict(status='frozen',scope_plan_sha256=c.file_sha256(scope),
                         models={'improved3d':{'checkpoint_sha256':'a'*64}}))
    report=dict(status='completed',split='test',synthetic_fixture=True,
        scientific_status='SYNTHETIC previously inspected TEST simulation; no independent blind-test claim.',
        scope_plan_sha256=c.file_sha256(scope),models={},
        input_files={'scope_plan':bound(scope),'v1_manifest':bound(manifests['v1']),
                     'new_manifest':bound(manifests['new']),'v1_cache_identity':bound(book),
                     'selection':bound(selection),'evaluations':{}})
    for name,family,_ in g.ORDER:
        version='new' if name=='hard_improved3d' else 'v1'
        evpath=(artifacts/'hard_test/improved3d/evaluation.json' if version=='new'
                else project/f'artifacts/test_final/{family}/evaluation.json')
        cases,common_cases=[],[]
        for patient,visit in keys:
            low=target.copy()
            if name=='v1_unet2d' and (patient,visit)==('SYN0','Week0'):
                low[low==5]=1
            if name=='hard_improved3d' and visit=='Week0':
                if patient in ('SYN0','SYN1'):low[low==5]=0
                elif patient=='SYN2':low[1:3][low[1:3]==5]=0
                elif patient=='SYN3':low[2,2,2]=4;low[2,0,0]=5
                elif patient=='SYN4':low[0,2,2]=5
            prediction=low if version=='v1' else c.categorical_to_grid(low,oldaffine,newaffine,newshape)[0]
            gt=target if version=='v1' else high_target
            path=evpath.parent/f'predictions/{patient}_{visit}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,prediction=prediction,spacing_dhw=[1.5]*3 if version=='v1' else [1.]*3,
                label_ids=np.arange(6),affine_xyz=oldaffine if version=='v1' else newaffine)
            cases.append(dict(patient=patient,visit=visit,prediction_file=path.relative_to(evpath.parent).as_posix(),
                              **c.class_metrics(prediction,gt)))
            scored=(prediction if version=='v1' else c.categorical_to_grid(prediction,newaffine,oldaffine,oldshape)[0])
            common_cases.append(dict(patient=patient,visit=visit,**c.class_metrics(scored,target),
                prostate_surface=c.prostate_surface_metrics(scored,target,spacing),
                prediction_evidence=bound(path)))
        checkpoint='a'*64 if version=='new' else c.FROZEN_V1_CHECKPOINTS[family]
        evaluation=dict(split='test',cases=cases,checkpoint_sha256=checkpoint)
        if version=='new':
            evaluation['identity']=dict(model=family,manifest_sha256=c.file_sha256(manifests[version]))
        else:
            evaluation.update(manifest_sha256=c.file_sha256(manifests[version]),model_config={'model':family})
        write(evpath,evaluation)
        report['input_files']['evaluations'][name]=bound(evpath)
        report['models'][name]=dict(cases=common_cases,aggregate=c.aggregate(common_cases),checkpoint_sha256=checkpoint)
    report_path=root/'comparison.json';write(report_path,report)
    return SimpleNamespace(v1_project=project,artifacts_root=artifacts,report=report_path,output_dir=root/'gallery'),c.file_sha256(book)


class HardGalleryTests(unittest.TestCase):
    def test_complete_synthetic_pipeline_renders_ten_views_without_hiding_new_failure(self):
        with TemporaryDirectory() as directory:
            args,book_sha=synthetic_study(Path(directory))
            with patch.object(c,'FROZEN_V1_CACHE_IDENTITY_SHA256',book_sha):
                receipt=g.build_gallery(args)
            self.assertEqual(receipt['status'],'completed')
            self.assertEqual(receipt['all_test_cases_retained_in_report'],30)
            self.assertEqual(receipt['model_order'],[n for n,_,_ in g.ORDER])
            self.assertEqual(receipt['selected_case_keys'][0],['SYN0','Week0'])
            self.assertEqual(len(receipt['selected_case_keys']),5)
            self.assertEqual(len({p for p,v in receipt['selected_case_keys']}),5)
            self.assertEqual(len(list(args.output_dir.glob('*.png'))),10)
            first=receipt['cases'][0]
            self.assertEqual(first['complete_volume_metrics']['hard_improved3d']['dice'],0.)
            self.assertEqual(first['complete_volume_metrics']['hard_improved3d']['empty_status'],'missed')
            self.assertFalse(receipt['raw_arrays_saved'])
            for case in receipt['cases']:
                self.assertEqual(set(case['figures']),{'full_fov','display_roi'})
                for item in case['figures'].values():
                    path=args.output_dir/item['file']
                    self.assertEqual(c.file_sha256(path),item['sha256'])
                    self.assertGreater(float(np.asarray(Image.open(path).convert('RGB')).std()),10)
                for values in case['complete_volume_metrics'].values():
                    self.assertEqual(sum(values['false_positive_by_true_class'].values()),values['fp'])
                    self.assertEqual(sum(values['false_negative_by_predicted_class'].values()),values['fn'])
            preview=Path('/private/tmp/hard-gallery-SYNTHETIC-preview.png')
            shutil.copyfile(args.output_dir/first['figures']['full_fov']['file'],preview)
            shutil.copyfile(args.output_dir/first['figures']['display_roi']['file'],
                            '/private/tmp/hard-gallery-SYNTHETIC-roi-preview.png')
            with self.assertRaises(FileExistsError):g.build_gallery(args)

    def test_contour_linewidth_is_not_overwritten_by_image_width(self):
        axis=Mock();mask=np.ones((128,128),dtype=bool)
        g._contour(axis,mask,'green',1.7,'-')
        self.assertEqual(axis.contour.call_args.kwargs['linewidths'],1.7)
        self.assertEqual(axis.contour.call_args.kwargs['linestyles'],'-')

    def test_degenerate_single_depth_view_keeps_all_text_inside_canvas(self):
        import matplotlib.figure
        from matplotlib.text import Text
        target=np.zeros((3,12,12),dtype=np.uint8);target[1,5,5]=5
        image=np.linspace(-2,2,target.size,dtype=np.float32).reshape(target.shape)
        predictions={name:target.copy() for name,_,_ in g.ORDER}
        metrics={name:dict(g.prostate_counts(pred,target),**c.prostate_surface_metrics(pred,target,[2.,2.,2.]))
                 for name,pred in predictions.items()}
        original_save=matplotlib.figure.Figure.savefig
        checked=[]
        def inspect_and_save(figure,*args,**kwargs):
            figure.canvas.draw();renderer=figure.canvas.get_renderer()
            for item in figure.findobj(Text):
                if not item.get_visible() or not item.get_text():continue
                box=item.get_window_extent(renderer)
                self.assertGreaterEqual(box.x0,-1,item.get_text())
                self.assertGreaterEqual(box.y0,-1,item.get_text())
                self.assertLessEqual(box.x1,figure.bbox.width+1,item.get_text())
                self.assertLessEqual(box.y1,figure.bbox.height+1,item.get_text())
            checked.append(True)
            return original_save(figure,*args,**kwargs)
        with TemporaryDirectory() as directory,patch.object(matplotlib.figure.Figure,'savefig',inspect_and_save):
            g.render_case(image,target,predictions,metrics,[2.,2.,2.],g.display_slices(target),
                [0,12,0,12],'SYNTHIN','Week0',Path(directory)/'SYNTHETIC.png',synthetic=True)
        self.assertEqual(checked,[True])

    def test_invalid_completed_scope_is_rejected_before_pixels(self):
        with TemporaryDirectory() as directory:
            args,book_sha=synthetic_study(Path(directory))
            report=c.read_json(args.report)
            for change in [('status','running'),('split','validation'),('scientific_status','independent blind test')]:
                wrong=deepcopy(report);wrong[change[0]]=change[1];write(args.report,wrong)
                with patch.object(np,'load',side_effect=AssertionError('Pixels must not open')),self.assertRaises(ValueError):
                    g.build_gallery(args)
                self.assertFalse(args.output_dir.exists())
            wrong=deepcopy(report);wrong['models']['new_unet2d']={};write(args.report,wrong)
            with self.assertRaises(ValueError):g.build_gallery(args)

    def test_scored_prediction_or_metric_tampering_does_not_make_a_gallery(self):
        for tamper in ('metric','prediction'):
            with self.subTest(tamper=tamper),TemporaryDirectory() as directory:
                args,book_sha=synthetic_study(Path(directory))
                report=c.read_json(args.report)
                if tamper=='metric':
                    report['models']['v1_improved3d']['cases'][0]['per_class'][5]['dice']=.123
                    write(args.report,report)
                else:
                    path=Path(report['models']['v1_improved3d']['cases'][0]['prediction_evidence']['path'])
                    with np.load(path,allow_pickle=False) as saved:data={k:saved[k] for k in saved.files}
                    data['prediction']=np.zeros_like(data['prediction']);np.savez_compressed(path,**data)
                with patch.object(c,'FROZEN_V1_CACHE_IDENTITY_SHA256',book_sha),self.assertRaises(ValueError):
                    g.build_gallery(args)
                self.assertEqual(c.read_json(args.output_dir/'gallery_receipt.json')['status'],'failed')
                self.assertFalse(list(args.output_dir.glob('*.png')))

    def test_gt_guided_display_roi_and_label_extent_roles_are_only_display_helpers(self):
        gt=np.zeros((5,40,48),dtype=np.uint8);gt[2,20,24]=5
        pred=gt.copy();pred[0,0,0]=5
        self.assertEqual(g.display_slices(gt),[{'z':2,'roles':['inferior label extent','largest GT area','superior label extent']}])
        self.assertEqual(g.display_roi(gt,[pred]),[0,29,0,33])
        counts=g.prostate_counts(pred,gt)
        self.assertEqual(counts['outside_gt_depth_false_positive_voxels'],1)
        self.assertEqual(counts['false_positive_by_true_class']['0'],1)


if __name__=='__main__':unittest.main()
