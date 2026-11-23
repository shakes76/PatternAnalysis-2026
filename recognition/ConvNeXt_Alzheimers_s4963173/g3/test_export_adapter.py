"""Synthetic regression for aggregate replay-boolean export and fail-closed guard."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import dataset
from g3 import export_adapter as fix
from g3 import results


def summary():
    return {'study':'G3_Seed42','rows':[{'name':'G3','replay':{'checks':{'attention':{
        'absolute_tolerance':2e-5,'max_absolute_difference':0.,'slice_order_unchanged':True}}}}]}


class ExportTests(unittest.TestCase):
    def test_plan_with_same_study_name_has_no_rows_and_passes_unchanged(self):
        plan={'study':'G3_Seed42','configuration':{'variant':'G3'},'control_mode':'REUSE_G2'}
        before=deepcopy(plan);self.assertIs(fix.sanitize_summary(plan),plan)
        self.assertEqual(plan,before);results.privacy(plan)
        with self.assertRaises(dataset.DataError):fix.sanitize_summary({'study':'G3_Seed42','rows':None})

    def test_real_receipt_shape_rejected_before_and_safe_after(self):
        value=summary();original=deepcopy(value)
        with self.assertRaises(dataset.DataError):results.privacy(value)
        public=fix.sanitize_summary(deepcopy(value));results.privacy(public)
        self.assertTrue(public['rows'][0]['replay']['checks']['attention']['ordering_unchanged'])
        self.assertEqual(value,original)

    def test_identifiers_paths_arrays_and_unknown_fields_still_rejected(self):
        for key,value in [('subject_id','synthetic-person'),('scan_id','synthetic-scan'),
                          ('image_path','synthetic/image.jpeg'),('slice_order',['synthetic/image.jpeg']),
                          ('attention_weights',[.5,.5]),('path','C:/Users/private'),('path','C:\\Users\\private'),
                          ('note','PRIVATE_pred.json')]:
            bad=summary();bad['unexpected']={key:value}
            with self.assertRaises(dataset.DataError):results.privacy(fix.sanitize_summary(bad))

    def test_failed_non_boolean_unexpected_or_conflicting_order_checks_rejected(self):
        for value in (False,1,'true',[],{}):
            bad=summary();bad['rows'][0]['replay']['checks']['attention']['slice_order_unchanged']=value
            with self.assertRaises(dataset.DataError):fix.sanitize_summary(bad)
        bad=summary();bad['rows'][0]['name']='unexpected'
        with self.assertRaises(dataset.DataError):fix.sanitize_summary(bad)
        bad=summary();bad['rows'][0]['replay']['checks']['attention']['ordering_unchanged']=True
        with self.assertRaises(dataset.DataError):fix.sanitize_summary(bad)

    def test_scoped_adapter_calls_original_guard_and_restores_it(self):
        guard=results.privacy;seen=[]
        def fake_compare(args):
            value=summary();results.privacy(value);seen.append(value)
        with patch.object(results,'compare',side_effect=fake_compare):fix.compare(None)
        self.assertIs(results.privacy,guard);guard(seen[0])
        bad=summary();bad['patient_predictions']={'subject_id':'synthetic'}
        with patch.object(results,'compare',side_effect=lambda args:results.privacy(bad)):
            with self.assertRaises(dataset.DataError):fix.compare(None)
        self.assertIs(results.privacy,guard)

    def test_check_only_stops_before_export_and_restores_guard(self):
        guard=results.privacy;after=[]
        def fake_compare(args):results.privacy(summary());after.append('would write')
        with patch.object(results,'compare',side_effect=fake_compare):fix.compare(None,True)
        self.assertEqual(after,[]);self.assertIs(results.privacy,guard)

    def test_repaired_summary_passes_frozen_bundle_guard_allowlist_roundtrip(self):
        import json,zipfile
        from types import SimpleNamespace
        from g3 import storage as st
        with tempfile.TemporaryDirectory(prefix='g3 export repair spaces ') as tmp:
            root=Path(tmp);folder=root/'aggregate_exports'/('a'*20);folder.mkdir(parents=True)
            value=fix.sanitize_summary(summary());st.immutable_json(folder/'comparison.json',value)
            (folder/'PRIVATE_extra.pt').write_bytes(b'synthetic excluded')
            st.immutable_json(root/'latest_aggregate.json',{'directory':folder.name})
            args=SimpleNamespace(study_root=root)
            with patch.object(st,'context',return_value=({},root,{},None,{})):
                results.bundle(args);results.bundle(args)
            with zipfile.ZipFile(folder/'COMP3710_G3_Seed42_Results.zip') as z:
                self.assertEqual(z.namelist(),['comparison.json']);self.assertEqual(json.loads(z.read('comparison.json')),value)

    def test_complete_frozen_comparison_plan_history_plot_and_zip_pipeline(self):
        import zipfile
        from types import SimpleNamespace
        import train_convnext_v2 as v2
        from g012 import core
        from g012.test_g012 import rows,fetch
        from g012.engine import identity as old_identity
        from g3 import storage as st,execution as ex,operations as ops,model as gm
        from g3.test_g3 import Toy
        v2.seed_everything(42)
        import torch
        metrics,pred=ops.predict(Toy(),core.group_records(rows('validation'),'validation'),fetch,torch.device('cpu'),'G3')
        a1={k:deepcopy(metrics[k]) for k in ('slice','scan','patient')}
        a1['patient']['macro_f1']=.8012151782643586
        with tempfile.TemporaryDirectory(prefix='g3 full export pipeline spaces ') as tmp:
            root=Path(tmp);historical=root/'historical';historical.mkdir()
            plan={'study':'G3_Seed42','sources':{},'configuration':gm.configuration(),
                  'control_mode':'REUSE_G2','a1_configuration':{'synthetic':True}}
            bundle=SimpleNamespace(manifest_hashes={});args=SimpleNamespace(study_root=root,historical_root=historical)
            hist=[{'epoch':e,'patient_accuracy':metrics['patient']['accuracy'],'patient_macro_f1':metrics['patient']['macro_f1'],
                'patient_log_loss':metrics['patient']['log_loss'],'patient_brier':metrics['patient']['brier'],
                'ad_recall':metrics['patient']['per_class']['AD']['recall'],'nc_recall':metrics['patient']['per_class']['NC']['recall'],
                'selected_as_best':e==1} for e in range(1,13)]
            schedules=[{'epoch':e,'order_digest':'synthetic','augmentation_digest':'synthetic'} for e in range(1,13)]
            st.immutable_json(root/'PRIVATE_matched_schedule.json',{'epochs':schedules})
            st.immutable_json(root/'PRIVATE_a1_reference.json',{'predictions':pred,'metrics':a1,'history':hist,
                'completed_epoch':21,'parameters':27818594,'recorded_resources':{}})
            st.immutable_json(root/'preparation.json',{'execution_seconds':1.,'images_decoded':0})
            st.immutable_json(root/'initialization.json',{'shared_digest':'synthetic','checkpoint_sha256':'synthetic'})
            replay={'passed':True,'checks':{'attention':deepcopy(summary()['rows'][0]['replay']['checks']['attention'])},
                    'absolute_probability_tolerance':2e-5,'inference_seconds':1.,'resources':{}}
            for name,folder in [('G2',historical/'G2'),('G3',root/'G3')]:
                folder.mkdir();identity=old_identity(plan,'G2',{}) if name=='G2' else ex.identity(plan,'G3',{})
                st.immutable_json(folder/'PRIVATE_selected_predictions.json',pred)
                st.immutable_json(folder/'training_summary.json',{'identity':identity,'metrics':metrics,'selected_epoch':1,
                    'completed_epoch':12,'stop_reason':'PATIENCE_AFTER_MINIMUM_12','committed_epoch_loop_seconds':10.})
                st.write_text_atomic(folder/'history.csv',st.history_csv(hist))
                st.immutable_json(folder/'PRIVATE_order_augmentation_digests.json',schedules)
                st.immutable_json(folder/'completion.json',{'identity':identity,'hashes':{}})
                st.immutable_json(folder/'active_budget.json',{'charged_seconds':11.,'sessions':[]})
                st.immutable_json(folder/'run_status.json',{'status':'COMPLETED_TRAINING'})
                st.immutable_json(folder/'checkpoint_replay.json',replay)
            guard=results.privacy
            with patch.object(st,'context',return_value=(plan,root,{},bundle,{})),\
                 patch('g012.engine.verify_replay',return_value={**replay,'checks':{}}),\
                 patch.object(ex,'verify_replay',return_value=replay):
                fix.compare(args);fix.compare(args);results.bundle(args);results.bundle(args)
            self.assertIs(results.privacy,guard)
            folder=root/'aggregate_exports'/st.read(root/'latest_aggregate.json')['directory']
            comparison=st.read(folder/'comparison.json');public_plan=st.read(folder/'plan.json')
            self.assertEqual(public_plan['study'],'G3_Seed42');self.assertNotIn('rows',public_plan)
            self.assertTrue(comparison['rows'][-1]['replay']['checks']['attention']['ordering_unchanged'])
            with zipfile.ZipFile(folder/'COMP3710_G3_Seed42_Results.zip') as z:
                self.assertEqual(set(z.namelist()),set(results.ALLOWLIST))
                for name in z.namelist():self.assertEqual(z.read(name),(folder/name).read_bytes())


if __name__=='__main__':unittest.main(verbosity=2)
