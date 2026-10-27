"""Portable image-only entry for the already selected Hard checkpoint.

The frozen infer_v2 adapter's optional selection-freeze expects the stopped
2D/3D pair. This wrapper instead validates the genuine Hard-only freeze before
calling the unchanged adapter and writes an explicit Hard-only receipt after it.
No target label, cache manifest, model search or new postprocessing is used.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json

from verify_package import digest, load, require

HERE = Path(__file__).resolve().parent

def check_selection(selection, scope_digest, checkpoint_digest):
    require(selection.get('status') == 'frozen', 'Selection is not frozen')
    require(selection.get('scope_plan_sha256') == scope_digest, 'Hard scope identity changed')
    require(set(selection.get('models', {})) == {'improved3d'}, 'Use the real Hard-only selection')
    choice = selection['models']['improved3d']
    require(choice.get('completed_epochs') == 30, 'Incomplete Hard training')
    require(choice.get('checkpoint_sha256') == checkpoint_digest, 'Weights differ from the selected Hard checkpoint')
    return choice

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=HERE.parent)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args(argv)
    selection_path = HERE / 'artifacts/hard_selection.json'
    scope_path = HERE / 'hard_followup_protocol.json'
    checkpoint_digest = digest(args.checkpoint)
    check_selection(load(selection_path), digest(scope_path), checkpoint_digest)
    # Import tensor execution only after explicit selected-checkpoint validation.
    import infer_v2
    infer_v2.main(['--image', str(args.image), '--checkpoint', str(args.checkpoint),
                  '--protocol', str(HERE / 'protocol_v2.json'), '--project-root', str(args.project_root),
                  '--device', args.device, '--output-dir', str(args.output_dir)])
    image_receipt = args.output_dir / 'receipt.json'
    require(load(image_receipt)['checkpoint_sha256'] == checkpoint_digest, 'Inference checkpoint mismatch')
    receipt = {'status': 'completed', 'image_only_inference': True,
               'checkpoint_sha256': checkpoint_digest,
               'selection_freeze_sha256': digest(selection_path),
               'scope_plan_sha256': digest(scope_path),
               'image_only_receipt_sha256': digest(image_receipt),
               'integration_adapter_sha256': digest(__file__),
               'native_resolution_accuracy_claimed': False}
    with (args.output_dir / 'hard_scope_receipt.json').open('x') as output:
        output.write(json.dumps(receipt, indent=2) + '\n')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
