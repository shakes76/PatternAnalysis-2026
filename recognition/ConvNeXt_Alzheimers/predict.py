"""Reproducible COMP3710 prediction and final-test interface.

freeze: check development evidence and save the exact final-test plan (no images).
evaluate: explicitly open the frozen test, load trusted weights, infer, report,
          and generate PRIVATE representative examples; resumes the same plan.
recover-lock: after a reboot, archive a stale lock without deleting predictions.

Use this installed file in recognition/ConvNeXt_Alzheimers. See
FINAL_EVALUATION_METHOD.md for protocol, limitations, privacy and AI disclosure.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == 'infer':
        from submission_prediction import main as inference_main
        inference_main(sys.argv[2:])
        return 0
    # Legacy evaluation keeps its original integrity/explicit-test guards.
    # Its frozen source revision is documented in HISTORICAL_REPLAY.md.
    from final_evaluation import evaluate, freeze, recover_lock
    # Revised A1 evaluation is explicit; historical commands are not redirected.
    if len(sys.argv) > 1 and sys.argv[1] in (
            "revised-freeze", "revised-evaluate", "revised-publish"):
        from revised_evaluation import main as revised_main
        return revised_main(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'evaluate', 'recover-lock'):
        p = commands.add_parser(name)
        p.add_argument('--private-root', type=Path, required=True)
        if name == 'freeze':
            p.add_argument('--device', choices=('cuda','cpu'), default='cuda')
        elif name == 'evaluate':
            p.add_argument('--confirm-final-test', action='store_true',
                help='Acknowledges that this opens the held-out test; no subsequent test-guided tuning.')
        else:
            p.add_argument('--confirm-no-evaluation-running', action='store_true')
    a = parser.parse_args()
    try:
        if a.command == 'freeze': freeze(a.private_root, a.device)
        elif a.command == 'evaluate':
            if not a.confirm_final_test:
                parser.error('evaluate requires --confirm-final-test; first freeze and commit the plan.')
            evaluate(a.private_root)
        else: recover_lock(a.private_root, a.confirm_no_evaluation_running)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'STEP10_STOPPED: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('Keep all existing results. Do not retrain, change cutoffs or create a different test split.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
