"""Explicit G012 seed-42 study commands. No real operation runs on import.

Codex assisted implementation and synthetic verification. See G012_PLAN.md.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g012 import core, storage as st, engine, results
import train_convnext_v2 as v2


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('verify',help='Read-only committed source/plan identity check.')
    for command in ('prepare','historical','smoke','run','resume','status','replay','compare','bundle','_worker','_replay_worker'):
        s=sub.add_parser(command)
        for name in ('data-root','splits-dir','study-root'): s.add_argument('--'+name,type=Path,required=True)
        if command in ('smoke','run','resume','replay','_worker','_replay_worker'): s.add_argument('--variant',choices=core.VARIANTS,required=True)
        if command=='prepare':
            for name in ('summary','predictions','history','checkpoint','replay-receipt','replayed-predictions'):
                s.add_argument('--a1-'+name,type=Path,required=True)
        if command in ('_worker','_replay_worker'):s.add_argument('--gate',type=Path,required=True,help=argparse.SUPPRESS)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if a.command=='verify':
        plan=st.implementation()
        print('G012_COMMITTED_IMPLEMENTATION_VERIFIED',core.digest(plan));return
    if a.command=='prepare':st.prepare(a)
    elif a.command=='historical':
        # Serial with training/status writers; reuses cached validation evidence.
        with v2.run_lock(a.study_root):st.historical(a)
    elif a.command=='smoke':engine.smoke(a)
    elif a.command in ('run','resume'):engine.supervised(a,a.command=='resume')
    elif a.command=='replay':engine.replay(a)
    elif a.command=='compare':results.compare(a)
    elif a.command=='bundle':results.bundle(a)
    elif a.command=='status':
        plan=st.implementation()
        if not a.study_root.exists():print('NOT_PREPARED');return
        try:
            with v2.run_lock(a.study_root):
                plan,root,binding,bundle,groups=st.context(a)
                for variant in core.VARIANTS:
                    folder=root/variant
                    status=st.read(folder/'run_status.json') if (folder/'run_status.json').exists() else {'status':'NOT_STARTED'}
                    charged=st.read(folder/'active_budget.json')['charged_seconds'] if (folder/'active_budget.json').exists() else 0.
                    print(variant,status['status'],'CHARGED_SECONDS',charged,'CAP',st.CAP)
        except RuntimeError as exc:
            if 'locked by another process' not in str(exc):raise
            print('STUDY_ACTIVE_LOCKED; read append-only G0/G1/G2 execution.log; do not monitor live atomic JSON with editors.')
    else:
        try:
            if a.command == '_replay_worker':engine.replay_worker(a)
            else:engine.worker(a)
        except BaseException as exc:
            import torch
            status='BLOCKED_OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'BLOCKED_HEADROOM' if 'HEADROOM_BLOCKER' in str(exc) else 'DISK_BLOCKED' if 'DISK_BLOCKER' in str(exc) else 'FAILED_NEEDS_INVESTIGATION'
            v2.atomic_json(a.study_root/a.variant/'worker_failure.json',{'status':status,'exception_type':type(exc).__name__})
            raise


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if isinstance(exc,SystemExit):raise
        traceback.print_exc();sys.exit(1)
