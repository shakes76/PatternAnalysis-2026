"""Explicit fixed G3 commands. Real MRI operations require user invocation."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g012 import core
from g3 import storage as st,execution as ex,results
import train_convnext_v2 as v2


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('verify')
    for command in ('prepare','control','smoke','run','resume','status','replay','compare','bundle','_worker','_replay_worker'):
        s=sub.add_parser(command)
        for name in ('data-root','splits-dir','study-root','historical-root','historical-source'):s.add_argument('--'+name,type=Path,required=True)
        if command in ('run','resume','smoke','replay','_worker','_replay_worker'):s.add_argument('--variant',choices=('G3',),required=True)
        if command in ('_worker','_replay_worker'):s.add_argument('--gate',type=Path,required=True)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if a.command=='verify':print('G3_COMMITTED_IMPLEMENTATION_VERIFIED',core.digest(st.implementation()));return
    if a.command=='prepare':st.prepare(a)
    elif a.command=='control':
        plan,root,binding,bundle,groups=st.context(a)
        print('G3_CONTROL_VERIFIED_REUSE_G2_NO_NEW_INFERENCE')
    elif a.command=='smoke':ex.smoke(a)
    elif a.command in ('run','resume'):ex.supervised(a,a.command=='resume')
    elif a.command=='replay':ex.supervised(a,replay_only=True)
    elif a.command=='compare':results.compare(a)
    elif a.command=='bundle':results.bundle(a)
    elif a.command=='status':
        st.implementation()
        if not a.study_root.exists():print('NOT_PREPARED');return
        try:
            with v2.run_lock(a.study_root):
                plan,root,binding,bundle,groups=st.context(a);folder=root/'G3'
                status=st.read(folder/'run_status.json')['status'] if (folder/'run_status.json').exists() else 'NOT_STARTED'
                charged=st.read_budget(folder/'active_budget.json')['charged_seconds'] if (folder/'active_budget.json').exists() else 0.
                print('G3',status,'CHARGED_SECONDS',charged,'CAP',10800,'HISTORICAL_CONTROL','G2')
        except RuntimeError as e:
            if 'locked by another process' not in str(e):raise
            print('STUDY_ACTIVE_LOCKED; inspect append-only G3/execution.log, not live atomic JSON.')
    else:
        try:(ex.worker if a.command=='_worker' else ex.replay_worker)(a)
        except BaseException as e:
            import torch
            status='BLOCKED_OOM' if isinstance(e,torch.cuda.OutOfMemoryError) else 'BLOCKED_HEADROOM' if 'HEADROOM_BLOCKER' in str(e) else 'DISK_BLOCKED' if 'DISK_BLOCKER' in str(e) else 'FAILED_NEEDS_INVESTIGATION'
            v2.atomic_json(a.study_root/'G3'/'worker_failure.json',{'status':status,'exception_type':type(e).__name__});raise


if __name__=='__main__':main()
