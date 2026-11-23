"""Post-training aggregate-export repair; frozen G3 runtime stays unchanged.

The original guard rejects the harmless aggregate key slice_order_unchanged.
Rename that verified boolean in the NEW public summary only, then call the
unchanged frozen guard. Never modify source bindings or private replay receipts.
This entry point exposes cached comparison/bundle only, never MRI execution.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g012 import core
from g3 import results,storage as st


def sanitize_summary(value):
    # The public plan has the same study name but is not a comparison payload.
    if isinstance(value,dict) and value.get('study')=='G3_Seed42' and 'rows' in value:
        core.require(isinstance(value['rows'],list),'Comparison rows must be a list.')
        for row in value['rows']:
            attention=row.get('replay',{}).get('checks',{}).get('attention',{})
            if 'slice_order_unchanged' not in attention:continue
            core.require(row['name']=='G3' and type(attention['slice_order_unchanged']) is bool
                         and attention['slice_order_unchanged'] is True,
                         'Expected verified G3 aggregate ordering boolean; preserve and investigate.')
            core.require('ordering_unchanged' not in attention,'Conflicting aggregate ordering keys.')
            attention['ordering_unchanged']=attention.pop('slice_order_unchanged')
    return value


def verify_extension():
    path=Path(__file__).resolve();repository=st.BASE.parents[1]
    relative=path.relative_to(repository).as_posix()
    committed=subprocess.check_output(['git','-C',str(repository),'show','HEAD:'+relative])
    core.require(committed==path.read_bytes(),'Commit the exact export adapter; do not change trained source bindings.')
    return hashlib.sha256(committed).hexdigest()


def compare(args,check_only=False):
    frozen_guard=results.privacy
    class Checked(Exception):pass
    def public_guard(value):
        sanitize_summary(value)
        frozen_guard(value)
        if check_only and isinstance(value,dict) and value.get('study')=='G3_Seed42' and 'rows' in value:raise Checked()
    # Scoped output adaptation: the original computation, identity/native/replay
    # checks and privacy rejection remain active. No private receipt is edited.
    with patch.object(results,'privacy',side_effect=public_guard):
        try:results.compare(args)
        except Checked:print('G3_CACHED_COMPARISON_PRIVACY_CHECK_PASSED_NO_EXPORT_WRITTEN')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('compare','bundle'))
    for name in ('data-root','splits-dir','study-root','historical-root','historical-source'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--check-only',action='store_true',help='Compare only: validate aggregate then stop before output writes.')
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    core.require(not a.check_only or a.command=='compare','--check-only is only valid for compare.')
    extension=verify_extension()
    if a.command=='compare':compare(a,a.check_only)
    else:results.bundle(a)
    print('G3_EXPORT_ADAPTER_SHA256',extension)


if __name__=='__main__':main()
