"""Reproduce the already approved fixed assignments from explicit data inputs.

Reuses the reviewed grouping/allocation/CSV helpers verbatim. No image pixels,
private pointers, model predictions, or prior audit directory are needed here.
This reproduces recorded approval provenance; it does not issue new approval.
Codex assisted with packaging and focused tests.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from preparation_helpers import prepare_patient_split as reviewed

BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reproduce(root,output,reference=None):
    from dataset import load_finalized_splits, require
    root,output=Path(root).resolve(),Path(output).resolve()
    protocol=json.loads((BASE/'APPROVED_PARTITIONS.json').read_text(encoding='utf-8'))
    require(root.is_dir() and not output.exists(),'Use an existing data root and a NEW output directory.')
    require(root!=output and root not in output.parents,'Output must be outside the dataset.')
    require(not any((p/'.git').exists() for p in (output,*output.parents)),
            'Patient manifests must remain outside Git.')
    metadata_path=root/'meta_data_with_label.json'
    require(sha(metadata_path)==protocol['metadata_sha256'],
            'Dataset metadata differs from the recorded approved dataset; do not invent approval or repartition.')
    metadata=reviewed.read_json(metadata_path)
    rows=[];scan_info={};subject_labels={};scan_slices=defaultdict(set)
    for folder in reviewed.FOLDERS:
        source_split,source_class=folder.split('/')
        directory=root/'AD_NC'/folder
        require(directory.is_dir(),'A source folder is missing.')
        for path in sorted(directory.iterdir()):
            require(not path.is_dir(),'Unexpected nested data directory.')
            if path.suffix.lower() not in ('.jpeg','.jpg'):continue
            prefix,sep,index=path.stem.rpartition('_')
            require(bool(sep) and index.isdecimal() and prefix in metadata,'Unmapped scan/slice filename.')
            subject,_=reviewed.subject_for_record(metadata[prefix])
            label=1 if source_class=='AD' else 0
            require(type(metadata[prefix].get('label')) is int and metadata[prefix]['label']==2*label,
                    'Metadata diagnosis and source class disagree.')
            info=(source_split,source_class,subject,label)
            require(prefix not in scan_info or scan_info[prefix]==info,'Conflicting scan mapping.')
            require(subject not in subject_labels or subject_labels[subject]==label,'Conflicting patient diagnosis.')
            require(int(index) not in scan_slices[prefix],'Duplicate slice index.')
            scan_info[prefix]=info;subject_labels[subject]=label;scan_slices[prefix].add(int(index))
            rows.append(dict(image_path=path.relative_to(root).as_posix(),subject_id=subject,scan_id=prefix,
                slice_index=int(index),source_split=source_split,source_class=source_class,
                metadata_label=2*label,label=label))
    require(all(len(x)==20 for x in scan_slices.values()),'Expected 20 slices per scan.')
    assigned=reviewed.assign_split(rows)
    encoded={f'{s}.csv':reviewed.csv_bytes([r for r in assigned if r['split']==s]) for s in reviewed.PARTITIONS}
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in encoded.items()}
    counts=reviewed.split_counts(assigned)
    require(hashes==protocol['manifest_sha256'] and counts==protocol['counts'],
            'Recreated assignments differ from the approved hashes/counts; stop without writing new partitions.')
    if reference is not None:
        reference=Path(reference).resolve()
        for name,value in encoded.items():
            require((reference/name).read_bytes()==value,'Original and reproduced manifest bytes differ.')
    # Provenance is the existing reported approval of these exact assignments,
    # not a new attestation from the person executing this reproduction command.
    summary={'format_version':1,'status':'FINALIZED_UNDER_USER_CONFIRMED_PATIENT_RULE_AND_TUTOR_APPROVAL',
        'mapping_structural_checks_passed':True,'patient_identifier_semantics_confirmed_by_user':True,
        'tutor_split_change_approved_as_reported_by_user':True,'zero_group_overlap_verified':True,
        'splits_finalized':True,'new_pairwise_candidate_subject_overlap':0,
        'model_label_map':{'NC':0,'AD':1},'metadata_to_model_labels':{'0':0,'2':1},
        'manifest_sha256':hashes,'proposed_or_final_counts':counts,
        'evidence':{'images':len(rows),'scan_prefixes':len(scan_info),'candidate_subjects':len(subject_labels),
                    'metadata_sha256':protocol['metadata_sha256']},
        'mapping_rule':protocol['mapping_rule'],'seed':42,'approval_provenance':protocol['approval_provenance'],
        'reproduction_only':True,'new_approval_granted':False,'images_decoded':0,
        'image_integrity_audit_repeated':False,'source_rangpur_copy_hash_verified':False,
        'caution':'Recorded user-reported approval for exact existing assignments; date/verbatim feedback unrecorded. '
                  'Filename/metadata reconstruction is not a new image integrity audit or semantic identity certification.'}
    output.mkdir(parents=True)
    for name,value in encoded.items():(output/name).write_bytes(value)
    (output/'split_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    bundle=load_finalized_splits(root,output)
    receipt={'manifest_sha256':bundle.manifest_hashes,'counts':counts,'exact_original_bytes_compared':reference is not None,
        'public_protocol_sha256':sha(BASE/'APPROVED_PARTITIONS.json'),'images_decoded':0,
        'reviewed_helper_sha256':sha(BASE/'preparation_helpers'/'prepare_patient_split.py')}
    (output/'reproduction_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('APPROVED_MANIFESTS_REPRODUCED; image pixels not decoded; original partitions unchanged')
    print(json.dumps(receipt,indent=2))
    return receipt

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare',));p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--reference-splits-dir',type=Path)
    a=p.parse_args(argv);reproduce(a.data_root,a.output_dir,a.reference_splits_dir)

if __name__=='__main__':main()
