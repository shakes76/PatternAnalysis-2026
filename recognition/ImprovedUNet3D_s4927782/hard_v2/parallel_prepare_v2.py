"""One bounded CPU engineering recovery; frozen v2 preprocessing is unchanged.

The original sequential preparation timed out before its manifest was committed.
This helper rebuilds all TRAIN/VAL caches in a fresh directory using four worker
processes. It never reuses unregistered partial arrays or opens TEST pixels.
The scientific protocol and all eighteen frozen source files remain unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing
from pathlib import Path

import v2_pipeline as v2


def validate_amendment(amendment, protocol, helper_digest, workers):
    if (amendment.get('experiment_id') != protocol['experiment_id']
            or amendment.get('original_protocol_sha256') != protocol['_sha256']
            or amendment.get('failed_job_id') != 704519
            or amendment.get('engineering_only') is not True
            or amendment.get('scientific_configuration_changed') is not False
            or not isinstance(amendment.get('reason'), str) or not amendment['reason'].strip()
            or amendment.get('recovery') != {'workers': 4, 'maximum_walltime_minutes': 20}
            or amendment.get('helper_sha256') != helper_digest or workers != 4):
        raise ValueError('Recovery must bind the unchanged protocol, source digest and one four-worker/20-minute engineering amendment.')


def initial_record(source_record, configuration, dependencies):
    record = {key: source_record[key] for key in ('patient', 'visit', 'split', 'source_image', 'source_mask')}
    record['cache_path'] = 'volumes/' + dependencies.utils.safe_case_name(record['patient'], record['visit']) + '.npz'
    record.update(v2.expected_geometry(source_record, configuration['target_shape_dhw']))
    record['pending'] = record['split'] == 'test'
    return record


def build_manifest(source, records, configuration, source_digest, protocol_digest, dependencies):
    """Exactly the scientific manifest schema produced by frozen v2.prepare."""
    result = {key: copy.deepcopy(source[key]) for key in
              ('num_classes', 'label_ids', 'raw_label_to_index', 'patient_splits', 'audit_sha256')}
    counts = {split: sum(r['split'] == split for r in source['records'])
              for split in ('train', 'validation', 'test')}
    result.update({'schema_version': 1, 'v2_schema': v2.SCHEMA, 'records': records,
                   'label_semantics': copy.deepcopy(source.get('label_semantics', {})),
                   'class_names': v2.CLASS_NAMES, 'target_shape_dhw': configuration['target_shape_dhw'],
                   'partial': False, 'development_only': False,
                   'metrics_grid': 'downsampled_resampled_grid', 'native_resolution_accuracy_claimed': False,
                   'original_manifest_sha256': source_digest, 'protocol_sha256': protocol_digest,
                   'preprocessor_source_sha256': dependencies.source_hashes['tools/prepare_cache.py'],
                   'preprocessing_fit_scope': 'each individual image only; no split-level fitting',
                   'pending_test_policy': 'TEST entries have no pixel-derived new-grid metadata; prepare-test may fill cache files without changing this manifest.',
                   'prepared_record_count': counts['train'] + counts['validation'],
                   'source_audit_record_count': len(records)})
    return result


def complete_record(record, output, raw_root, configuration, dependencies):
    """One worker invokes the same frozen preprocessor, geometry guard and writer."""
    if record['split'] not in ('train', 'validation') or record['pending']:
        raise ValueError('A recovery worker may open TRAIN/VAL pixels only.')
    image, mask = v2._source_paths(raw_root, record)
    arrays, metadata = dependencies.prepare.preprocess_pair(
        image, mask, configuration['target_shape_dhw'], list(range(6)))
    v2._check_source_geometry(record, metadata)
    record = copy.deepcopy(record)
    record.update(metadata)
    record['cache_sha256'] = v2._write_cache(Path(output) / record['cache_path'], arrays)
    record['source_image_sha256'], record['source_mask_sha256'] = v2.sha256(image), v2.sha256(mask)
    return record


_WORKER = None


def initialise_worker(project_root, protocol_path, output, raw_root):
    global _WORKER
    protocol = v2.load_protocol(protocol_path)
    v2.verify_review_sources(protocol, Path(protocol_path).resolve().parent)
    dependencies = v2.load_dependencies(project_root, protocol)
    _WORKER = (Path(output), Path(raw_root), protocol['configuration'], dependencies)


def worker_record(payload):
    index, record = payload
    if _WORKER is None:
        raise RuntimeError('Recovery worker was not initialised.')
    return index, complete_record(record, *_WORKER)


def recover(project_root, protocol_path, original_manifest, raw_root, output_dir, amendment_path, workers=4):
    protocol_path, original_manifest = Path(protocol_path).resolve(strict=True), Path(original_manifest).resolve(strict=True)
    protocol = v2.load_protocol(protocol_path)
    helper_digest = v2.sha256(__file__)
    amendment = v2.read_json(amendment_path)
    validate_amendment(amendment, protocol, helper_digest, workers)
    v2.verify_review_sources(protocol, protocol_path.parent)
    dependencies = v2.load_dependencies(project_root, protocol)
    source = dependencies.dataset.load_manifest(original_manifest)
    if (source.get('partial') or source.get('development_only') or source['num_classes'] != 6
            or source['label_ids'] != list(range(6))):
        raise ValueError('Recovery requires the complete original six-class manifest.')
    counts = {split: sum(r['split'] == split for r in source['records']) for split in ('train', 'validation', 'test')}
    if counts != {'train': 143, 'validation': 38, 'test': 30}:
        raise ValueError('Recovery must retain exactly the original 143/38/30 cases.')
    source_digest = v2.sha256(original_manifest)
    if source_digest != protocol['required_original_sha256']['artifacts/cache64/manifest.json']:
        raise ValueError('Original manifest identity differs from the fixed scientific protocol.')
    output = v2._fresh_directory(output_dir)
    receipt = {'status': 'recovering_train_validation', 'started_at': v2.utc_now(),
               'source_manifest_sha256': source_digest, 'protocol_sha256': protocol['_sha256'],
               'source_hashes': dependencies.source_hashes, 'v2_pipeline_sha256': v2.sha256(v2.__file__),
               'recovery_helper_sha256': helper_digest, 'engineering_amendment_sha256': v2.sha256(amendment_path),
               'failed_original_job_id': 704519, 'workers': workers, 'maximum_walltime_minutes': 20,
               'scientific_configuration_changed': False, 'test_pixels_read': False,
               'partial_arrays_reused': False, 'records_completed': 0}
    v2.write_json(output / 'preparation_receipt.json', receipt)
    try:
        records = [initial_record(r, protocol['configuration'], dependencies) for r in source['records']]
        payloads = [(i, r) for i, r in enumerate(records) if not r['pending']]
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialise_worker,
                                 initargs=(str(project_root), str(protocol_path), str(output), str(raw_root))) as pool:
            for index, result in pool.map(worker_record, payloads, chunksize=1):
                records[index] = result
                receipt['records_completed'] += 1
                v2.write_json(output / 'preparation_receipt.json', receipt)
                print(f"Recovery {receipt['records_completed']}/181: {result['patient']}/{result['visit']}", flush=True)
        manifest = build_manifest(source, records, protocol['configuration'], source_digest,
                                  protocol['_sha256'], dependencies)
        manifest_path = output / 'manifest.json'
        v2.write_json(manifest_path, manifest)
        checked = v2.load_new_manifest(manifest_path, protocol, dependencies)
        decoded = 0
        for split in ('train', 'validation'):
            data = v2.CachedVolumes(manifest_path, checked, split)
            for index in range(len(data)):
                data.load(index)  # Actual bytes, CRC/shape/dtype/finiteness/counts/spacing validation.
                decoded += 1
        if decoded != 181 or any((output / r['cache_path']).exists() for r in records if r['pending']):
            raise ValueError('Recovery scope differs from precisely 181 TRAIN/VAL caches and zero TEST caches.')
        v2.verify_review_sources(protocol, protocol_path.parent)
        v2.load_dependencies(project_root, protocol)
        if v2.sha256(original_manifest) != source_digest or v2.sha256(protocol_path) != protocol['_sha256']:
            raise RuntimeError('Frozen source manifest or scientific protocol changed during recovery.')
        if v2.sha256(__file__) != helper_digest or v2.sha256(amendment_path) != receipt['engineering_amendment_sha256']:
            raise RuntimeError('Engineering recovery source/amendment changed during execution.')
        receipt.update(status='train_validation_complete_test_pending', completed_at=v2.utc_now(),
                       manifest_sha256=v2.sha256(manifest_path), decoded_cache_records_verified=decoded,
                       frozen_scientific_sources_unchanged=True)
        v2.write_json(output / 'preparation_receipt.json', receipt)
        return manifest_path
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        v2.write_json(output / 'preparation_receipt.json', receipt)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'protocol', 'original-manifest', 'raw-root', 'output-dir', 'amendment'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args(argv)
    result = recover(a.project_root, a.protocol, a.original_manifest, a.raw_root, a.output_dir, a.amendment, a.workers)
    print(f'Engineering recovery completed: {result}', flush=True)


if __name__ == '__main__':
    main()
