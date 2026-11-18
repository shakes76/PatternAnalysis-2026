"""Train-only class/patient balancing with unchanged epoch draw count.

WeightedRandomSampler API: https://docs.pytorch.org/docs/2.6/data.html
Weights implement equal class mass, equal patients within class, then uniform
slices within each patient. Repeated scans do not increase a patient's mass.
"""
from collections import Counter
import math

SAMPLING_NAMES = ('slice_uniform', 'class_patient_balanced')


def sampling_weights(rows: list[dict], name: str) -> tuple[list[float] | None, dict]:
    """Describe expected sampling mass using training identifiers only."""
    if name not in SAMPLING_NAMES or not rows:
        raise ValueError('Unsupported sampling mode or empty training rows.')
    if name == 'slice_uniform':
        return None, {'name': name, 'replacement': False, 'draws_per_epoch': len(rows),
                      'loss_class_weight_source': 'original_training_slice_counts'}
    owners: dict[str, int] = {}
    counts = Counter()
    for row in rows:
        if row.get('partition') != 'development':
            raise ValueError('Balanced sampling requires development training rows.')
        patient = row['patient_id']
        label = int(row['label'])
        if label not in (0, 1) or (patient in owners and owners[patient] != label):
            raise ValueError('Balanced sampling requires one binary diagnosis per patient.')
        owners[patient] = label
        counts[patient] += 1
    class_patients = Counter(owners.values())
    if set(class_patients) != {0, 1}:
        raise ValueError('Balanced sampling requires both patient classes.')
    weights = [1.0 / (2 * class_patients[int(r['label'])] * counts[r['patient_id']]) for r in rows]
    if not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-12):
        raise ValueError('Invalid total sampling mass.')
    return weights, {'name': name, 'replacement': True, 'draws_per_epoch': len(rows),
                     'patients_by_class': {str(k):v for k,v in sorted(class_patients.items())},
                     'expected_class_probability': {'0':0.5, '1':0.5},
                     'slices_per_patient_min':min(counts.values()), 'slices_per_patient_max':max(counts.values()),
                     'loss_class_weight_source':'balanced_sampler_prior_pos_weight_one',
                     'sampler_seed_offset':1000003}
