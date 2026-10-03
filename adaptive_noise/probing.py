"""Pure CPU control of cumulative frozen-checkpoint probing."""
from fractions import Fraction
import math

from adaptive_noise.seeds import derive_rollout_seed

SCALES = (0.0, .25, .5, 1.0, 2.0)
SOURCE_MAX_TOKENS = {'gsm8k_aug': 128, 'dapo_math': 4096}


def decide_probe(rows, split):
    if split not in {'train', 'validation'}:
        raise ValueError('probing selection accepts only train/validation')
    buckets = {s: {} for s in SCALES}
    identities = set()
    for row in rows:
        scale, index, reward = row['scale'], row['rollout_id'], row['reward']
        if isinstance(scale, bool) or not math.isfinite(float(scale)) or float(scale) not in buckets:
            raise ValueError('unknown scale in approved probing grid')
        scale = float(scale)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= (1 if scale == 0 else 7):
            raise ValueError('rollout_id exceeds approved budget')
        if row.get('status') != 'completed' or reward not in (0, 1) or reward is None:
            raise ValueError('only completed binary-reward rollouts can be counted')
        if index in buckets[scale]:
            raise ValueError('duplicate rollout identity')
        buckets[scale][index] = int(reward)
        identities.add(row['problem_id'])
    if len(identities) > 1:
        raise ValueError('decision rows must refer to one problem_id')
    for bucket in buckets.values():
        if sorted(bucket) != list(range(len(bucket))):
            raise ValueError('non-contiguous rollout IDs')
    counts = {s: len(buckets[s]) for s in SCALES}
    summary = [{'scale': s, 'completed_count': counts[s], 'success_count': sum(buckets[s].values()),
                'q_hat': sum(buckets[s].values()) / counts[s] if counts[s] else None} for s in SCALES]
    last_complete = None
    # Decisions only use complete rounds. A terminal round cannot have later rows.
    for m in (3, 5, 7):
        if counts[0] == 1 and all(counts[s] >= m for s in SCALES if s != 0):
            rates = {s: Fraction(sum(buckets[s][i] for i in range(1 if s == 0 else m)), 1 if s == 0 else m) for s in SCALES}
            delta = max(rates.values()) - min(rates.values())
            reason = None
            if all(q == 0 for q in rates.values()): reason = 'all_wrong'
            elif all(q == 1 for q in rates.values()): reason = 'all_correct'
            elif delta > Fraction(2, 5): reason = 'noise_sensitive'
            elif m == 7: reason = 'not_sensitive_at_cap'
            if reason:
                if any(counts[s] > (1 if s == 0 else m) for s in SCALES):
                    raise ValueError('rows collected after terminal decision')
                best = min(s for s in SCALES if rates[s] == max(rates.values()))
                return {'terminal': True, 'reason': reason, 'stage': m, 'target_m': m,
                        'keep': split == 'validation' or reason == 'noise_sensitive',
                        'delta': float(delta), 's_star': best, 'summary': summary}
            last_complete = {'stage': m, 'delta': float(delta)}
        else:
            if any(counts[s] > m for s in SCALES if s != 0):
                raise ValueError('later-stage rows before preceding round completed')
            return {'terminal': False, 'reason': 'continue_probing' if last_complete and all(counts[s] == last_complete['stage'] for s in SCALES if s != 0) else 'incomplete',
                    'stage': last_complete['stage'] if last_complete else None, 'target_m': m,
                    'keep': None, 'delta': last_complete['delta'] if last_complete else None,
                    's_star': None, 'summary': summary}
        # A completed non-terminal round advances to the next cumulative target.
    raise AssertionError('unreachable probing state')


def generation_settings(source, scale):
    if source not in SOURCE_MAX_TOKENS:
        raise ValueError('unknown training source')
    return {'max_new_tokens': SOURCE_MAX_TOKENS[source], 'temperature': .6,
            'top_p': .95, 'max_topk': 10, 'gumbel_softmax_temperature': 1.0,
            'noise_scale': float(scale), 'add_noise_gumbel_softmax': True,
            'use_one_sided_gumbel_noise': False}


def plan_requests(problem_id, split, source, rows):
    if source not in SOURCE_MAX_TOKENS:
        raise ValueError('unknown training source')
    if any(row['problem_id'] != problem_id for row in rows):
        raise ValueError('foreign problem_id in rollout history')
    decision = decide_probe(rows, split)
    if decision['terminal']:
        return []
    existing = {(float(row['scale']), row['rollout_id']) for row in rows}
    requests = []
    for scale in SCALES:
        for index in range(1 if scale == 0 else decision['target_m']):
            if (scale, index) in existing:
                continue
            requests.append({'problem_id': problem_id, 'source': source, 'split': split,
                             'scale': scale, 'rollout_id': index,
                             'seed': derive_rollout_seed(problem_id, scale, index),
                             **generation_settings(source, scale)})
    return requests
