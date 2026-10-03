"""Stable seed identities for data construction and frozen-model probing."""
import hashlib
import json
import math

DATA_ROLES = ('gsm8k_aug_sample', 'dapo_sample', 'train_validation_split')


def _uint32(data):
    return int.from_bytes(hashlib.sha256(data).digest()[:4], 'big')


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return value


def derive_dataset_seeds(master_seed):
    _integer(master_seed, 'master_seed')
    return {role: _uint32(f'adanoise.dataset-seeds.v1:{master_seed}:{role}'.encode('utf-8'))
            for role in DATA_ROLES}


def derive_rollout_seed(problem_id, scale, rollout_id, root_seed=12345):
    _integer(root_seed, 'root_seed')
    _integer(rollout_id, 'rollout_id')
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError('problem_id must be non-empty')
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or float(scale) < 0:
        raise ValueError('scale must be finite and non-negative')
    scale_string = format(float(scale) if float(scale) != 0 else 0.0, '.17g')
    payload = json.dumps([root_seed, problem_id, scale_string, rollout_id],
                         ensure_ascii=True, separators=(',', ':')).encode('ascii')
    return _uint32(b'adanoise.probe-seeds.v1:' + payload)
