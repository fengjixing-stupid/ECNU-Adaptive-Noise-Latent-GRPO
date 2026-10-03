import pytest
from adaptive_noise.probing import decide_probe, plan_requests
from adaptive_noise.seeds import derive_rollout_seed

SCALES = [0, .25, .5, 1, 2]


def records(m, counts):
    rows = []
    for s, successes in zip(SCALES, counts):
        n = 1 if s == 0 else m
        for i in range(n):
            rows.append({'problem_id': 'p', 'scale': s, 'rollout_id': i,
                         'reward': int(i < successes), 'status': 'completed'})
    return rows


def test_initial_requests_and_seed_identity():
    requests = plan_requests('p', 'train', 'gsm8k_aug', [])
    assert len(requests) == 13
    assert sum(r['scale'] == 0 for r in requests) == 1
    assert all(r['seed'] == derive_rollout_seed('p', r['scale'], r['rollout_id']) for r in requests)
    assert all(r['max_new_tokens'] == 128 and r['add_noise_gumbel_softmax'] is True for r in requests)
    assert plan_requests('p', 'validation', 'dapo_math', [])[0]['max_new_tokens'] == 4096


@pytest.mark.parametrize('split,keep', [('train', False), ('validation', True)])
@pytest.mark.parametrize('correct,reason', [(False, 'all_wrong'), (True, 'all_correct')])
def test_all_same_priority_and_validation_kept(split, keep, correct, reason):
    counts = [1, 3, 3, 3, 3] if correct else [0, 0, 0, 0, 0]
    decision = decide_probe(records(3, counts), split)
    assert decision['reason'] == reason and decision['keep'] is keep
    assert decision['terminal'] is True
    assert plan_requests('p', split, 'dapo_math', records(3, counts)) == []


def test_sensitive_and_best_scale_tie():
    decision = decide_probe(records(3, [0, 2, 2, 1, 1]), 'train')
    assert decision['reason'] == 'noise_sensitive' and decision['keep'] is True
    assert decision['s_star'] == .25


def test_cumulative_three_five_seven_never_repeats_zero():
    rows = records(3, [0, 1, 1, 1, 1])
    add = plan_requests('p', 'train', 'dapo_math', rows)
    assert len(add) == 8
    assert all(r['scale'] != 0 and r['rollout_id'] in (3, 4) for r in add)
    assert len(plan_requests('p', 'train', 'dapo_math', rows + [dict(add[0], reward=0, status='completed')])) == 7
    rows5 = records(5, [0, 2, 2, 2, 2])
    # The second success arrives in the added round, not the first three.
    for row in rows5:
        if row['scale'] != 0:
            row['reward'] = int(row['rollout_id'] in (0, 3))
    decision = decide_probe(rows5, 'train')
    assert decision['delta'] == .4 and not decision['terminal']
    add = plan_requests('p', 'train', 'dapo_math', rows5)
    assert len(add) == 8 and all(r['rollout_id'] in (5, 6) for r in add)
    for split, keep in [('train', False), ('validation', True)]:
        rows7 = records(7, [0, 2, 2, 2, 2])
        for row in rows7:
            if row['scale'] != 0:
                row['reward'] = int(row['rollout_id'] in (0, 3))
        decision = decide_probe(rows7, split)
        assert decision['reason'] == 'not_sensitive_at_cap' and decision['keep'] is keep
        assert decision['terminal'] is True


def test_partial_stage_must_not_select_and_resumes_missing_requests():
    rows = records(3, [0, 1, 1, 1, 1])[:-1]
    assert decide_probe(rows, 'train')['reason'] == 'incomplete'
    todo = plan_requests('p', 'train', 'dapo_math', rows)
    assert len(todo) == 1 and todo[0]['scale'] == 2 and todo[0]['rollout_id'] == 2


@pytest.mark.parametrize('mutate', ['duplicate', 'gap', 'zero_extra', 'unfinished', 'bad_reward', 'foreign_scale', 'after_terminal'])
def test_malformed_or_overcollected_results_stop(mutate):
    rows = records(3, [0, 1, 1, 1, 1])
    if mutate == 'duplicate': rows.append(rows[-1].copy())
    if mutate == 'gap': rows[-1]['rollout_id'] = 4
    if mutate == 'zero_extra': rows.append(dict(rows[0], rollout_id=1))
    if mutate == 'unfinished': rows[-1]['status'] = 'error'
    if mutate == 'bad_reward': rows[-1]['reward'] = None
    if mutate == 'foreign_scale': rows[-1]['scale'] = .75
    if mutate == 'after_terminal':
        rows = records(3, [0, 0, 0, 0, 0]) + [dict(rows[-1], rollout_id=3)]
    with pytest.raises(ValueError):
        decide_probe(rows, 'train')


def test_test_split_cannot_be_probed_for_selection():
    with pytest.raises(ValueError): decide_probe([], 'test')
