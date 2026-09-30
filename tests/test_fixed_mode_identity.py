"""Catch changes to minfix RNG order, support, mixture, scale, and masks."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from adaptive_noise.sampling_kernel import sample_latent_kernel, streaming_noisy_topk
from tests.reference_fixed_sampler import load_upstream_helper, reference_sample


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("one_sided", [False, True])
@pytest.mark.parametrize("top_p", [0.2, 0.95])
def test_fixed_identity(batch, seed, one_sided, top_p):
    logits = torch.linspace(-2, 2, 32).repeat(batch, 1)
    scales = torch.full((batch, 1), 0.75)
    ps = torch.full((batch, 1), top_p)
    sided = torch.full((batch, 1), one_sided)
    active = torch.ones(batch, dtype=torch.bool)
    torch.manual_seed(seed)
    expected = reference_sample(logits, scales, ps, 10, 1.0, sided, active, 0)
    torch.manual_seed(seed)
    actual = sample_latent_kernel(logits, scales, ps, 10, 1.0, sided, active,
                                 latent_end_token_id=0)
    for value, want in zip(actual, expected):
        assert torch.equal(value, want)


@pytest.mark.parametrize("vocab", [8192, 8205])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_chunk_boundary_matches_actual_minfix(vocab, seed):
    logits = torch.linspace(-3, 3, vocab).repeat(3, 1)
    support = torch.ones_like(logits, dtype=torch.bool)
    support[:, 17::23] = False
    sided = torch.tensor([[False], [True], [False]])
    scales = torch.full((3, 1), 0.75)
    torch.manual_seed(seed)
    expected = load_upstream_helper()(logits, support, 10, scales[0], sided)
    torch.manual_seed(seed)
    actual = streaming_noisy_topk(logits, support, 10, scales, sided)
    for value, want in zip(actual, expected):
        assert torch.equal(value, want)


def test_nonlatent_and_configured_end_token_use_clean_mixture():
    logits = torch.zeros(3, 32)
    logits[:, 7] = 2
    active = torch.tensor([False, True, True])
    scales = torch.ones(3, 1)
    ps = torch.ones(3, 1)
    sided = torch.ones(3, 1, dtype=torch.bool)
    ids, probs, next_ids = sample_latent_kernel(
        logits, scales, ps, 10, 1.0, sided, active, latent_end_token_id=7)
    clean_logits, clean_ids = logits.topk(10, dim=-1)
    assert torch.equal(ids, clean_ids)
    assert torch.equal(probs, clean_logits.softmax(-1))
    assert next_ids.tolist() == [7, 7, 7]


def test_zero_scale_removes_noise_and_preserves_topk():
    logits = torch.linspace(-2, 2, 8205).view(1, -1)
    scores, ids, noise = streaming_noisy_topk(
        logits, torch.ones_like(logits, dtype=torch.bool), 10,
        torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool))
    clean, clean_ids = logits.topk(10, dim=-1)
    assert torch.equal(ids, clean_ids)
    assert torch.count_nonzero(noise) == 0
    torch.testing.assert_close(scores, clean - logits.logsumexp(-1, keepdim=True))


def test_inactive_row_stays_clean_while_neighbor_samples():
    logits = torch.arange(32).float().view(1, -1).repeat(2, 1) * 0.01
    raw_noise = torch.full_like(logits, -1.5)
    raw_noise[:, 0] = 3
    ids, _, next_ids = sample_latent_kernel(
        logits, torch.ones(2, 1), torch.ones(2, 1), 10, 1.0,
        torch.zeros(2, 1, dtype=torch.bool), torch.tensor([False, True]),
        latent_end_token_id=7, gumbels=raw_noise)
    assert next_ids.tolist() == [31, 0]
    assert torch.equal(ids[0], logits[0].topk(10).indices)


def test_per_row_scales_do_not_broadcast_first_request():
    logits = torch.linspace(-0.1, 0.1, 32).repeat(3, 1)
    support = torch.ones_like(logits, dtype=torch.bool)
    scales = torch.tensor([[0.0], [0.5], [1.0]])
    sided = torch.ones(3, 1, dtype=torch.bool)
    torch.manual_seed(2)
    expected = load_upstream_helper()(logits, support, 10, scales, sided)
    torch.manual_seed(2)
    actual = streaming_noisy_topk(logits, support, 10, scales, sided)
    for value, want in zip(actual, expected):
        assert torch.equal(value, want)
    assert not torch.equal(actual[1][0], actual[1][2])


def test_noise_can_select_outside_clean_topk():
    logits = torch.arange(32).float().view(1, -1) * 0.01
    raw_noise = torch.full_like(logits, -1.5)
    raw_noise[0, 0] = 3
    _, ids, _ = streaming_noisy_topk(
        logits, torch.ones_like(logits, dtype=torch.bool), 10,
        torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.bool), gumbels=raw_noise)
    assert 0 in ids[0].tolist()
    assert 0 not in logits.topk(10, dim=-1).indices[0].tolist()


class ShapeRecorder(TorchDispatchMode):
    def __init__(self):
        self.shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        def record(value):
            if isinstance(value, torch.Tensor):
                self.shapes.append(tuple(value.shape))
            elif isinstance(value, (tuple, list)):
                for item in value:
                    record(item)
        record(output)
        return output


def test_noisy_stage_workspace_stays_chunk_bounded():
    logits = torch.linspace(-3, 3, 8205).repeat(3, 1)
    support = torch.ones_like(logits, dtype=torch.bool)
    scales = torch.ones(3, 1)
    sided = torch.ones(3, 1, dtype=torch.bool)
    recorder = ShapeRecorder()
    with recorder:
        result = streaming_noisy_topk(logits, support, 10, scales, sided)
    assert all(x.shape == (3, 10) for x in result)
    assert all(shape[-1] <= 8192 for shape in recorder.shapes if len(shape) == 2)


@pytest.mark.parametrize("bad_k", [0, 33])
def test_rejects_invalid_k(bad_k):
    with pytest.raises(ValueError):
        sample_latent_kernel(torch.zeros(1, 32), torch.ones(1, 1), torch.ones(1, 1),
                             bad_k, 1.0, torch.ones(1, 1, dtype=torch.bool),
                             torch.ones(1, dtype=torch.bool), latent_end_token_id=7)
