"""Minfix-compatible latent mixture, without an SGLang engine dependency.

Streaming algorithm adapted from minfix 0b7e85f, under the MIT license.
Copyright (c) 2025 Zhi Zheng. See THIRD_PARTY_NOTICES.md.
"""

from typing import NamedTuple

import torch


class LatentSample(NamedTuple):
    indices: torch.Tensor
    probs: torch.Tensor
    next_latent_id: torch.Tensor


@torch.no_grad()
def streaming_noisy_topk(
    logits_f32: torch.Tensor,
    support_mask: torch.Tensor,
    max_topk: int,
    noise_scale: torch.Tensor,
    use_one_sided_gumbel_noise: torch.Tensor,
    *,
    gumbels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return noisy log scores, IDs, selected noise; workspace width <=8192.

    Optional gumbels are *untransformed* supplied noise, used for controlled
    reference checks. Normal inference draws once per vocabulary chunk.
    Full-vocabulary support/logits are inputs, not noisy workspaces.
    """
    if not 1 <= max_topk <= logits_f32.shape[-1]:
        raise ValueError("max_topk must be within the vocabulary")
    if support_mask.shape != logits_f32.shape:
        raise ValueError("support mask must match logits")
    if gumbels is not None and gumbels.shape != logits_f32.shape:
        raise ValueError("supplied raw Gumbels must match logits")
    log_z = torch.logsumexp(logits_f32, dim=-1, keepdim=True)
    running_scores = running_indices = running_noise = None
    for start in range(0, logits_f32.shape[-1], 8192):
        end = min(start + 8192, logits_f32.shape[-1])
        chunk = logits_f32[:, start:end]
        noise = (-torch.empty_like(chunk).exponential_().log()
                 if gumbels is None else gumbels[:, start:end].clone())
        noise.clamp_(-1.5, 3)
        if use_one_sided_gumbel_noise.any().item():
            noise = torch.where(use_one_sided_gumbel_noise, noise + 1.5, noise)
        noise = noise_scale * noise
        scores = (chunk + noise).masked_fill(~support_mask[:, start:end], float("-inf"))
        local_scores, offsets = scores.topk(min(max_topk, end - start), dim=-1)
        local_ids = offsets + start
        local_noise = noise.gather(-1, offsets)
        if running_scores is None:
            running_scores, running_indices, running_noise = local_scores, local_ids, local_noise
        else:
            scores = torch.cat((running_scores, local_scores), dim=-1)
            ids = torch.cat((running_indices, local_ids), dim=-1)
            noises = torch.cat((running_noise, local_noise), dim=-1)
            running_scores, offsets = scores.topk(min(max_topk, scores.shape[-1]), dim=-1)
            running_indices = ids.gather(-1, offsets)
            running_noise = noises.gather(-1, offsets)
    return running_scores - log_z, running_indices, running_noise


@torch.no_grad()
def sample_latent_kernel(
    logits: torch.Tensor,
    fixed_scales: torch.Tensor,
    top_p: torch.Tensor,
    K: int,
    gumbel_temperature: float,
    one_sided: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    latent_end_token_id: int,
    gumbels: torch.Tensor | None = None,
) -> LatentSample:
    """Compute only the latent mixture boundary, not explicit token sampling.

    active_mask combines latent mode and add_noise_gumbel_softmax flags.
    Inactive/end-token rows use clean top-K, matching minfix's noisy branch.
    Homogeneous scales reproduce minfix; heterogeneous scales apply per row.
    """
    if logits.ndim != 2 or not 1 <= K <= logits.shape[-1]:
        raise ValueError("logits must be [B,V] with 1 <= K <= V")
    batch = logits.shape[0]
    if any(t.shape != (batch, 1) for t in (fixed_scales, top_p, one_sided)):
        raise ValueError("scales, top_p and one_sided must be [B,1]")
    if active_mask.shape != (batch,):
        raise ValueError("active_mask must be [B]")
    if gumbel_temperature <= 0:
        raise ValueError("Gumbel temperature must be positive")
    if not 0 <= latent_end_token_id < logits.shape[-1]:
        raise ValueError("latent end token must be in the vocabulary")
    logits_f32 = logits.float()
    probs = logits_f32.softmax(-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True, dim=-1)
    sorted_support = (sorted_probs.cumsum(-1) - sorted_probs) < top_p
    sorted_support[:, :K] = True
    support = torch.zeros_like(logits_f32, dtype=torch.bool).scatter_(1, sorted_ids, sorted_support)
    del probs, sorted_probs, sorted_ids, sorted_support
    scores, noisy_ids, _ = streaming_noisy_topk(
        logits_f32, support, K, fixed_scales, one_sided, gumbels=gumbels)
    del support
    noisy_probs = (scores / gumbel_temperature).softmax(-1)
    clean_logits, clean_ids = logits_f32.topk(K, dim=-1)
    active = active_mask & (clean_ids[:, 0] != latent_end_token_id)
    ids = torch.where(active[:, None], noisy_ids, clean_ids)
    mixture = torch.where(active[:, None], noisy_probs, clean_logits.softmax(-1))
    return LatentSample(ids, mixture, ids[:, 0])
