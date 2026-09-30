"""Independent reference from minfix commit 0b7e85f, without engine imports."""

import ast
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "latent_grpo_minfix/Latent-GRPO-final"
SAMPLER = UPSTREAM / "Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py"
UPSTREAM_COMMIT = "0b7e85f15e9859033653964282348e518f9f6291"


def load_upstream_helper():
    tree = ast.parse(SAMPLER.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_streaming_latent_noisy_topk")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SAMPLER), "exec"), namespace)
    return namespace[node.name]


def reference_sample(logits, scales, top_p, k, temperature, one_sided, active, end_id):
    logits = logits.float()
    probs = logits.softmax(-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True, dim=-1)
    support_sorted = (sorted_probs.cumsum(-1) - sorted_probs) < top_p
    support_sorted[:, :k] = True
    support = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, sorted_ids, support_sorted)
    scores, noisy_ids, _ = load_upstream_helper()(logits, support, k, scales[0], one_sided)
    noisy_probs = (scores / temperature).softmax(-1)
    clean_logits, clean_ids = logits.topk(k, dim=-1)
    mask = active & (clean_ids[:, 0] != end_id)
    ids = torch.where(mask[:, None], noisy_ids, clean_ids)
    mixture = torch.where(mask[:, None], noisy_probs, clean_logits.softmax(-1))
    return ids, mixture, ids[:, 0]
