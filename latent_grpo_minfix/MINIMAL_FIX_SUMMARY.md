# Latent-GRPO minimal fix summary

## What changed

- Updated the stale 3-GPU Low configuration contract test to the actual frozen values:
  `actor_micro=16`, `logprob_micro=32`, and SGLang `gpu_memory_utilization=0.45`.
- Added a direct regression test asserting that disabled observer mode emits no worker observer payload.
- Synchronized the release checklist, runbook, handoff, hyperparameter deviation/audit, and package notes.

## What did not change

No runtime implementation, algorithm, optimizer, sampler, trainer, worker, YAML configuration,
dependency pin, launcher, or shell tool was changed relative to parent commit
`8c9ce4932234844620d4da0b99df982123d35ae7`.

## Verification

- Full unit suite: `241 passed, 1 skipped`.
- Local release gate: `PASS`.
- Current commit: `910989600df1896258a448f3341b010cc265a296`.

The local release report was generated with Python 3.13.5. The target server must repeat the
release gate with its locked Python 3.11.15 environment and run a fresh 3-GPU validation bound
to this commit before a new formal run.
