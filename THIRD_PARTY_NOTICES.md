---
id: DOC-LICENSE-001
type: guide
status: active
title: Third-party notices
created: 2026-09-30
updated: 2026-10-03
---
# Third-party notices

The streaming Top-K algorithm in `adaptive_noise/sampling_kernel.py` is adapted from the minfix Latent-GRPO sampler, commit `0b7e85f15e9859033653964282348e518f9f6291`.
The test reference executes that upstream helper without importing its CUDA engine.
`adaptive_noise/answer_verifier.py` executes only the allowlisted pure scoring helpers from the same pinned minfix evaluation scripts, with no upstream entrypoint or CUDA engine imports.
Source: [minfix repository](https://github.com/fengjixing-stupid/Latent-GRPO/tree/0b7e85f15e9859033653964282348e518f9f6291).
Return to [project README](README.md).

MIT License

Copyright (c) 2025 Zhi Zheng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
