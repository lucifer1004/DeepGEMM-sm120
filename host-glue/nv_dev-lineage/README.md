# nv_dev-lineage host glue (reference copy)

Reference host layer for forks on the **nv_dev lineage** of DeepGEMM
(DeepSeek's `nv_dev` branch and forks tracking it).

Source: `lucifer1004/DeepGEMM` (nv_dev-lineage fork), commit `0aa7482`
(`test(sm120): collect SM120 tests into dedicated files`).

Layout mirrors the source repository:

- `csrc/apis/sm120_gemm.hpp` — public C++ entry points (`fp8_fp4_gemm_nt_sm120`).
- `csrc/jit_kernels/heuristics/sm120.hpp` — tile/stage/launch heuristics.
- `csrc/jit_kernels/heuristics/sm120_config.hpp` — GEMM descriptor and config types.
- `csrc/jit_kernels/impls/sm120_*.hpp` — per-kernel JIT codegen + launch
  (bf16 GEMM, fp8_fp4 1d1d GEMM, bmk_bnk_mn, mqa/paged logits, sparse logits,
  padding, tf32 hc-prenorm, runtime helpers).

These are **reference copies**: they are not vendored by forks and nothing in
this repository compiles them. They show how a complete host layer on this
lineage builds kernel arguments against the vendored device headers' current
template signatures. If you fix a host bug, fix it in the fork (or upstream
the device part here); do not edit these copies to "match" — refresh them
from the source repo instead.
