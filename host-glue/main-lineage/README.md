# main-lineage host glue (reference copy)

Reference host layer for forks on the **main lineage** of DeepGEMM
(vllm-project/DeepGEMM and forks tracking upstream `main`).

Source: vllm-project/DeepGEMM `dev` branch + `kgroup-fix` work, commit
`75d3c96` (`test(sm120): move SM120 regressions into dedicated test files`).

Layout mirrors the source repository:

- `csrc/apis/sm120_dispatch.hpp` — arch dispatch into the SM120 paths.
- `csrc/jit_kernels/heuristics/sm120.hpp` — tile/stage/launch heuristics.
- `csrc/jit_kernels/impls/sm120_*.hpp` — per-kernel JIT codegen + launch
  (bf16 GEMM, fp8_fp4 1d1d GEMM, bmk_bnk_mn, mqa/paged logits,
  tf32 hc-prenorm).

## Adaptation status: old template signatures

These reference files are **pre-vendoring**: they generate and launch the
kernels with the fork's OLD SM120 template signatures. When adopting the
vendored device tree, this host layer must be updated for the merged
(superset) signatures, in particular:

- `sm120_bf16_gemm_impl`: new trailing template parameters
  `kKGroupedConstantStride` and `kKAlignment`; the runtime argument list now
  takes the epilogue object (`const epilogue_type_t epilogue`) and splits
  output strides into `stride_d_m`, `stride_c_m`, `stride_d_batch`
  (previously a single `stride_cd_m`/`stride_cd_batch` pair, with the
  epilogue carried only at the type level).
- `sm120_fp8_fp4_gemm_1d1d_impl`: adds `kKAlignment`, a `shape_cd_m`
  runtime argument (logical CD rows, distinct from `shape_m` under k-grouped
  tensormap patching), `stride_c_m`, split-K workspace pointer, and the
  k-grouped tensormap patch buffer protocol.
- The GEMM impls include `deep_gemm/scheduler/sm120_gemm.cuh`
  (`deep_gemm::sched_sm120::Scheduler`); the old copies included the shared
  `deep_gemm/scheduler/gemm.cuh`. Vendored headers already carry the new
  include — only remove the fork's old sm120 copies, do not re-point them.

The `nv_dev-lineage/` reference shows the target state for these call sites.

These are **reference copies**: not vendored by forks, not compiled from
this repository. Refresh them from the source repo rather than editing here.
