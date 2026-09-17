# Design

## Why this repository exists

DeepSeek DeepGEMM does not support SM120 (GeForce RTX 5090 / `sm_120a`), and
upstream has declined to take SM120 support. Two forks therefore carry their
own SM120 device-kernel ports: one on the `nv_dev` lineage, one on the
vllm-project `main` lineage (with SGLang's fork expected to follow). The two
ports were near-identical but not identical; each fork fixed some bugs the
other had not. Maintaining two copies guarantees permanent drift.

This repository is the single source of truth for the SM120 **device layer**.
Forks vendor `deep_gemm/include/deep_gemm/**` byte-for-byte; fixes land here
once and propagate mechanically. The vendored tree keeps DeepGEMM's exact
directory layout so vendoring is a plain copy over the fork's tree.

## Layering

- **Device layer (this repo, vendored):** the 16 headers under
  `deep_gemm/include/deep_gemm/` — kernel impls, the SM120 MMA wrappers, the
  SM120-owned schedulers, SM120 common utilities, and the sparse-MQA KV
  layout. Forks copy them over their own tree at the same paths.
- **Host layer (per-lineage, not vendored):** JIT descriptor construction,
  tile heuristics, launch, and Python API wiring. This is per-lineage because
  the nv_dev and main JIT infrastructures have diverged (argument
  marshalling, launch helpers, runtime types). `host-glue/` keeps read-only
  reference copies for both lineages.
- **Tests stay per-fork:** the host Python/C++ API surface differs between
  lineages, so the correctness suites cannot be shared either. The
  vendored-tree correctness gate is (a) this repository's compile gate run
  against both lineages' bases (CI), and (b) each fork's own SM120 test
  suite, rerun after every sync.

## Superset merge decisions

The vendored tree is a merged superset of the two forks' SM120 copies. Where
the copies disagreed, one mechanism was kept and the other superseded:

1. **Masked-boundary handling (GEMM epilogue).** Kept: the in-kernel
   `row_is_valid` / swizzle-aware masking that guards every global-memory
   store. Superseded: the fork's boundary-tile TMA-store skip, which
   interacted with tile-granularity edge cases the in-kernel guard covers.
2. **`sm120_split_k_reduce`.** Kept: the superset variant with `gmem_c`
   (C-accumulate), `alpha`, explicit C/D strides, and HeadSplits support.
   Superseded: the fork's boolean `kWithAccumulation` template variant.
3. **Paged MQA logits.** Kept: the `PAGE_KV` ∈ {32, 64, 128, 256} generalization
   in which `BLOCK_KV` is derived from `PAGE_KV` and
   within-page addressing is done with a `page_offset` TMA coordinate.
   Superseded: the fork's `BLOCK_KV == page size` assumption.
4. **SM120-owned GEMM scheduler.** The GEMM impls include
   `deep_gemm/scheduler/sm120_gemm.cuh` and use `deep_gemm::sched_sm120::`
   instead of the shared `scheduler/gemm.cuh`, whose template surface differs
   between lineages. Main-lineage forks carrying older SM120 copies switch
   two `#include`s when vendoring (see docs/vendoring.md); fresh vendors get
   ours directly.
5. **`tensor_map_replace_global_dim_in_smem`** lives in the vendored
   `common/sm120_utils.cuh` because main-lineage `ptx/tma.cuh` lacks it
   (nv_dev has it). Call sites use the `sm120::` qualification, so both bases
   work unmodified.
6. **CUDA ≥ 13 compile-time guard**, duplicated in `common/sm120_utils.cuh`
   and `mma/sm120.cuh`. Pre-13 ptxas was reported to silently drop the
   `block_scale` qualifier when targeting `sm_120a` — the failure is silent
   (wrong numerics, no diagnostic), so 12.x is rejected as a precaution. On
   13.x the qualifier is verified kept (SASS `QMMA.SF.*`, PTX `block_scale`);
   the compile gate asserts this on every run.

## Compatibility contract with bases

Vendored files may include CUTLASS/CUDA/standard headers, other files in this
tree, and the fixed allowlist of **base-provided shared headers**
(`common/{cute_tie,math,tma_copy,types,utils}.cuh`,
`epilogue/transform.cuh`, `ptx/{ld_st,tma,utils}.cuh`), enforced by
`tools/self_containment.py`. The allowlisted headers **differ textually
between lineages**: the contract with them is symbol-level, not byte-level.
Vendored code may only rely on symbols present in every supported lineage's
version. Adding a dependency on a symbol missing from any supported lineage
is a **breaking change**: it requires a minor version bump and a note in the
affected `host-glue/` README.

**Supported base lineages.** The SM120 layer targets the post-"Public
Release 26/09" (deepseek-ai/DeepGEMM#432) API surface — the new
`EpilogueArgs`-based epilogue transforms, current scaling-factor layouts,
and DeepJIT. The compile gate therefore validates against
**deepseek-ai/DeepGEMM `main`** and **vllm-project/DeepGEMM `dev`**. The
pristine DeepSeek `nv_dev` branch predates that API surface and is *not* a
compile target; the nv_dev-lineage vehicle is the PR #447 branch (nv_dev +
main-API migration), which carries this layer and is covered by its own GPU
test suite. (v0.1.0 briefly gated on pristine nv_dev and `common/packing.cuh`
was allowlisted; v0.1.1 inlined the single packing helper into
`layout/sparse_mqa_logits.cuh` and corrected the gate bases.)

## Versioning and sync protocol

- Tags: `vX.Y.Z`. Breaking symbol-contract changes bump the minor version;
  pure bug fixes and new vendored kernels bump patch/minor per impact.
- Forks pin a release by copying the vendored tree and naming the tag in
  the commit message. Nothing vendoring-specific is committed to a fork and
  the vendored files carry no marker of any kind: the per-file sha256
  manifest (`VENDOR-sm120.json`) lives in this repo per release, and
  `tools/check_vendor.py --fork <path>` runs from a DeepGEMM-sm120 checkout
  against any fork.
- Fixes happen here once; forks sync by re-copying at a new tag, running
  the drift check, and rerunning their SM120 tests. See
  docs/vendoring.md for the step-by-step.
- Downstreams can get *ahead* of this repo (a fork merges a third-party PR
  touching the vendored files; the files themselves carry no pointer back
  here). The round trip is closed by
  `tools/watch_downstreams.py`: a scheduled workflow
  (.github/workflows/drift-watch.yml) fetches every registered downstream's
  vendored tree daily and compares it against all release manifests
  (.github/downstreams.json). A tree matching no release tag is drift: the
  workflow opens a `vendor-drift` issue, the change is absorbed here and
  re-tagged, and the downstream re-vendors. Downstreams commit nothing for
  this — watching is the upstream's job.
