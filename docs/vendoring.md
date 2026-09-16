# Vendoring playbook

Step-by-step for adopting the SM120 device layer in a **new** DeepGEMM fork.
This is also the SGLang playbook. For forks that already carry their own
SM120 copies, the same steps apply — the copy in step 1 simply overwrites
them (see step 4 for the one include-level caveat).

## 1. Copy the device tree

Copy `deep_gemm/include/deep_gemm/**` from a tagged release of this
repository into the fork at the **same paths**. Do not rename, relocate, or
edit files: byte-for-byte identity is what makes drift checking and
mechanical syncs possible.

## 2. Install the drift check

- Copy `VENDOR-sm120.json` to the fork repository root. It records the
  upstream repo, the tag you vendored, and the sha256 of every file.
- Copy `tools/check_vendor.py` into the fork's tooling and wire it into the
  fork's CI: `python3 tools/check_vendor.py` (run from the fork root).
  It fails on missing/changed vendored files and on fork-side sm120-named
  headers not covered by the manifest.

## 3. Implement the host layer

The device layer is vendored; the host layer is yours. Start from the closest
reference lineage under `host-glue/` (`nv_dev-lineage/` or `main-lineage/`)
and implement, per kernel:

- **`sm120_bf16_gemm_impl`** — A/B/CD TMA descriptors; grouped-layout pointer
  for m/k-grouped modes; runtime `epilogue` object matching the
  `epilogue_type_t` template argument; output strides `stride_d_m`,
  `stride_c_m`, `stride_d_batch`; tile/stage heuristics satisfying the
  header's `DG_STATIC_ASSERT`s.
- **`sm120_fp8_fp4_gemm_1d1d_impl`** — the above, plus: SF-A/SF-B TMA
  descriptors and granularity (`kGranKA`/`kGranKB`) plumbing; **k-grouped
  tensormap patching** (the `tensor_map_buffer` protocol and
  `kKGroupedConstantStride`/`kKAlignment` template parameters); the runtime
  `epilogue` and `shape_cd_m` (logical CD rows, which differ from `shape_m`
  for k-grouped shapes); and the **split-K workspace** pointer used with
  `kSplitKFactor > 1` together with `sm120_split_k_reduce_impl`.
- **MQA logits** (`sm120_fp8_mqa_logits`, `sm120_fp4_mqa_logits`) — Q/KV/
  KV-scales/weights TMA descriptors; `cu_seq_len_k_start/end`; the logits
  buffer and its stride; `BLOCK_KV = (kNumMathThreads/32) * 16` (i.e. 128
  with the standard 256 math threads); FP4 additionally requires
  `head_dim == 128` and SF-Q/SF-KV descriptors.
- **Paged MQA logits** (`sm120_fp8_paged_mqa_logits`,
  `sm120_fp4_paged_mqa_logits`) — block tables and their stride, context
  lens, indices, and the schedule metadata produced by
  `sched::sm120_paged_mqa_logits_metadata`; `PAGE_KV` ∈ {64, 128, 256}
  (FP4 also 32) with `SPLIT_KV = BLOCK_KV * num_groups`.
- **Sparse MQA logits** (`sm120_fp8_fp4_sparse_mqa_logits_kernel`) — the
  sparse metadata blob (`layout/sparse_mqa_logits.cuh` layout), top-k
  indices / block tables for the paged variant, and the shared-memory sizing
  formula mirrored in the reference host glue.
- **`sm120_clear_padding` / `sm120_copy_grouped_output`** — zero-padding for
  grouped layouts and masked/psum output extraction.
- **`sm120_bmn_bnk_mn_gemm_impl`** — batched BF16 contraction
  (`BLOCK_M = 128`, 128 TMA + 256 math threads, even `kNTiles`).
- **`sm120_tf32_hc_prenorm_gemm_impl`** — TF32 hyperconnection prenorm GEMM
  with the `sqr_sum` side-output (`BLOCK_K = 64`, one M-tile per warp).

`tools/gate_instantiations.cu` contains one known-good minimal configuration
per template — use it as the ground truth for signatures and assert-satisfying
tile shapes.

## 4. Main-lineage include caveat (existing SM120 copies only)

The vendored GEMM impls include the SM120-owned scheduler
`deep_gemm/scheduler/sm120_gemm.cuh` (`deep_gemm::sched_sm120::`). If your
fork carries **older** SM120 copies that include the shared
`deep_gemm/scheduler/gemm.cuh`, those copies are replaced by the vendored
files — no include switching is needed in the vendored tree itself. Only if
you keep fork-local SM120 variants of the two GEMM impls for some reason must
you switch their `#include <deep_gemm/scheduler/gemm.cuh>` to
`scheduler/sm120_gemm.cuh`. Fresh vendors get the correct includes directly.

## 5. Run your SM120 tests

Rerun the fork's full SM120 test suite (and compute-sanitizer subsets if you
have them) against the vendored tree before merging the sync.

## 6. Future syncs

1. Re-copy `deep_gemm/include/deep_gemm/**` and `VENDOR-sm120.json` at the
   new tag.
2. Run `python3 tools/check_vendor.py --fork /path/to/the/fork` from the
   DeepGEMM-sm120 checkout at that tag — must be clean.
3. Clear the fork's JIT cache (`DG_JIT_CACHE_DIR`) before testing: DeepJIT's
   cache key covers the generated translation unit and compiler options but
   NOT the contents of `#include`d headers, so re-vendored headers alone do
   not invalidate cached cubins. A fresh cache directory per sync is the
   safe default.
3. Rerun the fork's SM120 tests.

Never patch vendored files in the fork: fix here, tag, sync.
