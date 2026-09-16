# Host glue

This repository vendors only the **device layer** (`deep_gemm/include/**`),
byte-for-byte. The **host layer** — JIT descriptor construction, tile
heuristics, launch, and Python API wiring — is intentionally per-lineage,
because the nv_dev and main DeepGEMM lineages have divergent JIT
infrastructure (argument marshalling, launch helpers, and runtime types do
not match and cannot be shared verbatim).

A fork adopting the vendored device layer must provide:

1. **Host code that builds kernel arguments matching the vendored kernels'
   signatures** — TMA descriptors (`cute::TmaDescriptor`, passed as
   `__grid_constant__`), grouped-layout pointers, tensor-map patch buffers
   for k-grouped GEMMs, block tables for paged attention kernels, etc.
2. **Heuristics choosing tile configs** — `BLOCK_M/N/K`, swizzle modes,
   stage counts, warp splits that satisfy the `DG_STATIC_ASSERT`s in the
   vendored headers (see `tools/gate_instantiations.cu` for known-good
   minimal configurations).
3. **`GemmType` / scheduler plumbing** — the vendored GEMM kernels use the
   SM120-owned scheduler in `deep_gemm/scheduler/sm120_gemm.cuh`
   (`deep_gemm::sched_sm120::Scheduler`), not the base's shared
   `scheduler/gemm.cuh`.

Two reference host layers, one per lineage, are kept here as
**reference copies** (not vendored, not compiled from this repository):

- [`nv_dev-lineage/`](nv_dev-lineage/) — from the nv_dev-lineage fork
  (lucifer1004/DeepGEMM).
- [`main-lineage/`](main-lineage/) — from the main-lineage fork
  (vllm-project/DeepGEMM `dev` + kgroup-fix).

Start from whichever reference matches your fork's lineage; see
[docs/vendoring.md](../docs/vendoring.md) for the full adoption playbook.

## Host-side obligations that are easy to miss

These are part of the device layer's contract but enforced on the host —
the vendored kernels assume the host upholds them:

- **Grouped-GEMM boundary tiles need the scalar epilogue.** The GEMM
  kernels' TMA-store epilogue writes full `BLOCK_M`-high tiles. TMA clamps
  only at the tensor map's outermost global dim, so for **m-grouped masked**
  and **k-grouped** GEMMs — where each group's output slab is *interior* to
  the flat `[groups * m, n]` allocation — a partial boundary tile would
  spill rows into the next group's slab. When `m % BLOCK_M != 0` on those
  paths, the host MUST force the scalar store epilogue, which carries the
  per-group row bounds (`row_is_valid` / `total_shape_m`):
  `config.storage_config.swizzle_cd_mode = 0;`
  Both reference lineages show the guard placement (right after the
  heuristics pick a config, before descriptor construction). Dense and
  m-grouped contiguous paths do not need it (boundaries are outermost /
  block-aligned).
- **m-grouped contiguous (labels mode): group starts must be multiples of
  the runtime mk alignment.** The GEMM kernels select B/SFB per `BLOCK_M`
  tile from the label of the tile's first row, and the reference heuristics
  guarantee `BLOCK_M` divides `get_mk_alignment_for_contiguous_layout()`.
  Labels built at a finer granularity than the runtime setting put a group
  boundary inside a tile and silently compute the straddled rows with the
  wrong group's B (this presented as "an empty middle group corrupts later
  groups" — the real trigger is any boundary misaligned to the runtime
  setting). A host-side opt-in checker (`DG_CHECK_CONTIGUOUS_LABELS=1`) in
  the reference glue turns such violations into a loud error.
- **CUDA >= 13 toolkit** for any TU that instantiates the vendored kernels;
  the vendored headers `#error` out on earlier toolkits during SM120 device
  passes (see `common/sm120_utils.cuh`).
