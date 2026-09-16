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
