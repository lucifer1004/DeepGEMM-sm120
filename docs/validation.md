# Validation evidence

This file summarizes the public-safe validation story for the vendored tree.
Machine-specific facts (hosts, board models, cluster details, local paths)
are intentionally omitted; execution targets were SM120a-class consumer GPUs.

## Cross-compile gate (this repository, no GPU required)

The merged tree cross-compiles for `sm_120a` against **both** supported base
lineages with CUDA 13.2:

- nv_dev-lineage base (DeepSeek `nv_dev` tip at the time of the v0.1.0
  merge), and
- main-lineage base (vllm-project/DeepGEMM `dev` + kgroup-fix),

using `tools/compile_gate.sh`, which

1. self-compiles each of the 16 vendored headers to cubin (device pass,
   `__CUDA_ARCH__ >= 1200` path),
2. compiles one representative explicit instantiation per kernel template
   (`tools/gate_instantiations.cu`), and
3. asserts on the generated code:
   - BF16 GEMM: SASS contains `HMMA.16816` (mma.sync m16n8k16 bf16).
   - FP8/FP4 block-scaled 1D1D GEMM: SASS contains a block-scaled MMA
     mnemonic — `QMMA.SF.16832.F32.E4M3.E4M3.E8` for FP8 and
     `OMMA.SF.16864.F32.E2M1.E2M1.E8` for FP4 (OMMA is the sm_120a FP4
     opcode family). The `.SF.` qualifier proves ptxas did **not** drop
     block scaling.
   - MQA / paged / sparse logits kernels: each kernel's SASS section
     contains `HMMA`/`QMMA`/`OMMA`-family opcodes.
   - The same instantiation TU compiled to PTX mentions `block_scale`.

CI runs all of the above on every change, against both pinned bases.

## Include-closure gate

`tools/self_containment.py` verifies that vendored headers include only
CUTLASS/CUDA/standard headers, other vendored files, or the fixed allowlist
of base-provided shared headers (see docs/design.md). Runs in CI.

## GPU correctness evidence (in the fork PRs)

End-to-end numerical correctness was validated on SM120a-class consumer GPUs
in the two lineage forks; see the PRs rather than this repository for logs:

- nv_dev lineage: **deepseek-ai/DeepGEMM#447** — 741 SM120 tests, including
  compute-sanitizer memcheck and racecheck subsets.
- main lineage: **vllm-project/DeepGEMM#9** — 46 test nodes × 3 modes.

Every fork is expected to rerun its own SM120 suite after each sync from this
repository (see docs/vendoring.md); this repository itself contains no tests
because the host API surface that tests exercise is per-lineage.
