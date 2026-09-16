# DeepGEMM-sm120

SM120a device-kernel layer for [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
forks, targeting GeForce RTX 5090-class GPUs (`sm_120a`).

Upstream DeepSeek has declined SM120 support, so multiple forks (an
`nv_dev`-lineage fork, a vllm-project `main`-lineage fork, later SGLang's)
each maintained their own copy of the SM120 device kernels. This repository
extracts that layer into a single source of truth that forks vendor
byte-for-byte, instead of each maintaining its own diverging copy. It is
derived from the SM120 ports developed for **deepseek-ai/DeepGEMM#447**
(nv_dev lineage) and **vllm-project/DeepGEMM#9** (main lineage), themselves
derived from DeepSeek DeepGEMM and distributed under the same MIT license
(see [LICENSE](LICENSE)).

**Status: v0.1.0** — merged superset of both forks' SM120 device trees.

## Scope

- Vendored: the 16 device headers under `deep_gemm/include/deep_gemm/`
  (GEMM impls, MQA/paged/sparse logits impls, SM120 MMA wrappers, SM120
  schedulers, SM120 common utilities, sparse-MQA layout).
- Not vendored: host-side JIT/heuristics/launch glue (per-lineage;
  reference copies in [`host-glue/`](host-glue/)), and tests (per-fork).

## Quickstart

To vendor this layer into your fork, follow
[docs/vendoring.md](docs/vendoring.md). In short: copy
`deep_gemm/include/deep_gemm/**` over your tree and record the release tag in
the commit message. Forks commit nothing extra and the files carry no
marker -- `tools/check_vendor.py --fork <path>` (run from a DeepGEMM-sm120
checkout) verifies drift, and a scheduled watcher monitors registered
downstreams.

## Development gates

- `python3 tools/self_containment.py` — include-closure check for the
  vendored tree (stdlib only).
- `./tools/compile_gate.sh <base_include_dir> <cutlass_include_dir>` —
  cross-compiles every vendored header plus one representative instantiation
  per kernel for `sm_120a`, then asserts on SASS/PTX opcodes (HMMA.16816,
  `QMMA.*.SF.*`, `block_scale`). Run it against each supported base lineage;
  CI does this for both.

See [docs/design.md](docs/design.md) for the layering, merge decisions, and
compatibility contract, and [docs/validation.md](docs/validation.md) for the
validation evidence summary.

## Requirements

- CUDA toolkit ≥ 13 (pre-13 ptxas was reported to silently drop `block_scale`
  under `sm_120a`; the vendored headers refuse to compile there).
- Cross-compiling (the gates above) needs only the toolkit — no GPU.
- Executing the kernels needs SM120a hardware, exercised via each fork's own
  test suite.
