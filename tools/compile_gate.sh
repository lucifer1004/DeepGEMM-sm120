#!/usr/bin/env bash
# SM120 cross-compile gate. No GPU required.
#
# Usage:
#   ./tools/compile_gate.sh <base_include_dir> <cutlass_include_dir> [out_dir]
#
# Example (base DeepGEMM checkout at $BASE):
#   ./tools/compile_gate.sh $BASE/deep_gemm/include $BASE/third-party/cutlass/include
#
# Steps:
#   1. Header self-compile: every vendored .cuh is compiled alone to cubin.
#      Headers are guarded on __CUDA_ARCH__ >= 1200, so this exercises the
#      device pass for sm_120a.
#   2. Representative instantiations: tools/gate_instantiations.cu holds one
#      explicit instantiation per kernel template; compiled to cubin.
#   3. Opcode assertions: nvdisasm -c on the instantiations cubin, then
#        - BF16 GEMM kernel        -> SASS has HMMA.16816
#        - FP8/FP4 1D1D GEMM       -> SASS has a QMMA (FP8) / OMMA (FP4)
#                                     mnemonic containing .SF. (proves ptxas
#                                     kept block scaling)
#        - MQA/paged/sparse logits -> each kernel section has HMMA/QMMA/OMMA
#      and a PTX compile of the same TU must mention `block_scale`.
#
# Include order matters: this repo's tree FIRST so vendored headers shadow
# any same-path files in the base, then the base include dir (for the
# allowlisted base-provided shared headers), then CUTLASS.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 <base_include_dir> <cutlass_include_dir> [out_dir]" >&2
    exit 2
fi

BASE_INC="$(cd "$1" && pwd)"
CUTLASS_INC="$(cd "$2" && pwd)"
OUT_DIR="${3:-$REPO_ROOT/build/compile_gate}"
mkdir -p "$OUT_DIR"

for d in "$BASE_INC/deep_gemm" "$CUTLASS_INC/cutlass"; do
    [[ -d "$d" ]] || { echo "error: expected directory missing: $d" >&2; exit 2; }
done

NVCC="${NVCC:-$(command -v nvcc || true)}"
[[ -n "$NVCC" && -x "$NVCC" ]] || { echo "error: nvcc not found (set \$NVCC or put it on PATH)" >&2; exit 2; }

NVCC_VERSION="$("$NVCC" --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
CUDA_MAJOR="${NVCC_VERSION%%.*}"
if [[ -z "$NVCC_VERSION" || "$CUDA_MAJOR" -lt 13 ]]; then
    echo "error: CUDA >= 13 required (found nvcc release '${NVCC_VERSION:-unknown}')." >&2
    echo "       pre-13 ptxas was reported to silently drop block_scale under sm_120a." >&2
    exit 2
fi

NVDISASM="$(dirname "$NVCC")/nvdisasm"
[[ -x "$NVDISASM" ]] || NVDISASM="$(command -v nvdisasm || true)"
[[ -n "$NVDISASM" ]] || { echo "error: nvdisasm not found next to nvcc or on PATH" >&2; exit 2; }

echo "nvcc:      $NVCC (release $NVCC_VERSION)"
echo "nvdisasm:  $NVDISASM"
echo "base inc:  $BASE_INC"
echo "cutlass:   $CUTLASS_INC"
echo "out dir:   $OUT_DIR"
echo

FLAGS=(--gpu-architecture=sm_120a -O3 -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda)
INCLUDES=(--include-path "$REPO_ROOT/deep_gemm/include"
          --include-path "$BASE_INC"
          --include-path "$CUTLASS_INC")

failures=0

# --------------------------------------------------------------------------
echo "== Step 1: header self-compile (cubin per vendored header) =="
step1_fail=0
while IFS= read -r -d '' header; do
    rel="${header#"$REPO_ROOT"/}"
    inc="${rel#deep_gemm/include/}"
    tu="$OUT_DIR/step1_${inc//\//__}.cu"
    bin="$OUT_DIR/step1_${inc//\//__}.cubin"
    printf '#include <%s>\n' "$inc" > "$tu"
    if "$NVCC" "${FLAGS[@]}" "${INCLUDES[@]}" -cubin "$tu" -o "$bin" 2> "$bin.log"; then
        echo "  PASS  $inc"
    else
        echo "  FAIL  $inc (log: $bin.log)"
        sed 's/^/        /' "$bin.log" | head -20
        step1_fail=1
    fi
done < <(find "$REPO_ROOT/deep_gemm/include" -name '*.cuh' -print0 | sort -z)
[[ $step1_fail -eq 0 ]] || { echo "Step 1: FAIL"; exit 1; }
echo "Step 1: PASS"
echo

# --------------------------------------------------------------------------
echo "== Step 2: representative kernel instantiations =="
INST_CU="$REPO_ROOT/tools/gate_instantiations.cu"
INST_CUBIN="$OUT_DIR/gate_instantiations.cubin"
INST_PTX="$OUT_DIR/gate_instantiations.ptx"
if "$NVCC" "${FLAGS[@]}" "${INCLUDES[@]}" -cubin "$INST_CU" -o "$INST_CUBIN" 2> "$INST_CUBIN.log"; then
    echo "Step 2: PASS ($INST_CUBIN)"
else
    echo "Step 2: FAIL (log: $INST_CUBIN.log)"
    sed 's/^/  /' "$INST_CUBIN.log" | head -40
    exit 1
fi
echo

# --------------------------------------------------------------------------
echo "== Step 3: opcode assertions =="
"$NVDISASM" -c "$INST_CUBIN" > "$OUT_DIR/gate_instantiations.sass" 2>/dev/null
"$NVCC" "${FLAGS[@]}" "${INCLUDES[@]}" -ptx "$INST_CU" -o "$INST_PTX" 2> "$INST_PTX.log"

if python3 - "$OUT_DIR/gate_instantiations.sass" "$INST_PTX" <<'PYEOF'
import re
import sys

sass_path, ptx_path = sys.argv[1], sys.argv[2]
sass = open(sass_path).read()
ptx = open(ptx_path).read()

# Split SASS into per-kernel sections keyed by the mangled .text name.
sections = {}
current = None
for line in sass.splitlines():
    if line.startswith("//") and ".text." in line:
        m = re.search(r"\.text\.(\S+)", line)
        current = m.group(1) if m else line
        sections[current] = []
    elif current is not None:
        sections[current].append(line)

def find_sections(substr):
    return {name: "\n".join(body) for name, body in sections.items() if substr in name}

def mnemonics(body):
    # SASS listing lines look like:  /*0040*/  HMMA.16816.F32.BF16 ... ;
    return re.findall(r"/\*[0-9a-f]+\*/\s+([A-Z@!][A-Z0-9._@!\[\]%]*)", body)

def has_mma(body, pattern):
    return any(re.search(pattern, m) for m in mnemonics(body))

checks = []
def check(name, ok, detail=""):
    checks.append((name, ok, detail))

# BF16 GEMM -> HMMA.16816
bf16 = find_sections("sm120_bf16_gemm_impl")
check("bf16_gemm: section present", bool(bf16))
check("bf16_gemm: HMMA.16816 in SASS",
      any(has_mma(b, r"^HMMA\.16816") for b in bf16.values()))

# FP8 / FP4 block-scaled 1D1D GEMM -> QMMA (FP8) / OMMA (FP4) with .SF. in
# the mnemonic. The .SF. qualifier proves ptxas did NOT drop block scaling.
gemm1d1d = find_sections("sm120_fp8_fp4_gemm_1d1d_impl")
check("fp8_fp4_gemm_1d1d: 2 instantiation sections present", len(gemm1d1d) == 2,
      f"found {len(gemm1d1d)}")
for name, body in gemm1d1d.items():
    sf_ops = [m for m in mnemonics(body)
              if m.startswith(("QMMA", "OMMA")) and ".SF." in m]
    check(f"fp8_fp4_gemm_1d1d: [QO]MMA.*.SF.* in section ...{name[-60:]}",
          bool(sf_ops), f"e.g. {sf_ops[0] if sf_ops else 'none'}")

# MQA / paged / sparse logits kernels -> some HMMA/QMMA/OMMA family opcode.
for label in ("sm120_fp8_mqa_logits", "sm120_fp4_mqa_logits",
              "sm120_fp8_paged_mqa_logits", "sm120_fp4_paged_mqa_logits",
              "sm120_fp8_fp4_sparse_mqa_logits_kernel"):
    secs = find_sections(label)
    check(f"{label}: section(s) present", bool(secs))
    for name, body in secs.items():
        check(f"{label}: HMMA/QMMA/OMMA in section ...{name[-60:]}",
              has_mma(body, r"^(HMMA|QMMA|OMMA)\."))

# PTX of the same TU must keep the block_scale qualifier.
check("instantiations PTX mentions block_scale", "block_scale" in ptx)

width = max(len(n) for n, _, _ in checks)
failed = 0
for name, ok, detail in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    failed += 0 if ok else 1
print()
if failed:
    print(f"Step 3: FAIL ({failed} assertions failed)")
    sys.exit(1)
print("Step 3: PASS")
PYEOF
then
    :
else
    failures=1
fi
echo

if [[ $failures -ne 0 ]]; then
    echo "COMPILE GATE: FAIL"
    exit 1
fi
echo "COMPILE GATE: PASS (base: $BASE_INC)"
