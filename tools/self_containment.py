#!/usr/bin/env python3
"""Include-closure gate for the vendored SM120 device tree.

Every `.cuh` under `deep_gemm/include/` may only include:

  1. CUTLASS / CUDA / standard C++ headers:
     `cute/...`, `cutlass/...`, `cuda/...`, `cuda_bf16.h`, and standard C/C++
     headers (e.g. `cstdint`, `cuda_runtime.h`).
  2. Other headers physically present in this repository's
     `deep_gemm/include/` tree (resolved dynamically, so a newly vendored
     file is automatically allowed).
  3. The fixed allowlist of base-provided shared headers below. Each fork
     base (nv_dev lineage and vllm-project main lineage) supplies these at
     the same paths; they differ textually between lineages, so the contract
     with them is symbol-level, not byte-level.

Any other `#include <deep_gemm/...>` is a violation: it would silently pick up
a base-checkout header that is not part of the compatibility contract.

Exit code 0 on success, 1 on any violation. Stdlib only.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INCLUDE_ROOT = REPO_ROOT / "deep_gemm" / "include"

# Base-provided shared headers (category 3). Each vendoring fork supplies
# these; compatibility is symbol-level. Extending this list is a deliberate
# contract change and must be reflected in docs/design.md.
#
# NOTE: `common/packing.cuh` is included by the vendored
# `layout/sparse_mqa_logits.cuh` and is provided byte-identically by both
# supported base lineages, so it is part of the contract.
BASE_PROVIDED_ALLOWLIST = frozenset(
    "deep_gemm/" + p
    for p in (
        "common/cute_tie.cuh",
        "common/math.cuh",
        "common/packing.cuh",
        "common/tma_copy.cuh",
        "common/types.cuh",
        "common/utils.cuh",
        "epilogue/transform.cuh",
        "ptx/ld_st.cuh",
        "ptx/tma.cuh",
        "ptx/utils.cuh",
    )
)

# Category 1: header-name prefixes that are always allowed (CUTLASS/CUDA).
EXTERNAL_PREFIXES = ("cute/", "cutlass/", "cuda/")

# Well-known CUDA/C/C++ headers without a directory component.
EXTERNAL_HEADERS = frozenset({
    "cuda_bf16.h",
    "cuda_fp16.h",
    "cuda_fp8.h",
    "cuda_fp4.h",
    "cuda_runtime.h",
    "cuda_runtime_api.h",
    "device_launch_parameters.h",
})

# Standard C/C++ headers are extensionless (`cstdint`, `tuple`) or the C
# compatibility headers (`stdint.h`, ...). Anything else without a directory
# is suspicious and reported as unclassified.
STD_HEADER_RE = re.compile(r"^[a-z0-9_]+(\.h)?$")

INCLUDE_RE = re.compile(r"""^\s*#\s*include\s*[<"]([^>"]+)[>"]""")


def classify(header: str, repo_headers: frozenset) -> tuple:
    """Return (category, detail). category is one of:
    'system', 'repo', 'base', 'violation'."""
    if header.startswith("deep_gemm/"):
        if header in repo_headers:
            return "repo", "vendored in this tree"
        if header in BASE_PROVIDED_ALLOWLIST:
            return "base", "base-provided (allowlist)"
        return "violation", "not vendored here and not in the base-provided allowlist"
    if header.startswith(EXTERNAL_PREFIXES) or header in EXTERNAL_HEADERS:
        return "system", "CUTLASS/CUDA"
    if STD_HEADER_RE.match(header):
        return "system", "standard C/C++"
    return "violation", "unrecognized external header (not cute/cutlass/cuda/std)"


def main() -> int:
    if not INCLUDE_ROOT.is_dir():
        print(f"error: {INCLUDE_ROOT} not found", file=sys.stderr)
        return 1

    cuh_files = sorted(INCLUDE_ROOT.rglob("*.cuh"))
    if not cuh_files:
        print(f"error: no .cuh files under {INCLUDE_ROOT}", file=sys.stderr)
        return 1

    # Header names resolvable inside this repo, in `#include <...>` spelling.
    repo_headers = frozenset(
        str(p.relative_to(INCLUDE_ROOT)) for p in INCLUDE_ROOT.rglob("*.cuh")
    )

    violations = []
    counts = {"system": 0, "repo": 0, "base": 0}
    for path in cuh_files:
        rel = path.relative_to(REPO_ROOT)
        entries = []
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            m = INCLUDE_RE.match(line)
            if not m:
                continue
            header = m.group(1)
            category, detail = classify(header, repo_headers)
            if category == "violation":
                violations.append((rel, lineno, header, detail))
            else:
                counts[category] += 1
            entries.append((lineno, header, category, detail))

        print(f"{rel}")
        for lineno, header, category, detail in entries:
            marker = "VIOLATION" if category == "violation" else category
            print(f"  L{lineno:<4} {header:<45} [{marker}] {detail}")

    print()
    print(
        f"checked {len(cuh_files)} headers: "
        f"{counts['repo']} vendored-repo includes, "
        f"{counts['base']} base-provided includes, "
        f"{counts['system']} system includes, "
        f"{len(violations)} violations"
    )
    if violations:
        print("\nVIOLATIONS (each must be resolved or allowlisted deliberately):")
        for rel, lineno, header, detail in violations:
            print(f"  {rel}:{lineno}: <{header}> -- {detail}")
        return 1
    print("self_containment: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
