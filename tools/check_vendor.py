#!/usr/bin/env python3
"""Vendor drift checker for the SM120 device layer.

Runs from the DeepGEMM-sm120 side against any fork checkout; forks commit
nothing extra (no manifest, no this script) -- provenance travels in the
vendored files' header comments and in the vendoring commit message.

    python3 tools/check_vendor.py --fork <fork_repo_root> [--manifest <path>]

The manifest defaults to the `VENDOR-sm120.json` of this repository (which
describes the release at its current tag); pass --manifest to check against
a different release's manifest.

Checks, in both directions:

  1. Manifest -> fork: every file listed in the manifest exists in the fork
     at the same repo-relative path with a matching sha256.
  2. Fork -> manifest: every sm120-named header under the fork's
     `deep_gemm/include/` tree (plus the known non-prefixed vendored surface
     `deep_gemm/include/deep_gemm/layout/sparse_mqa_logits.cuh`) is covered
     by the manifest. Uncovered files are drift: either local fork edits
     that belong upstream, or a partial sync.

Exit code 0 when clean, 1 on any mismatch. Stdlib only.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Vendored files whose basename does not contain "sm120" but are part of the
# vendored surface and must be covered by the manifest.
EXTRA_VENDORED_PATHS = frozenset({
    "deep_gemm/include/deep_gemm/layout/sparse_mqa_logits.cuh",
})


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_sm120_named(rel: str) -> bool:
    return "sm120" in Path(rel).name.lower() or rel in EXTRA_VENDORED_PATHS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fork", required=True, help="fork repository root to check")
    parser.add_argument("--manifest", default=None,
                        help="manifest to check against (default: this repo's VENDOR-sm120.json)")
    args = parser.parse_args()

    root = Path(args.fork).resolve()
    manifest_path = (Path(args.manifest).resolve() if args.manifest
                     else Path(__file__).resolve().parents[1] / "VENDOR-sm120.json")
    if not manifest_path.is_file():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        manifest = json.loads(manifest_path.read_text())
        upstream = manifest["upstream"]
        tag = manifest["tag"]
        files = manifest["files"]
        assert isinstance(files, dict) and files
    except (KeyError, AssertionError, json.JSONDecodeError) as e:
        print(f"error: malformed {manifest_path}: {e}", file=sys.stderr)
        return 1

    print(f"manifest: {upstream} @ {tag} ({len(files)} files)")
    print(f"fork:     {root}")

    problems = 0

    # Direction 1: manifest -> fork.
    for rel, want in sorted(files.items()):
        p = root / rel
        if not p.is_file():
            print(f"  MISSING  {rel}")
            print(f"           listed in the {tag} manifest but absent from the fork;")
            print(f"           re-vendor {tag} (see docs/vendoring.md)")
            problems += 1
            continue
        got = sha256_of(p)
        if got != want:
            print(f"  CHANGED  {rel}")
            print(f"           sha256 {got[:16]}... != manifest {want[:16]}...")
            print(f"           local edit or partial sync; restore the {tag} copy")
            problems += 1

    # Direction 2: fork -> manifest (drift detection).
    include_root = root / "deep_gemm" / "include"
    if include_root.is_dir():
        for p in sorted(include_root.rglob("*.cuh")):
            rel = str(p.relative_to(root))
            if is_sm120_named(rel) and rel not in files:
                print(f"  DRIFT    {rel}")
                print(f"           sm120-named file not covered by the {tag} manifest;")
                print(f"           remove it, or upstream it and re-vendor at a new tag")
                problems += 1
    else:
        print(f"  note: {include_root} absent; nothing to check", file=sys.stderr)
        return 1

    if problems:
        print(f"\ncheck_vendor: FAIL ({problems} problem(s)) against {upstream} @ {tag}")
        return 1
    print(f"check_vendor: PASS ({len(files)} files match {upstream} @ {tag}; no drift)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
