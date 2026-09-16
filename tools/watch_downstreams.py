#!/usr/bin/env python3
"""Downstream drift watcher for the SM120 device layer.

The canonical repo watches its downstreams, not the other way around: any
fork can merge a PR that edits the vendored headers directly, and this tool
is how the upstream finds out. For each registered downstream ref, it fetches
the vendored files over HTTPS and compares their sha256 set against the
manifest of every release tag:

  * exact match against some tag -> in sync at that tag;
  * no exact match -> drift: someone changed the files downstream (or a
    partial sync). The fix is to absorb the change here and re-tag, after
    which the downstream re-vendors.

Usage (from a full clone of this repo, tags fetched):

    python3 tools/watch_downstreams.py [--config .github/downstreams.json]
                                       [--report drift-report.md]

Exit code 0 when every downstream matches some tag, 1 on any drift or
fetch failure. Stdlib only.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def fetch_sha256(url: str) -> str | None:
    """None when the file does not exist at that ref (404)."""
    req = urllib.request.Request(url, headers={"User-Agent": "deepgemm-sm120-watch"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return hashlib.sha256(resp.read()).hexdigest()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def tag_manifests(repo_root: Path) -> dict[str, dict]:
    """Map tag -> manifest for every tag carrying VENDOR-sm120.json."""
    tags = subprocess.run(["git", "tag", "--list", "v*"], cwd=repo_root,
                          capture_output=True, text=True, check=True).stdout.split()
    manifests = {}
    for tag in sorted(tags):
        show = subprocess.run(["git", "show", f"{tag}:VENDOR-sm120.json"], cwd=repo_root,
                              capture_output=True, text=True)
        if show.returncode == 0:
            manifests[tag] = json.loads(show.stdout)["files"]
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", default=repo_root / ".github" / "downstreams.json")
    parser.add_argument("--report", default=None, help="write a markdown report here")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    manifests = tag_manifests(repo_root)
    if not manifests:
        print("error: no release tag manifests found; run from a full clone with tags",
              file=sys.stderr)
        return 1
    print(f"known release tags: {', '.join(manifests)}")

    drifted = []
    lines = ["# Downstream vendor drift report", ""]
    for ds in config["downstreams"]:
        name, repo, ref = ds["name"], ds["repo"], ds["ref"]
        # Union of all files across tags (the vendored surface is append-mostly).
        paths = sorted({p for m in manifests.values() for p in m})
        got = {}
        try:
            for rel in paths:
                got[rel] = fetch_sha256(RAW.format(repo=repo, ref=ref, path=rel))
        except Exception as e:
            print(f"{name}: fetch FAILED: {e}")
            drifted.append(name)
            lines += [f"## {name} (`{repo}@{ref}`)", "", f"fetch failed: `{e}`", ""]
            continue

        matched = [tag for tag, files in manifests.items()
                   if all(got.get(rel) == want for rel, want in files.items())]
        if matched:
            status = f"in sync @ {' / '.join(matched)}"
            print(f"{name}: {status}")
            lines += [f"## {name} (`{repo}@{ref}`)", "", f"OK: {status}", ""]
            continue

        drifted.append(name)
        latest = next(reversed(manifests))
        files = manifests[latest]
        missing = [rel for rel in files if got.get(rel) is None]
        changed = [rel for rel in files if got.get(rel) not in (None, files[rel])]
        extra = [rel for rel, sha in got.items()
                 if sha is not None and all(rel not in m for m in manifests.values())]
        print(f"{name}: DRIFT vs {latest} (missing {len(missing)}, changed {len(changed)}, "
              f"extra {len(extra)})")
        lines += [f"## {name} (`{repo}@{ref}`)", "",
                  f"**DRIFT** — matches no release tag (latest: {latest}).", ""]
        for rel in missing:
            lines.append(f"- MISSING `{rel}`")
        for rel in changed:
            lines.append(f"- CHANGED `{rel}`")
        for rel in extra:
            lines.append(f"- EXTRA `{rel}` (not part of any release)")
        lines.append("")

    if args.report:
        Path(args.report).write_text("\n".join(lines) + "\n")
    if drifted:
        print(f"\ndrift: {', '.join(drifted)}")
        return 1
    print("\nall downstreams in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
