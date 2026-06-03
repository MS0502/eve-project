#!/usr/bin/env python3
"""Validate and optionally restore the Round96 code-only handoff package."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
MANIFEST = PACKAGE_DIR / "eve_v3_round96_code_only_manifest.json"
ARCHIVE = PACKAGE_DIR / "eve_v3_round96_code_only_no_medium_vectors.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="extract the package into the repository root")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    size = ARCHIVE.stat().st_size
    digest = sha256(ARCHIVE)
    size_ok = size == int(manifest["size_bytes"])
    sha_ok = digest == manifest["sha256"]
    with zipfile.ZipFile(ARCHIVE) as zf:
        bad_member = zf.testzip()
        names = zf.namelist()
        if args.extract:
            zf.extractall(REPO_ROOT)
    result = {
        "archive": str(ARCHIVE.relative_to(REPO_ROOT)),
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "size_bytes": size,
        "size_matches_manifest": size_ok,
        "sha256": digest,
        "sha256_matches_manifest": sha_ok,
        "zip_integrity_ok": bad_member is None,
        "bad_member": bad_member,
        "member_count": len(names),
        "extracted": bool(args.extract),
        "extract_root": str(REPO_ROOT) if args.extract else None,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if size_ok and sha_ok and bad_member is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
