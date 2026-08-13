#!/usr/bin/env python3
"""Stage 5: record digest-only human review without loading EVE."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m3_c_r_resumable_phone_goal_window import (  # noqa: E402
    canonical_json,
    private_paths,
    record_public_review,
)


def _clean_exact_head(expected_head: str) -> None:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if actual != expected_head:
        raise SystemExit(
            f"repository head mismatch: expected {expected_head}, got {actual}"
        )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    if dirty.strip():
        raise SystemExit("stage 5 requires a clean exact checkout")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--reviewed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _clean_exact_head(args.expected_head)
    if not args.reviewed:
        raise SystemExit(
            "--reviewed is required after reviewing the exact stage-4 receipt"
        )
    paths = private_paths(args.private_root)
    review = record_public_review(paths["root"])
    print(
        canonical_json(
            {"engine_loaded": False, "public_review": review, "stage": 5}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
