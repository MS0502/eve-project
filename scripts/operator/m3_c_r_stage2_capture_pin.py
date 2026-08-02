#!/usr/bin/env python3
"""Stage 2: capture or verify the local pin without loading EVE."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m3_c_p_private_device_goal_window_authorization_pin import (  # noqa: E402
    binding_from_private_package,
)
from core.m3_c_r_resumable_phone_goal_window import (  # noqa: E402
    canonical_json,
    capture_or_reuse_local_pin,
    load_canonical_mapping,
    load_local_pin_for_binding,
    private_paths,
    validate_immutable_inputs,
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
        raise SystemExit("stage 2 requires a clean exact checkout")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--private-root", required=True)
    return parser


def _completed_capture_receipt(private_root: Path) -> dict | None:
    paths = private_paths(private_root)
    if not paths["consumed_pin"].exists():
        return None
    if paths["pin"].exists():
        raise SystemExit("both active and consumed local pin paths exist")
    package, _, _ = validate_immutable_inputs(paths["root"])
    load_local_pin_for_binding(paths["root"], binding_from_private_package(package))
    return load_canonical_mapping(
        paths["pin_receipt"], field="authorization capture receipt"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _clean_exact_head(args.expected_head)
    paths = private_paths(args.private_root)
    receipt = _completed_capture_receipt(paths["root"])
    if receipt is None:
        receipt = capture_or_reuse_local_pin(paths["root"])
    print(
        canonical_json(
            {
                "authorization_capture": receipt,
                "engine_loaded": False,
                "stage": 2,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
