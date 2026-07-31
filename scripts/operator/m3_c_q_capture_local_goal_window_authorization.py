#!/usr/bin/env python3
"""Capture one private M3-C-Q local reviewed authorization pin artifact.

This command validates one canonical M3-C-O private package and writes one
single-use private M3-C-P pin file.  It does not install the pin or execute the
M3-C-O operator.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.m3_c_o_private_device_goal_dual_read_operator as m3_c_o_operator  # noqa: E402
from core.m3_c_o_private_device_goal_dual_read_operator import (  # noqa: E402
    read_canonical_private_package,
)
from core.m3_c_p_private_device_goal_window_authorization_pin import (  # noqa: E402
    active_local_reviewed_authorization_pin,
    binding_from_private_package,
)
from core.m3_c_q_local_goal_window_authorization_capture import (  # noqa: E402
    capture_local_reviewed_authorization,
)


def _canonical(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _outside_repository(path: Path, field: str) -> Path:
    if not path.is_absolute():
        raise SystemExit(f"{field} must be absolute")
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise SystemExit(f"{field} must remain outside the repository")


def _repository_head(expected_head: str) -> str:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if actual != expected_head:
        raise SystemExit(
            f"repository head mismatch: expected {expected_head}, got {actual}"
        )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
    )
    if dirty.strip():
        raise SystemExit("capture requires a clean exact repository checkout")
    return actual


def _require_private_file(path: Path, field: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"{field} is absent")
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise SystemExit(f"{field} must not grant group/other permissions")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--package-file", required=True)
    parser.add_argument("--expected-package-digest", required=True)
    parser.add_argument("--pin-output", required=True)
    parser.add_argument(
        "--reviewed",
        action="store_true",
        help="confirm that the exact canonical private package was reviewed locally",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    launch_head = _repository_head(args.expected_head)
    if not args.reviewed:
        raise SystemExit("--reviewed is required after local human review")

    # The capture command requires every runtime activation seam to remain closed.
    try:
        active_local_reviewed_authorization_pin()
    except Exception as exc:
        if "no active local reviewed" not in str(exc):
            raise
    else:
        raise SystemExit("a local authorization pin is already active")
    if (
        m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD is not None
        or m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST is not None
    ):
        raise SystemExit("M3-C-O operator pin seam must remain closed")

    package_path = _require_private_file(
        _outside_repository(Path(args.package_file), "package file"),
        "package file",
    )
    pin_output = _outside_repository(Path(args.pin_output), "pin output")
    package = read_canonical_private_package(package_path)
    if package.package_digest != args.expected_package_digest:
        raise SystemExit("private package digest differs from reviewed digest")
    binding = binding_from_private_package(package)
    pin, receipt = capture_local_reviewed_authorization(
        binding,
        private_output_path=pin_output,
        human_review_confirmed=True,
    )
    print(
        _canonical(
            {
                "capture_receipt": receipt.to_mapping(),
                "capture_receipt_digest": receipt.receipt_digest,
                "launch_repository_head": launch_head,
                "local_pin_digest": pin.pin_digest,
                "operator_executed": False,
                "private_output_path_plaintext_public": False,
                "raw_private_text_public": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
