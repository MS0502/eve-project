#!/usr/bin/env python3
"""Run one exact-reviewed M3-C-O private-device goal dual-read window.

The command is dormant until a later isolated exact pin is merged.  It requires
one clean checkout, one canonical locally reviewed package, four distinct new
private paths, and the complete forbidden prior-path digest set.  It never
installs a default runtime hook and refuses every existing output.
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

from core.m3_c_o_private_device_goal_dual_read_operator import (  # noqa: E402
    active_reviewed_operator_pin,
    build_private_path_binding,
    execute_private_device_goal_dual_read_window,
    read_canonical_private_package,
    require_single_use_private_paths,
    verify_active_operator_authorization,
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
        raise SystemExit("operator requires a clean exact repository checkout")
    return actual


def _require_private_file(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit("private package file is absent")
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise SystemExit("private package file must not grant group/other permissions")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--package-file", required=True)
    parser.add_argument("--working-store", required=True)
    parser.add_argument("--baseline-backup", required=True)
    parser.add_argument("--separate-restore", required=True)
    parser.add_argument(
        "--forbidden-path-digest",
        action="append",
        required=True,
        help="lowercase SHA-256 digest of every prior private path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    launch_head = _repository_head(args.expected_head)

    # Fail before resolving or reading any private path while the checked-in
    # implementation remains default-absent.
    active_reviewed_operator_pin()

    package_path = _outside_repository(Path(args.package_file), "package file")
    working_store = _outside_repository(Path(args.working_store), "working store")
    baseline_backup = _outside_repository(
        Path(args.baseline_backup), "baseline backup"
    )
    separate_restore = _outside_repository(
        Path(args.separate_restore), "separate restore"
    )
    package_path = _require_private_file(package_path)
    path_binding = build_private_path_binding(
        package_path=package_path,
        working_store_path=working_store,
        baseline_backup_path=baseline_backup,
        separate_restore_path=separate_restore,
        forbidden_existing_path_digests=tuple(
            sorted(args.forbidden_path_digest)
        ),
    )
    require_single_use_private_paths(
        package_path=package_path,
        working_store_path=working_store,
        baseline_backup_path=baseline_backup,
        separate_restore_path=separate_restore,
        path_binding=path_binding,
    )
    package = read_canonical_private_package(package_path)
    verify_active_operator_authorization(package.authorization)

    from main import build_full_engine

    engine = build_full_engine()
    goal_adapter = engine.goal_adapter
    if goal_adapter.production_origin_shadow_tap is not None:
        raise SystemExit("default engine unexpectedly contains a goal shadow tap")
    receipt, window_receipt = execute_private_device_goal_dual_read_window(
        package,
        goal_adapter=goal_adapter,
        path_binding=path_binding,
        working_store_path=working_store,
        baseline_backup_path=baseline_backup,
        separate_restore_path=separate_restore,
        launch_repository_head=launch_head,
    )
    print(
        _canonical(
            {
                "operator_receipt": receipt.to_mapping(),
                "operator_receipt_digest": receipt.receipt_digest,
                "private_path_plaintext_public": False,
                "raw_private_text_public": False,
                "window_receipt": window_receipt.to_mapping(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
