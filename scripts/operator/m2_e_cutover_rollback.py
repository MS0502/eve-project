#!/usr/bin/env python3
"""Write the private one-command M2-E operational rollback control record."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m2_e_cutover_activation import (  # noqa: E402
    build_operational_rollback_record,
    canonical_rollback_json,
)

DEFAULT_PRIVATE_ROOT = Path("~/.local/share/eve-m2e-window-private")
ROLLBACK_FILENAME = "m2_e_cutover_operational_rollback.json"


def _outside_repository(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise SystemExit("rollback control must remain outside the repository")


def _write_once(path: Path, text: str) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8").rstrip("\n")
        if existing == text:
            os.chmod(path, 0o600)
            return
        raise SystemExit("different rollback control already exists; refusing overwrite")
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(text + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--requested-by", default="김민석")
    parser.add_argument("--reason", default="operator_requested_fail_closed_rollback")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    private_root = _outside_repository(args.private_root)
    private_root.mkdir(parents=True, exist_ok=True)
    os.chmod(private_root, 0o700)
    record = build_operational_rollback_record(
        requested_by=args.requested_by,
        reason=args.reason,
    )
    text = canonical_rollback_json(record)
    target = private_root / ROLLBACK_FILENAME
    _write_once(target, text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
