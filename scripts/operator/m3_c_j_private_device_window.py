#!/usr/bin/env python3
"""Execute one exact-reviewed four-event M3-C-J private-device window.

The command is single-use. It refuses an existing database or SQLite sidecar,
requires a clean exact repository head, keeps raw goal/drive input and path
plaintext private, and never replays the #211 witness or retained sequences 1-5.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m3_c_j_private_device_operator import (  # noqa: E402
    PrivateDeviceGoalInput,
    active_reviewed_private_device_operator_authorization_packet,
    execute_private_device_observation_window,
    verify_active_private_device_operator_authorization,
)

PRIVATE_FILENAME = "m3_c_j_private_device_bundle_v1.json"
PUBLIC_FILENAME = "m3_c_j_public_review_v1.json"
JOURNAL_FILENAME = "m3_c_j_operator_journal_private_v1.json"


def _canonical(value: Any) -> str:
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
        raise SystemExit("operator window requires a clean exact repository checkout")
    return actual


def _read_private_nonce(path: Path) -> bytes:
    stat_result = path.stat()
    if os.name != "nt" and stat_result.st_mode & 0o077:
        raise SystemExit("private nonce file must not grant group/other permissions")
    value = path.read_bytes()
    if len(value) < 32:
        raise SystemExit("private nonce file must contain at least 32 bytes")
    return value


def _read_input(path: Path) -> PrivateDeviceGoalInput:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("operator input file is unreadable or invalid JSON") from exc
    if not isinstance(value, Mapping) or _canonical(value) != path.read_text(
        encoding="utf-8"
    ).strip():
        raise SystemExit("operator input must be canonical JSON")
    return PrivateDeviceGoalInput.from_mapping(value)


def _prepare_private_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    for name in (PRIVATE_FILENAME, PUBLIC_FILENAME, JOURNAL_FILENAME):
        if (path / name).exists():
            raise SystemExit("single-use operator output already exists")
    for child in (path / "backups", path / "restore"):
        if child.exists():
            raise SystemExit("single-use backup or restore directory already exists")
    return path


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    if path.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite {path.name}")
    temporary.write_text(_canonical(value) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(_canonical(value) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--database-path", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--nonce-file", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--launch-attestation-id", required=True)
    parser.add_argument("--runtime-instance-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_head = _repository_head(args.expected_head)
    packet = active_reviewed_private_device_operator_authorization_packet()
    verify_active_private_device_operator_authorization(packet)

    database_path = _outside_repository(Path(args.database_path), "database path")
    private_root = _outside_repository(Path(args.private_root), "private root")
    nonce_path = _outside_repository(Path(args.nonce_file), "private nonce file")
    input_path = _outside_repository(Path(args.input_file), "operator input file")
    private_root = _prepare_private_root(private_root)
    private_nonce = _read_private_nonce(nonce_path)
    operator_input = _read_input(input_path)

    journal_path = private_root / JOURNAL_FILENAME
    _atomic_json(
        journal_path,
        {
            "database_path_plaintext_public": False,
            "operator_authorization_digest": packet.authorization_digest,
            "operator_implementation_head": packet.operator_implementation_head,
            "phone_witness_replayed": False,
            "retained_sequences_replayed": False,
            "schema_version": "eve.m3-c-j.private-device-operator-journal.v1",
            "stage": "authorized_before_database_access",
        },
    )

    bundle = execute_private_device_observation_window(
        packet,
        operator_input=operator_input,
        private_nonce=private_nonce,
        repository_head=repository_head,
        launch_attestation_id=args.launch_attestation_id,
        runtime_instance_id=args.runtime_instance_id,
        database_path=database_path,
        backup_directory=private_root / "backups",
        restore_path=private_root / "restore" / "goal_lifecycle_baseline.sqlite3",
    )
    _atomic_json(private_root / PRIVATE_FILENAME, bundle.private_mapping())
    public_mapping = bundle.public_review_mapping()
    _atomic_json(private_root / PUBLIC_FILENAME, public_mapping)
    _replace_json(
        journal_path,
        {
            "database_path_plaintext_public": False,
            "operator_authorization_digest": packet.authorization_digest,
            "operator_implementation_head": packet.operator_implementation_head,
            "operator_receipt_digest": bundle.operator_receipt.receipt_digest,
            "phone_witness_replayed": False,
            "retained_sequences_replayed": False,
            "schema_version": "eve.m3-c-j.private-device-operator-journal.v1",
            "stage": "complete_writer_disabled_restore_verified",
            "window_receipt_digest": bundle.window_receipt.receipt_digest,
        },
    )
    print(_canonical(public_mapping))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
