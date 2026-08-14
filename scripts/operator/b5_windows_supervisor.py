"""Windows supervision boundary for the B2 authoritative store.

WinSW/NSSM starts this process, never EVE directly.  The supervisor latches an
authority failure (child exit 86) in a durable sentinel, emits a durable local
alert, and returns success so Windows Service Recovery cannot reinterpret the
authority failure as a restartable service crash.  Other non-zero child exits
are restarted with bounded exponential backoff; child exit zero is final.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.authoritative_store import AUTHORITY_FAILURE_EXIT_CODE  # noqa: E402
from scripts.operator.b5_runtime_environment import (  # noqa: E402
    RuntimeEnvironmentError,
    load_and_verify_receipt,
)

SENTINEL_SCHEMA = "eve.b5-authority-stop-sentinel.v1"
AUDIT_SCHEMA = "eve.b5-supervisor-audit-record.v1"
STATE_SCHEMA = "eve.b5-supervisor-state.v1"
GENESIS_HASH = "0" * 64


class SupervisorError(RuntimeError):
    """The supervisor cannot prove a safe launch or state transition."""


@dataclass(frozen=True)
class SupervisorPaths:
    sentinel: Path
    audit_log: Path
    alert_log: Path
    state_file: Path
    authority_store: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(_canonical({"argv": list(command)})).hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    if exclusive:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                path.unlink()
            except OSError:
                pass
            raise
        _fsync_directory(path.parent)
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SupervisorError(
                    f"audit log JSON is invalid at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise SupervisorError("audit log record is not an object")
            records.append(value)
    return records


def _audit_tail(path: Path) -> tuple[int, str]:
    previous = GENESIS_HASH
    records = _read_json_lines(path)
    for index, record in enumerate(records, 1):
        claimed = record.get("record_sha256")
        unsigned = dict(record)
        unsigned.pop("record_sha256", None)
        if (
            unsigned.get("schema") != AUDIT_SCHEMA
            or unsigned.get("sequence") != index
            or unsigned.get("previous_record_sha256") != previous
            or claimed != hashlib.sha256(_canonical(unsigned)).hexdigest()
        ):
            raise SupervisorError(f"audit hash chain differs at line {index}")
        previous = str(claimed)
    return len(records), previous


def _append_record(path: Path, event: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, previous = _audit_tail(path)
    record: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "sequence": count + 1,
        "timestamp_utc": _utc_now(),
        "event": event,
        "previous_record_sha256": previous,
        **fields,
    }
    record["record_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def _write_state(path: Path, status: str, **fields: Any) -> None:
    _atomic_json(
        path,
        {
            "schema": STATE_SCHEMA,
            "timestamp_utc": _utc_now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            **fields,
        },
        exclusive=False,
    )


def _sentinel_payload(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"sentinel_unreadable:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "sentinel_not_object"
    claimed = payload.get("sentinel_sha256")
    unsigned = dict(payload)
    unsigned.pop("sentinel_sha256", None)
    if payload.get("schema") != SENTINEL_SCHEMA:
        return None, "sentinel_schema_differs"
    if claimed != hashlib.sha256(_canonical(unsigned)).hexdigest():
        return None, "sentinel_digest_differs"
    return payload, "valid"


def _write_sentinel(
    paths: SupervisorPaths,
    *,
    command: Sequence[str],
    child_pid: int,
    before_sha256: str,
    after_sha256: str,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": SENTINEL_SCHEMA,
        "latched_at_utc": _utc_now(),
        "reason": "authority_unprovable_child_exit_86",
        "child_exit_code": AUTHORITY_FAILURE_EXIT_CODE,
        "supervisor_pid": os.getpid(),
        "child_pid": child_pid,
        "command_sha256": _command_sha256(command),
        "authority_store": str(paths.authority_store.resolve()),
        "store_sha256_before": before_sha256,
        "store_sha256_after": after_sha256,
        "automatic_clear_permitted": False,
        "operator_clear_required": True,
    }
    packet["sentinel_sha256"] = hashlib.sha256(_canonical(packet)).hexdigest()
    try:
        _atomic_json(paths.sentinel, packet, exclusive=True)
    except FileExistsError:
        existing, state = _sentinel_payload(paths.sentinel)
        if existing is None:
            raise SupervisorError(f"existing sentinel is invalid: {state}")
        return existing
    return packet


def _alert(paths: SupervisorPaths, sentinel: Mapping[str, Any]) -> None:
    record = _append_record(
        paths.alert_log,
        "AUTHORITY_STOP_LATCHED",
        {
            "severity": "critical",
            "child_exit_code": AUTHORITY_FAILURE_EXIT_CODE,
            "sentinel": str(paths.sentinel.resolve()),
            "sentinel_sha256": sentinel["sentinel_sha256"],
            "message": "EVE authority is unprovable; restart is prohibited until operator clear.",
        },
    )
    print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)
    if os.name == "nt":
        message = (
            "EVE authority unprovable; child exit 86; sentinel "
            f"{paths.sentinel.resolve()}"
        )
        result = subprocess.run(
            [
                "eventcreate.exe",
                "/L",
                "APPLICATION",
                "/T",
                "ERROR",
                "/ID",
                "86",
                "/SO",
                "EVE-B5-Supervisor",
                "/D",
                message,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        _append_record(
            paths.audit_log,
            "windows_event_alert_attempt",
            {
                "return_code": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            },
        )


@contextmanager
def _instance_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if handle.tell() == 0 and handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise SupervisorError("another supervisor holds the instance lock") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise SupervisorError("another supervisor holds the instance lock") from exc
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def run_supervisor(
    paths: SupervisorPaths,
    runtime_receipt: Path,
    command: Sequence[str],
    *,
    backoff_initial: float,
    backoff_max: float,
    max_restarts: int | None,
) -> int:
    if not command:
        raise SupervisorError("child command is empty")
    if not paths.authority_store.is_file():
        raise SupervisorError("authority store is absent")
    receipt = load_and_verify_receipt(runtime_receipt)
    receipt_python = Path(receipt["python"]["interpreter"]).resolve()
    child_python = Path(command[0]).resolve()
    if not child_python.is_file() or not child_python.samefile(receipt_python):
        raise SupervisorError("child is not launched by the pinned interpreter")
    lock_path = paths.sentinel.with_suffix(paths.sentinel.suffix + ".supervisor.lock")
    with _instance_lock(lock_path):
        if paths.sentinel.exists():
            sentinel, sentinel_state = _sentinel_payload(paths.sentinel)
            _append_record(
                paths.audit_log,
                "startup_blocked_by_sentinel",
                {
                    "sentinel_state": sentinel_state,
                    "sentinel_sha256": None
                    if sentinel is None
                    else sentinel["sentinel_sha256"],
                    "child_launched": False,
                },
            )
            _write_state(
                paths.state_file,
                "STOPPED_LATCHED",
                sentinel_state=sentinel_state,
                child_launched=False,
            )
            return 0
        restarts = 0
        while True:
            if paths.sentinel.exists():
                _append_record(
                    paths.audit_log,
                    "restart_blocked_by_sentinel",
                    {"restart_count": restarts, "child_launched": False},
                )
                _write_state(paths.state_file, "STOPPED_LATCHED", child_launched=False)
                return 0
            store_before = _sha256(paths.authority_store)
            started_at = _utc_now()
            child = subprocess.Popen(list(command), cwd=ROOT)
            _append_record(
                paths.audit_log,
                "child_started",
                {
                    "child_pid": child.pid,
                    "started_at_utc": started_at,
                    "restart_count": restarts,
                    "command_sha256": _command_sha256(command),
                    "store_sha256_before": store_before,
                    "numpy_version": receipt["numpy_version"],
                },
            )
            _write_state(
                paths.state_file,
                "RUNNING_CHILD",
                child_pid=child.pid,
                restart_count=restarts,
                store_sha256=store_before,
            )
            exit_code = child.wait()
            store_after = _sha256(paths.authority_store)
            _append_record(
                paths.audit_log,
                "child_exited",
                {
                    "child_pid": child.pid,
                    "child_exit_code": exit_code,
                    "restart_count": restarts,
                    "store_sha256_before": store_before,
                    "store_sha256_after": store_after,
                },
            )
            if exit_code == 0:
                _write_state(
                    paths.state_file,
                    "STOPPED_NORMAL",
                    child_exit_code=0,
                    child_pid=child.pid,
                    restart_count=restarts,
                )
                return 0
            if exit_code == AUTHORITY_FAILURE_EXIT_CODE:
                sentinel = _write_sentinel(
                    paths,
                    command=command,
                    child_pid=child.pid,
                    before_sha256=store_before,
                    after_sha256=store_after,
                )
                _append_record(
                    paths.audit_log,
                    "authority_stop_latched",
                    {
                        "child_exit_code": exit_code,
                        "sentinel_sha256": sentinel["sentinel_sha256"],
                        "automatic_restart": False,
                    },
                )
                _alert(paths, sentinel)
                _write_state(
                    paths.state_file,
                    "STOPPED_LATCHED",
                    child_exit_code=exit_code,
                    sentinel_sha256=sentinel["sentinel_sha256"],
                    child_launched=False,
                )
                return 0
            if max_restarts is not None and restarts >= max_restarts:
                _append_record(
                    paths.audit_log,
                    "restart_budget_exhausted",
                    {"child_exit_code": exit_code, "restart_count": restarts},
                )
                _write_state(
                    paths.state_file,
                    "STOPPED_RESTART_BUDGET",
                    child_exit_code=exit_code,
                    restart_count=restarts,
                )
                return exit_code if 0 < exit_code < 256 else 1
            delay = min(backoff_initial * (2**restarts), backoff_max)
            restarts += 1
            _append_record(
                paths.audit_log,
                "restart_scheduled",
                {
                    "child_exit_code": exit_code,
                    "restart_count": restarts,
                    "backoff_seconds": delay,
                },
            )
            _write_state(
                paths.state_file,
                "BACKOFF",
                child_exit_code=exit_code,
                restart_count=restarts,
                backoff_seconds=delay,
            )
            time.sleep(delay)


def clear_sentinel(
    sentinel_path: Path,
    audit_log: Path,
    *,
    operator: str,
    reason: str,
    expected_sha256: str,
) -> Path:
    if not operator.strip() or not reason.strip():
        raise SupervisorError("operator and reason are required")
    if not sentinel_path.exists():
        raise SupervisorError("active sentinel is absent")
    payload, state = _sentinel_payload(sentinel_path)
    if payload is None:
        raise SupervisorError(f"invalid sentinel cannot be cleared: {state}")
    if payload["sentinel_sha256"] != expected_sha256:
        raise SupervisorError("sentinel digest changed; refusing clear")
    archive = sentinel_path.with_name(
        f"{sentinel_path.name}.cleared-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    os.replace(sentinel_path, archive)
    _fsync_directory(sentinel_path.parent)
    _append_record(
        audit_log,
        "operator_sentinel_clear",
        {
            "operator": operator.strip(),
            "reason": reason.strip(),
            "sentinel_sha256": expected_sha256,
            "archived_sentinel": str(archive.resolve()),
            "automatic": False,
        },
    )
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run")
    for name in ("sentinel", "audit-log", "alert-log", "state-file", "authority-store"):
        run_parser.add_argument(f"--{name}", type=Path, required=True)
    run_parser.add_argument("--runtime-receipt", type=Path, required=True)
    run_parser.add_argument("--backoff-initial", type=float, default=1.0)
    run_parser.add_argument("--backoff-max", type=float, default=60.0)
    run_parser.add_argument("--max-restarts", type=int)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    clear_parser = subparsers.add_parser("clear-sentinel")
    clear_parser.add_argument("--sentinel", type=Path, required=True)
    clear_parser.add_argument("--audit-log", type=Path, required=True)
    clear_parser.add_argument("--operator", required=True)
    clear_parser.add_argument("--reason", required=True)
    clear_parser.add_argument("--expected-sentinel-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "clear-sentinel":
            archive = clear_sentinel(
                args.sentinel,
                args.audit_log,
                operator=args.operator,
                reason=args.reason,
                expected_sha256=args.expected_sentinel_sha256,
            )
            print(json.dumps({"cleared": True, "archive": str(archive)}, sort_keys=True))
            return 0
        command = args.command
        if command and command[0] == "--":
            command = command[1:]
        if args.backoff_initial <= 0 or args.backoff_max < args.backoff_initial:
            raise SupervisorError("backoff bounds are invalid")
        if args.max_restarts is not None and args.max_restarts < 0:
            raise SupervisorError("max restarts cannot be negative")
        return run_supervisor(
            SupervisorPaths(
                sentinel=args.sentinel,
                audit_log=args.audit_log,
                alert_log=args.alert_log,
                state_file=args.state_file,
                authority_store=args.authority_store,
            ),
            args.runtime_receipt,
            command,
            backoff_initial=args.backoff_initial,
            backoff_max=args.backoff_max,
            max_restarts=args.max_restarts,
        )
    except (SupervisorError, RuntimeEnvironmentError, OSError) as exc:
        print(f"supervisor fail-closed: {exc}", file=sys.stderr)
        return AUTHORITY_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
