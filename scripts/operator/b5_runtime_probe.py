"""Controlled child used to prove B5 supervision without activating EVE runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.authoritative_store import (  # noqa: E402
    AUTHORITY_FAILURE_EXIT_CODE,
    AuthorityPersistenceError,
    AuthorityUnprovable,
    AuthoritativeStore,
)

CONTROL_SCHEMA = "eve.b5-runtime-probe-control.v1"
READY_SCHEMA = "eve.b5-runtime-probe-ready.v1"


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


def _atomic_json(path: Path, packet: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(
            json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
            + b"\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_raw(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    packet = {"timestamp_utc": _utc_now(), "event": event, "pid": os.getpid(), **fields}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_control(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CONTROL_SCHEMA:
        raise AuthorityUnprovable("B5 runtime probe control differs")
    if payload.get("action") not in {"verify_once", "verify_and_hold", "crash_93_once_then_hold"}:
        raise AuthorityUnprovable("B5 runtime probe action differs")
    return payload


def _boot_identity() -> dict[str, Any]:
    if os.name != "nt":
        return {"platform": os.name, "last_boot_utc": None}
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetTickCount64.argtypes = []
    kernel32.GetTickCount64.restype = ctypes.c_ulonglong
    uptime_ms = int(kernel32.GetTickCount64())
    boot_epoch = time.time() - (uptime_ms / 1000.0)
    return {
        "platform": "nt",
        "uptime_milliseconds": uptime_ms,
        "last_boot_utc": datetime.fromtimestamp(boot_epoch, timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--raw-log", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        control = _load_control(args.control)
        action = str(control["action"])
        marker_value = control.get("crash_marker")
        marker = Path(str(marker_value)).resolve() if marker_value else None
        _append_raw(
            args.raw_log,
            "probe_started",
            action=action,
            database=str(args.database.resolve()),
            store_sha256=_sha256(args.database),
        )
        store = AuthoritativeStore(args.database)
        try:
            startup = store.open()
            verification = store.verify()
        finally:
            store.close()
        _append_raw(
            args.raw_log,
            "startup_tail_chain_verified",
            startup=asdict(startup),
            verification=asdict(verification),
            store_sha256=_sha256(args.database),
        )
        if action == "crash_93_once_then_hold":
            if marker is None:
                raise AuthorityUnprovable("crash marker is required")
            marker.parent.mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(_utc_now() + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                _append_raw(args.raw_log, "intentional_crash", exit_code=93)
                os._exit(93)
        ready: dict[str, Any] = {
            "schema": READY_SCHEMA,
            "ready_at_utc": _utc_now(),
            "pid": os.getpid(),
            "authority_active_for_runtime": False,
            "t0_started": False,
            "python": platform.python_version(),
            "numpy_version": __import__("numpy").__version__,
            "database": {
                "path": str(args.database.resolve()),
                "sha256": _sha256(args.database),
                "startup": asdict(startup),
                "verification": asdict(verification),
            },
            "boot": _boot_identity(),
            "action": action,
        }
        ready["receipt_sha256"] = hashlib.sha256(_canonical(ready)).hexdigest()
        _atomic_json(args.ready, ready)
        _append_raw(args.raw_log, "probe_ready", receipt_sha256=ready["receipt_sha256"])
        if action == "verify_once":
            return 0
        stop_value = control.get("stop_file")
        stop_file = Path(str(stop_value)).resolve() if stop_value else None
        while stop_file is None or not stop_file.exists():
            time.sleep(0.25)
        _append_raw(args.raw_log, "operator_stop_observed", stop_file=str(stop_file))
        return 0
    except (AuthorityUnprovable, AuthorityPersistenceError, OSError, ValueError) as exc:
        try:
            _append_raw(
                args.raw_log,
                "authority_unprovable",
                exit_code=AUTHORITY_FAILURE_EXIT_CODE,
                error_type=type(exc).__name__,
                error=str(exc),
                store_sha256=_sha256(args.database) if args.database.is_file() else None,
            )
        except OSError:
            pass
        print(f"authority unprovable: {exc}", file=sys.stderr, flush=True)
        return AUTHORITY_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
