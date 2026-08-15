"""Collect and validate B5 Windows/8840U physical supervision evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import subprocess
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
    AuthorityPersistenceError,
    AuthorityUnprovable,
    AuthoritativeStore,
)
from scripts.operator.b5_windows_supervisor import (  # noqa: E402
    SENTINEL_SCHEMA,
    _sentinel_payload,
)

CAPTURE_SCHEMA = "eve.b5-windows-physical-capture.v1"
INJECTION_SCHEMA = "eve.b5-tail-mismatch-injection.v1"
RESTART_CONTINUITY_SCHEMA = "eve.b5-windows-restart-continuity-proof.v1"
GATE_SCHEMA = "eve.b5-windows-8840u-supervision-gate.v2"


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


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _cpu_identity(
    *, os_name: str | None = None, platform_processor: str | None = None
) -> dict[str, Any]:
    reported_processor = (
        platform.processor() if platform_processor is None else platform_processor
    ).strip()
    effective_os_name = os.name if os_name is None else os_name
    if effective_os_name != "nt":
        return {
            "source": "platform.processor",
            "name": reported_processor,
            "platform_processor": reported_processor,
            "raw": None,
        }

    argv = [
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        (
            "$c=Get-CimInstance Win32_Processor | Select-Object -First 1;"
            "[ordered]@{Name=[string]$c.Name;Manufacturer=[string]$c.Manufacturer;"
            "ProcessorId=[string]$c.ProcessorId;Description=[string]$c.Description}"
            "|ConvertTo-Json -Compress"
        ),
    ]
    result = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    raw = {
        "argv": argv,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode != 0:
        raise RuntimeError("Win32_Processor observation failed")
    try:
        observed = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("Win32_Processor observation is not JSON") from exc
    if not isinstance(observed, dict) or not str(observed.get("Name", "")).strip():
        raise RuntimeError("Win32_Processor name is absent")
    return {
        "source": "Win32_Processor",
        "name": str(observed["Name"]).strip(),
        "manufacturer": str(observed.get("Manufacturer", "")).strip(),
        "processor_id": str(observed.get("ProcessorId", "")).strip(),
        "description": str(observed.get("Description", "")).strip(),
        "platform_processor": reported_processor,
        "raw": raw,
    }


def _outside_repository(path: Path, field: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise RuntimeError(f"{field} must remain outside the repository")


def _write(path: Path, packet: dict[str, Any], schema_hash_field: str = "receipt_sha256") -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {path}")
    payload = dict(packet)
    payload[schema_hash_field] = hashlib.sha256(_canonical(packet)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_hashed(path: Path, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_sha256", None)
    if payload.get("schema") != schema or claimed != hashlib.sha256(_canonical(payload)).hexdigest():
        raise RuntimeError(f"evidence receipt differs: {path}")
    payload["receipt_sha256"] = claimed
    return payload


def _validated_ready(capture: Mapping[str, Any], role: str) -> dict[str, Any]:
    ready = capture.get("ready", {}).get("content", {})
    if (
        not isinstance(ready, dict)
        or ready.get("schema") != "eve.b5-runtime-probe-ready.v1"
    ):
        raise RuntimeError(f"{role} startup ready evidence differs")
    unsigned = dict(ready)
    claimed = unsigned.pop("receipt_sha256", None)
    if claimed != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise RuntimeError(f"{role} startup ready receipt differs")
    store = capture.get("store", {})
    database = ready.get("database", {})
    if (
        not isinstance(database, dict)
        or database.get("sha256") != store.get("sha256")
        or database.get("verification") != store.get("verification")
    ):
        raise RuntimeError(f"{role} startup tail verification evidence differs")
    return ready


def prove_restart_continuity(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate gate-d from immutable before/after physical captures."""
    before_boot = before.get("boot", {}).get("last_boot_utc")
    after_boot = after.get("boot", {}).get("last_boot_utc")
    if not isinstance(before_boot, str) or not isinstance(after_boot, str):
        raise RuntimeError("reboot identity is absent")
    if before_boot == after_boot:
        raise RuntimeError("reboot identity did not change")

    before_service = before.get("service", {}).get("parsed", {})
    after_service = after.get("service", {}).get("parsed", {})
    if not isinstance(before_service, dict) or before_service.get("Status") != "Running":
        raise RuntimeError("service is not running before reboot")
    if not isinstance(after_service, dict) or after_service.get("Status") != "Running":
        raise RuntimeError("service is not running after reboot")
    if not (
        after_service.get("StartType") == "Automatic"
        or after_service.get("StartMode") == "Auto"
    ):
        raise RuntimeError("service is not Automatic after reboot")

    before_store = before.get("store", {})
    after_store = after.get("store", {})
    if (
        before_store.get("valid") is not True
        or after_store.get("valid") is not True
        or before_store.get("sha256") != after_store.get("sha256")
        or before_store.get("verification") != after_store.get("verification")
    ):
        raise RuntimeError("accepted event continuity across reboot differs")

    before_ready = _validated_ready(before, "pre-reboot")
    after_ready = _validated_ready(after, "post-reboot")
    return {
        "schema": RESTART_CONTINUITY_SCHEMA,
        "passed": True,
        "verdict": "ACCEPTED",
        "reason": "restart continuity proven",
        "before": {
            "label": before.get("label"),
            "capture_receipt_sha256": before.get("receipt_sha256"),
            "boot": before.get("boot"),
            "service": before_service,
            "store_sha256": before_store.get("sha256"),
            "store_verification": before_store.get("verification"),
            "ready_receipt_sha256": before_ready.get("receipt_sha256"),
        },
        "after": {
            "label": after.get("label"),
            "capture_receipt_sha256": after.get("receipt_sha256"),
            "boot": after.get("boot"),
            "service": after_service,
            "store_sha256": after_store.get("sha256"),
            "store_verification": after_store.get("verification"),
            "ready_receipt_sha256": after_ready.get("receipt_sha256"),
        },
    }


def load_restart_continuity_proof(before_path: Path, after_path: Path) -> dict[str, Any]:
    before_path = _outside_repository(before_path, "before-reboot capture")
    after_path = _outside_repository(after_path, "after-reboot capture")
    before = _load_hashed(before_path, CAPTURE_SCHEMA)
    after = _load_hashed(after_path, CAPTURE_SCHEMA)
    proof = prove_restart_continuity(before, after)
    proof["capture_files"] = {
        "before": {
            "path": str(before_path.resolve()),
            "sha256": _sha256(before_path),
        },
        "after": {
            "path": str(after_path.resolve()),
            "sha256": _sha256(after_path),
        },
    }
    return proof


def _run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "argv": command,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _boot_identity() -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("physical capture requires Windows")
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetTickCount64.argtypes = []
    kernel32.GetTickCount64.restype = ctypes.c_ulonglong
    uptime_ms = int(kernel32.GetTickCount64())
    boot_epoch = time.time() - uptime_ms / 1000.0
    return {
        "uptime_milliseconds": uptime_ms,
        "last_boot_utc": datetime.fromtimestamp(boot_epoch, timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }


def _service(service_name: str) -> dict[str, Any]:
    escaped = service_name.replace("'", "''")
    command = (
        f"$s=Get-Service -Name '{escaped}' -ErrorAction Stop;"
        "$w=Get-CimInstance Win32_Service -Filter \"Name='$($s.Name)'\";"
        "[ordered]@{Name=$s.Name;Status=[string]$s.Status;StartType=[string]$s.StartType;"
        "ProcessId=$w.ProcessId;PathName=$w.PathName;StartMode=$w.StartMode;StartName=$w.StartName}"
        "|ConvertTo-Json -Compress"
    )
    raw = _run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command]
    )
    parsed = None
    if raw["return_code"] == 0:
        try:
            parsed = json.loads(raw["stdout"])
        except json.JSONDecodeError:
            pass
    return {
        "parsed": parsed,
        "powershell": raw,
        "queryex": _run(["sc.exe", "queryex", service_name]),
        "qc": _run(["sc.exe", "qc", service_name]),
        "qfailure": _run(["sc.exe", "qfailure", service_name]),
    }


def _file_observation(path: Path, *, json_value: bool = False, lines: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.resolve()), "exists": False}
    data: Any = None
    error: str | None = None
    try:
        text = path.read_text(encoding="utf-8")
        data = [json.loads(line) for line in text.splitlines() if line] if lines else (
            json.loads(text) if json_value else text
        )
    except (OSError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(path.resolve()),
        "exists": True,
        "sha256": _sha256(path),
        "size": path.stat().st_size,
        "content": data,
        "read_error": error,
    }


def _store_observation(path: Path) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
    }
    if not path.is_file():
        return observation
    store = AuthoritativeStore(path)
    try:
        startup = store.open()
        verification = store.verify()
        observation.update(
            {
                "valid": True,
                "startup": asdict(startup),
                "verification": asdict(verification),
            }
        )
    except (AuthorityUnprovable, AuthorityPersistenceError, OSError, sqlite3.Error) as exc:
        observation.update(
            {"valid": False, "error_type": type(exc).__name__, "error": str(exc)}
        )
    finally:
        store.close()
    return observation


def inject_tail_mismatch(source: Path, target: Path, output: Path) -> dict[str, Any]:
    source = _outside_repository(source, "source database")
    target = _outside_repository(target, "injected database")
    output = _outside_repository(output, "injection receipt")
    if not source.is_file() or target.exists():
        raise RuntimeError("tail injection requires an existing source and absent target")
    source_before = _sha256(source)
    shutil.copy2(source, target)
    target_before = _sha256(target)
    with sqlite3.connect(target) as connection:
        count = int(connection.execute("SELECT COUNT(*) FROM accepted_tail").fetchone()[0])
        if count < 1:
            raise RuntimeError("tail injection source has no accepted tail")
        triggers = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='trigger' AND tbl_name='accepted_tail' ORDER BY name"
        ).fetchall()
        update_triggers = [(str(name), str(sql)) for name, sql in triggers if "UPDATE" in str(sql).upper()]
        if not update_triggers:
            raise RuntimeError("accepted-tail update protection trigger is absent")
        before_tail = str(
            connection.execute(
                "SELECT tail_hash FROM accepted_tail ORDER BY revision DESC LIMIT 1"
            ).fetchone()[0]
        )
        mismatch = "f" * 64 if before_tail != "f" * 64 else "e" * 64
        connection.execute("BEGIN IMMEDIATE")
        for name, _sql in update_triggers:
            connection.execute(f'DROP TRIGGER "{name.replace(chr(34), chr(34) * 2)}"')
        connection.execute(
            "UPDATE accepted_tail SET tail_hash=? WHERE revision=(SELECT MAX(revision) FROM accepted_tail)",
            (mismatch,),
        )
        for _name, sql in update_triggers:
            connection.execute(sql)
        connection.commit()
        after_tail = str(
            connection.execute(
                "SELECT tail_hash FROM accepted_tail ORDER BY revision DESC LIMIT 1"
            ).fetchone()[0]
        )
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    packet = {
        "schema": INJECTION_SCHEMA,
        "injected_at_utc": _utc_now(),
        "source": {
            "path": str(source),
            "sha256_before": source_before,
            "sha256_after": _sha256(source),
        },
        "target": {
            "path": str(target),
            "sha256_before_injection": target_before,
            "sha256_after_injection": _sha256(target),
            "accepted_tail_count": count,
            "tail_hash_before": before_tail,
            "tail_hash_after": after_tail,
            "integrity_check": integrity,
            "restored_update_triggers": [name for name, _sql in update_triggers],
        },
        "source_unchanged": source_before == _sha256(source),
        "target_is_copy": True,
    }
    _write(output, packet)
    packet["receipt_sha256"] = hashlib.sha256(_canonical(packet)).hexdigest()
    return packet


def capture(
    *,
    label: str,
    service_name: str,
    store: Path,
    sentinel: Path,
    audit_log: Path,
    alert_log: Path,
    state_file: Path,
    ready_file: Path,
    child_log: Path,
    output: Path,
) -> dict[str, Any]:
    output = _outside_repository(output, "physical capture")
    sentinel_payload, sentinel_state = (
        _sentinel_payload(sentinel) if sentinel.exists() else (None, "absent")
    )
    cpu = _cpu_identity()
    packet = {
        "schema": CAPTURE_SCHEMA,
        "label": label,
        "captured_at_utc": _utc_now(),
        "cpu": cpu["name"],
        "cpu_observation": cpu,
        "boot": _boot_identity(),
        "service": _service(service_name),
        "store": _store_observation(store),
        "sentinel": {
            **_file_observation(sentinel, json_value=True),
            "validation": sentinel_state,
            "valid_payload": sentinel_payload,
        },
        "state": _file_observation(state_file, json_value=True),
        "ready": _file_observation(ready_file, json_value=True),
        "audit_log": _file_observation(audit_log, lines=True),
        "alert_log": _file_observation(alert_log, lines=True),
        "child_log": _file_observation(child_log, lines=True),
        "cleared_sentinel_archives": [
            _file_observation(path, json_value=True)
            for path in sorted(sentinel.parent.glob(sentinel.name + ".cleared-*"))
        ],
    }
    _write(output, packet)
    packet["receipt_sha256"] = hashlib.sha256(_canonical(packet)).hexdigest()
    return packet


def _audit_events(capture_packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = capture_packet.get("audit_log", {}).get("content")
    return content if isinstance(content, list) else []


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("finalization requires the Windows Ryzen 7 8840U host")
    cpu = _cpu_identity()
    if "8840U" not in cpu["name"].upper():
        raise RuntimeError("finalization requires the Windows Ryzen 7 8840U host")
    if _git("rev-parse", "HEAD") != args.expected_head or _git("status", "--porcelain"):
        raise RuntimeError("finalization requires the clean exact expected head")
    captures = {
        name: _load_hashed(getattr(args, name.replace("-", "_")), CAPTURE_SCHEMA)
        for name in (
            "before_reboot",
            "after_reboot",
            "crash_recovered",
            "exit86_latched",
            "sentinel_restart_blocked",
            "operator_recovered",
        )
    }
    injection = _load_hashed(args.injection, INJECTION_SCHEMA)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    preflight_hash = preflight.pop("receipt_sha256", None)
    if (
        preflight.get("schema") != "eve.b5-windows-continuity-preflight.v2"
        or preflight_hash != hashlib.sha256(_canonical(preflight)).hexdigest()
        or preflight.get("passed") is not True
    ):
        raise RuntimeError("Windows preflight is not green")
    before = captures["before_reboot"]
    after = captures["after_reboot"]
    restart_continuity = load_restart_continuity_proof(
        args.before_reboot, args.after_reboot
    )
    update_check = preflight.get("checks", {}).get(
        "windows_update_automatic_reboot", {}
    )
    if (
        update_check.get("verdict") != "ACCEPTED"
        or update_check.get("restart_continuity") != restart_continuity
    ):
        raise RuntimeError("preflight is not bound to the gate-d captures")

    crash = captures["crash_recovered"]
    crash_events = _audit_events(crash)
    exit_indexes = [
        index
        for index, event in enumerate(crash_events)
        if event.get("event") == "child_exited" and event.get("child_exit_code") == 93
    ]
    if not exit_indexes or not any(
        event.get("event") == "restart_scheduled" and index > exit_indexes[-1]
        for index, event in enumerate(crash_events)
    ) or crash["service"].get("parsed", {}).get("Status") != "Running":
        raise RuntimeError("exit 93 restart proof differs")

    latched = captures["exit86_latched"]
    sentinel = latched.get("sentinel", {}).get("valid_payload")
    if (
        not isinstance(sentinel, dict)
        or sentinel.get("schema") != SENTINEL_SCHEMA
        or sentinel.get("child_exit_code") != 86
        or latched.get("state", {}).get("content", {}).get("status") != "STOPPED_LATCHED"
        or latched["service"].get("parsed", {}).get("Status") != "Stopped"
    ):
        raise RuntimeError("exit 86 latch proof differs")
    if latched["store"]["sha256"] != sentinel.get("store_sha256_after"):
        raise RuntimeError("exit 86 sentinel is not bound to the observed store")

    blocked = captures["sentinel_restart_blocked"]
    blocked_events = _audit_events(blocked)
    if (
        blocked.get("sentinel", {}).get("valid_payload", {}).get("sentinel_sha256")
        != sentinel.get("sentinel_sha256")
        or blocked["service"].get("parsed", {}).get("Status") != "Stopped"
        or not any(event.get("event") == "startup_blocked_by_sentinel" for event in blocked_events)
    ):
        raise RuntimeError("service restart did not remain sentinel-blocked")

    recovered = captures["operator_recovered"]
    recovered_events = _audit_events(recovered)
    if (
        recovered.get("sentinel", {}).get("exists") is not False
        or recovered["service"].get("parsed", {}).get("Status") != "Running"
        or not recovered.get("cleared_sentinel_archives")
        or not any(event.get("event") == "operator_sentinel_clear" for event in recovered_events)
        or recovered["store"].get("valid") is not True
    ):
        raise RuntimeError("operator sentinel recovery proof differs")

    t0_before = args.t0_sha256.lower()
    if _sha256(args.t0_database) != t0_before:
        raise RuntimeError("t=0 establishment database changed")
    source_sha = injection["source"]["sha256_before"]
    if _sha256(args.source_database) != source_sha:
        raise RuntimeError("source physical-gate database changed")

    packet = {
        "schema": GATE_SCHEMA,
        "passed": True,
        "captured_at_utc": _utc_now(),
        "authority_active_for_runtime": False,
        "t0_started": False,
        "repository": {
            "commit_sha": args.expected_head,
            "tree_sha": _git("rev-parse", "HEAD^{tree}"),
            "clean_checkout": True,
        },
        "environment": {
            "os_name": os.name,
            "platform": platform.platform(),
            "cpu": cpu["name"],
            "cpu_observation": cpu,
            "python": platform.python_version(),
            "numpy": preflight["runtime_environment"]["numpy"],
        },
        "preflight": {
            "file_sha256": _sha256(args.preflight),
            "receipt_sha256": preflight_hash,
            "checks": preflight["checks"],
        },
        "proofs": {
            "exit_86_tail_mismatch_latched": {
                "passed": True,
                "child_exit_code": 86,
                "sentinel": sentinel,
                "injection": injection,
                "capture": latched,
            },
            "service_restart_remains_stopped": {"passed": True, "capture": blocked},
            "exit_93_restarted": {"passed": True, "capture": crash},
            "physical_reboot_continuity": {
                "passed": True,
                "proof": restart_continuity,
                "before": before,
                "after": after,
            },
            "operator_sentinel_clear_recovered": {"passed": True, "capture": recovered},
        },
        "protected_inputs": {
            "t0_establishment_database": {
                "path": str(args.t0_database.resolve()),
                "sha256_before": t0_before,
                "sha256_after": _sha256(args.t0_database),
                "unchanged": True,
            },
            "source_physical_database": {
                "path": str(args.source_database.resolve()),
                "sha256_before": source_sha,
                "sha256_after": _sha256(args.source_database),
                "unchanged": True,
            },
        },
    }
    _write(args.output, packet)
    packet["receipt_sha256"] = hashlib.sha256(_canonical(packet)).hexdigest()
    return packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    injection = subparsers.add_parser("inject-tail-mismatch")
    injection.add_argument("--source", type=Path, required=True)
    injection.add_argument("--target", type=Path, required=True)
    injection.add_argument("--output", type=Path, required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--label", required=True)
    capture_parser.add_argument("--service-name", required=True)
    for name in (
        "store",
        "sentinel",
        "audit-log",
        "alert-log",
        "state-file",
        "ready-file",
        "child-log",
        "output",
    ):
        capture_parser.add_argument(f"--{name}", type=Path, required=True)
    final = subparsers.add_parser("finalize")
    final.add_argument("--expected-head", required=True)
    final.add_argument("--preflight", type=Path, required=True)
    final.add_argument("--injection", type=Path, required=True)
    for name in (
        "before-reboot",
        "after-reboot",
        "crash-recovered",
        "exit86-latched",
        "sentinel-restart-blocked",
        "operator-recovered",
    ):
        final.add_argument(f"--{name}", type=Path, required=True)
    final.add_argument("--t0-database", type=Path, required=True)
    final.add_argument("--t0-sha256", required=True)
    final.add_argument("--source-database", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "inject-tail-mismatch":
            packet = inject_tail_mismatch(args.source, args.target, args.output)
        elif args.action == "capture":
            packet = capture(
                label=args.label,
                service_name=args.service_name,
                store=args.store,
                sentinel=args.sentinel,
                audit_log=args.audit_log,
                alert_log=args.alert_log,
                state_file=args.state_file,
                ready_file=args.ready_file,
                child_log=args.child_log,
                output=args.output,
            )
        else:
            packet = finalize(args)
        print(
            json.dumps(
                {
                    "schema": packet["schema"],
                    "receipt_sha256": packet["receipt_sha256"],
                    "passed": packet.get("passed"),
                },
                sort_keys=True,
            )
        )
        return 0
    except (RuntimeError, OSError, ValueError, sqlite3.Error) as exc:
        print(f"B5 physical proof unprovable: {exc}", file=sys.stderr)
        return 86


if __name__ == "__main__":
    raise SystemExit(main())
