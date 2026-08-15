"""Fail-closed Windows continuity preflight for B5 supervision."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.operator.b5_runtime_environment import (  # noqa: E402
    RuntimeEnvironmentError,
    load_and_verify_receipt,
)
from scripts.operator.b5_windows_physical_gate import (  # noqa: E402
    RESTART_CONTINUITY_SCHEMA,
    load_restart_continuity_proof,
)

SCHEMA = "eve.b5-windows-continuity-preflight.v2"


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


def _powershell_json(script: str) -> tuple[Any | None, dict[str, Any]]:
    raw = _run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ]
    )
    if raw["return_code"] != 0:
        return None, raw
    try:
        return json.loads(raw["stdout"]), raw
    except json.JSONDecodeError:
        return None, raw


def _power_indexes(output: str) -> dict[str, list[int]]:
    ac: list[int] = []
    dc: list[int] = []
    unlabeled: list[int] = []
    for line in output.splitlines():
        match = re.search(r"0x([0-9a-fA-F]+)\s*$", line)
        if not match:
            continue
        lowered = line.lower()
        if " ac " in f" {lowered} " or "현재 ac" in lowered:
            ac.append(int(match.group(1), 16))
        elif " dc " in f" {lowered} " or "현재 dc" in lowered:
            dc.append(int(match.group(1), 16))
        else:
            unlabeled.append(int(match.group(1), 16))
    if not ac and not dc and len(unlabeled) == 2:
        return {"ac": [unlabeled[0]], "dc": [unlabeled[1]]}
    return {"ac": ac, "dc": dc}


def _defender_exclusion_state(authority_dir: Path, exclusions: list[str]) -> dict[str, Any]:
    target = authority_dir.resolve()
    exact_matches: list[str] = []
    broader_parent_matches: list[str] = []
    for value in exclusions:
        try:
            candidate = Path(os.path.expandvars(value)).expanduser().resolve()
            if os.path.normcase(str(candidate)) == os.path.normcase(str(target)):
                exact_matches.append(value)
                continue
            target.relative_to(candidate)
            broader_parent_matches.append(value)
        except (OSError, ValueError):
            continue
    return {
        "target": str(target),
        "exact_match": bool(exact_matches),
        "exact_matches": exact_matches,
        "broader_parent_matches": broader_parent_matches,
    }


def _defender_access_probe(authority_dir: Path) -> dict[str, Any]:
    authority_dir.mkdir(parents=True, exist_ok=True)
    probe = authority_dir / f".eve-b5-defender-probe-{os.getpid()}"
    moved = probe.with_suffix(".moved")
    payload = b"eve-b5-defender-access-probe\n"
    try:
        with probe.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        before = _sha256(probe)
        os.replace(probe, moved)
        after = _sha256(moved)
        moved.unlink()
        return {
            "passed": before == after,
            "sha256_before_rename": before,
            "sha256_after_rename": after,
            "probe_removed": not probe.exists() and not moved.exists(),
        }
    except OSError as exc:
        for path in (probe, moved):
            try:
                path.unlink()
            except OSError:
                pass
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def _restart_continuity_evidence(
    before_reboot_capture: Path | None,
    after_reboot_capture: Path | None,
) -> dict[str, Any]:
    if before_reboot_capture is None or after_reboot_capture is None:
        return {
            "schema": RESTART_CONTINUITY_SCHEMA,
            "passed": False,
            "verdict": "UNRESOLVED",
            "reason": "gate-d before/after captures were not both supplied",
            "before_capture": (
                str(before_reboot_capture.resolve())
                if before_reboot_capture is not None
                else None
            ),
            "after_capture": (
                str(after_reboot_capture.resolve())
                if after_reboot_capture is not None
                else None
            ),
        }
    try:
        return load_restart_continuity_proof(
            before_reboot_capture.resolve(), after_reboot_capture.resolve()
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "schema": RESTART_CONTINUITY_SCHEMA,
            "passed": False,
            "verdict": "UNRESOLVED",
            "reason": f"gate-d evidence differs: {type(exc).__name__}: {exc}",
            "before_capture": str(before_reboot_capture.resolve()),
            "after_capture": str(after_reboot_capture.resolve()),
        }


def classify_windows_update(
    current_setting: Any, restart_continuity: Mapping[str, Any]
) -> dict[str, Any]:
    pending_state_clear = bool(
        isinstance(current_setting, dict)
        and current_setting.get("CBSRebootPending") is False
        and current_setting.get("WURebootRequired") is False
        and current_setting.get("PendingFileRename") is False
    )
    accepted = bool(
        restart_continuity.get("schema") == RESTART_CONTINUITY_SCHEMA
        and restart_continuity.get("passed") is True
        and restart_continuity.get("verdict") == "ACCEPTED"
        and pending_state_clear
    )
    return {
        "verdict": "ACCEPTED" if accepted else "UNRESOLVED",
        "reason": (
            "restart continuity proven"
            if accepted
            else "gate-d is incomplete or a pending reboot indicator remains"
        ),
        "pending_state_clear": pending_state_clear,
        "registry_policy_required": False,
    }


def collect(
    authority_dir: Path,
    runtime_receipt_path: Path,
    change_record: Path,
    before_reboot_capture: Path | None = None,
    after_reboot_capture: Path | None = None,
) -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("B5 Windows preflight requires os.name == 'nt'")
    receipt = load_and_verify_receipt(runtime_receipt_path)
    change = json.loads(change_record.read_text(encoding="utf-8"))
    if change.get("schema") != "eve.b5-windows-host-policy-record.v1":
        raise RuntimeError("host policy change record schema differs")

    update, update_raw = _powershell_json(
        "$p='HKLM:\\SOFTWARE\\Policies\\Microsoft\\Windows\\WindowsUpdate\\AU';"
        "$s='HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager';"
        "$v=Get-ItemProperty -LiteralPath $p -ErrorAction SilentlyContinue;"
        "[ordered]@{NoAutoRebootWithLoggedOnUsers=$v.NoAutoRebootWithLoggedOnUsers;"
        "AlwaysAutoRebootAtScheduledTime=$v.AlwaysAutoRebootAtScheduledTime;"
        "AUOptions=$v.AUOptions;"
        "CBSRebootPending=(Test-Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Component Based Servicing\\RebootPending');"
        "WURebootRequired=(Test-Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\WindowsUpdate\\Auto Update\\RebootRequired');"
        "PendingFileRename=($null -ne (Get-ItemProperty -LiteralPath $s -Name PendingFileRenameOperations -ErrorAction SilentlyContinue).PendingFileRenameOperations)}"
        "|ConvertTo-Json -Compress"
    )
    restart_continuity = _restart_continuity_evidence(
        before_reboot_capture, after_reboot_capture
    )
    update_classification = classify_windows_update(update, restart_continuity)

    sleep_raw = {
        "standby_idle": _run(
            ["powercfg.exe", "/qh", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"]
        ),
        "hibernate_idle": _run(
            ["powercfg.exe", "/qh", "SCHEME_CURRENT", "SUB_SLEEP", "HIBERNATEIDLE"]
        ),
    }
    disk_raw = _run(
        ["powercfg.exe", "/qh", "SCHEME_CURRENT", "SUB_DISK", "DISKIDLE"]
    )
    lid_raw = _run(
        ["powercfg.exe", "/qh", "SCHEME_CURRENT", "SUB_BUTTONS", "LIDACTION"]
    )
    active_raw = _run(["powercfg.exe", "/getactivescheme"])
    sleep_indexes = {
        name: _power_indexes(raw["stdout"]) for name, raw in sleep_raw.items()
    }
    disk_indexes = _power_indexes(disk_raw["stdout"])
    lid_indexes = _power_indexes(lid_raw["stdout"])
    sleep_passed = all(raw["return_code"] == 0 for raw in sleep_raw.values()) and all(
        indexes["ac"] and indexes["ac"][0] == 0
        for indexes in sleep_indexes.values()
    )
    disk_passed = bool(
        disk_raw["return_code"] == 0
        and disk_indexes["ac"]
        and disk_indexes["ac"][0] == 0
    )
    lid_verdict = "PASS" if lid_indexes["ac"] and lid_indexes["ac"][0] == 0 else "UNRESOLVED"
    chassis, chassis_raw = _powershell_json(
        "@(Get-CimInstance Win32_SystemEnclosure).ChassisTypes|ConvertTo-Json -Compress"
    )
    chassis_values = chassis if isinstance(chassis, list) else [chassis]
    if not lid_indexes["ac"] and any(value in {30, 31, 32} for value in chassis_values):
        lid_verdict = "NOT_APPLICABLE" if 30 in chassis_values else "UNRESOLVED"

    fast, fast_raw = _powershell_json(
        "$v=(Get-ItemProperty -LiteralPath 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Power' -Name HiberbootEnabled -ErrorAction SilentlyContinue).HiberbootEnabled;"
        "@{HiberbootEnabled=$v}|ConvertTo-Json -Compress"
    )
    fast_passed = isinstance(fast, dict) and fast.get("HiberbootEnabled") == 0

    defender, defender_raw = _powershell_json(
        "$a=$true; try {$p=Get-MpPreference -ErrorAction Stop} catch {$a=$false;$p=$null};"
        "$s=Get-MpComputerStatus;"
        "[ordered]@{ExclusionQueryAvailable=$a;AntivirusEnabled=$s.AntivirusEnabled;"
        "RealTimeProtectionEnabled=$s.RealTimeProtectionEnabled;"
        "BehaviorMonitorEnabled=$s.BehaviorMonitorEnabled;"
        "DisableRealtimeMonitoring=$p.DisableRealtimeMonitoring;"
        "ExclusionPath=@($p.ExclusionPath)}|ConvertTo-Json -Compress -Depth 4"
    )
    access_probe = _defender_access_probe(authority_dir)
    exclusions: list[str] = []
    if isinstance(defender, dict) and isinstance(defender.get("ExclusionPath"), list):
        exclusions = [str(value) for value in defender["ExclusionPath"]]
    exclusion_state = _defender_exclusion_state(authority_dir, exclusions)
    defender_passed = bool(
        isinstance(defender, dict)
        and defender.get("ExclusionQueryAvailable") is True
        and defender.get("AntivirusEnabled") is True
        and defender.get("RealTimeProtectionEnabled") is True
        and defender.get("DisableRealtimeMonitoring") is False
        and exclusion_state["exact_match"] is True
        and not exclusion_state["broader_parent_matches"]
        and access_probe.get("passed") is True
    )

    service_raw = _run(["sc.exe", "qc", "EveB5Supervisor"])
    packet: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "passed": False,
        "authority_active_for_runtime": False,
        "t0_started": False,
        "repository": {
            "commit_sha": _git("rev-parse", "HEAD"),
            "tree_sha": _git("rev-parse", "HEAD^{tree}"),
            "clean_checkout": not bool(_git("status", "--porcelain")),
        },
        "runtime_environment": {
            "receipt_path": str(runtime_receipt_path.resolve()),
            "receipt_file_sha256": _sha256(runtime_receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
            "dependency_source": "requirements-lock.txt",
            "require_hashes": True,
            "requirements_runtime_used": False,
            "python": receipt["python"]["installed_version"],
            "numpy": receipt["numpy_version"],
        },
        "host_policy_change_record": {
            "path": str(change_record.resolve()),
            "sha256": _sha256(change_record),
            "action": change.get("action"),
            "changes": change.get("changes"),
        },
        "checks": {
            "windows_update_automatic_reboot": {
                "verdict": update_classification["verdict"],
                "reason": update_classification["reason"],
                "current_setting": update,
                "restart_continuity": restart_continuity,
                "pending_state_clear": update_classification[
                    "pending_state_clear"
                ],
                "registry_policy_required": update_classification[
                    "registry_policy_required"
                ],
                "required_policy": (
                    "gate-d proves Automatic-service restart continuity across a real "
                    "Windows reboot; all pending-reboot indicators are clear"
                ),
                "raw": update_raw,
            },
            "sleep_hibernate": {
                "verdict": "PASS" if sleep_passed else "UNRESOLVED",
                "current_setting": sleep_indexes,
                "required_policy": "plugged-in sleep and hibernate idle timeouts are disabled",
                "raw": sleep_raw,
            },
            "disk_idle": {
                "verdict": "PASS" if disk_passed else "UNRESOLVED",
                "current_setting": disk_indexes,
                "required_policy": "plugged-in disk idle timeout is disabled (AC index 0)",
                "raw": disk_raw,
            },
            "lid_close": {
                "verdict": lid_verdict,
                "current_setting": {"indexes": lid_indexes, "chassis_types": chassis_values},
                "required_policy": "lid close does nothing, or the device has no lid",
                "raw": {"lid": lid_raw, "chassis": chassis_raw},
            },
            "fast_startup": {
                "verdict": "PASS" if fast_passed else "UNRESOLVED",
                "current_setting": fast,
                "required_policy": "HiberbootEnabled=0 so startup executes a full service path",
                "raw": fast_raw,
            },
            "defender_authority_directory": {
                "verdict": "PASS" if defender_passed else "UNRESOLVED",
                "current_setting": defender,
                "authority_directory_exclusion": exclusion_state,
                "access_probe": access_probe,
                "required_policy": (
                    "Defender real-time protection remains enabled, the exact authority "
                    "store directory (not a broader parent) is excluded, and file access succeeds"
                ),
                "raw": defender_raw,
            },
            "plugged_in_power_plan": {
                "verdict": "PASS"
                if (
                    active_raw["return_code"] == 0
                    and sleep_passed
                    and disk_passed
                    and lid_verdict in {"PASS", "NOT_APPLICABLE"}
                )
                else "UNRESOLVED",
                "current_setting": active_raw["stdout"],
                "required_policy": (
                    "an identified active plan with plugged-in sleep, hibernate, and disk "
                    "idle disabled and lid close set to do nothing when applicable"
                ),
                "raw": active_raw,
            },
            "service_configuration": {
                "verdict": "PASS" if service_raw["return_code"] == 0 else "UNRESOLVED",
                "current_setting": service_raw["stdout"],
                "required_policy": "EveB5Supervisor is installed as an automatic Windows service",
                "raw": service_raw,
            },
        },
    }
    verdicts = [value["verdict"] for value in packet["checks"].values()]
    packet["unresolved"] = [
        name for name, value in packet["checks"].items() if value["verdict"] == "UNRESOLVED"
    ]
    packet["passed"] = all(
        value in {"PASS", "ACCEPTED", "NOT_APPLICABLE"} for value in verdicts
    )
    packet["receipt_sha256"] = hashlib.sha256(_canonical(packet)).hexdigest()
    return packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-directory", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--host-policy-record", type=Path, required=True)
    parser.add_argument("--before-reboot-capture", type=Path)
    parser.add_argument("--after-reboot-capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise RuntimeError("refusing to overwrite preflight receipt")
        packet = collect(
            args.authority_directory.resolve(),
            args.runtime_receipt.resolve(),
            args.host_policy_record.resolve(),
            args.before_reboot_capture.resolve()
            if args.before_reboot_capture is not None
            else None,
            args.after_reboot_capture.resolve()
            if args.after_reboot_capture is not None
            else None,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"passed": packet["passed"], "unresolved": packet["unresolved"]}, sort_keys=True))
        return 0 if packet["passed"] else 86
    except (RuntimeError, RuntimeEnvironmentError, OSError, ValueError) as exc:
        print(f"Windows preflight unprovable: {exc}", file=sys.stderr)
        return 86


if __name__ == "__main__":
    raise SystemExit(main())
