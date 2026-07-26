#!/usr/bin/env python3
"""Run three real full-engine phone interactions and emit an energy-budget review witness.

Raw CPU/memory/battery/process observations and the private nonce stay outside the
repository. The v2 collector falls back from Android-restricted proc/sysfs surfaces
only to explicit kernel/API observations and records the exact method used.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m3_b_phone_energy_budget_witness_v2 import (
    BATTERY_METHOD_SYSFS,
    BATTERY_METHOD_TERMUX_API,
    CPU_METHOD_LOADAVG,
    CPU_METHOD_PROC_STAT,
    DEFAULT_SOURCE_INSTANCE_ID,
    ENTRYPOINT_ID,
    MEMORY_METHOD_PROC_MEMINFO,
    MEMORY_METHOD_SYSCONF,
    PhoneEnergyBudgetRuntimeSnapshot,
    build_phone_energy_budget_witness,
)
from main import build_full_engine

PRIVATE_FILENAME = "energy_budget_witness_private_v2.json"
PUBLIC_FILENAME = "energy_budget_witness_public_review_v2.json"
DEFAULT_BATTERY_CAPACITY_FILE = "/sys/class/power_supply/battery/capacity"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _outside_repository(path: Path, field: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise SystemExit(f"{field} must remain outside the repository")


def _private_nonce(path: str) -> bytes:
    nonce_path = _outside_repository(Path(path), "private nonce file")
    stat_result = nonce_path.stat()
    if os.name != "nt" and stat_result.st_mode & 0o077:
        raise SystemExit("private nonce file must not grant group/other permissions")
    nonce = nonce_path.read_bytes()
    if len(nonce) < 32:
        raise SystemExit("private nonce file must contain at least 32 bytes")
    return nonce


def _prepare_private_root(path: str) -> Path:
    root = _outside_repository(Path(path), "private witness root")
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    return root


def _repository_head(expected_head: str) -> str:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual != expected_head:
        raise SystemExit(f"repository head mismatch: expected {expected_head}, got {actual}")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise SystemExit("operator witness requires a clean exact repository checkout")
    return actual


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(_canonical(value) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _proc_cpu_counters(path: Path = Path("/proc/stat")) -> tuple[int, int]:
    line = path.read_text(encoding="utf-8").splitlines()[0]
    fields = line.split()
    if not fields or fields[0] != "cpu" or len(fields) < 5:
        raise ValueError("aggregate CPU row is unavailable")
    values = [int(value) for value in fields[1:]]
    total = sum(values)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    if total <= 0 or idle < 0 or idle > total:
        raise ValueError("aggregate CPU counters are invalid")
    return total, idle


def _load_average_1m() -> float:
    try:
        value = float(os.getloadavg()[0])
    except (AttributeError, OSError) as exc:
        raise OSError("kernel load average is unavailable") from exc
    if value < 0.0:
        raise OSError("kernel load average is invalid")
    return value


def _cpu_probe_start() -> tuple[str, int | None, int | None, float | None]:
    try:
        total, idle = _proc_cpu_counters()
        return CPU_METHOD_PROC_STAT, total, idle, None
    except (OSError, ValueError, IndexError):
        try:
            return CPU_METHOD_LOADAVG, None, None, _load_average_1m()
        except OSError as exc:
            raise SystemExit("cannot observe system CPU headroom: /proc/stat is blocked and kernel load average is unavailable") from exc


def _cpu_probe_finish(method: str) -> tuple[int | None, int | None, float | None]:
    if method == CPU_METHOD_PROC_STAT:
        try:
            total, idle = _proc_cpu_counters()
        except (OSError, ValueError, IndexError) as exc:
            raise SystemExit("/proc/stat became unavailable during the interaction window") from exc
        return total, idle, None
    if method == CPU_METHOD_LOADAVG:
        try:
            return None, None, _load_average_1m()
        except OSError as exc:
            raise SystemExit("kernel load average became unavailable during the interaction window") from exc
    raise SystemExit("unsupported CPU measurement method")


def _process_cpu_seconds() -> float:
    usage = os.times()
    result = float(usage.user + usage.system)
    if result < 0.0:
        raise SystemExit("process CPU time is invalid")
    return result


def _proc_memory_budget(path: Path = Path("/proc/meminfo")) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    values: dict[str, int] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0])
        except ValueError:
            continue
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None or total <= 0 or not 0 <= available <= total:
        raise ValueError("MemTotal/MemAvailable are unavailable")
    return total, available


def _sysconf_memory_budget() -> tuple[int, int]:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (OSError, ValueError, KeyError) as exc:
        raise OSError("sysconf memory counters are unavailable") from exc
    if page_size <= 0 or total_pages <= 0 or available_pages < 0 or available_pages > total_pages:
        raise OSError("sysconf memory counters are invalid")
    return (total_pages * page_size) // 1024, (available_pages * page_size) // 1024


def _memory_budget() -> tuple[str, int, int]:
    try:
        total, available = _proc_memory_budget()
        return MEMORY_METHOD_PROC_MEMINFO, total, available
    except (OSError, ValueError):
        try:
            total, available = _sysconf_memory_budget()
            return MEMORY_METHOD_SYSCONF, total, available
        except OSError as exc:
            raise SystemExit("cannot observe memory headroom from /proc/meminfo or sysconf") from exc


def _termux_api_battery_capacity() -> int:
    executable = shutil.which("termux-battery-status")
    if executable is None:
        raise OSError("termux-battery-status is not installed")
    try:
        result = subprocess.run([executable], check=True, capture_output=True, text=True, timeout=10)
        payload = json.loads(result.stdout)
        capacity = int(payload["percentage"])
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise OSError("termux-battery-status did not return a valid percentage") from exc
    if not 0 <= capacity <= 100:
        raise OSError("termux-battery-status percentage is outside [0,100]")
    return capacity


def _battery_capacity(path: Path) -> tuple[str, int]:
    try:
        raw = path.expanduser().read_text(encoding="utf-8").strip()
        capacity = int(raw)
        if not 0 <= capacity <= 100:
            raise ValueError("capacity outside [0,100]")
        return BATTERY_METHOD_SYSFS, capacity
    except (OSError, ValueError):
        try:
            return BATTERY_METHOD_TERMUX_API, _termux_api_battery_capacity()
        except OSError as exc:
            raise SystemExit("cannot observe battery capacity: sysfs is blocked and termux-battery-status is unavailable; install Termux:API or provide a readable --battery-capacity-file") from exc


def _run_interaction(engine: Any, text: str) -> None:
    for chunk in engine.chat_stream(text):
        print(chunk, end="", file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)


def _capture_window(engine: Any, text: str, *, logical_tick: int, source_instance_id: str, battery_capacity_file: Path) -> PhoneEnergyBudgetRuntimeSnapshot:
    cpu_method, cpu_total_before, cpu_idle_before, load_before = _cpu_probe_start()
    process_before = _process_cpu_seconds()
    wall_before = time.monotonic()
    _run_interaction(engine, text)
    wall_after = time.monotonic()
    process_after = _process_cpu_seconds()
    cpu_total_after, cpu_idle_after, load_after = _cpu_probe_finish(cpu_method)
    memory_method, mem_total, mem_available = _memory_budget()
    battery_method, battery_capacity = _battery_capacity(battery_capacity_file)
    process_cpu_delta = process_after - process_before
    wall_delta = wall_after - wall_before
    if process_cpu_delta < 0.0 or wall_delta <= 0.0:
        raise SystemExit("process/wall counters did not advance monotonically")
    cpu_total_delta: int | None = None
    cpu_idle_delta: int | None = None
    if cpu_method == CPU_METHOD_PROC_STAT:
        assert cpu_total_before is not None and cpu_idle_before is not None
        assert cpu_total_after is not None and cpu_idle_after is not None
        cpu_total_delta = cpu_total_after - cpu_total_before
        cpu_idle_delta = cpu_idle_after - cpu_idle_before
        if cpu_total_delta <= 0 or cpu_idle_delta < 0:
            raise SystemExit("CPU counters did not advance monotonically during interaction window")
    return PhoneEnergyBudgetRuntimeSnapshot(
        source_instance_id=source_instance_id,
        logical_tick=logical_tick,
        cpu_measurement_method=cpu_method,
        cpu_total_delta=cpu_total_delta,
        cpu_idle_delta=cpu_idle_delta,
        load_average_1m_before=load_before,
        load_average_1m_after=load_after,
        process_cpu_seconds=process_cpu_delta,
        wall_seconds=wall_delta,
        cpu_count=max(1, os.cpu_count() or 1),
        mem_total_kib=mem_total,
        mem_available_kib=mem_available,
        memory_measurement_method=memory_method,
        battery_capacity_percent=battery_capacity,
        battery_measurement_method=battery_method,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonce-file", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--runtime-instance-id", required=True)
    parser.add_argument("--launch-attestation-id", required=True)
    parser.add_argument("--source-instance-id", default=DEFAULT_SOURCE_INSTANCE_ID)
    parser.add_argument("--battery-capacity-file", default=DEFAULT_BATTERY_CAPACITY_FILE, help="Readable integer 0..100 battery-capacity file. If blocked, Termux:API is tried automatically.")
    parser.add_argument("--input", action="append", dest="inputs", required=True, help="Real operator interaction text. Supply exactly three times.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.inputs) != 3:
        raise SystemExit("phone energy-budget witness requires exactly three --input values")
    for text in args.inputs:
        if not isinstance(text, str) or not text.strip():
            raise SystemExit("witness input must be non-empty text")
    repository_head = _repository_head(args.expected_head)
    private_root = _prepare_private_root(args.private_root)
    private_nonce = _private_nonce(args.nonce_file)
    battery_capacity_file = Path(args.battery_capacity_file)
    engine = build_full_engine()
    snapshots = tuple(_capture_window(engine, text, logical_tick=index, source_instance_id=args.source_instance_id, battery_capacity_file=battery_capacity_file) for index, text in enumerate(args.inputs))
    witness = build_phone_energy_budget_witness(private_nonce=private_nonce, runtime_instance_id=args.runtime_instance_id, source_instance_id=args.source_instance_id, repository_head_sha=repository_head, launch_attestation_id=args.launch_attestation_id, snapshots=snapshots, launch_logical_tick=0, entrypoint_id=ENTRYPOINT_ID)
    private_mapping = witness.private_mapping()
    public_mapping = witness.public_review_mapping()
    _atomic_json(private_root / PRIVATE_FILENAME, private_mapping)
    _atomic_json(private_root / PUBLIC_FILENAME, public_mapping)
    print(_canonical(public_mapping))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
