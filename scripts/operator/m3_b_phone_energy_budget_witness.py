#!/usr/bin/env python3
"""Run three real full-engine phone interactions and emit an energy-budget review witness.

Raw CPU/memory/battery/process counters and the private nonce stay outside the
repository.  The final stdout object is the only material intended for review.
This command does not register trust/verifiers, retain an observation, start the
M3-B observation window, or authorize cutover.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m3_b_phone_energy_budget_witness import (
    DEFAULT_SOURCE_INSTANCE_ID,
    ENTRYPOINT_ID,
    PhoneEnergyBudgetRuntimeSnapshot,
    build_phone_energy_budget_witness,
)
from main import build_full_engine

PRIVATE_FILENAME = "energy_budget_witness_private.json"
PUBLIC_FILENAME = "energy_budget_witness_public_review.json"
DEFAULT_BATTERY_CAPACITY_FILE = "/sys/class/power_supply/battery/capacity"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if actual != expected_head:
        raise SystemExit(f"repository head mismatch: expected {expected_head}, got {actual}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
    )
    if dirty.strip():
        raise SystemExit("operator witness requires a clean exact repository checkout")
    return actual


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(_canonical(value) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _cpu_counters(path: Path = Path("/proc/stat")) -> tuple[int, int]:
    try:
        line = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError) as exc:
        raise SystemExit(f"cannot read CPU counters from {path}") from exc
    fields = line.split()
    if not fields or fields[0] != "cpu" or len(fields) < 5:
        raise SystemExit("/proc/stat aggregate CPU row is unavailable")
    try:
        values = [int(value) for value in fields[1:]]
    except ValueError as exc:
        raise SystemExit("/proc/stat aggregate CPU row is malformed") from exc
    total = sum(values)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    if total <= 0 or idle < 0 or idle > total:
        raise SystemExit("/proc/stat aggregate CPU counters are invalid")
    return total, idle


def _process_cpu_seconds() -> float:
    usage = os.times()
    result = float(usage.user + usage.system)
    if result < 0.0:
        raise SystemExit("process CPU time is invalid")
    return result


def _memory_budget(path: Path = Path("/proc/meminfo")) -> tuple[int, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"cannot read memory counters from {path}") from exc
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
        raise SystemExit("/proc/meminfo lacks valid MemTotal/MemAvailable values")
    return total, available


def _battery_capacity(path: Path) -> int:
    try:
        raw = path.expanduser().read_text(encoding="utf-8").strip()
        capacity = int(raw)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read integer battery capacity from {path}") from exc
    if not 0 <= capacity <= 100:
        raise SystemExit("battery capacity must be inside [0,100]")
    return capacity


def _run_interaction(engine: Any, text: str) -> None:
    for chunk in engine.chat_stream(text):
        print(chunk, end="", file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)


def _capture_window(
    engine: Any,
    text: str,
    *,
    logical_tick: int,
    source_instance_id: str,
    battery_capacity_file: Path,
) -> PhoneEnergyBudgetRuntimeSnapshot:
    cpu_total_before, cpu_idle_before = _cpu_counters()
    process_before = _process_cpu_seconds()
    wall_before = time.monotonic()

    _run_interaction(engine, text)

    wall_after = time.monotonic()
    process_after = _process_cpu_seconds()
    cpu_total_after, cpu_idle_after = _cpu_counters()
    mem_total, mem_available = _memory_budget()
    battery_capacity = _battery_capacity(battery_capacity_file)

    cpu_total_delta = cpu_total_after - cpu_total_before
    cpu_idle_delta = cpu_idle_after - cpu_idle_before
    process_cpu_delta = process_after - process_before
    wall_delta = wall_after - wall_before
    if cpu_total_delta <= 0 or cpu_idle_delta < 0:
        raise SystemExit("CPU counters did not advance monotonically during interaction window")
    if process_cpu_delta < 0.0 or wall_delta <= 0.0:
        raise SystemExit("process/wall counters did not advance monotonically")

    return PhoneEnergyBudgetRuntimeSnapshot(
        source_instance_id=source_instance_id,
        logical_tick=logical_tick,
        cpu_total_delta=cpu_total_delta,
        cpu_idle_delta=cpu_idle_delta,
        process_cpu_seconds=process_cpu_delta,
        wall_seconds=wall_delta,
        cpu_count=max(1, os.cpu_count() or 1),
        mem_total_kib=mem_total,
        mem_available_kib=mem_available,
        battery_capacity_percent=battery_capacity,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonce-file", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--runtime-instance-id", required=True)
    parser.add_argument("--launch-attestation-id", required=True)
    parser.add_argument("--source-instance-id", default=DEFAULT_SOURCE_INSTANCE_ID)
    parser.add_argument(
        "--battery-capacity-file",
        default=DEFAULT_BATTERY_CAPACITY_FILE,
        help="Readable integer 0..100 battery-capacity file. Defaults to Android/Linux power_supply path.",
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        required=True,
        help="Real operator interaction text. Supply exactly three times.",
    )
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
    snapshots = tuple(
        _capture_window(
            engine,
            text,
            logical_tick=index,
            source_instance_id=args.source_instance_id,
            battery_capacity_file=battery_capacity_file,
        )
        for index, text in enumerate(args.inputs)
    )

    witness = build_phone_energy_budget_witness(
        private_nonce=private_nonce,
        runtime_instance_id=args.runtime_instance_id,
        source_instance_id=args.source_instance_id,
        repository_head_sha=repository_head,
        launch_attestation_id=args.launch_attestation_id,
        snapshots=snapshots,
        launch_logical_tick=0,
        entrypoint_id=ENTRYPOINT_ID,
    )
    private_mapping = witness.private_mapping()
    public_mapping = witness.public_review_mapping()
    _atomic_json(private_root / PRIVATE_FILENAME, private_mapping)
    _atomic_json(private_root / PUBLIC_FILENAME, public_mapping)

    # Only this final canonical object is intended to leave the private companion.
    print(_canonical(public_mapping))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
