"""Generate a Ryzen 7 8840U B2 authoritative-persistence receipt.

This is a physical workstation gate, not a GitHub CI workflow.  It must run
from a clean, exact merged checkout with the repository's pinned Python and
hash-pinned validation environment.  The database and receipt must remain
outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.authoritative_store import AuthoritativeStore  # noqa: E402
from core.event_kernel import EventEnvelope  # noqa: E402

SCHEMA = "eve.b2-authority-8840u-physical-gate.v1"
WORKER = ROOT / "scripts" / "audit" / "b2_authority_fault_worker.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _database_artifacts(path: Path) -> dict[str, dict[str, int | str]]:
    connection = sqlite3.connect(path)
    try:
        mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if mode == "wal":
            result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result is None or int(result[0]) != 0:
                raise RuntimeError("final WAL checkpoint did not complete")
    finally:
        connection.close()
    artifacts: dict[str, dict[str, int | str]] = {}
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            artifacts[candidate.name] = {
                "bytes": candidate.stat().st_size,
                "sha256": _sha256(candidate),
            }
    return artifacts


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor().strip() or platform.machine().strip()


def _rss_bytes() -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _percentile(sorted_values: list[int], percentile: float) -> int:
    if not sorted_values:
        raise ValueError("latency sample is empty")
    index = round((len(sorted_values) - 1) * percentile)
    return sorted_values[index]


def _event(sequence: int) -> EventEnvelope:
    return EventEnvelope.create(
        event_id=f"b2:physical:{sequence}",
        event_type="authority.physical-gate",
        stream_id="authority:physical-gate",
        sequence=sequence,
        producer="scripts.operator.b2_authority_physical_gate",
        producer_version="1.0.0",
        correlation_id="corr:b2-physical-gate",
        causation_id=None if sequence == 1 else f"b2:physical:{sequence - 1}",
        payload={"delta": sequence},
        causal_context={"gate": "ryzen-7-8840u"},
    )


def _require_outside_repository(path: Path, *, field: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise SystemExit(f"{field} must remain outside the repository")


def _validate_environment(expected_head: str, workspace: Path) -> tuple[str, dict[str, Any]]:
    actual_head = _git("rev-parse", "HEAD")
    if actual_head != expected_head:
        raise SystemExit(f"repository head mismatch: expected {expected_head}, got {actual_head}")
    if _git("status", "--porcelain"):
        raise SystemExit("physical gate requires a clean exact checkout")
    pin = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    actual_python = platform.python_version()
    if actual_python != pin:
        raise SystemExit(f"expected Python {pin}, got {actual_python}")
    subprocess.run([sys.executable, "-m", "pip", "check"], cwd=ROOT, check=True)
    tree = _git("rev-parse", "HEAD^{tree}")
    identity_path = workspace / "validation-identity.json"
    env = os.environ.copy()
    env.update(
        {
            "EVE_VALIDATION_COMMIT": actual_head,
            "EVE_VALIDATION_TREE": tree,
            "EVE_VALIDATION_IDENTITY_OUTPUT": str(identity_path),
        }
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit" / "validation_preflight.py")],
        cwd=ROOT,
        env=env,
        check=True,
    )
    return tree, json.loads(identity_path.read_text(encoding="utf-8"))


def _write_receipt(path: Path, packet: dict[str, Any]) -> None:
    if path.exists():
        raise SystemExit("refusing to overwrite an existing physical-gate receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(packet)
    payload["receipt_sha256"] = hashlib.sha256(_canonical(packet)).hexdigest()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--minimum-events", type=int, default=1000)
    parser.add_argument("--minimum-duration-seconds", type=float, default=60.0)
    parser.add_argument("--crash-interval", type=int, default=100)
    args = parser.parse_args(argv)
    if args.minimum_events < 1000:
        raise SystemExit("physical gate requires at least 1000 accepted events")
    if args.minimum_duration_seconds < 60.0:
        raise SystemExit("physical gate requires at least 60 sustained seconds")
    if args.crash_interval < 2 or args.crash_interval > 100:
        raise SystemExit("physical gate crash interval must be between 2 and 100 events")
    workspace = _require_outside_repository(args.workspace, field="workspace")
    output = _require_outside_repository(args.output, field="output")
    if workspace.exists():
        raise SystemExit("physical-gate workspace must not already exist")
    workspace.mkdir(parents=True)

    cpu = _cpu_name()
    if "8840U" not in cpu.upper():
        raise SystemExit(f"physical gate requires Ryzen 7 8840U, got {cpu}")
    tree, identity = _validate_environment(args.expected_head, workspace)
    database = workspace / "authority-gate.sqlite3"
    latency_ns: list[int] = []
    rss_samples: list[dict[str, int]] = [{"accepted": 0, "rss_bytes": _rss_bytes()}]
    crashes: list[dict[str, int | str]] = []
    sequence = 0
    start_ns = time.perf_counter_ns()
    store = AuthoritativeStore(database)
    startup = store.open()
    try:
        while sequence < args.minimum_events or (
            time.perf_counter_ns() - start_ns
        ) < int(args.minimum_duration_seconds * 1_000_000_000):
            next_sequence = sequence + 1
            if next_sequence > 1 and next_sequence % args.crash_interval == 0:
                store.close()
                worker = subprocess.run(
                    [
                        sys.executable,
                        str(WORKER),
                        "--database",
                        str(database),
                        "--sequence",
                        str(next_sequence),
                        "--fault-point",
                        "after_event_transaction_commit",
                        "--physical-gate",
                    ],
                    cwd=ROOT,
                    check=False,
                )
                if worker.returncode != 93:
                    raise RuntimeError(f"crash worker returned {worker.returncode}")
                recovered = store.open()
                if recovered.recovered_candidate_count != 1:
                    raise RuntimeError("restart did not prove and remove exactly one residue row")
                crashes.append(
                    {
                        "at_sequence": next_sequence,
                        "return_code": worker.returncode,
                        "recovered_candidate_count": recovered.recovered_candidate_count,
                    }
                )
            before = time.perf_counter_ns()
            store.append(_event(next_sequence))
            latency_ns.append(time.perf_counter_ns() - before)
            sequence = next_sequence
            if sequence == 1 or sequence % max(1, args.crash_interval // 2) == 0:
                rss_samples.append({"accepted": sequence, "rss_bytes": _rss_bytes()})
        verification = store.verify()
        replay_one = store.replay(0, lambda state, item: state + int(item.payload["delta"]))
        replay_two = store.replay(0, lambda state, item: state + int(item.payload["delta"]))
    finally:
        store.close()
    elapsed_ns = time.perf_counter_ns() - start_ns
    expected_replay = sequence * (sequence + 1) // 2
    ordered_latency = sorted(latency_ns)
    memory_values = [sample["rss_bytes"] for sample in rss_samples]
    packet: dict[str, Any] = {
        "schema": SCHEMA,
        "passed": (
            replay_one == replay_two == expected_replay
            and verification.accepted_event_count == sequence
            and verification.candidate_count == 0
            and len(crashes) == sequence // args.crash_interval
        ),
        "repository": {
            "commit_sha": args.expected_head,
            "tree_sha": tree,
            "clean_checkout": True,
        },
        "validation_identity": identity,
        "environment": {
            "cpu": cpu,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
        },
        "configuration": {
            "minimum_events": args.minimum_events,
            "minimum_duration_seconds": args.minimum_duration_seconds,
            "crash_interval": args.crash_interval,
        },
        "durability": {
            "journal_mode": startup.journal_mode,
            "wal_enabled": startup.wal_enabled,
            "rollback_fallback": startup.rollback_fallback,
            "synchronous": startup.synchronous,
            "candidate_then_atomic_event_and_tail": True,
        },
        "workload": {
            "accepted_events": sequence,
            "elapsed_ns": elapsed_ns,
            "append_latency_ns": latency_ns,
            "latency_summary_ns": {
                "min": ordered_latency[0],
                "p50": _percentile(ordered_latency, 0.50),
                "p95": _percentile(ordered_latency, 0.95),
                "p99": _percentile(ordered_latency, 0.99),
                "max": ordered_latency[-1],
                "mean": round(statistics.fmean(ordered_latency)),
            },
        },
        "restart_crash_injection": crashes,
        "memory": {
            "rss_samples": rss_samples,
            "start_rss_bytes": memory_values[0],
            "final_rss_bytes": memory_values[-1],
            "peak_sampled_rss_bytes": max(memory_values),
            "growth_bytes": memory_values[-1] - memory_values[0],
        },
        "replay": {
            "expected": expected_replay,
            "first": replay_one,
            "second": replay_two,
            "deterministic": replay_one == replay_two == expected_replay,
        },
        "final_chain": {
            "accepted_event_count": verification.accepted_event_count,
            "event_chain_head": verification.event_chain_head,
            "accepted_tail_hash": verification.accepted_tail_hash,
            "candidate_count": verification.candidate_count,
        },
        "database_artifacts": _database_artifacts(database),
    }
    _write_receipt(output, packet)
    print(json.dumps({"passed": packet["passed"], "output": str(output)}, sort_keys=True))
    return 0 if packet["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
