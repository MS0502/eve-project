#!/usr/bin/env python3
"""Run three real full-engine phone interactions and emit a recovery-need witness.

Each interaction is followed by the versioned fixed quiet cooldown window. Raw
process CPU, wall-clock, kernel load-average observations, and the private nonce
stay outside the repository. The public file contains only bounded evidence,
method identifiers, and digests needed for later review.
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

from core.m3_b_phone_recovery_need_witness import (
    DEFAULT_SOURCE_INSTANCE_ID,
    ENTRYPOINT_ID,
    FIXED_COOLDOWN_SECONDS,
    PhoneRecoveryNeedRuntimeSnapshot,
    build_phone_recovery_need_witness,
)
from main import build_full_engine

PRIVATE_FILENAME = "recovery_need_witness_private_v1.json"
PUBLIC_FILENAME = "recovery_need_witness_public_review_v1.json"


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
        raise SystemExit(
            f"repository head mismatch: expected {expected_head}, got {actual}"
        )
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


def _process_cpu_seconds() -> float:
    usage = os.times()
    result = float(usage.user + usage.system)
    if result < 0.0:
        raise SystemExit("process CPU time is invalid")
    return result


def _load_average_1m() -> float:
    try:
        result = float(os.getloadavg()[0])
    except (AttributeError, OSError) as exc:
        raise SystemExit("kernel one-minute load average is unavailable") from exc
    if result < 0.0:
        raise SystemExit("kernel one-minute load average is invalid")
    return result


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
) -> PhoneRecoveryNeedRuntimeSnapshot:
    load_before = _load_average_1m()
    active_cpu_before = _process_cpu_seconds()
    active_wall_before = time.monotonic()

    _run_interaction(engine, text)

    active_wall_after = time.monotonic()
    active_cpu_after = _process_cpu_seconds()
    load_after_active = _load_average_1m()

    cooldown_cpu_before = _process_cpu_seconds()
    cooldown_wall_before = time.monotonic()
    time.sleep(FIXED_COOLDOWN_SECONDS)
    cooldown_wall_after = time.monotonic()
    cooldown_cpu_after = _process_cpu_seconds()
    load_after_cooldown = _load_average_1m()

    active_cpu_delta = active_cpu_after - active_cpu_before
    active_wall_delta = active_wall_after - active_wall_before
    cooldown_cpu_delta = cooldown_cpu_after - cooldown_cpu_before
    cooldown_wall_delta = cooldown_wall_after - cooldown_wall_before
    if (
        active_cpu_delta < 0.0
        or active_wall_delta <= 0.0
        or cooldown_cpu_delta < 0.0
        or cooldown_wall_delta <= 0.0
    ):
        raise SystemExit("recovery counters did not advance monotonically")

    return PhoneRecoveryNeedRuntimeSnapshot(
        source_instance_id=source_instance_id,
        logical_tick=logical_tick,
        active_process_cpu_seconds=active_cpu_delta,
        active_wall_seconds=active_wall_delta,
        cooldown_process_cpu_seconds=cooldown_cpu_delta,
        cooldown_wall_seconds=cooldown_wall_delta,
        cpu_count=max(1, os.cpu_count() or 1),
        load_average_1m_before=load_before,
        load_average_1m_after_active=load_after_active,
        load_average_1m_after_cooldown=load_after_cooldown,
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
        raise SystemExit(
            "phone recovery-need witness requires exactly three --input values"
        )
    for text in args.inputs:
        if not isinstance(text, str) or not text.strip():
            raise SystemExit("witness input must be non-empty text")

    repository_head = _repository_head(args.expected_head)
    private_root = _prepare_private_root(args.private_root)
    private_nonce = _private_nonce(args.nonce_file)
    engine = build_full_engine()
    snapshots = tuple(
        _capture_window(
            engine,
            text,
            logical_tick=index,
            source_instance_id=args.source_instance_id,
        )
        for index, text in enumerate(args.inputs)
    )
    witness = build_phone_recovery_need_witness(
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
    print(_canonical(public_mapping))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
