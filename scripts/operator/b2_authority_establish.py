"""Preflight and t=0 establishment for a physically accepted B2 store."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.authoritative_store import GENESIS_HASH, AuthoritativeStore  # noqa: E402

PHYSICAL_SCHEMA = "eve.b2-authority-8840u-physical-gate.v1"
ESTABLISHMENT_SCHEMA = "eve.b2-authority-establishment.v1"


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _outside_repository(path: Path, field: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise SystemExit(f"{field} must remain outside the repository")


def _load_physical_receipt(path: Path, expected_head: str) -> dict[str, Any]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    receipt_hash = packet.pop("receipt_sha256", None)
    if receipt_hash != hashlib.sha256(_canonical(packet)).hexdigest():
        raise SystemExit("physical receipt digest differs")
    if packet.get("schema") != PHYSICAL_SCHEMA or packet.get("passed") is not True:
        raise SystemExit("physical receipt is not a green B2 8840U gate")
    if packet.get("repository", {}).get("commit_sha") != expected_head:
        raise SystemExit("physical receipt is bound to a different commit")
    if "8840U" not in str(packet.get("environment", {}).get("cpu", "")).upper():
        raise SystemExit("physical receipt does not identify Ryzen 7 8840U")
    pin = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    if packet.get("environment", {}).get("python") != pin:
        raise SystemExit("physical receipt Python does not match the repository pin")
    repository = packet.get("repository", {})
    if repository.get("tree_sha") != _git("rev-parse", "HEAD^{tree}"):
        raise SystemExit("physical receipt is bound to a different repository tree")
    configuration = packet.get("configuration", {})
    minimum_events = configuration.get("minimum_events")
    minimum_duration = configuration.get("minimum_duration_seconds")
    crash_interval = configuration.get("crash_interval")
    if (
        isinstance(minimum_events, bool)
        or not isinstance(minimum_events, int)
        or minimum_events < 1000
        or isinstance(minimum_duration, bool)
        or not isinstance(minimum_duration, (int, float))
        or minimum_duration < 60.0
        or isinstance(crash_interval, bool)
        or not isinstance(crash_interval, int)
        or crash_interval < 2
        or crash_interval > 100
    ):
        raise SystemExit("physical receipt does not satisfy sustained fault-gate bounds")
    durability = packet.get("durability", {})
    if (
        durability.get("journal_mode") not in {"wal", "delete"}
        or durability.get("synchronous") != "FULL"
        or durability.get("candidate_then_atomic_event_and_tail") is not True
    ):
        raise SystemExit("physical receipt durability contract differs")
    workload = packet.get("workload", {})
    accepted_events = workload.get("accepted_events")
    crashes = packet.get("restart_crash_injection")
    final_chain = packet.get("final_chain", {})
    replay = packet.get("replay", {})
    if (
        isinstance(accepted_events, bool)
        or not isinstance(accepted_events, int)
        or accepted_events < minimum_events
        or not isinstance(crashes, list)
        or len(crashes) != accepted_events // crash_interval
        or final_chain.get("accepted_event_count") != accepted_events
        or final_chain.get("candidate_count") != 0
        or replay.get("deterministic") is not True
        or replay.get("first") != replay.get("expected")
        or replay.get("second") != replay.get("expected")
    ):
        raise SystemExit("physical receipt workload or replay proof differs")
    packet["receipt_sha256"] = receipt_hash
    return packet


def _write(path: Path, packet: dict[str, Any]) -> None:
    if path.exists():
        raise SystemExit("refusing to overwrite an establishment receipt")
    payload = dict(packet)
    payload["receipt_sha256"] = hashlib.sha256(_canonical(packet)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--physical-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args(argv)
    database = _outside_repository(args.database, "database")
    receipt_path = _outside_repository(args.physical_receipt, "physical receipt")
    output = _outside_repository(args.output, "output")
    if database.exists():
        raise SystemExit("t=0 establishment requires a new authority database")
    actual_head = _git("rev-parse", "HEAD")
    if actual_head != args.expected_head:
        raise SystemExit("repository head differs from requested establishment head")
    if _git("status", "--porcelain"):
        raise SystemExit("establishment requires a clean exact checkout")
    pin = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    if platform.python_version() != pin:
        raise SystemExit(f"expected Python {pin}, got {platform.python_version()}")
    physical = _load_physical_receipt(receipt_path, args.expected_head)

    store = AuthoritativeStore(database)
    try:
        startup = store.open()
        verification = store.verify()
        if (
            verification.accepted_event_count != 0
            or verification.event_chain_head != GENESIS_HASH
            or verification.accepted_tail_hash != GENESIS_HASH
        ):
            raise SystemExit("new authority database does not verify at genesis")
    finally:
        store.close()
    packet = {
        "schema": ESTABLISHMENT_SCHEMA,
        "established": True,
        "authority_active_for_runtime": False,
        "repository": {
            "commit_sha": actual_head,
            "tree_sha": _git("rev-parse", "HEAD^{tree}"),
            "clean_checkout": True,
        },
        "python": platform.python_version(),
        "physical_receipt": {
            "path": str(receipt_path),
            "file_sha256": _sha256(receipt_path),
            "receipt_sha256": physical["receipt_sha256"],
        },
        "database": {
            "path": str(database),
            "file_sha256": _sha256(database),
            "startup": asdict(startup),
            "verification": asdict(verification),
        },
    }
    _write(output, packet)
    print(json.dumps({"established": True, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
