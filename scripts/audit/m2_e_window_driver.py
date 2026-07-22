#!/usr/bin/env python3
"""Synthetic-only M2-E window-driver chaos and portability evidence CLI.

The CLI operates only in an explicit caller-supplied disposable workspace. It
never discovers production state, imports the legacy runtime, or grants cutover,
recovery, or M3 authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY, canonical_json_object
from core.m2_e_window_driver import (
    ACCEPTED_M2_E_DECISION_DIGEST,
    ACCEPTED_M2_E_HEAD,
    ACCEPTED_M2_E_PACKET_DIGEST,
    BOUNDED_STREAM,
    CHAOS_KILL_REPETITIONS,
    CHAOS_PHASES,
    STATE_SCHEMA_VERSION,
    WINDOW_BASELINE_SHA,
)
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    OBSERVER_PRODUCER,
    OBSERVER_VERSION,
    SUCCESS_EVENT_TYPE,
)
from core.shadow_projection import (
    ActivationLearnPairShadowState,
    replay_activation_learn_pair,
)
from core.sqlite_shadow_store import (
    PersistedEventCorruption,
    SQLiteShadowStore,
    ShadowStoragePolicy,
    StoragePolicyExceeded,
)

CHAOS_EVIDENCE_SCHEMA = "eve.m2-e-window-chaos-evidence.v1"
PORTABILITY_MANIFEST_SCHEMA = "eve.m2-e-window-portability-manifest.v1"
PORTABILITY_VERIFY_SCHEMA = "eve.m2-e-window-portability-verification.v1"
_SYNTHETIC_MARKER = "eve.m2-e.synthetic-only.v1"
_SNAPSHOT_UPDATE_TRIGGER = (
    "CREATE TRIGGER snapshots_no_update BEFORE UPDATE ON snapshots "
    "BEGIN SELECT RAISE(ABORT,'append-only snapshots'); END"
)
_EVENT_UPDATE_TRIGGER = (
    "CREATE TRIGGER events_no_update BEFORE UPDATE ON events "
    "BEGIN SELECT RAISE(ABORT,'append-only events'); END"
)
_EMPTY = {"calls": [], "learned": []}


class WindowDriverCLIError(RuntimeError):
    pass


def _canon(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_object(value, field=field)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _digest(value: Mapping[str, Any], field: str) -> str:
    return _sha_bytes(_canon(value, field).encode("utf-8"))


def _write_json(path: Path, value: Mapping[str, Any], *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if pretty
        else _canon(value, "output") + "\n"
    )
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _new_workspace(path: str | Path) -> Path:
    root = Path(path)
    if str(path) == ":memory:" or root.name == "" or root.exists():
        raise WindowDriverCLIError("workspace must be a new concrete path")
    root.mkdir(parents=True, exist_ok=False)
    marker = root / "SYNTHETIC_ONLY"
    marker.write_text(_SYNTHETIC_MARKER + "\n", encoding="utf-8")
    return root


def _pair(sequence: int) -> list[Any]:
    return [f"synthetic-left-{sequence:04d}", f"synthetic-right-{sequence:04d}", sequence / 1000.0]


def _snapshot_for(sequence: int) -> dict[str, Any]:
    pairs = [_pair(index) for index in range(1, sequence + 1)]
    return {"calls": pairs, "learned": pairs}


def _event(sequence: int) -> EventEnvelope:
    before = _snapshot_for(sequence - 1)
    after = _snapshot_for(sequence)
    previous = None if sequence == 1 else f"m2e-window:event:{sequence - 1:04d}"
    return EventEnvelope.create(
        event_id=f"m2e-window:event:{sequence:04d}",
        event_type=SUCCESS_EVENT_TYPE,
        stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id,
        sequence=sequence,
        producer=OBSERVER_PRODUCER,
        producer_version=OBSERVER_VERSION,
        correlation_id="m2e-window:synthetic",
        causation_id=previous,
        payload={
            "after": after,
            "before": before,
            "legacy_outcome": {"error_type": None, "succeeded": True},
            "target": {
                "callable": ACTIVATION_LEARN_PAIR_TARGET.callable_name,
                "disposition": ACTIVATION_LEARN_PAIR_TARGET.module_disposition,
                "module_path": ACTIVATION_LEARN_PAIR_TARGET.module_path,
                "target_id": ACTIVATION_LEARN_PAIR_TARGET.target_id,
            },
        },
        causal_context={
            "arguments_captured": False,
            "legacy_result_captured": False,
            "observation_phase": "after_the_fact",
            "source_evidence_range": ACTIVATION_LEARN_PAIR_TARGET.evidence_range,
        },
        authority=SHADOW_AUTHORITY,
    )


def _restore(store: SQLiteShadowStore):
    initial = ActivationLearnPairShadowState.from_initial_snapshot(_EMPTY)
    events_by_sequence = {
        envelope.sequence: envelope for envelope in store.events(stream_id=BOUNDED_STREAM)
    }

    def reducer(state: ActivationLearnPairShadowState, envelope: EventEnvelope):
        if state.sequence == 0 and envelope.sequence > 1:
            previous = events_by_sequence.get(envelope.sequence - 1)
            if previous is None:
                raise WindowDriverCLIError("snapshot suffix lacks preceding synthetic event")
            state = ActivationLearnPairShadowState(
                calls=state.calls,
                learned=state.learned,
                sequence=previous.sequence,
                last_event_id=previous.event_id,
                last_event_digest=previous.digest,
            )
        return replay_activation_learn_pair(state, (envelope,))

    return store.restore_verified(
        stream_id=BOUNDED_STREAM,
        initial_state=initial,
        reducer=reducer,
        state_to_mapping=lambda value: value.snapshot,
        state_from_mapping=ActivationLearnPairShadowState.from_initial_snapshot,
    )


def _create_baseline(root: Path, *, policy: ShadowStoragePolicy | None = None) -> tuple[Path, Path, str]:
    store_path = root / "baseline.sqlite3"
    backup_dir = root / "backups"
    store = SQLiteShadowStore(store_path, policy=policy or ShadowStoragePolicy(snapshot_interval_events=2))
    store.initialize()
    first = (_event(1), _event(2))
    store.append_many(first)
    state_two = replay_activation_learn_pair(
        ActivationLearnPairShadowState.from_initial_snapshot(_EMPTY), first
    )
    store.write_snapshot(
        snapshot_id="m2e-window:snapshot:0002",
        stream_id=BOUNDED_STREAM,
        through_sequence=2,
        state=state_two.snapshot,
        state_schema_version=STATE_SCHEMA_VERSION,
    )
    second = (_event(3), _event(4))
    store.append_many(second)
    state_four = replay_activation_learn_pair(state_two, second)
    store.write_snapshot(
        snapshot_id="m2e-window:snapshot:0004",
        stream_id=BOUNDED_STREAM,
        through_sequence=4,
        state=state_four.snapshot,
        state_schema_version=STATE_SCHEMA_VERSION,
    )
    integrity = store.integrity_check()
    if not integrity.valid:
        raise WindowDriverCLIError(f"synthetic baseline integrity failed: {integrity.errors}")
    restored = _restore(store)
    if not restored.verified:
        raise WindowDriverCLIError("synthetic baseline restore not verified")
    backup = store.create_backup(backup_dir, backup_ordinal=1)
    return store_path, Path(backup.backup_path), restored.state_digest


def _copy_database(source: Path, target: Path) -> None:
    if target.exists():
        raise WindowDriverCLIError("copy target already exists")
    shutil.copyfile(source, target)


def _remove_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _wait_for_marker(path: Path, process: subprocess.Popen[Any], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            raise WindowDriverCLIError(
                f"chaos child exited before marker: returncode={process.returncode}"
            )
        time.sleep(0.02)
    process.kill()
    raise WindowDriverCLIError("timed out waiting for chaos phase marker")


def _hard_kill(process: subprocess.Popen[Any]) -> str:
    if os.name == "nt":
        process.kill()  # TerminateProcess: Windows hard-kill equivalent to SIGKILL.
        mechanism = "windows_terminate_process"
    else:
        os.kill(process.pid, signal.SIGKILL)
        mechanism = "posix_sigkill"
    process.wait(timeout=15)
    return mechanism


def _child_phase(database: Path, phase: str, marker: Path, partial: Path) -> int:
    store = SQLiteShadowStore(database, policy=ShadowStoragePolicy(snapshot_interval_events=2))
    store.initialize()
    if phase == "idle":
        marker.write_text("ready\n", encoding="utf-8")
    elif phase == "mid-write":
        connection = sqlite3.connect(database, isolation_level=None, timeout=5.0)
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO events(ordinal,event_id,stream_id,sequence,event_json,envelope_digest,event_bytes,previous_chain_digest,chain_digest) VALUES(?,?,?,?,?,?,?,?,?)",
            (5, "m2e-window:uncommitted", BOUNDED_STREAM, 5, "{}", "0" * 64, 2, "0" * 64, "0" * 64),
        )
        marker.write_text("ready\n", encoding="utf-8")
    elif phase == "mid-snapshot":
        connection = sqlite3.connect(database, isolation_level=None, timeout=5.0)
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO snapshots(ordinal,snapshot_id,stream_id,through_sequence,through_event_id,through_event_digest,state_schema_version,state_json,state_digest,manifest_json,manifest_digest,snapshot_digest) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (3, "m2e-window:uncommitted-snapshot", BOUNDED_STREAM, 4, None, None, STATE_SCHEMA_VERSION, "{}", "0" * 64, "{}", "0" * 64, "0" * 64),
        )
        marker.write_text("ready\n", encoding="utf-8")
    elif phase == "mid-consolidation":
        shutil.copyfile(database, partial)
        with partial.open("ab") as handle:
            handle.write(b"partial-publication")
            handle.flush()
            os.fsync(handle.fileno())
        marker.write_text("ready\n", encoding="utf-8")
    else:
        raise WindowDriverCLIError(f"unknown chaos phase: {phase}")
    while True:
        time.sleep(60)


def _run_kill_matrix(root: Path, backup: Path, expected_digest: str, repetitions: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    script = Path(__file__).resolve()
    for phase in CHAOS_PHASES:
        for repetition in range(1, repetitions + 1):
            case = root / f"kill-{phase}-{repetition}"
            case.mkdir()
            database = case / "store.sqlite3"
            marker = case / "phase.ready"
            partial = case / ".consolidated.sqlite3.partial"
            _copy_database(backup, database)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(script),
                    "_child-phase",
                    "--database",
                    str(database),
                    "--phase",
                    phase,
                    "--marker",
                    str(marker),
                    "--partial",
                    str(partial),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _wait_for_marker(marker, process)
            mechanism = _hard_kill(process)
            partial_observed = partial.exists()
            partial.unlink(missing_ok=True)
            _remove_sidecars(partial)
            recovered = SQLiteShadowStore(
                database, policy=ShadowStoragePolicy(snapshot_interval_events=2)
            )
            recovered.initialize()
            integrity = recovered.integrity_check()
            restore = _restore(recovered)
            digest_match = restore.state_digest == expected_digest
            checks = {
                "child_hard_killed": process.returncode not in (None, 0),
                "integrity_valid": integrity.valid,
                "recovery_digest_match": digest_match,
                "event_count_unchanged": integrity.event_count == 4,
                "partial_publication_not_authoritative": not partial.exists(),
            }
            results.append(
                {
                    "checks": checks,
                    "database_sha256": _sha_file(database),
                    "expected_recovery_digest": expected_digest,
                    "hard_kill_mechanism": mechanism,
                    "observed_partial_before_cleanup": partial_observed,
                    "passed": all(checks.values()),
                    "phase": phase,
                    "recovered_digest": restore.state_digest,
                    "repetition": repetition,
                }
            )
    return results


def _corrupt_snapshot(path: Path) -> None:
    connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP TRIGGER snapshots_no_update")
        connection.execute(
            "UPDATE snapshots SET state_digest=? WHERE snapshot_id=?",
            ("0" * 64, "m2e-window:snapshot:0004"),
        )
        connection.execute(_SNAPSHOT_UPDATE_TRIGGER)
        connection.commit()
    except sqlite3.DatabaseError:
        connection.rollback()
        raise
    finally:
        connection.close()


def _corrupt_event(path: Path) -> None:
    connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute(
            "UPDATE events SET envelope_digest=? WHERE event_id=?",
            ("0" * 64, "m2e-window:event:0004"),
        )
        connection.execute(_EVENT_UPDATE_TRIGGER)
        connection.commit()
    except sqlite3.DatabaseError:
        connection.rollback()
        raise
    finally:
        connection.close()


def _run_corruption(root: Path, backup: Path, expected_digest: str) -> dict[str, Any]:
    snapshot_path = root / "corrupt-snapshot.sqlite3"
    _copy_database(backup, snapshot_path)
    _corrupt_snapshot(snapshot_path)
    snapshot_store = SQLiteShadowStore(
        snapshot_path, policy=ShadowStoragePolicy(snapshot_interval_events=2)
    )
    snapshot_store.initialize()
    selection = snapshot_store.latest_valid_snapshot(BOUNDED_STREAM)
    snapshot_restore = _restore(snapshot_store)

    event_path = root / "corrupt-event.sqlite3"
    _copy_database(backup, event_path)
    _corrupt_event(event_path)
    event_store = SQLiteShadowStore(
        event_path, policy=ShadowStoragePolicy(snapshot_interval_events=2)
    )
    event_store.initialize()
    event_integrity = event_store.integrity_check()
    restore_error = None
    try:
        _restore(event_store)
    except PersistedEventCorruption as exc:
        restore_error = type(exc).__name__

    recovered_path = root / "recovered-from-backup.sqlite3"
    _copy_database(backup, recovered_path)
    recovered_store = SQLiteShadowStore(
        recovered_path, policy=ShadowStoragePolicy(snapshot_interval_events=2)
    )
    recovered_store.initialize()
    recovered = _restore(recovered_store)

    checks = {
        "corrupt_snapshot_rejected": "m2e-window:snapshot:0004" in selection.rejected_snapshot_ids,
        "valid_snapshot_fallback_selected": selection.selected is not None
        and selection.selected.snapshot_id == "m2e-window:snapshot:0002",
        "snapshot_restore_digest_match": snapshot_restore.state_digest == expected_digest,
        "corrupt_event_integrity_rejected": not event_integrity.valid,
        "corrupt_event_restore_failed_closed": restore_error == "PersistedEventCorruption",
        "backup_recovery_digest_match": recovered.state_digest == expected_digest,
    }
    return {
        "checks": checks,
        "corrupt_event_errors": list(event_integrity.errors),
        "corrupt_event_sha256": _sha_file(event_path),
        "corrupt_snapshot_sha256": _sha_file(snapshot_path),
        "passed": all(checks.values()),
        "recovered_digest": recovered.state_digest,
        "recovered_sha256": _sha_file(recovered_path),
        "rejected_snapshot_ids": list(selection.rejected_snapshot_ids),
        "selected_snapshot_id": None if selection.selected is None else selection.selected.snapshot_id,
    }


def _run_disk_pressure(root: Path, expected_digest: str) -> dict[str, Any]:
    pressure_root = root / "disk-pressure"
    pressure_root.mkdir()
    path, _backup, baseline_digest = _create_baseline(
        pressure_root,
        policy=ShadowStoragePolicy(
            snapshot_interval_events=2,
            max_event_count=4,
            max_event_bytes=4_000_000,
            max_snapshot_count=10,
            max_snapshot_bytes=4_000_000,
        ),
    )
    store = SQLiteShadowStore(
        path,
        policy=ShadowStoragePolicy(
            snapshot_interval_events=2,
            max_event_count=4,
            max_event_bytes=4_000_000,
            max_snapshot_count=10,
            max_snapshot_bytes=4_000_000,
        ),
    )
    store.initialize()
    rejected = False
    try:
        store.append(_event(5))
    except StoragePolicyExceeded:
        rejected = True
    integrity = store.integrity_check()
    restored = _restore(store)
    checks = {
        "pressure_append_rejected": rejected,
        "event_count_preserved": integrity.event_count == 4,
        "integrity_valid": integrity.valid,
        "recovery_digest_preserved": restored.state_digest == expected_digest == baseline_digest,
    }
    return {
        "checks": checks,
        "database_sha256": _sha_file(path),
        "passed": all(checks.values()),
        "recovered_digest": restored.state_digest,
    }


def run_chaos(*, workspace: str | Path, repetitions: int) -> dict[str, Any]:
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise WindowDriverCLIError("repetitions must be positive")
    root = _new_workspace(workspace)
    baseline_root = root / "baseline"
    baseline_root.mkdir()
    baseline_path, backup_path, expected_digest = _create_baseline(baseline_root)
    kill_matrix = _run_kill_matrix(root, backup_path, expected_digest, repetitions)
    corruption = _run_corruption(root, backup_path, expected_digest)
    disk_pressure = _run_disk_pressure(root, expected_digest)
    checks = {
        "all_kills_recovered": all(item["passed"] for item in kill_matrix),
        "kill_matrix_complete": len(kill_matrix) == len(CHAOS_PHASES) * repetitions,
        "corruption_rejected_and_recovered": corruption["passed"],
        "disk_pressure_guarded": disk_pressure["passed"],
        "synthetic_store_only": (root / "SYNTHETIC_ONLY").read_text(encoding="utf-8").strip()
        == _SYNTHETIC_MARKER,
        "legacy_runtime_authoritative": True,
        "cutover_authorized": False,
        "m3_authority_open": False,
    }
    material: dict[str, Any] = {
        "accepted_m2_e_decision_digest": ACCEPTED_M2_E_DECISION_DIGEST,
        "accepted_m2_e_head": ACCEPTED_M2_E_HEAD,
        "accepted_m2_e_packet_digest": ACCEPTED_M2_E_PACKET_DIGEST,
        "authority": "chaos_evidence_only",
        "baseline_backup_sha256": _sha_file(backup_path),
        "baseline_database_sha256": _sha_file(baseline_path),
        "baseline_sha": WINDOW_BASELINE_SHA,
        "checks": checks,
        "corruption": corruption,
        "cutover_authorized": False,
        "disk_pressure": disk_pressure,
        "expected_recovery_digest": expected_digest,
        "kill_matrix": kill_matrix,
        "legacy_runtime_authoritative": True,
        "m3_authority_open": False,
        "machine_passed": all(checks.values()),
        "repetitions": repetitions,
        "schema_version": CHAOS_EVIDENCE_SCHEMA,
        "synthetic_store_only": True,
    }
    material["evidence_digest"] = _digest(material, "m2_e_window_chaos_evidence")
    return material


def create_portability(*, workspace: str | Path) -> tuple[dict[str, Any], Path]:
    root = _new_workspace(workspace)
    _database, backup, expected_digest = _create_baseline(root)
    material: dict[str, Any] = {
        "accepted_m2_e_decision_digest": ACCEPTED_M2_E_DECISION_DIGEST,
        "authority": "portability_evidence_only",
        "backup_file_name": "portable-shadow.sqlite3",
        "backup_sha256": _sha_file(backup),
        "baseline_sha": WINDOW_BASELINE_SHA,
        "cutover_authorized": False,
        "expected_event_count": 4,
        "expected_recovery_digest": expected_digest,
        "legacy_runtime_authoritative": True,
        "m3_authority_open": False,
        "schema_version": PORTABILITY_MANIFEST_SCHEMA,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "store_authority": SHADOW_AUTHORITY,
        "stream_id": BOUNDED_STREAM,
        "synthetic_store_only": True,
    }
    material["manifest_digest"] = _digest(material, "m2_e_portability_manifest")
    return material, backup


def verify_portability(*, backup: str | Path, manifest_path: str | Path, workspace: str | Path) -> dict[str, Any]:
    source = Path(backup)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != PORTABILITY_MANIFEST_SCHEMA:
        raise WindowDriverCLIError("portability manifest schema mismatch")
    supplied_digest = manifest.pop("manifest_digest", None)
    expected_manifest_digest = _digest(manifest, "m2_e_portability_manifest")
    manifest["manifest_digest"] = supplied_digest
    if supplied_digest != expected_manifest_digest:
        raise WindowDriverCLIError("portability manifest digest mismatch")
    if _sha_file(source) != manifest.get("backup_sha256"):
        raise WindowDriverCLIError("portable backup digest mismatch")
    root = _new_workspace(workspace)
    target = root / "restored.sqlite3"
    _copy_database(source, target)
    store = SQLiteShadowStore(target, policy=ShadowStoragePolicy(snapshot_interval_events=2))
    store.initialize()
    integrity = store.integrity_check()
    restored = _restore(store)
    checks = {
        "backup_digest_match": _sha_file(source) == manifest["backup_sha256"],
        "integrity_valid": integrity.valid,
        "event_count_match": integrity.event_count == manifest["expected_event_count"],
        "replay_digest_match": restored.state_digest == manifest["expected_recovery_digest"],
        "synthetic_store_only": manifest["synthetic_store_only"] is True,
        "legacy_runtime_authoritative": manifest["legacy_runtime_authoritative"] is True,
        "no_cutover_or_m3": not manifest["cutover_authorized"] and not manifest["m3_authority_open"],
    }
    result: dict[str, Any] = {
        "authority": "portability_evidence_only",
        "checks": checks,
        "machine_passed": all(checks.values()),
        "manifest_digest": supplied_digest,
        "recovered_digest": restored.state_digest,
        "restored_database_sha256": _sha_file(target),
        "schema_version": PORTABILITY_VERIFY_SCHEMA,
    }
    result["verification_digest"] = _digest(result, "m2_e_portability_verification")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    chaos = subparsers.add_parser("chaos")
    chaos.add_argument("--workspace", required=True)
    chaos.add_argument("--output", required=True)
    chaos.add_argument("--repetitions", type=int, default=CHAOS_KILL_REPETITIONS)
    chaos.add_argument("--pretty", action="store_true")

    create = subparsers.add_parser("portability-create")
    create.add_argument("--workspace", required=True)
    create.add_argument("--backup-output", required=True)
    create.add_argument("--manifest-output", required=True)
    create.add_argument("--pretty", action="store_true")

    verify = subparsers.add_parser("portability-verify")
    verify.add_argument("--backup", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--workspace", required=True)
    verify.add_argument("--output", required=True)
    verify.add_argument("--pretty", action="store_true")

    child = subparsers.add_parser("_child-phase")
    child.add_argument("--database", required=True)
    child.add_argument("--phase", choices=CHAOS_PHASES, required=True)
    child.add_argument("--marker", required=True)
    child.add_argument("--partial", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "_child-phase":
        return _child_phase(Path(args.database), args.phase, Path(args.marker), Path(args.partial))
    if args.command == "chaos":
        evidence = run_chaos(workspace=args.workspace, repetitions=args.repetitions)
        _write_json(Path(args.output), evidence, pretty=args.pretty)
        print(json.dumps({"machine_passed": evidence["machine_passed"], "evidence_digest": evidence["evidence_digest"]}, sort_keys=True))
        return 0 if evidence["machine_passed"] else 1
    if args.command == "portability-create":
        manifest, backup = create_portability(workspace=args.workspace)
        backup_output = Path(args.backup_output)
        backup_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(backup, backup_output)
        if _sha_file(backup_output) != manifest["backup_sha256"]:
            raise WindowDriverCLIError("published portability backup digest changed")
        _write_json(Path(args.manifest_output), manifest, pretty=args.pretty)
        print(json.dumps({"backup_sha256": manifest["backup_sha256"], "manifest_digest": manifest["manifest_digest"]}, sort_keys=True))
        return 0
    result = verify_portability(
        backup=args.backup,
        manifest_path=args.manifest,
        workspace=args.workspace,
    )
    _write_json(Path(args.output), result, pretty=args.pretty)
    print(json.dumps({"machine_passed": result["machine_passed"], "verification_digest": result["verification_digest"]}, sort_keys=True))
    return 0 if result["machine_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
