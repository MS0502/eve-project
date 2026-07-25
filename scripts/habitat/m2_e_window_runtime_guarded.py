#!/usr/bin/env python3
"""Guarded Termux habitat runtime for the bounded M2-E shadow window.

This is the operational successor to the original habitat runtime.  The public
entrypoint remains ``m2_e_window_runtime.py`` for phone/boot compatibility, but
that file delegates here so recovery failures cannot be silently collapsed.

All evidence stays in the operator-private companion directory.  This module
never imports or mutates the legacy application runtime, installs no cutover
hook, and cannot open M3 authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.m2_e_window_driver as window_driver
from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY, canonical_json_object
from core.m2_e_window_driver import (
    BOUNDED_STREAM,
    SCRIPTED_STIMULI_PER_SIM_HOUR,
    SIM_HOUR_SECONDS,
    STATE_SCHEMA_VERSION,
    WindowConfig,
    WindowState,
    advance_runtime,
    apply_watchdog,
    evidence_record,
    freeze_shadow,
    mark_notification_sent,
    maybe_seal,
    one_line_status,
    record_discrete_stimulus,
    record_power_cycle,
    record_recovery,
)
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    OBSERVER_PRODUCER,
    OBSERVER_VERSION,
    SUCCESS_EVENT_TYPE,
)
from core.shadow_projection import ActivationLearnPairShadowState, replay_activation_learn_pair
from core.sqlite_shadow_store import (
    SQLiteShadowStore,
    ShadowPersistenceError,
    ShadowStoragePolicy,
    StoragePolicyExceeded,
)

PRIVATE_SCHEMA = "eve.m2-e-window-private-companion.v1"
RAW_EVIDENCE_SCHEMA = "eve.m2-e-window-habitat-raw.v1"
SEAL_SCHEMA = "eve.m2-e-window-seal.v1"
DEFAULT_PRIVATE_ROOT = Path.home() / ".local" / "share" / "eve-m2e-window-private"
STIMULUS_INTERVAL_SECONDS = SIM_HOUR_SECONDS // SCRIPTED_STIMULI_PER_SIM_HOUR
CHECKPOINT_INTERVAL_SECONDS = 10
WATCHDOG_INTERVAL_SECONDS = 60
SNAPSHOT_INTERVAL_EVENTS = 24
BACKUP_INTERVAL_EVENTS = 72
IO_BACKOFF_SECONDS = (1, 2, 4)
IO_FAILURE = "io_failure"
_EMPTY = {"calls": [], "learned": []}
_STOP = False


class HabitatError(RuntimeError):
    pass


def _enable_io_failure_reason() -> None:
    """Extend only this habitat process' frozen-state vocabulary deterministically."""

    if IO_FAILURE not in window_driver.HALT_REASONS:
        window_driver.HALT_REASONS = (*window_driver.HALT_REASONS, IO_FAILURE)


def _canon(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_object(value, field=field)


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canon(value, field).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(_canon(value, "private_json") + "\n", encoding="utf-8")
    os.chmod(temporary, mode)
    temporary.replace(path)


def _append_raw(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canon(value, "raw_evidence") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def _paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "state": root / "window_state.json",
        "database": root / "shadow.sqlite3",
        "backups": root / "backups",
        "raw": root / "raw_evidence.jsonl",
        "running": root / "worker.running",
        "seal": root / "sealed_digest.json",
        "private_manifest": root / "private_manifest.json",
    }


def _ensure_private_root(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    paths = _paths(root)
    paths["backups"].mkdir(parents=True, exist_ok=True)
    os.chmod(paths["backups"], 0o700)
    manifest = {
        "authority": SHADOW_AUTHORITY,
        "private_root": str(root),
        "raw_publication_allowed": False,
        "schema_version": PRIVATE_SCHEMA,
        "synthetic_scripted_stimuli_only": True,
    }
    _atomic_json(paths["private_manifest"], manifest)
    return paths


def _pair(sequence: int) -> list[Any]:
    return [f"habitat-left-{sequence:06d}", f"habitat-right-{sequence:06d}", sequence / 10000.0]


def _snapshot_for(sequence: int) -> dict[str, Any]:
    pairs = [_pair(index) for index in range(1, sequence + 1)]
    return {"calls": pairs, "learned": pairs}


def _hour_counts_for_events(event_count: int, config: WindowConfig) -> tuple[tuple[int, int], ...]:
    counts: list[tuple[int, int]] = []
    remaining = event_count
    hour = 0
    while remaining > 0:
        count = min(config.event_cap_per_sim_hour, remaining)
        counts.append((hour, count))
        remaining -= count
        hour += 1
    return tuple(counts)


def _event(sequence: int) -> EventEnvelope:
    return EventEnvelope.create(
        event_id=f"m2e-habitat:event:{sequence:06d}",
        event_type=SUCCESS_EVENT_TYPE,
        stream_id=BOUNDED_STREAM,
        sequence=sequence,
        producer=OBSERVER_PRODUCER,
        producer_version=OBSERVER_VERSION,
        correlation_id="m2e-habitat:scripted-window",
        causation_id=None if sequence == 1 else f"m2e-habitat:event:{sequence - 1:06d}",
        payload={
            "after": _snapshot_for(sequence),
            "before": _snapshot_for(sequence - 1),
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
    events = {event.sequence: event for event in store.events(stream_id=BOUNDED_STREAM)}

    def reducer(state: ActivationLearnPairShadowState, envelope: EventEnvelope):
        if state.sequence == 0 and envelope.sequence > 1:
            previous = events.get(envelope.sequence - 1)
            if previous is None:
                raise HabitatError("snapshot suffix lacks preceding event")
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


def _load_state(paths: Mapping[str, Path], *, window_id: str) -> WindowState:
    state_path = paths["state"]
    if not state_path.exists():
        state = WindowState.create(window_id=window_id, local_date=date.today().isoformat())
        _save_state(paths, state)
        return state
    try:
        value = json.loads(state_path.read_text(encoding="utf-8"))
        return WindowState.from_mapping(value)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HabitatError("private window state is malformed") from exc


def _save_state(paths: Mapping[str, Path], state: WindowState) -> None:
    _atomic_json(paths["state"], state.to_mapping())


def _store(paths: Mapping[str, Path], config: WindowConfig) -> SQLiteShadowStore:
    policy = ShadowStoragePolicy(
        snapshot_interval_events=SNAPSHOT_INTERVAL_EVENTS,
        max_event_count=max(config.event_quota + 24, 512),
        max_event_bytes=min(config.disk_budget_bytes // 2, 256 * 1024 * 1024),
        max_snapshot_count=128,
        max_snapshot_bytes=min(config.disk_budget_bytes // 4, 128 * 1024 * 1024),
        max_backups=3,
    )
    store = SQLiteShadowStore(paths["database"], policy=policy)
    store.initialize()
    return store


def _disk_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _exception_evidence(
    paths: Mapping[str, Path],
    *,
    context: str,
    exc: BaseException,
    attempt: int,
) -> str:
    trace_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    record: dict[str, Any] = {
        "attempt": attempt,
        "context": context,
        "exception_message": str(exc),
        "exception_type": type(exc).__name__,
        "schema_version": RAW_EVIDENCE_SCHEMA,
        "traceback_digest": hashlib.sha256(trace_text.encode("utf-8")).hexdigest(),
        "type": "habitat_exception",
    }
    record["exception_digest"] = _digest(record, "habitat_exception")
    _append_raw(paths["raw"], record)
    return record["exception_digest"]


def _freeze_with_evidence(
    paths: Mapping[str, Path],
    state: WindowState,
    reason: str,
    *,
    context: str,
    trigger_exception_digest: str | None = None,
) -> WindowState:
    if state.frozen:
        return state
    frozen = freeze_shadow(state, reason)
    record: dict[str, Any] = {
        "before_state_digest": state.digest,
        "context": context,
        "freeze_reason": reason,
        "frozen_state_digest": frozen.digest,
        "schema_version": RAW_EVIDENCE_SCHEMA,
        "trigger_exception_digest": trigger_exception_digest,
        "type": "freeze_transition",
    }
    record["freeze_record_digest"] = _digest(record, "freeze_transition")
    _append_raw(paths["raw"], record)
    return frozen


def _integrity_with_io_backoff(
    paths: Mapping[str, Path],
    store: SQLiteShadowStore,
    *,
    context: str,
):
    last_digest: str | None = None
    for attempt in range(len(IO_BACKOFF_SECONDS) + 1):
        try:
            return store.integrity_check(), last_digest
        except OSError as exc:
            last_digest = _exception_evidence(
                paths,
                context=context,
                exc=exc,
                attempt=attempt + 1,
            )
            if attempt == len(IO_BACKOFF_SECONDS):
                return None, last_digest
            time.sleep(IO_BACKOFF_SECONDS[attempt])
    raise AssertionError("unreachable integrity retry state")


def _restore_with_io_backoff(
    paths: Mapping[str, Path],
    store: SQLiteShadowStore,
    *,
    context: str,
):
    last_digest: str | None = None
    for attempt in range(len(IO_BACKOFF_SECONDS) + 1):
        try:
            return _restore(store), None, last_digest
        except OSError as exc:
            last_digest = _exception_evidence(
                paths,
                context=context,
                exc=exc,
                attempt=attempt + 1,
            )
            integrity, integrity_exc = _integrity_with_io_backoff(
                paths,
                store,
                context=f"{context}:integrity_after_oserror",
            )
            if integrity_exc is not None:
                last_digest = integrity_exc
            if integrity is not None and not integrity.valid:
                return None, "unrecoverable_corruption", last_digest
            if attempt == len(IO_BACKOFF_SECONDS):
                return None, IO_FAILURE, last_digest
            time.sleep(IO_BACKOFF_SECONDS[attempt])
        except (ShadowPersistenceError, ValueError, HabitatError) as exc:
            last_digest = _exception_evidence(
                paths,
                context=context,
                exc=exc,
                attempt=attempt + 1,
            )
            integrity, integrity_exc = _integrity_with_io_backoff(
                paths,
                store,
                context=f"{context}:integrity_after_restore_error",
            )
            if integrity_exc is not None:
                last_digest = integrity_exc
            if integrity is None:
                return None, IO_FAILURE, last_digest
            if not integrity.valid:
                return None, "unrecoverable_corruption", last_digest
            return None, "recovery_digest_mismatch", last_digest
    raise AssertionError("unreachable restore retry state")


def _classify_nonretryable_failure(
    paths: Mapping[str, Path],
    store: SQLiteShadowStore,
    *,
    context: str,
    trigger_exception_digest: str,
) -> tuple[str, str]:
    integrity, integrity_exc = _integrity_with_io_backoff(
        paths,
        store,
        context=f"{context}:integrity_classification",
    )
    digest = integrity_exc or trigger_exception_digest
    if integrity is None:
        return IO_FAILURE, digest
    if not integrity.valid:
        return "unrecoverable_corruption", digest
    return "recovery_digest_mismatch", digest


def _notify_once(paths: Mapping[str, Path], state: WindowState) -> WindowState:
    if not state.frozen or state.notification_sent:
        return state
    message = f"EVE M2-E shadow window frozen: {state.freeze_reason}"
    try:
        subprocess.run(
            ["termux-notification", "--id", "m2e-window", "--title", "EVE window halted", "--content", message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    updated = mark_notification_sent(state)
    _append_raw(
        paths["raw"],
        {
            "notification_attempted": True,
            "reason": state.freeze_reason,
            "schema_version": RAW_EVIDENCE_SCHEMA,
            "state_digest": state.digest,
            "type": "watchdog_notification_once",
        },
    )
    return updated


def _recover(
    paths: Mapping[str, Path],
    state: WindowState,
    config: WindowConfig,
    *,
    boot: bool,
):
    store = _store(paths, config)
    integrity, integrity_exc = _integrity_with_io_backoff(
        paths,
        store,
        context="startup_integrity",
    )
    if integrity is None:
        state = _freeze_with_evidence(
            paths,
            state,
            IO_FAILURE,
            context="startup_integrity",
            trigger_exception_digest=integrity_exc,
        )
        return state, store, None
    state = apply_watchdog(
        state,
        config=config,
        disk_bytes=_disk_bytes(paths["root"]),
        integrity_valid=integrity.valid,
    )
    if not integrity.valid and not state.frozen:
        state = _freeze_with_evidence(paths, state, "unrecoverable_corruption", context="startup_integrity")
    if state.frozen:
        return state, store, None

    restored, failure_reason, failure_digest = _restore_with_io_backoff(
        paths,
        store,
        context="startup_restore",
    )
    if restored is None:
        state = _freeze_with_evidence(
            paths,
            state,
            failure_reason or IO_FAILURE,
            context="startup_restore",
            trigger_exception_digest=failure_digest,
        )
        return state, store, None

    unclean = paths["running"].exists()
    persisted_count = integrity.event_count
    if persisted_count == state.event_count + 1 and unclean:
        events = store.events(stream_id=BOUNDED_STREAM)
        pending = events[-1]
        expected_pending = _event(persisted_count)
        if pending != expected_pending:
            state = _freeze_with_evidence(paths, state, "recovery_digest_mismatch", context="pending_commit_reconcile")
        else:
            state = replace(
                state,
                cumulative_runtime_seconds=max(
                    state.cumulative_runtime_seconds,
                    persisted_count * STIMULUS_INTERVAL_SECONDS,
                ),
                event_count=persisted_count,
                hour_counts=_hour_counts_for_events(persisted_count, config),
                expected_recovery_digest=restored.state_digest,
                last_recovery_digest=restored.state_digest,
            )
            _append_raw(
                paths["raw"],
                {
                    "event_id": pending.event_id,
                    "recovered_digest": restored.state_digest,
                    "schema_version": RAW_EVIDENCE_SCHEMA,
                    "sequence": persisted_count,
                    "type": "deterministic_pending_commit_reconciled",
                },
            )
    elif persisted_count != state.event_count:
        state = _freeze_with_evidence(paths, state, "recovery_digest_mismatch", context="event_count_reconcile")

    expected = state.expected_recovery_digest or restored.state_digest
    if unclean and not state.frozen:
        state = record_recovery(
            state,
            expected_digest=expected,
            recovered_digest=restored.state_digest,
        )
        _append_raw(
            paths["raw"],
            {
                "expected_digest": expected,
                "matched": expected == restored.state_digest,
                "recovered_digest": restored.state_digest,
                "schema_version": RAW_EVIDENCE_SCHEMA,
                "type": "unclean_exit_recovery",
            },
        )
    elif not state.frozen:
        if state.expected_recovery_digest is not None and expected != restored.state_digest:
            state = _freeze_with_evidence(paths, state, "recovery_digest_mismatch", context="expected_restore_digest")
        else:
            state = replace(
                state,
                expected_recovery_digest=expected,
                last_recovery_digest=restored.state_digest,
            )
    if boot and not state.frozen:
        state = record_power_cycle(state, local_date=date.today().isoformat())
        _append_raw(
            paths["raw"],
            {
                "power_cycle_count": state.power_cycle_count,
                "recovered_digest": restored.state_digest,
                "schema_version": RAW_EVIDENCE_SCHEMA,
                "type": "power_cycle_continuity",
            },
        )
    paths["running"].unlink(missing_ok=True)
    state = _notify_once(paths, state)
    _save_state(paths, state)
    return state, store, restored


def _resume_denied(
    paths: Mapping[str, Path],
    state: WindowState,
    *,
    freeze_record_digest: str,
    reason: str,
    integrity_report_digest: str | None = None,
    recovered_digest: str | None = None,
) -> int:
    _append_raw(
        paths["raw"],
        {
            "freeze_record_digest": freeze_record_digest,
            "integrity_report_digest": integrity_report_digest,
            "reason": reason,
            "recovered_digest": recovered_digest,
            "schema_version": RAW_EVIDENCE_SCHEMA,
            "state_digest": state.digest,
            "type": "freeze_reviewed_resume_denied",
        },
    )
    return 22


def resume_reviewed(*, private_root: Path, window_id: str) -> int:
    """Clear a freeze only after recomputable store/recovery checks pass."""

    _enable_io_failure_reason()
    paths = _ensure_private_root(private_root)
    config = WindowConfig()
    state = _load_state(paths, window_id=window_id)
    if state.sealed or not state.frozen:
        return 2

    freeze_record: dict[str, Any] = {
        "freeze_reason": state.freeze_reason,
        "frozen_state_digest": state.digest,
        "schema_version": RAW_EVIDENCE_SCHEMA,
        "type": "freeze_review_snapshot",
    }
    freeze_record["freeze_record_digest"] = _digest(freeze_record, "freeze_review_snapshot")
    _append_raw(paths["raw"], freeze_record)
    freeze_record_digest = freeze_record["freeze_record_digest"]

    store = _store(paths, config)
    integrity, integrity_exc = _integrity_with_io_backoff(
        paths,
        store,
        context="reviewed_resume_integrity",
    )
    if integrity is None:
        return _resume_denied(
            paths,
            state,
            freeze_record_digest=freeze_record_digest,
            reason=IO_FAILURE,
        )
    if not integrity.valid:
        return _resume_denied(
            paths,
            state,
            freeze_record_digest=freeze_record_digest,
            reason="unrecoverable_corruption",
            integrity_report_digest=integrity.report_digest,
        )

    persisted_count = integrity.event_count
    reconciled_pending = False
    if persisted_count == state.event_count:
        pass
    elif persisted_count == state.event_count + 1:
        events = store.events(stream_id=BOUNDED_STREAM)
        if not events or events[-1] != _event(persisted_count):
            return _resume_denied(
                paths,
                state,
                freeze_record_digest=freeze_record_digest,
                reason="pending_commit_digest_mismatch",
                integrity_report_digest=integrity.report_digest,
            )
        reconciled_pending = True
    else:
        return _resume_denied(
            paths,
            state,
            freeze_record_digest=freeze_record_digest,
            reason="event_count_mismatch",
            integrity_report_digest=integrity.report_digest,
        )

    restored, failure_reason, failure_digest = _restore_with_io_backoff(
        paths,
        store,
        context="reviewed_resume_restore",
    )
    if restored is None:
        return _resume_denied(
            paths,
            state,
            freeze_record_digest=freeze_record_digest,
            reason=failure_reason or IO_FAILURE,
            integrity_report_digest=integrity.report_digest,
        )

    if not reconciled_pending and state.expected_recovery_digest is not None:
        if restored.state_digest != state.expected_recovery_digest:
            return _resume_denied(
                paths,
                state,
                freeze_record_digest=freeze_record_digest,
                reason="recovery_digest_mismatch",
                integrity_report_digest=integrity.report_digest,
                recovered_digest=restored.state_digest,
            )

    updated = state
    if reconciled_pending:
        updated = replace(
            updated,
            cumulative_runtime_seconds=max(
                updated.cumulative_runtime_seconds,
                persisted_count * STIMULUS_INTERVAL_SECONDS,
            ),
            event_count=persisted_count,
            hour_counts=_hour_counts_for_events(persisted_count, config),
        )
    updated = replace(
        updated,
        expected_recovery_digest=restored.state_digest,
        last_recovery_digest=restored.state_digest,
        freeze_reason=None,
        notification_sent=False,
    )

    resume_record: dict[str, Any] = {
        "after_state_digest": updated.digest,
        "before_state_digest": state.digest,
        "freeze_record_digest": freeze_record_digest,
        "integrity_report_digest": integrity.report_digest,
        "reconciled_pending_commit": reconciled_pending,
        "recovered_digest": restored.state_digest,
        "schema_version": RAW_EVIDENCE_SCHEMA,
        "trigger_exception_digest": failure_digest or integrity_exc,
        "type": "freeze_reviewed_resume",
    }
    resume_record["resume_record_digest"] = _digest(resume_record, "freeze_reviewed_resume")
    _append_raw(paths["raw"], resume_record)
    paths["running"].unlink(missing_ok=True)
    _save_state(paths, updated)
    return 0


def _seal(paths: Mapping[str, Path], state: WindowState, config: WindowConfig) -> None:
    evidence = evidence_record(state, config=config)
    public = {
        "authority": SHADOW_AUTHORITY,
        "checks": evidence["checks"],
        "config_digest": evidence["config_digest"],
        "cutover_authorized": False,
        "evidence_digest": evidence["evidence_digest"],
        "legacy_runtime_authoritative": True,
        "m3_authority_open": False,
        "raw_evidence_sha256": hashlib.sha256(paths["raw"].read_bytes()).hexdigest()
        if paths["raw"].exists()
        else hashlib.sha256(b"").hexdigest(),
        "raw_location": "local_private_companion_only",
        "schema_version": SEAL_SCHEMA,
        "seal_reason": state.seal_reason,
        "state_digest": state.digest,
    }
    public["seal_digest"] = _digest(public, "m2_e_window_seal")
    _atomic_json(paths["seal"], public)


def _handle_signal(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def run_worker(*, private_root: Path, window_id: str, boot: bool) -> int:
    global _STOP
    _enable_io_failure_reason()
    _STOP = False
    paths = _ensure_private_root(private_root)
    config = WindowConfig()
    state = _load_state(paths, window_id=window_id)
    state, store, restored = _recover(paths, state, config, boot=boot)
    if state.frozen:
        _save_state(paths, _notify_once(paths, state))
        return 20
    if state.sealed:
        _seal(paths, state, config)
        return 0
    if restored is None:
        raise HabitatError("recovery did not produce a verified state")

    paths["running"].write_text(f"pid={os.getpid()}\n", encoding="utf-8")
    os.chmod(paths["running"], 0o600)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    last = time.monotonic()
    checkpoint_elapsed = watchdog_elapsed = 0
    try:
        while not _STOP:
            time.sleep(1)
            now = time.monotonic()
            elapsed = max(0, int(now - last))
            if elapsed == 0:
                continue
            last += elapsed
            checkpoint_elapsed += elapsed
            watchdog_elapsed += elapsed
            state = advance_runtime(
                state,
                elapsed_seconds=elapsed,
                local_date=date.today().isoformat(),
            )

            due_count = min(
                config.event_quota,
                state.cumulative_runtime_seconds // STIMULUS_INTERVAL_SECONDS,
            )
            while not state.frozen and not state.sealed and state.event_count < due_count:
                tentative = record_discrete_stimulus(state, config=config)
                if tentative.frozen:
                    state = tentative
                    break
                sequence = tentative.event_count
                try:
                    receipt = store.append(_event(sequence))
                    restored_candidate, restore_failure, restore_exc = _restore_with_io_backoff(
                        paths,
                        store,
                        context="post_append_restore",
                    )
                    if restored_candidate is None:
                        state = _freeze_with_evidence(
                            paths,
                            state,
                            restore_failure or IO_FAILURE,
                            context="post_append_restore",
                            trigger_exception_digest=restore_exc,
                        )
                        break
                    restored = restored_candidate
                    state = replace(
                        tentative,
                        expected_recovery_digest=restored.state_digest,
                        last_recovery_digest=restored.state_digest,
                    )
                    _append_raw(
                        paths["raw"],
                        {
                            "append_transition_hash": receipt.transition_hash,
                            "envelope_digest": receipt.envelope_digest,
                            "event_id": receipt.event_id,
                            "recovery_digest": restored.state_digest,
                            "schema_version": RAW_EVIDENCE_SCHEMA,
                            "sequence": sequence,
                            "type": "scripted_discrete_stimulus",
                        },
                    )
                    _save_state(paths, state)
                    if store.snapshot_due(BOUNDED_STREAM):
                        store.write_snapshot(
                            snapshot_id=f"m2e-habitat:snapshot:{sequence:06d}",
                            stream_id=BOUNDED_STREAM,
                            through_sequence=sequence,
                            state=restored.state.snapshot,
                            state_schema_version=STATE_SCHEMA_VERSION,
                        )
                    if sequence % BACKUP_INTERVAL_EVENTS == 0:
                        ordinal = sequence // BACKUP_INTERVAL_EVENTS
                        store.create_backup(paths["backups"], backup_ordinal=ordinal)
                except StoragePolicyExceeded as exc:
                    exc_digest = _exception_evidence(
                        paths,
                        context="append_snapshot_backup",
                        exc=exc,
                        attempt=1,
                    )
                    state = _freeze_with_evidence(
                        paths,
                        state,
                        "disk_budget_exceeded",
                        context="append_snapshot_backup",
                        trigger_exception_digest=exc_digest,
                    )
                except OSError as exc:
                    exc_digest = _exception_evidence(
                        paths,
                        context="append_snapshot_backup",
                        exc=exc,
                        attempt=1,
                    )
                    integrity, integrity_exc = _integrity_with_io_backoff(
                        paths,
                        store,
                        context="append_snapshot_backup:io_probe",
                    )
                    reason = (
                        "unrecoverable_corruption"
                        if integrity is not None and not integrity.valid
                        else IO_FAILURE
                    )
                    state = _freeze_with_evidence(
                        paths,
                        state,
                        reason,
                        context="append_snapshot_backup",
                        trigger_exception_digest=integrity_exc or exc_digest,
                    )
                except (ShadowPersistenceError, ValueError, HabitatError) as exc:
                    exc_digest = _exception_evidence(
                        paths,
                        context="append_snapshot_backup",
                        exc=exc,
                        attempt=1,
                    )
                    reason, trigger = _classify_nonretryable_failure(
                        paths,
                        store,
                        context="append_snapshot_backup",
                        trigger_exception_digest=exc_digest,
                    )
                    state = _freeze_with_evidence(
                        paths,
                        state,
                        reason,
                        context="append_snapshot_backup",
                        trigger_exception_digest=trigger,
                    )

            if watchdog_elapsed >= WATCHDOG_INTERVAL_SECONDS and not state.frozen:
                watchdog_elapsed = 0
                integrity, integrity_exc = _integrity_with_io_backoff(
                    paths,
                    store,
                    context="watchdog_integrity",
                )
                if integrity is None:
                    state = _freeze_with_evidence(
                        paths,
                        state,
                        IO_FAILURE,
                        context="watchdog_integrity",
                        trigger_exception_digest=integrity_exc,
                    )
                else:
                    state = apply_watchdog(
                        state,
                        config=config,
                        disk_bytes=_disk_bytes(private_root),
                        integrity_valid=integrity.valid,
                    )
                    if not integrity.valid and state.frozen:
                        _append_raw(
                            paths["raw"],
                            {
                                "freeze_reason": "unrecoverable_corruption",
                                "integrity_report_digest": integrity.report_digest,
                                "schema_version": RAW_EVIDENCE_SCHEMA,
                                "state_digest": state.digest,
                                "type": "integrity_failure_freeze_evidence",
                            },
                        )
                    if not state.frozen:
                        watchdog_restore, restore_failure, restore_exc = _restore_with_io_backoff(
                            paths,
                            store,
                            context="watchdog_restore",
                        )
                        if watchdog_restore is None:
                            state = _freeze_with_evidence(
                                paths,
                                state,
                                restore_failure or IO_FAILURE,
                                context="watchdog_restore",
                                trigger_exception_digest=restore_exc,
                            )
                        else:
                            expected = state.expected_recovery_digest or watchdog_restore.state_digest
                            if watchdog_restore.state_digest != expected:
                                state = _freeze_with_evidence(
                                    paths,
                                    state,
                                    "recovery_digest_mismatch",
                                    context="watchdog_restore_digest",
                                )

            state = maybe_seal(state, config=config, local_date=date.today().isoformat())
            if state.frozen:
                state = _notify_once(paths, state)
            if checkpoint_elapsed >= CHECKPOINT_INTERVAL_SECONDS or state.frozen or state.sealed:
                checkpoint_elapsed = 0
                _save_state(paths, state)
            if state.sealed:
                _seal(paths, state, config)
                paths["running"].unlink(missing_ok=True)
                return 0
            if state.frozen:
                paths["running"].unlink(missing_ok=True)
                return 20
    finally:
        _save_state(paths, state)
        if _STOP:
            paths["running"].unlink(missing_ok=True)
    return 0


def status(*, private_root: Path, window_id: str) -> str:
    _enable_io_failure_reason()
    paths = _ensure_private_root(private_root)
    state = _load_state(paths, window_id=window_id)
    return one_line_status(state, config=WindowConfig())


def seal_now(*, private_root: Path, window_id: str) -> int:
    _enable_io_failure_reason()
    paths = _ensure_private_root(private_root)
    config = WindowConfig()
    state = _load_state(paths, window_id=window_id)
    state = maybe_seal(state, config=config, local_date=date.today().isoformat())
    _save_state(paths, state)
    if state.sealed:
        _seal(paths, state, config)
        return 0
    return 2


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--private-root",
        default=os.environ.get("EVE_M2E_PRIVATE_ROOT", str(DEFAULT_PRIVATE_ROOT)),
    )
    parser.add_argument("--window-id", default="m2-e:phone-habitat:v1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--boot", action="store_true")
    subparsers.add_parser("status")
    subparsers.add_parser("seal")
    resume = subparsers.add_parser("resume")
    resume.add_argument("--reviewed", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.private_root).expanduser()
    if args.command == "run":
        return run_worker(private_root=root, window_id=args.window_id, boot=args.boot)
    if args.command == "status":
        print(status(private_root=root, window_id=args.window_id))
        return 0
    if args.command == "resume":
        if not args.reviewed:
            return 2
        return resume_reviewed(private_root=root, window_id=args.window_id)
    return seal_now(private_root=root, window_id=args.window_id)


if __name__ == "__main__":
    raise SystemExit(main())
