"""M2-D bounded recovery and rollback rehearsal evidence.

The harness runs only when a caller explicitly invokes :func:`run_recovery_rehearsal`
with a new disposable workspace and a bounded M1-B/C event window. Import and
construction perform no I/O. The resulting packet is machine evidence only: it
cannot grant recovery authority, activate production dual read, change defaults,
transfer legacy authority, or authorize cutover.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY, canonical_json_object
from core.m2_c_migration import StateEvidence
from core.shadow_observer import ACTIVATION_LEARN_PAIR_TARGET
from core.shadow_projection import (
    PROJECTION_SCHEMA_VERSION,
    ActivationLearnPairShadowState,
    ShadowProjectionError,
    replay_activation_learn_pair,
)
from core.sqlite_shadow_store import (
    PersistedEventCorruption,
    SQLiteShadowStore,
    ShadowStoragePolicy,
)

OBSERVATION_WINDOW_SCHEMA_VERSION = "eve.m2-d-observation-window.v1"
SCENARIO_EVIDENCE_SCHEMA_VERSION = "eve.m2-d-scenario-evidence.v1"
REHEARSAL_PACKET_SCHEMA_VERSION = "eve.m2-d-rehearsal-packet.v1"
REHEARSAL_AUTHORITY = "rehearsal_only"
HUMAN_REVIEW_REQUIRED = "required_not_performed"

ACCEPTED_M2_C_PR = 164
ACCEPTED_M2_C_HEAD = "3e7e484e91460f5cc46e7bc2e67bac4a5bb51d14"
ACCEPTED_M2_C_WORKFLOW = 29912952289
ACCEPTED_M2_C_ARTIFACT_SHA256 = (
    "af4f69b75c1316033b695b79362f2968058f31a5e37785f82b5f30626db90c3a"
)

REQUIRED_SCENARIOS = (
    "snapshot_restore",
    "full_replay_equivalence",
    "corrupt_snapshot_fallback",
    "corrupt_event_fail_closed",
    "forced_termination",
    "rollback_rehearsal",
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SNAPSHOT_UPDATE_TRIGGER = (
    "CREATE TRIGGER snapshots_no_update BEFORE UPDATE ON snapshots "
    "BEGIN SELECT RAISE(ABORT,'append-only snapshots'); END"
)
_EVENT_UPDATE_TRIGGER = (
    "CREATE TRIGGER events_no_update BEFORE UPDATE ON events "
    "BEGIN SELECT RAISE(ABORT,'append-only events'); END"
)


class M2DRehearsalError(ValueError):
    """Malformed or out-of-scope M2-D rehearsal input/evidence."""


def _canon(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_object(value, field=field)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Mapping[str, Any], field: str) -> str:
    return _sha_text(_canon(value, field))


def _require_digest(value: str, field: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise M2DRehearsalError(f"{field} must be a lowercase SHA-256 digest")


def _require_commit(value: str, field: str) -> None:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise M2DRehearsalError(f"{field} must be a lowercase 40-character commit SHA")


def _require_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise M2DRehearsalError(f"{field} must be a canonical identifier")


def _state_record(value: StateEvidence) -> dict[str, Any]:
    if not isinstance(value, StateEvidence):
        raise M2DRehearsalError("expected_final_state must be StateEvidence")
    return {
        "manifest_digest": value.manifest_digest,
        "manifest_json": value.manifest_json,
        "schema_version": value.schema_version,
        "snapshot_digest": value.snapshot_digest,
        "snapshot_json": value.snapshot_json,
    }


@dataclass(frozen=True, slots=True)
class ObservationWindowSpec:
    window_id: str
    baseline_event_count: int
    snapshot_sequence: int
    expected_final_state: StateEvidence
    m2_c_pr: int = ACCEPTED_M2_C_PR
    m2_c_head: str = ACCEPTED_M2_C_HEAD
    m2_c_workflow: int = ACCEPTED_M2_C_WORKFLOW
    m2_c_artifact_sha256: str = ACCEPTED_M2_C_ARTIFACT_SHA256
    stream_id: str = ACTIVATION_LEARN_PAIR_TARGET.stream_id
    state_schema_version: str = PROJECTION_SCHEMA_VERSION
    schema_version: str = OBSERVATION_WINDOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_identifier(self.window_id, "window_id")
        if (
            isinstance(self.baseline_event_count, bool)
            or not isinstance(self.baseline_event_count, int)
            or self.baseline_event_count < 2
        ):
            raise M2DRehearsalError("baseline_event_count must be at least two")
        if (
            isinstance(self.snapshot_sequence, bool)
            or not isinstance(self.snapshot_sequence, int)
            or not 1 <= self.snapshot_sequence < self.baseline_event_count
        ):
            raise M2DRehearsalError(
                "snapshot_sequence must be within the baseline event window and leave replay suffix"
            )
        _state_record(self.expected_final_state)
        fixed = (
            (self.m2_c_pr, ACCEPTED_M2_C_PR),
            (self.m2_c_head, ACCEPTED_M2_C_HEAD),
            (self.m2_c_workflow, ACCEPTED_M2_C_WORKFLOW),
            (self.m2_c_artifact_sha256, ACCEPTED_M2_C_ARTIFACT_SHA256),
            (self.stream_id, ACTIVATION_LEARN_PAIR_TARGET.stream_id),
            (self.state_schema_version, PROJECTION_SCHEMA_VERSION),
            (self.schema_version, OBSERVATION_WINDOW_SCHEMA_VERSION),
        )
        if any(actual != expected for actual, expected in fixed):
            raise M2DRehearsalError("observation window is not pinned to accepted M2-C scope")
        _require_commit(self.m2_c_head, "m2_c_head")
        _require_digest(self.m2_c_artifact_sha256, "m2_c_artifact_sha256")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "baseline_event_count": self.baseline_event_count,
            "expected_final_state": _state_record(self.expected_final_state),
            "m2_c_artifact_sha256": self.m2_c_artifact_sha256,
            "m2_c_head": self.m2_c_head,
            "m2_c_pr": self.m2_c_pr,
            "m2_c_workflow": self.m2_c_workflow,
            "schema_version": self.schema_version,
            "snapshot_sequence": self.snapshot_sequence,
            "state_schema_version": self.state_schema_version,
            "stream_id": self.stream_id,
            "window_id": self.window_id,
        }

    @property
    def digest(self) -> str:
        return _digest(self.canonical_record, "m2_d_observation_window")


@dataclass(frozen=True, slots=True)
class ScenarioEvidence:
    scenario_id: str
    passed: bool
    checks_json: str
    observations_json: str
    before_integrity_digest: str
    after_integrity_digest: str
    transition_hash: str
    evidence_digest: str
    schema_version: str = SCENARIO_EVIDENCE_SCHEMA_VERSION
    authority: str = REHEARSAL_AUTHORITY
    runtime_integrated: bool = False
    legacy_authority_retained: bool = True
    authoritative_recovery: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        if self.scenario_id not in REQUIRED_SCENARIOS:
            raise M2DRehearsalError("unknown M2-D scenario")
        if not isinstance(self.passed, bool):
            raise M2DRehearsalError("scenario passed must be boolean")
        for field in (
            "before_integrity_digest",
            "after_integrity_digest",
            "transition_hash",
            "evidence_digest",
        ):
            _require_digest(getattr(self, field), field)
        try:
            checks = json.loads(self.checks_json)
            observations = json.loads(self.observations_json)
        except json.JSONDecodeError as exc:
            raise M2DRehearsalError("scenario evidence JSON is malformed") from exc
        if not isinstance(checks, dict) or not checks or not all(
            isinstance(key, str) and isinstance(value, bool)
            for key, value in checks.items()
        ):
            raise M2DRehearsalError("scenario checks must be a non-empty boolean object")
        if not isinstance(observations, dict):
            raise M2DRehearsalError("scenario observations must be an object")
        if _canon(checks, "m2_d_checks") != self.checks_json:
            raise M2DRehearsalError("scenario checks must be canonical JSON")
        if _canon(observations, "m2_d_observations") != self.observations_json:
            raise M2DRehearsalError("scenario observations must be canonical JSON")
        if self.passed != all(checks.values()):
            raise M2DRehearsalError("scenario result disagrees with raw checks")
        fixed = (
            (self.schema_version, SCENARIO_EVIDENCE_SCHEMA_VERSION),
            (self.authority, REHEARSAL_AUTHORITY),
            (self.runtime_integrated, False),
            (self.legacy_authority_retained, True),
            (self.authoritative_recovery, False),
            (self.cutover_authorized, False),
        )
        if any(actual != expected for actual, expected in fixed):
            raise M2DRehearsalError("scenario evidence changed authority boundary")
        if _digest(self.transition_material, "m2_d_scenario_transition") != self.transition_hash:
            raise M2DRehearsalError("scenario transition hash mismatch")
        if _digest(self.evidence_material, "m2_d_scenario_evidence") != self.evidence_digest:
            raise M2DRehearsalError("scenario evidence digest mismatch")

    @property
    def checks(self) -> dict[str, bool]:
        return json.loads(self.checks_json)

    @property
    def observations(self) -> dict[str, Any]:
        return json.loads(self.observations_json)

    @property
    def transition_material(self) -> dict[str, Any]:
        return {
            "after_integrity_digest": self.after_integrity_digest,
            "before_integrity_digest": self.before_integrity_digest,
            "checks": self.checks,
            "observations": self.observations,
            "scenario_id": self.scenario_id,
        }

    @property
    def evidence_material(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "authoritative_recovery": self.authoritative_recovery,
            "cutover_authorized": self.cutover_authorized,
            "legacy_authority_retained": self.legacy_authority_retained,
            "passed": self.passed,
            "runtime_integrated": self.runtime_integrated,
            "schema_version": self.schema_version,
            "transition_hash": self.transition_hash,
            **self.transition_material,
        }

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {**self.evidence_material, "evidence_digest": self.evidence_digest}

    @classmethod
    def create(
        cls,
        *,
        scenario_id: str,
        checks: Mapping[str, bool],
        observations: Mapping[str, Any],
        before_integrity_digest: str,
        after_integrity_digest: str,
    ) -> "ScenarioEvidence":
        checks_json = _canon(checks, "m2_d_checks")
        observations_json = _canon(observations, "m2_d_observations")
        transition_material = {
            "after_integrity_digest": after_integrity_digest,
            "before_integrity_digest": before_integrity_digest,
            "checks": json.loads(checks_json),
            "observations": json.loads(observations_json),
            "scenario_id": scenario_id,
        }
        transition_hash = _digest(transition_material, "m2_d_scenario_transition")
        passed = all(checks.values())
        evidence_material = {
            "authority": REHEARSAL_AUTHORITY,
            "authoritative_recovery": False,
            "cutover_authorized": False,
            "legacy_authority_retained": True,
            "passed": passed,
            "runtime_integrated": False,
            "schema_version": SCENARIO_EVIDENCE_SCHEMA_VERSION,
            "transition_hash": transition_hash,
            **transition_material,
        }
        return cls(
            scenario_id=scenario_id,
            passed=passed,
            checks_json=checks_json,
            observations_json=observations_json,
            before_integrity_digest=before_integrity_digest,
            after_integrity_digest=after_integrity_digest,
            transition_hash=transition_hash,
            evidence_digest=_digest(evidence_material, "m2_d_scenario_evidence"),
        )


@dataclass(frozen=True, slots=True)
class M2DRehearsalPacket:
    window: ObservationWindowSpec
    scenarios: tuple[ScenarioEvidence, ...]
    passed_count: int
    failed_count: int
    machine_passed: bool
    eligible_for_human_review: bool
    packet_digest: str
    schema_version: str = REHEARSAL_PACKET_SCHEMA_VERSION
    human_review_status: str = HUMAN_REVIEW_REQUIRED
    human_accepted: bool = False
    authority: str = REHEARSAL_AUTHORITY
    shadow_authority: str = SHADOW_AUTHORITY
    legacy_authority_retained: bool = True
    runtime_integrated: bool = False
    production_dual_read: bool = False
    authoritative_recovery: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.window, ObservationWindowSpec):
            raise M2DRehearsalError("packet requires ObservationWindowSpec")
        if not isinstance(self.scenarios, tuple) or any(
            not isinstance(value, ScenarioEvidence) for value in self.scenarios
        ):
            raise M2DRehearsalError("packet scenarios must be immutable evidence values")
        if tuple(value.scenario_id for value in self.scenarios) != REQUIRED_SCENARIOS:
            raise M2DRehearsalError("packet scenario set/order is incomplete")
        passed = sum(value.passed for value in self.scenarios)
        failed = len(self.scenarios) - passed
        if (self.passed_count, self.failed_count) != (passed, failed):
            raise M2DRehearsalError("packet counts disagree with scenarios")
        computed = failed == 0
        if self.machine_passed != computed or self.eligible_for_human_review != computed:
            raise M2DRehearsalError("packet status disagrees with scenario evidence")
        fixed = (
            (self.schema_version, REHEARSAL_PACKET_SCHEMA_VERSION),
            (self.human_review_status, HUMAN_REVIEW_REQUIRED),
            (self.human_accepted, False),
            (self.authority, REHEARSAL_AUTHORITY),
            (self.shadow_authority, SHADOW_AUTHORITY),
            (self.legacy_authority_retained, True),
            (self.runtime_integrated, False),
            (self.production_dual_read, False),
            (self.authoritative_recovery, False),
            (self.cutover_authorized, False),
        )
        if any(actual != expected for actual, expected in fixed):
            raise M2DRehearsalError("packet cannot self-promote or grant authority")
        _require_digest(self.packet_digest, "packet_digest")
        if _digest(self.packet_material, "m2_d_rehearsal_packet") != self.packet_digest:
            raise M2DRehearsalError("packet digest mismatch")

    @property
    def packet_material(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "authoritative_recovery": self.authoritative_recovery,
            "cutover_authorized": self.cutover_authorized,
            "eligible_for_human_review": self.eligible_for_human_review,
            "failed_count": self.failed_count,
            "human_accepted": self.human_accepted,
            "human_review_status": self.human_review_status,
            "legacy_authority_retained": self.legacy_authority_retained,
            "machine_passed": self.machine_passed,
            "passed_count": self.passed_count,
            "production_dual_read": self.production_dual_read,
            "runtime_integrated": self.runtime_integrated,
            "scenarios": [value.canonical_record for value in self.scenarios],
            "schema_version": self.schema_version,
            "shadow_authority": self.shadow_authority,
            "window": self.window.canonical_record,
            "window_digest": self.window.digest,
        }

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {**self.packet_material, "packet_digest": self.packet_digest}

    @classmethod
    def create(
        cls,
        *,
        window: ObservationWindowSpec,
        scenarios: Sequence[ScenarioEvidence],
    ) -> "M2DRehearsalPacket":
        immutable = tuple(scenarios)
        passed = sum(value.passed for value in immutable)
        failed = len(immutable) - passed
        material = {
            "authority": REHEARSAL_AUTHORITY,
            "authoritative_recovery": False,
            "cutover_authorized": False,
            "eligible_for_human_review": failed == 0,
            "failed_count": failed,
            "human_accepted": False,
            "human_review_status": HUMAN_REVIEW_REQUIRED,
            "legacy_authority_retained": True,
            "machine_passed": failed == 0,
            "passed_count": passed,
            "production_dual_read": False,
            "runtime_integrated": False,
            "scenarios": [value.canonical_record for value in immutable],
            "schema_version": REHEARSAL_PACKET_SCHEMA_VERSION,
            "shadow_authority": SHADOW_AUTHORITY,
            "window": window.canonical_record,
            "window_digest": window.digest,
        }
        return cls(
            window=window,
            scenarios=immutable,
            passed_count=passed,
            failed_count=failed,
            machine_passed=failed == 0,
            eligible_for_human_review=failed == 0,
            packet_digest=_digest(material, "m2_d_rehearsal_packet"),
        )


def _validate_window_inputs(
    *,
    window: ObservationWindowSpec,
    baseline_events: Sequence[EventEnvelope],
    rollback_probe_event: EventEnvelope,
    initial_snapshot: Mapping[str, Any],
) -> tuple[tuple[EventEnvelope, ...], ActivationLearnPairShadowState]:
    if not isinstance(window, ObservationWindowSpec):
        raise M2DRehearsalError("window must be ObservationWindowSpec")
    events = tuple(baseline_events)
    if len(events) != window.baseline_event_count or any(
        not isinstance(value, EventEnvelope) for value in events
    ):
        raise M2DRehearsalError("baseline events do not match the observation window")
    if not isinstance(rollback_probe_event, EventEnvelope):
        raise M2DRehearsalError("rollback_probe_event must be EventEnvelope")
    if tuple(value.sequence for value in events) != tuple(range(1, len(events) + 1)):
        raise M2DRehearsalError("baseline event sequence must be contiguous and one-based")
    if any(value.stream_id != window.stream_id for value in events):
        raise M2DRehearsalError("baseline event escaped the accepted stream")
    if (
        rollback_probe_event.stream_id != window.stream_id
        or rollback_probe_event.sequence != len(events) + 1
    ):
        raise M2DRehearsalError("rollback probe must be the next event in the accepted stream")
    try:
        initial = ActivationLearnPairShadowState.from_initial_snapshot(initial_snapshot)
        final = replay_activation_learn_pair(initial, events)
        probe = replay_activation_learn_pair(final, (rollback_probe_event,))
    except (ShadowProjectionError, TypeError, ValueError) as exc:
        raise M2DRehearsalError("event window is not valid M1-C projection input") from exc
    if StateEvidence.from_snapshot(final.snapshot).snapshot_digest != window.expected_final_state.snapshot_digest:
        raise M2DRehearsalError("expected final state does not match baseline replay")
    if probe.digest == final.digest:
        raise M2DRehearsalError("rollback probe must change the bounded state")
    return events, initial


def _restore(store: SQLiteShadowStore, initial_snapshot: Mapping[str, Any]):
    initial = ActivationLearnPairShadowState.from_initial_snapshot(initial_snapshot)
    events_by_sequence = {
        envelope.sequence: envelope
        for envelope in store.events(stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id)
    }

    def reducer(
        state: ActivationLearnPairShadowState,
        envelope: EventEnvelope,
    ) -> ActivationLearnPairShadowState:
        if state.sequence == 0 and envelope.sequence > 1:
            previous = events_by_sequence.get(envelope.sequence - 1)
            if previous is None:
                raise M2DRehearsalError(
                    "snapshot replay boundary has no preceding accepted event"
                )
            state = ActivationLearnPairShadowState(
                calls=state.calls,
                learned=state.learned,
                sequence=previous.sequence,
                last_event_id=previous.event_id,
                last_event_digest=previous.digest,
            )
        return replay_activation_learn_pair(state, (envelope,))

    return store.restore_verified(
        stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id,
        initial_state=initial,
        reducer=reducer,
        state_to_mapping=lambda state: state.snapshot,
        state_from_mapping=ActivationLearnPairShadowState.from_initial_snapshot,
    )


def _copy_database(source: Path, target: Path) -> None:
    if target.exists():
        raise M2DRehearsalError("rehearsal database target already exists")
    shutil.copyfile(source, target)


def _remove_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _corrupt_snapshot(path: Path, snapshot_id: str) -> None:
    connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP TRIGGER snapshots_no_update")
        connection.execute(
            "UPDATE snapshots SET state_digest=? WHERE snapshot_id=?",
            ("0" * 64, snapshot_id),
        )
        connection.execute(_SNAPSHOT_UPDATE_TRIGGER)
        connection.commit()
    except sqlite3.DatabaseError:
        connection.rollback()
        raise
    finally:
        connection.close()


def _corrupt_event(path: Path, event_id: str) -> None:
    connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute(
            "UPDATE events SET envelope_digest=? WHERE event_id=?",
            ("0" * 64, event_id),
        )
        connection.execute(_EVENT_UPDATE_TRIGGER)
        connection.commit()
    except sqlite3.DatabaseError:
        connection.rollback()
        raise
    finally:
        connection.close()


def _force_uncommitted_process_exit(path: Path, *, ordinal: int, sequence: int) -> int:
    script = """
import os
import sqlite3
import sys
path, ordinal, sequence = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
connection.execute('PRAGMA synchronous=FULL')
connection.execute('BEGIN IMMEDIATE')
connection.execute(
    'INSERT INTO events(ordinal,event_id,stream_id,sequence,event_json,envelope_digest,event_bytes,previous_chain_digest,chain_digest) VALUES(?,?,?,?,?,?,?,?,?)',
    (ordinal, 'm2d:uncommitted', 'shadow:legacy.activation.learn_pair', sequence, '{}', '0' * 64, 2, '0' * 64, '0' * 64),
)
os._exit(97)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), str(ordinal), str(sequence)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return int(result.returncode)


def run_recovery_rehearsal(
    *,
    workspace: str | Path,
    window: ObservationWindowSpec,
    baseline_events: Sequence[EventEnvelope],
    rollback_probe_event: EventEnvelope,
    initial_snapshot: Mapping[str, Any],
) -> M2DRehearsalPacket:
    """Run all required M2-D scenarios in a new caller-selected sandbox.

    The function intentionally creates isolated SQLite files, injects corruption into
    copies, and replaces a disposable rollback target. It never discovers or opens a
    production path and never changes legacy/runtime authority.
    """

    root = Path(workspace)
    if str(workspace) == ":memory:" or root.name == "" or root.exists():
        raise M2DRehearsalError("workspace must be a new concrete path")
    events, initial = _validate_window_inputs(
        window=window,
        baseline_events=baseline_events,
        rollback_probe_event=rollback_probe_event,
        initial_snapshot=initial_snapshot,
    )
    root.mkdir(parents=True, exist_ok=False)

    policy = ShadowStoragePolicy(snapshot_interval_events=1, max_backups=2)
    baseline_path = root / "baseline.sqlite3"
    baseline = SQLiteShadowStore(baseline_path, policy=policy)
    baseline.initialize()
    prefix = events[: window.snapshot_sequence]
    suffix = events[window.snapshot_sequence :]
    baseline.append_many(prefix)
    checkpoint = replay_activation_learn_pair(
        ActivationLearnPairShadowState.from_initial_snapshot(initial.snapshot),
        prefix,
    )
    baseline.write_snapshot(
        snapshot_id="m2d:snapshot:good",
        stream_id=window.stream_id,
        through_sequence=window.snapshot_sequence,
        state=checkpoint.snapshot,
        state_schema_version=window.state_schema_version,
    )
    baseline.write_snapshot(
        snapshot_id="m2d:snapshot:newest",
        stream_id=window.stream_id,
        through_sequence=window.snapshot_sequence,
        state=checkpoint.snapshot,
        state_schema_version=window.state_schema_version,
    )
    baseline.append_many(suffix)
    baseline_integrity = baseline.integrity_check()
    if not baseline_integrity.valid:
        raise M2DRehearsalError("baseline rehearsal store failed integrity")
    baseline_restore = _restore(baseline, initial.snapshot)
    backup = baseline.create_backup(root / "backups", backup_ordinal=1)
    backup_path = Path(backup.backup_path)

    snapshot_restore = ScenarioEvidence.create(
        scenario_id="snapshot_restore",
        checks={
            "expected_final_state": baseline_restore.state_digest
            == window.expected_final_state.snapshot_digest,
            "repeated_restore_equal": baseline_restore.state_digest
            == baseline_restore.repeated_state_digest,
            "snapshot_selected": baseline_restore.snapshot_id == "m2d:snapshot:newest",
            "suffix_replayed": baseline_restore.replayed_event_count == len(suffix),
        },
        observations={
            "expected_state_digest": window.expected_final_state.snapshot_digest,
            "rejected_snapshot_ids": list(baseline_restore.rejected_snapshot_ids),
            "repeated_state_digest": baseline_restore.repeated_state_digest,
            "replayed_event_count": baseline_restore.replayed_event_count,
            "restore_state_digest": baseline_restore.state_digest,
            "restore_transition_hash": baseline_restore.transition_hash,
            "selected_snapshot_id": baseline_restore.snapshot_id,
        },
        before_integrity_digest=baseline_integrity.report_digest,
        after_integrity_digest=baseline.integrity_check().report_digest,
    )

    left = replay_activation_learn_pair(
        ActivationLearnPairShadowState.from_initial_snapshot(initial.snapshot), events
    )
    right = replay_activation_learn_pair(
        ActivationLearnPairShadowState.from_initial_snapshot(initial.snapshot), events
    )
    left_state_digest = StateEvidence.from_snapshot(left.snapshot).snapshot_digest
    right_state_digest = StateEvidence.from_snapshot(right.snapshot).snapshot_digest
    full_replay = ScenarioEvidence.create(
        scenario_id="full_replay_equivalence",
        checks={
            "expected_final_state": left_state_digest
            == window.expected_final_state.snapshot_digest,
            "repeated_digest_equal": left.digest == right.digest,
            "repeated_snapshot_equal": left.snapshot == right.snapshot,
            "restore_matches_full_replay": baseline_restore.state_digest
            == left_state_digest,
        },
        observations={
            "event_count": len(events),
            "expected_state_digest": window.expected_final_state.snapshot_digest,
            "first_projection_digest": left.digest,
            "first_state_digest": left_state_digest,
            "restore_state_digest": baseline_restore.state_digest,
            "second_projection_digest": right.digest,
            "second_state_digest": right_state_digest,
        },
        before_integrity_digest=baseline_integrity.report_digest,
        after_integrity_digest=baseline.integrity_check().report_digest,
    )

    corrupt_snapshot_path = root / "corrupt-snapshot.sqlite3"
    _copy_database(backup_path, corrupt_snapshot_path)
    corrupt_snapshot_store = SQLiteShadowStore(corrupt_snapshot_path, policy=policy)
    corrupt_snapshot_store.initialize()
    snapshot_before = corrupt_snapshot_store.integrity_check()
    _corrupt_snapshot(corrupt_snapshot_path, "m2d:snapshot:newest")
    snapshot_after = corrupt_snapshot_store.integrity_check()
    fallback_restore = _restore(corrupt_snapshot_store, initial.snapshot)
    corrupt_snapshot = ScenarioEvidence.create(
        scenario_id="corrupt_snapshot_fallback",
        checks={
            "corruption_visible": not snapshot_after.valid,
            "corrupt_snapshot_rejected": fallback_restore.rejected_snapshot_ids
            == ("m2d:snapshot:newest",),
            "fallback_selected": fallback_restore.snapshot_id == "m2d:snapshot:good",
            "fallback_state_matches": fallback_restore.state_digest
            == window.expected_final_state.snapshot_digest,
            "repeated_restore_equal": fallback_restore.state_digest
            == fallback_restore.repeated_state_digest,
        },
        observations={
            "after_errors": list(snapshot_after.errors),
            "expected_state_digest": window.expected_final_state.snapshot_digest,
            "fallback_state_digest": fallback_restore.state_digest,
            "rejected_snapshot_ids": list(fallback_restore.rejected_snapshot_ids),
            "selected_snapshot_id": fallback_restore.snapshot_id,
        },
        before_integrity_digest=snapshot_before.report_digest,
        after_integrity_digest=snapshot_after.report_digest,
    )

    corrupt_event_path = root / "corrupt-event.sqlite3"
    _copy_database(backup_path, corrupt_event_path)
    corrupt_event_store = SQLiteShadowStore(corrupt_event_path, policy=policy)
    corrupt_event_store.initialize()
    event_before = corrupt_event_store.integrity_check()
    _corrupt_event(corrupt_event_path, events[-1].event_id)
    event_after = corrupt_event_store.integrity_check()
    read_error = restore_error = None
    try:
        corrupt_event_store.events(stream_id=window.stream_id)
    except PersistedEventCorruption as exc:
        read_error = type(exc).__name__
    try:
        _restore(corrupt_event_store, initial.snapshot)
    except PersistedEventCorruption as exc:
        restore_error = type(exc).__name__
    event_after_read = corrupt_event_store.integrity_check()
    corrupt_event = ScenarioEvidence.create(
        scenario_id="corrupt_event_fail_closed",
        checks={
            "corruption_visible": not event_after.valid,
            "integrity_unchanged_by_failed_reads": event_after.report_digest
            == event_after_read.report_digest,
            "read_failed_closed": read_error == "PersistedEventCorruption",
            "restore_failed_closed": restore_error == "PersistedEventCorruption",
        },
        observations={
            "after_errors": list(event_after.errors),
            "read_error_type": read_error,
            "restore_error_type": restore_error,
        },
        before_integrity_digest=event_before.report_digest,
        after_integrity_digest=event_after_read.report_digest,
    )

    forced_path = root / "forced-termination.sqlite3"
    _copy_database(backup_path, forced_path)
    forced_store = SQLiteShadowStore(forced_path, policy=policy)
    forced_store.initialize()
    forced_before = forced_store.integrity_check()
    exit_code = _force_uncommitted_process_exit(
        forced_path,
        ordinal=window.baseline_event_count + 1,
        sequence=window.baseline_event_count + 1,
    )
    forced_reopened = SQLiteShadowStore(forced_path, policy=policy)
    forced_reopened.initialize()
    forced_after = forced_reopened.integrity_check()
    forced_restore = _restore(forced_reopened, initial.snapshot)
    forced_events = forced_reopened.events(stream_id=window.stream_id)
    forced = ScenarioEvidence.create(
        scenario_id="forced_termination",
        checks={
            "committed_event_count_preserved": len(forced_events)
            == window.baseline_event_count,
            "forced_exit_observed": exit_code == 97,
            "integrity_valid_after_reopen": forced_after.valid,
            "state_matches_baseline": forced_restore.state_digest
            == window.expected_final_state.snapshot_digest,
            "uncommitted_event_absent": all(
                value.event_id != "m2d:uncommitted" for value in forced_events
            ),
        },
        observations={
            "event_count_after_reopen": len(forced_events),
            "expected_event_count": window.baseline_event_count,
            "expected_state_digest": window.expected_final_state.snapshot_digest,
            "forced_exit_code": exit_code,
            "restored_state_digest": forced_restore.state_digest,
        },
        before_integrity_digest=forced_before.report_digest,
        after_integrity_digest=forced_after.report_digest,
    )

    rollback_path = root / "rollback.sqlite3"
    _copy_database(backup_path, rollback_path)
    rollback_store = SQLiteShadowStore(rollback_path, policy=policy)
    rollback_store.initialize()
    rollback_store.append(rollback_probe_event)
    mutated_integrity = rollback_store.integrity_check()
    mutated_restore = _restore(rollback_store, initial.snapshot)
    _remove_sidecars(rollback_path)
    rollback_path.unlink()
    _copy_database(backup_path, rollback_path)
    restored_store = SQLiteShadowStore(rollback_path, policy=policy)
    restored_store.initialize()
    restored_integrity = restored_store.integrity_check()
    restored = _restore(restored_store, initial.snapshot)
    restored_events = restored_store.events(stream_id=window.stream_id)
    rollback = ScenarioEvidence.create(
        scenario_id="rollback_rehearsal",
        checks={
            "probe_changed_state": mutated_restore.state_digest
            != window.expected_final_state.snapshot_digest,
            "probe_event_persisted_before_rollback": mutated_integrity.event_count
            == window.baseline_event_count + 1,
            "probe_removed_after_rollback": all(
                value.event_id != rollback_probe_event.event_id for value in restored_events
            ),
            "restored_event_count": len(restored_events) == window.baseline_event_count,
            "restored_integrity_valid": restored_integrity.valid,
            "restored_state_matches": restored.state_digest
            == window.expected_final_state.snapshot_digest,
        },
        observations={
            "expected_state_digest": window.expected_final_state.snapshot_digest,
            "mutated_event_count": mutated_integrity.event_count,
            "mutated_state_digest": mutated_restore.state_digest,
            "restored_event_count": len(restored_events),
            "restored_state_digest": restored.state_digest,
            "rollback_method": "verified_backup_file_replacement_in_disposable_workspace",
        },
        before_integrity_digest=mutated_integrity.report_digest,
        after_integrity_digest=restored_integrity.report_digest,
    )

    return M2DRehearsalPacket.create(
        window=window,
        scenarios=(
            snapshot_restore,
            full_replay,
            corrupt_snapshot,
            corrupt_event,
            forced,
            rollback,
        ),
    )
