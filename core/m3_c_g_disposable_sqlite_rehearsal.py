"""Disposable SQLite rehearsal for the bounded M3-C goal-lifecycle stream.

The public operation requires an explicit caller-created temporary directory and
synthetic replay-valid M3-C-D candidates. It installs no live writer, invents no
path default, touches no legacy goal database, changes no production setting,
transfers no legacy goal authority, and does not open M3-E.
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.event_kernel import EventEnvelope, canonical_json_object
from core.m2_e_cutover_activation import CutoverAuthorityState
from core.m3_c_c_goal_lifecycle_kernel import GoalLifecycleState
from core.m3_c_d_goal_lifecycle_event_preflight import (
    EVENT_STREAM,
    REDUCER_SNAPSHOT_VERSION,
    GoalLifecycleEventEnvelopeCandidate,
    GoalLifecycleReducerSnapshot,
    apply_event_candidate_in_memory,
    replay_event_candidates_in_memory,
)
from core.m3_c_e_goal_lifecycle_substrate_binding_preflight import (
    GoalLifecycleSubstrateBindingCandidate,
    build_substrate_binding_candidates,
    source_from_bound_envelope,
)
from core.sqlite_shadow_store import (
    AppendReceipt,
    SQLiteShadowStore,
    ShadowStoragePolicy,
)

REHEARSAL_VERSION = "eve.m3-c-g.disposable-sqlite-rehearsal.v1"
REHEARSAL_SCOPE = "m3_c_g_disposable_sqlite_rehearsal_only"
FORWARD_DATABASE_NAME = "m3c-goal-lifecycle-forward.sqlite3"
RESTORED_DATABASE_NAME = "m3c-goal-lifecycle-restored.sqlite3"
PREREQUISITE_PR = 223
PREREQUISITE_EXACT_HEAD = "395dea54fdbdac5fad7a2fab35fea1466a52a919"
PREREQUISITE_ARTIFACT_SHA256 = "a5e801992970cdcb2598f60ddee13a6e3957cc70db63335a3cca9a6c48bd3597"
PREREQUISITE_MERGE_SHA = "91470c1adace585995a2a92d39ebd3e330d57342"


class M3CDisposableSQLiteRehearsalError(RuntimeError):
    """Fail-closed disposable-rehearsal error."""


def _digest(value: Mapping[str, Any], *, field: str) -> str:
    canonical = canonical_json_object(value, field=field)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_hex(value: str, *, length: int, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M3CDisposableSQLiteRehearsalError(
            f"{field} must be lowercase {length}-character hex"
        )
    return value


def _snapshot_from_mapping(
    value: Mapping[str, Any],
) -> GoalLifecycleReducerSnapshot:
    plain = dict(value)
    if plain.get("schema_version") != REDUCER_SNAPSHOT_VERSION:
        raise M3CDisposableSQLiteRehearsalError(
            "snapshot schema version mismatch"
        )
    states_value = plain.get("states")
    steps_value = plain.get("last_logical_steps")
    transitions_value = plain.get("applied_transition_ids")
    if (
        not isinstance(states_value, dict)
        or not isinstance(steps_value, dict)
        or not isinstance(transitions_value, list)
    ):
        raise M3CDisposableSQLiteRehearsalError(
            "snapshot mapping shape mismatch"
        )
    try:
        return GoalLifecycleReducerSnapshot(
            states={
                str(candidate_id): GoalLifecycleState(**dict(state))
                for candidate_id, state in states_value.items()
            },
            last_logical_steps={
                str(candidate_id): int(step)
                for candidate_id, step in steps_value.items()
            },
            applied_transition_ids=tuple(
                str(item) for item in transitions_value
            ),
        )
    except (TypeError, ValueError) as exc:
        raise M3CDisposableSQLiteRehearsalError(
            "snapshot mapping is invalid"
        ) from exc


def _reducer(
    authority_state_digest: str,
) -> Callable[
    [GoalLifecycleReducerSnapshot, EventEnvelope],
    GoalLifecycleReducerSnapshot,
]:
    _require_hex(
        authority_state_digest,
        length=64,
        field="authority_state_digest",
    )

    def reduce(
        snapshot: GoalLifecycleReducerSnapshot,
        envelope: EventEnvelope,
    ) -> GoalLifecycleReducerSnapshot:
        source = source_from_bound_envelope(
            envelope,
            authority_state_digest=authority_state_digest,
        )
        return apply_event_candidate_in_memory(snapshot, source)[0]

    return reduce


def _validate_root(rehearsal_root: str | Path) -> Path:
    if isinstance(rehearsal_root, str) and not rehearsal_root.strip():
        raise M3CDisposableSQLiteRehearsalError(
            "rehearsal_root must be an explicit path"
        )
    root = Path(rehearsal_root)
    if not root.is_absolute():
        raise M3CDisposableSQLiteRehearsalError(
            "rehearsal_root must be absolute"
        )
    if not root.exists() or not root.is_dir():
        raise M3CDisposableSQLiteRehearsalError(
            "rehearsal_root must be a caller-created directory"
        )
    return root


def _append_one_verified(
    store: SQLiteShadowStore,
    binding: GoalLifecycleSubstrateBindingCandidate,
) -> AppendReceipt:
    envelope = binding.event_envelope
    before = store.events(stream_id=EVENT_STREAM)
    receipt = store.append(envelope)
    after = store.events(stream_id=EVENT_STREAM)
    if (
        receipt.event_id != envelope.event_id
        or receipt.stream_id != EVENT_STREAM
        or receipt.sequence != envelope.sequence
        or receipt.envelope_digest != envelope.digest
        or receipt.before_count + 1 != receipt.after_count
        or receipt.after_count != len(after)
        or len(after) != len(before) + 1
        or after[-1] != envelope
        or not receipt.readback_verified
        or not receipt.state_changed
    ):
        raise M3CDisposableSQLiteRehearsalError(
            "single append readback verification failed"
        )
    return receipt


@dataclass(frozen=True, slots=True)
class DisposableSQLiteRehearsalReceipt:
    binding_digests: tuple[str, ...]
    event_envelope_digests: tuple[str, ...]
    append_transition_hashes: tuple[str, ...]
    checkpoint_sequence: int
    appended_event_count: int
    forward_event_count: int
    restored_event_count: int
    checkpoint_snapshot_digest: str
    forward_direct_snapshot_digest: str
    forward_sqlite_snapshot_digest: str
    snapshot_suffix_snapshot_digest: str
    restored_checkpoint_snapshot_digest: str
    forward_chain_digest: str
    restored_chain_digest: str
    forward_integrity_report_digest: str
    restored_integrity_report_digest: str
    backup_sha256: str
    prerequisite_exact_head: str = PREREQUISITE_EXACT_HEAD
    prerequisite_artifact_sha256: str = PREREQUISITE_ARTIFACT_SHA256
    prerequisite_merge_sha: str = PREREQUISITE_MERGE_SHA
    prerequisite_pr: int = PREREQUISITE_PR
    schema_version: str = REHEARSAL_VERSION
    scope: str = REHEARSAL_SCOPE
    caller_owned_temporary_path: bool = True
    concrete_sqlite_file_used: bool = True
    memory_database_used: bool = False
    one_event_per_append_call: bool = True
    precommit_readback_verified: bool = True
    postcommit_readback_verified: bool = True
    event_chain_verified: bool = True
    direct_replay_equivalent: bool = True
    snapshot_suffix_equivalent: bool = True
    rollback_checkpoint_restored: bool = True
    failed_database_preservation_required: bool = True
    forward_database_preserved: bool = True
    restored_database_is_separate: bool = True
    disposable_sqlite_write_performed: bool = True
    production_authoritative_append_performed: bool = False
    live_writer_installed: bool = False
    production_integration_performed: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False
    writer_operationally_enabled: bool = False

    def __post_init__(self) -> None:
        sha256_values = (
            *self.binding_digests,
            *self.event_envelope_digests,
            *self.append_transition_hashes,
            self.checkpoint_snapshot_digest,
            self.forward_direct_snapshot_digest,
            self.forward_sqlite_snapshot_digest,
            self.snapshot_suffix_snapshot_digest,
            self.restored_checkpoint_snapshot_digest,
            self.forward_chain_digest,
            self.restored_chain_digest,
            self.forward_integrity_report_digest,
            self.restored_integrity_report_digest,
            self.backup_sha256,
            self.prerequisite_artifact_sha256,
        )
        for value in sha256_values:
            _require_hex(value, length=64, field="rehearsal digest")
        _require_hex(
            self.prerequisite_exact_head,
            length=40,
            field="prerequisite_exact_head",
        )
        _require_hex(
            self.prerequisite_merge_sha,
            length=40,
            field="prerequisite_merge_sha",
        )
        count = len(self.binding_digests)
        if (
            count < 2
            or count != len(self.event_envelope_digests)
            or count != len(self.append_transition_hashes)
            or self.appended_event_count != count
            or self.forward_event_count != count
            or not 0 < self.checkpoint_sequence < count
            or self.restored_event_count != self.checkpoint_sequence
        ):
            raise M3CDisposableSQLiteRehearsalError(
                "rehearsal counts are inconsistent"
            )
        if (
            self.forward_direct_snapshot_digest
            != self.forward_sqlite_snapshot_digest
            or self.forward_direct_snapshot_digest
            != self.snapshot_suffix_snapshot_digest
            or self.checkpoint_snapshot_digest
            != self.restored_checkpoint_snapshot_digest
        ):
            raise M3CDisposableSQLiteRehearsalError(
                "replay or rollback digest mismatch"
            )
        required_true = (
            self.caller_owned_temporary_path,
            self.concrete_sqlite_file_used,
            self.one_event_per_append_call,
            self.precommit_readback_verified,
            self.postcommit_readback_verified,
            self.event_chain_verified,
            self.direct_replay_equivalent,
            self.snapshot_suffix_equivalent,
            self.rollback_checkpoint_restored,
            self.failed_database_preservation_required,
            self.forward_database_preserved,
            self.restored_database_is_separate,
            self.disposable_sqlite_write_performed,
        )
        required_false = (
            self.memory_database_used,
            self.production_authoritative_append_performed,
            self.live_writer_installed,
            self.production_integration_performed,
            self.action_authorized,
            self.scheduler_authorized,
            self.speech_authorized,
            self.legacy_goal_authority_transferred,
            self.m3_e_authority_open,
            self.writer_operationally_enabled,
        )
        if (
            not all(required_true)
            or any(required_false)
            or self.prerequisite_pr != PREREQUISITE_PR
            or self.schema_version != REHEARSAL_VERSION
            or self.scope != REHEARSAL_SCOPE
        ):
            raise M3CDisposableSQLiteRehearsalError(
                "rehearsal authority boundary mismatch"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            field.name: (
                list(value)
                if isinstance((value := getattr(self, field.name)), tuple)
                else value
            )
            for field in fields(self)
        }

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping(), field="m3_c_g_rehearsal_receipt")


def run_disposable_sqlite_rehearsal(
    sources: Sequence[GoalLifecycleEventEnvelopeCandidate],
    *,
    rehearsal_root: str | Path,
    checkpoint_sequence: int,
    authority_state: CutoverAuthorityState | None = None,
) -> DisposableSQLiteRehearsalReceipt:
    """Run one synthetic rehearsal under an explicit disposable directory."""

    values = tuple(sources)
    bindings = build_substrate_binding_candidates(
        values,
        authority_state=authority_state,
    )
    if (
        isinstance(checkpoint_sequence, bool)
        or not isinstance(checkpoint_sequence, int)
        or not 0 < checkpoint_sequence < len(bindings)
    ):
        raise M3CDisposableSQLiteRehearsalError(
            "checkpoint_sequence must leave a non-empty suffix"
        )
    root = _validate_root(rehearsal_root)
    forward_path = root / "forward" / FORWARD_DATABASE_NAME
    backup_directory = root / "checkpoint-backup"
    restored_path = root / "restored" / RESTORED_DATABASE_NAME
    occupied = (
        forward_path,
        backup_directory,
        restored_path,
        Path(f"{forward_path}-wal"),
        Path(f"{forward_path}-shm"),
        Path(f"{restored_path}-wal"),
        Path(f"{restored_path}-shm"),
    )
    if any(path.exists() for path in occupied):
        raise M3CDisposableSQLiteRehearsalError(
            "rehearsal targets must not already exist"
        )

    policy = ShadowStoragePolicy(
        snapshot_interval_events=checkpoint_sequence,
        max_event_count=len(bindings),
        max_event_bytes=1_048_576,
        max_snapshot_count=1,
        max_snapshot_bytes=1_048_576,
        max_backups=1,
    )
    reducer = _reducer(bindings[0].authority_state_digest)
    forward_store = SQLiteShadowStore(forward_path, policy=policy)
    forward_store.initialize()

    append_receipts: list[AppendReceipt] = []
    checkpoint_snapshot = GoalLifecycleReducerSnapshot.empty()
    backup_receipt = None
    for index, binding in enumerate(bindings, 1):
        append_receipts.append(_append_one_verified(forward_store, binding))
        persisted_prefix = forward_store.events(stream_id=EVENT_STREAM)
        sqlite_prefix = GoalLifecycleReducerSnapshot.empty()
        for envelope in persisted_prefix:
            sqlite_prefix = reducer(sqlite_prefix, envelope)
        direct_prefix = replay_event_candidates_in_memory(values[:index])[0]
        if sqlite_prefix.snapshot_digest != direct_prefix.snapshot_digest:
            raise M3CDisposableSQLiteRehearsalError(
                "SQLite prefix replay diverges from direct reducer replay"
            )
        if index == checkpoint_sequence:
            checkpoint_snapshot = direct_prefix
            snapshot_receipt = forward_store.write_snapshot(
                snapshot_id=(
                    f"m3c:goal-lifecycle:checkpoint:{checkpoint_sequence}"
                ),
                stream_id=EVENT_STREAM,
                through_sequence=checkpoint_sequence,
                state=checkpoint_snapshot.to_mapping(),
                state_schema_version=REDUCER_SNAPSHOT_VERSION,
            )
            if (
                snapshot_receipt.state_digest
                != checkpoint_snapshot.snapshot_digest
            ):
                raise M3CDisposableSQLiteRehearsalError(
                    "checkpoint snapshot readback digest mismatch"
                )
            backup_receipt = forward_store.create_backup(
                backup_directory,
                backup_ordinal=1,
            )

    if backup_receipt is None:
        raise M3CDisposableSQLiteRehearsalError(
            "checkpoint backup was not created"
        )
    forward_integrity = forward_store.integrity_check()
    if not forward_integrity.valid:
        raise M3CDisposableSQLiteRehearsalError(
            "forward database integrity verification failed"
        )
    direct_final = replay_event_candidates_in_memory(values)[0]
    forward_restore = forward_store.restore_verified(
        stream_id=EVENT_STREAM,
        initial_state=GoalLifecycleReducerSnapshot.empty(),
        reducer=reducer,
        state_to_mapping=lambda state: state.to_mapping(),
        state_from_mapping=_snapshot_from_mapping,
    )
    if (
        not forward_restore.verified
        or forward_restore.state.snapshot_digest
        != direct_final.snapshot_digest
        or forward_restore.state_digest != direct_final.snapshot_digest
    ):
        raise M3CDisposableSQLiteRehearsalError(
            "full or snapshot-plus-suffix replay diverged"
        )

    restored_path.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(backup_receipt.backup_path), restored_path)
    restored_store = SQLiteShadowStore(restored_path, policy=policy)
    restored_store.initialize()
    restored_integrity = restored_store.integrity_check()
    if not restored_integrity.valid:
        raise M3CDisposableSQLiteRehearsalError(
            "restored database integrity verification failed"
        )
    restored = restored_store.restore_verified(
        stream_id=EVENT_STREAM,
        initial_state=GoalLifecycleReducerSnapshot.empty(),
        reducer=reducer,
        state_to_mapping=lambda state: state.to_mapping(),
        state_from_mapping=_snapshot_from_mapping,
    )
    if (
        not restored.verified
        or restored.state.snapshot_digest
        != checkpoint_snapshot.snapshot_digest
        or restored.state_digest != checkpoint_snapshot.snapshot_digest
    ):
        raise M3CDisposableSQLiteRehearsalError(
            "rollback checkpoint restoration diverged"
        )

    return DisposableSQLiteRehearsalReceipt(
        binding_digests=tuple(item.binding_digest for item in bindings),
        event_envelope_digests=tuple(
            item.event_envelope.digest for item in bindings
        ),
        append_transition_hashes=tuple(
            item.transition_hash for item in append_receipts
        ),
        checkpoint_sequence=checkpoint_sequence,
        appended_event_count=len(append_receipts),
        forward_event_count=forward_integrity.event_count,
        restored_event_count=restored_integrity.event_count,
        checkpoint_snapshot_digest=checkpoint_snapshot.snapshot_digest,
        forward_direct_snapshot_digest=direct_final.snapshot_digest,
        forward_sqlite_snapshot_digest=forward_restore.state_digest,
        snapshot_suffix_snapshot_digest=forward_restore.state.snapshot_digest,
        restored_checkpoint_snapshot_digest=restored.state_digest,
        forward_chain_digest=forward_integrity.chain_head_digest,
        restored_chain_digest=restored_integrity.chain_head_digest,
        forward_integrity_report_digest=forward_integrity.report_digest,
        restored_integrity_report_digest=restored_integrity.report_digest,
        backup_sha256=backup_receipt.backup_sha256,
    )
