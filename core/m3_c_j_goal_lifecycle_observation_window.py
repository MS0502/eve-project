"""Dormant M3-C-J bounded goal-lifecycle observation-window evaluator.

This module evaluates immutable append receipts and rollback-preservation evidence.
It performs no SQLite access, writer construction, append, backup, restore, runtime
integration, action, scheduling, speech, legacy goal-authority transfer, or M3-E
activation. Checked-in reviewed window pins remain absent until a later exact-head
review slice.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

from core.event_kernel import canonical_json_object
from core.m3_c_h_dormant_goal_lifecycle_writer import (
    DormantWriterAppendReceipt,
    WriterStorageLimits,
    active_reviewed_writer_authorization_packet,
)
from core.sqlite_shadow_store import IntegrityReport

WINDOW_AUTHORIZATION_SCHEMA = "eve.m3-c-j.goal-lifecycle-observation-authorization.v1"
WINDOW_BASELINE_SCHEMA = "eve.m3-c-j.goal-lifecycle-observation-baseline.v1"
WINDOW_ROLLBACK_SCHEMA = "eve.m3-c-j.goal-lifecycle-rollback-preservation.v1"
WINDOW_RECEIPT_SCHEMA = "eve.m3-c-j.goal-lifecycle-observation-receipt.v1"
M3_C_I_EXACT_HEAD = "bec44a796834e037c41fbb941d090de416cf1e23"
M3_C_I_EXACT_RUN = 30447974882
M3_C_I_FOCUSED_PASSED = 16
M3_C_I_FULL_PASSED = 3304
M3_C_I_ARTIFACT_SHA256 = (
    "650d11a611b9ae8dcf49fe540b117a26e49fedab5576c366f332eda9d7b92f0f"
)
M3_C_I_M2E_RUN = 30447974661
M3_C_I_MERGE_SHA = "51f682e00059698cbb301a75983e11dd4812f574"
DEFAULT_MAX_WINDOW_EVENTS = 32

# Deliberately absent in this preflight implementation. A later exact-reviewed
# slice may pin these after this tree has one accepted exact-head artifact.
_ACTIVE_REVIEWED_WINDOW_IMPLEMENTATION_HEAD: str | None = None
_ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST: str | None = None


class M3CObservationWindowError(RuntimeError):
    """Base fail-closed observation-window error."""


class M3CObservationWindowAuthorizationError(M3CObservationWindowError):
    """Window authorization is absent, malformed, or not exact-reviewed."""


class M3CObservationWindowEvidenceError(M3CObservationWindowError):
    """Append, integrity, replay, or rollback evidence is inconsistent."""


def _canonical(value: Mapping[str, Any], *, field: str) -> str:
    return canonical_json_object(value, field=field)


def _digest(value: Mapping[str, Any], *, field: str) -> str:
    return hashlib.sha256(_canonical(value, field=field).encode("utf-8")).hexdigest()


def _require_hex(value: str, *, length: int, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M3CObservationWindowError(
            f"{field} must be lowercase {length}-character hex"
        )
    return value


def _require_nonnegative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise M3CObservationWindowError(f"{field} must be a non-negative integer")
    return value


def _require_positive_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise M3CObservationWindowError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ObservationWindowAuthorizationPacket:
    window_implementation_head: str
    writer_authorization_digest: str
    writer_implementation_head: str
    database_path_digest: str
    storage_limits: WriterStorageLimits
    max_window_events: int = DEFAULT_MAX_WINDOW_EVENTS
    schema_version: str = WINDOW_AUTHORIZATION_SCHEMA
    prerequisite_exact_head: str = M3_C_I_EXACT_HEAD
    prerequisite_exact_run: int = M3_C_I_EXACT_RUN
    prerequisite_focused_passed: int = M3_C_I_FOCUSED_PASSED
    prerequisite_full_passed: int = M3_C_I_FULL_PASSED
    prerequisite_artifact_sha256: str = M3_C_I_ARTIFACT_SHA256
    prerequisite_m2e_run: int = M3_C_I_M2E_RUN
    prerequisite_m2e_passed: int = 6
    prerequisite_m2e_required: int = 6
    prerequisite_merge_sha: str = M3_C_I_MERGE_SHA
    human_reviewed: bool = True
    observation_window_authorized: bool = True
    production_append_authorized: bool = False
    runtime_integration_authorized: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    legacy_migration_authorized: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        _require_hex(
            self.window_implementation_head,
            length=40,
            field="window_implementation_head",
        )
        _require_hex(
            self.writer_authorization_digest,
            length=64,
            field="writer_authorization_digest",
        )
        _require_hex(
            self.writer_implementation_head,
            length=40,
            field="writer_implementation_head",
        )
        _require_hex(self.database_path_digest, length=64, field="database_path_digest")
        _require_hex(self.prerequisite_exact_head, length=40, field="prerequisite_exact_head")
        _require_hex(
            self.prerequisite_artifact_sha256,
            length=64,
            field="prerequisite_artifact_sha256",
        )
        _require_hex(self.prerequisite_merge_sha, length=40, field="prerequisite_merge_sha")
        if not isinstance(self.storage_limits, WriterStorageLimits):
            raise M3CObservationWindowAuthorizationError(
                "storage_limits must be WriterStorageLimits"
            )
        _require_positive_int(self.max_window_events, field="max_window_events")
        for field in (
            "prerequisite_exact_run",
            "prerequisite_focused_passed",
            "prerequisite_full_passed",
            "prerequisite_m2e_run",
        ):
            _require_positive_int(getattr(self, field), field=field)
        if self.schema_version != WINDOW_AUTHORIZATION_SCHEMA:
            raise M3CObservationWindowAuthorizationError(
                "unsupported window authorization schema"
            )
        if (
            self.prerequisite_exact_head != M3_C_I_EXACT_HEAD
            or self.prerequisite_exact_run != M3_C_I_EXACT_RUN
            or self.prerequisite_focused_passed != M3_C_I_FOCUSED_PASSED
            or self.prerequisite_full_passed != M3_C_I_FULL_PASSED
            or self.prerequisite_artifact_sha256 != M3_C_I_ARTIFACT_SHA256
            or self.prerequisite_m2e_run != M3_C_I_M2E_RUN
            or (self.prerequisite_m2e_passed, self.prerequisite_m2e_required) != (6, 6)
            or self.prerequisite_merge_sha != M3_C_I_MERGE_SHA
        ):
            raise M3CObservationWindowAuthorizationError(
                "M3-C-I prerequisite evidence does not match the reviewed boundary"
            )
        if not self.human_reviewed or not self.observation_window_authorized:
            raise M3CObservationWindowAuthorizationError(
                "window authorization must be explicitly human reviewed"
            )
        if any(
            (
                self.production_append_authorized,
                self.runtime_integration_authorized,
                self.action_authorized,
                self.scheduler_authorized,
                self.speech_authorized,
                self.legacy_goal_authority_transferred,
                self.legacy_migration_authorized,
                self.m3_e_authority_open,
            )
        ):
            raise M3CObservationWindowAuthorizationError(
                "window authorization escaped observation-only scope"
            )

    def to_mapping(self) -> dict[str, Any]:
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["storage_limits"] = self.storage_limits.to_mapping()
        return result

    @property
    def authorization_digest(self) -> str:
        return _digest(self.to_mapping(), field="m3_c_j_window_authorization")


def build_observation_window_authorization_candidate(
    *,
    window_implementation_head: str,
    max_window_events: int = DEFAULT_MAX_WINDOW_EVENTS,
) -> ObservationWindowAuthorizationPacket:
    """Build the deterministic candidate packet without activating the window."""

    writer_packet = active_reviewed_writer_authorization_packet()
    return ObservationWindowAuthorizationPacket(
        window_implementation_head=window_implementation_head,
        writer_authorization_digest=writer_packet.authorization_digest,
        writer_implementation_head=writer_packet.implementation_head,
        database_path_digest=writer_packet.database_path_digest,
        storage_limits=writer_packet.storage_limits,
        max_window_events=max_window_events,
    )


def verify_active_observation_window_authorization(
    packet: ObservationWindowAuthorizationPacket | None,
) -> str:
    """Fail closed until a later slice pins this evaluator's exact reviewed head."""

    if (
        _ACTIVE_REVIEWED_WINDOW_IMPLEMENTATION_HEAD is None
        or _ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST is None
    ):
        raise M3CObservationWindowAuthorizationError(
            "reviewed observation-window authorization is absent"
        )
    if not isinstance(packet, ObservationWindowAuthorizationPacket):
        raise M3CObservationWindowAuthorizationError(
            "packet must be ObservationWindowAuthorizationPacket"
        )
    if packet.window_implementation_head != _ACTIVE_REVIEWED_WINDOW_IMPLEMENTATION_HEAD:
        raise M3CObservationWindowAuthorizationError(
            "window implementation head is not the active reviewed head"
        )
    if packet.prerequisite_exact_head != M3_C_I_EXACT_HEAD:
        raise M3CObservationWindowAuthorizationError(
            "window prerequisite exact head mismatch"
        )
    if packet.authorization_digest != _ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST:
        raise M3CObservationWindowAuthorizationError(
            "window authorization digest is not the active reviewed packet"
        )
    return packet.authorization_digest


@dataclass(frozen=True, slots=True)
class ObservationWindowBaseline:
    authorization_digest: str
    database_path_digest: str
    start_sequence: int
    start_event_count: int
    start_chain_digest: str
    start_reducer_snapshot_digest: str
    integrity_report_digest: str
    backup_sha256: str
    backup_path_digest: str
    schema_version: str = WINDOW_BASELINE_SCHEMA
    database_integrity_valid: bool = True
    backup_integrity_verified: bool = True
    writer_disabled_during_baseline: bool = True
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        for field in (
            "authorization_digest",
            "database_path_digest",
            "start_chain_digest",
            "start_reducer_snapshot_digest",
            "integrity_report_digest",
            "backup_sha256",
            "backup_path_digest",
        ):
            _require_hex(getattr(self, field), length=64, field=field)
        _require_nonnegative_int(self.start_sequence, field="start_sequence")
        _require_nonnegative_int(self.start_event_count, field="start_event_count")
        if (
            self.schema_version != WINDOW_BASELINE_SCHEMA
            or not self.database_integrity_valid
            or not self.backup_integrity_verified
            or not self.writer_disabled_during_baseline
            or self.legacy_goal_authority_transferred
            or self.m3_e_authority_open
        ):
            raise M3CObservationWindowEvidenceError(
                "baseline does not prove a disabled, verified pre-window state"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @property
    def baseline_digest(self) -> str:
        return _digest(self.to_mapping(), field="m3_c_j_window_baseline")


@dataclass(frozen=True, slots=True)
class RollbackPreservationEvidence:
    authorization_digest: str
    database_path_digest: str
    backup_sha256: str
    backup_path_digest: str
    restore_path_digest: str
    pre_window_snapshot_digest: str
    restored_snapshot_digest: str
    restored_integrity_report_digest: str
    schema_version: str = WINDOW_ROLLBACK_SCHEMA
    writer_disabled: bool = True
    failed_database_preserved: bool = True
    restore_into_separate_path: bool = True
    restored_integrity_valid: bool = True
    restore_replay_verified: bool = True
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        for field in (
            "authorization_digest",
            "database_path_digest",
            "backup_sha256",
            "backup_path_digest",
            "restore_path_digest",
            "pre_window_snapshot_digest",
            "restored_snapshot_digest",
            "restored_integrity_report_digest",
        ):
            _require_hex(getattr(self, field), length=64, field=field)
        if self.restore_path_digest in {
            self.database_path_digest,
            self.backup_path_digest,
        }:
            raise M3CObservationWindowEvidenceError(
                "rollback restore path must be separate"
            )
        if self.pre_window_snapshot_digest != self.restored_snapshot_digest:
            raise M3CObservationWindowEvidenceError(
                "rollback restore snapshot differs from pre-window baseline"
            )
        if (
            self.schema_version != WINDOW_ROLLBACK_SCHEMA
            or not self.writer_disabled
            or not self.failed_database_preserved
            or not self.restore_into_separate_path
            or not self.restored_integrity_valid
            or not self.restore_replay_verified
            or self.legacy_goal_authority_transferred
            or self.m3_e_authority_open
        ):
            raise M3CObservationWindowEvidenceError(
                "rollback preservation evidence escaped the accepted contract"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @property
    def evidence_digest(self) -> str:
        return _digest(self.to_mapping(), field="m3_c_j_rollback_preservation")


@dataclass(frozen=True, slots=True)
class ObservationWindowReceipt:
    authorization_digest: str
    baseline_digest: str
    rollback_evidence_digest: str
    database_path_digest: str
    first_sequence: int
    last_sequence: int
    observed_event_count: int
    first_before_chain_digest: str
    final_chain_digest: str
    final_reducer_snapshot_digest: str
    final_integrity_report_digest: str
    append_receipt_digests: tuple[str, ...]
    event_envelope_digests: tuple[str, ...]
    transition_ids: tuple[str, ...]
    schema_version: str = WINDOW_RECEIPT_SCHEMA
    contiguous_sequences_verified: bool = True
    contiguous_counts_verified: bool = True
    chain_continuity_verified: bool = True
    unique_events_verified: bool = True
    direct_replay_equivalent: bool = True
    duplicate_acceptance_count: int = 0
    conflict_acceptance_count: int = 0
    rollback_preservation_verified: bool = True
    production_append_executed_by_evaluator: bool = False
    runtime_integration_performed: bool = False
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        for field in (
            "authorization_digest",
            "baseline_digest",
            "rollback_evidence_digest",
            "database_path_digest",
            "first_before_chain_digest",
            "final_chain_digest",
            "final_reducer_snapshot_digest",
            "final_integrity_report_digest",
        ):
            _require_hex(getattr(self, field), length=64, field=field)
        _require_positive_int(self.first_sequence, field="first_sequence")
        _require_positive_int(self.last_sequence, field="last_sequence")
        _require_positive_int(self.observed_event_count, field="observed_event_count")
        for values, field in (
            (self.append_receipt_digests, "append_receipt_digests"),
            (self.event_envelope_digests, "event_envelope_digests"),
            (self.transition_ids, "transition_ids"),
        ):
            if len(values) != self.observed_event_count:
                raise M3CObservationWindowEvidenceError(
                    f"{field} count does not match observed events"
                )
            for value in values:
                _require_hex(value, length=64, field=field)
        required_true = (
            self.contiguous_sequences_verified,
            self.contiguous_counts_verified,
            self.chain_continuity_verified,
            self.unique_events_verified,
            self.direct_replay_equivalent,
            self.rollback_preservation_verified,
        )
        if (
            self.schema_version != WINDOW_RECEIPT_SCHEMA
            or not all(required_true)
            or self.duplicate_acceptance_count != 0
            or self.conflict_acceptance_count != 0
            or self.production_append_executed_by_evaluator
            or self.runtime_integration_performed
            or self.legacy_goal_authority_transferred
            or self.m3_e_authority_open
        ):
            raise M3CObservationWindowEvidenceError(
                "observation receipt escaped accepted scope"
            )

    def to_mapping(self) -> dict[str, Any]:
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        for field in (
            "append_receipt_digests",
            "event_envelope_digests",
            "transition_ids",
        ):
            result[field] = list(result[field])
        return result

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping(), field="m3_c_j_window_receipt")


def evaluate_observation_window(
    authorization_packet: ObservationWindowAuthorizationPacket | None,
    *,
    baseline: ObservationWindowBaseline,
    append_receipts: Sequence[DormantWriterAppendReceipt],
    final_integrity_report: IntegrityReport,
    final_reducer_snapshot_digest: str,
    rollback_evidence: RollbackPreservationEvidence,
) -> ObservationWindowReceipt:
    """Evaluate evidence only; never construct or call a writer or SQLite store."""

    authorization_digest = verify_active_observation_window_authorization(
        authorization_packet
    )
    assert authorization_packet is not None
    if not isinstance(baseline, ObservationWindowBaseline):
        raise M3CObservationWindowEvidenceError(
            "baseline must be ObservationWindowBaseline"
        )
    if not isinstance(rollback_evidence, RollbackPreservationEvidence):
        raise M3CObservationWindowEvidenceError(
            "rollback_evidence must be RollbackPreservationEvidence"
        )
    if not isinstance(final_integrity_report, IntegrityReport):
        raise M3CObservationWindowEvidenceError(
            "final_integrity_report must be IntegrityReport"
        )
    _require_hex(
        final_reducer_snapshot_digest,
        length=64,
        field="final_reducer_snapshot_digest",
    )
    if baseline.authorization_digest != authorization_digest:
        raise M3CObservationWindowEvidenceError("baseline authorization mismatch")
    if baseline.database_path_digest != authorization_packet.database_path_digest:
        raise M3CObservationWindowEvidenceError("baseline database path mismatch")
    if rollback_evidence.authorization_digest != authorization_digest:
        raise M3CObservationWindowEvidenceError("rollback authorization mismatch")
    if rollback_evidence.database_path_digest != baseline.database_path_digest:
        raise M3CObservationWindowEvidenceError("rollback database path mismatch")
    if rollback_evidence.backup_sha256 != baseline.backup_sha256:
        raise M3CObservationWindowEvidenceError("rollback backup digest mismatch")
    if rollback_evidence.backup_path_digest != baseline.backup_path_digest:
        raise M3CObservationWindowEvidenceError("rollback backup path mismatch")
    if (
        rollback_evidence.pre_window_snapshot_digest
        != baseline.start_reducer_snapshot_digest
    ):
        raise M3CObservationWindowEvidenceError(
            "rollback baseline snapshot mismatch"
        )

    receipts = tuple(append_receipts)
    if not 1 <= len(receipts) <= authorization_packet.max_window_events:
        raise M3CObservationWindowEvidenceError(
            "observed event count is outside the reviewed bound"
        )
    expected_sequence = baseline.start_sequence + 1
    expected_count = baseline.start_event_count
    expected_chain = baseline.start_chain_digest
    receipt_digests: list[str] = []
    envelope_digests: list[str] = []
    transition_ids: list[str] = []
    for receipt in receipts:
        if not isinstance(receipt, DormantWriterAppendReceipt):
            raise M3CObservationWindowEvidenceError(
                "append evidence must be DormantWriterAppendReceipt"
            )
        if (
            receipt.authorization_digest
            != authorization_packet.writer_authorization_digest
            or receipt.implementation_head
            != authorization_packet.writer_implementation_head
            or receipt.database_path_digest
            != authorization_packet.database_path_digest
        ):
            raise M3CObservationWindowEvidenceError(
                "append receipt does not match reviewed writer identity"
            )
        if (
            not receipt.production_authoritative_append_performed
            or receipt.disposable_or_test_path_only
            or not receipt.transaction_committed
            or not receipt.precommit_readback_verified
            or not receipt.postcommit_readback_verified
            or not receipt.stream_sequence_advanced_by_one
            or not receipt.chain_advanced_and_verified
            or not receipt.direct_reducer_equivalent
            or not receipt.sqlite_write_performed
            or receipt.live_writer_installed
            or receipt.production_integration_performed
            or receipt.action_authorized
            or receipt.scheduler_authorized
            or receipt.speech_authorized
            or receipt.legacy_goal_authority_transferred
            or receipt.m3_e_authority_open
        ):
            raise M3CObservationWindowEvidenceError(
                "append receipt is not accepted production-path observation evidence"
            )
        if receipt.sequence != expected_sequence:
            raise M3CObservationWindowEvidenceError(
                "observation sequence is not contiguous"
            )
        if receipt.before_count != expected_count or receipt.after_count != expected_count + 1:
            raise M3CObservationWindowEvidenceError(
                "observation event counts are not contiguous"
            )
        if receipt.before_chain_digest != expected_chain:
            raise M3CObservationWindowEvidenceError(
                "observation event chain is not contiguous"
            )
        receipt_digests.append(receipt.receipt_digest)
        envelope_digests.append(receipt.event_envelope_digest)
        transition_ids.append(receipt.transition_id)
        expected_sequence += 1
        expected_count += 1
        expected_chain = receipt.after_chain_digest

    if len(set(receipt_digests)) != len(receipt_digests):
        raise M3CObservationWindowEvidenceError("duplicate append receipt accepted")
    if len(set(envelope_digests)) != len(envelope_digests):
        raise M3CObservationWindowEvidenceError("duplicate event envelope accepted")
    if len(set(transition_ids)) != len(transition_ids):
        raise M3CObservationWindowEvidenceError("duplicate lifecycle transition accepted")
    if not final_integrity_report.valid:
        raise M3CObservationWindowEvidenceError("final database integrity is invalid")
    if final_integrity_report.event_count != expected_count:
        raise M3CObservationWindowEvidenceError(
            "final integrity event count differs from observed window"
        )
    if final_integrity_report.chain_head_digest != expected_chain:
        raise M3CObservationWindowEvidenceError(
            "final integrity chain head differs from observed window"
        )
    if final_reducer_snapshot_digest != receipts[-1].reducer_snapshot_digest:
        raise M3CObservationWindowEvidenceError(
            "final reducer replay differs from the last append receipt"
        )

    return ObservationWindowReceipt(
        authorization_digest=authorization_digest,
        baseline_digest=baseline.baseline_digest,
        rollback_evidence_digest=rollback_evidence.evidence_digest,
        database_path_digest=authorization_packet.database_path_digest,
        first_sequence=receipts[0].sequence,
        last_sequence=receipts[-1].sequence,
        observed_event_count=len(receipts),
        first_before_chain_digest=baseline.start_chain_digest,
        final_chain_digest=expected_chain,
        final_reducer_snapshot_digest=final_reducer_snapshot_digest,
        final_integrity_report_digest=final_integrity_report.report_digest,
        append_receipt_digests=tuple(receipt_digests),
        event_envelope_digests=tuple(envelope_digests),
        transition_ids=tuple(transition_ids),
    )
