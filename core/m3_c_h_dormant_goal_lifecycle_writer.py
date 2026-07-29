"""Exact-reviewed bounded activation candidate for the M3-C lifecycle stream.

M3-C-H supplied the disconnected writer mechanism. M3-C-I pins one immutable
reviewed authorization packet to the exact validated M3-C-H implementation,
one private caller-owned absolute database-path digest, and one bounded storage
policy. Import and construction still perform no I/O. No runtime startup hook,
action, scheduler, speech, legacy goal-authority transfer, migration, or M3-E
authority is added.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY, canonical_json_object
from core.m2_e_cutover_activation import (
    CutoverAuthorityState,
    EVENT_STORE_ACTIVE_ROLE,
    active_cutover_authority,
)
from core.m3_c_d_goal_lifecycle_event_preflight import (
    EVENT_STREAM,
    EVENT_TYPE,
    REDUCER_SNAPSHOT_VERSION,
    GoalLifecycleReducerSnapshot,
    apply_event_candidate_in_memory,
)
from core.m3_c_e_goal_lifecycle_substrate_binding_preflight import (
    BINDING_AUTHORITY,
    PRODUCER_ID,
    PRODUCER_VERSION,
    GoalLifecycleSubstrateBindingCandidate,
    source_from_bound_envelope,
)
from core.sqlite_shadow_store import (
    STORE_SCHEMA_VERSION,
    AppendReceipt,
    SQLiteShadowStore,
    ShadowStoragePolicy,
)

DORMANT_WRITER_SCHEMA = "eve.m3-c-h.dormant-goal-lifecycle-writer.v1"
WRITER_AUTHORIZATION_SCHEMA = "eve.m3-c-f.goal-lifecycle-writer-authorization.v1"
WRITER_APPEND_RECEIPT_SCHEMA = "eve.m3-c-h.goal-lifecycle-writer-append-receipt.v1"
WRITER_ROLLBACK_CONTROL_SCHEMA = "eve.m3-c-f.goal-lifecycle-writer-rollback-control.v1"
DATABASE_PATH_OWNERSHIP = "reviewed_caller_owned_concrete_path"
ROLLBACK_PROCEDURE = "disable_preserve_restore_separate_verify_replay"
M3_C_E_PREREQUISITE_MERGE_SHA = "938dc3f9d00a8bd7fffe4e4cae38894531462947"
M3_C_E_PREREQUISITE_ARTIFACT_SHA256 = (
    "6e65b9da98ef8ce326187961bceec8e683b4d5ba0cdc9feef4e282ae815429ea"
)
M3_C_G_PREREQUISITE_MERGE_SHA = "b717b676ec84fd157eabf5b0a947f68c1c6617eb"
M3_C_G_PREREQUISITE_ARTIFACT_SHA256 = (
    "2705d825fc827624e71e4a86ba992e9a19f2b90a60d3d4603ac60ab553de86c2"
)

# M3-C-I exact-reviewed pins. The private absolute path itself is deliberately
# not stored in this public repository; only its lexical SHA-256 is public.
_ACTIVE_REVIEWED_IMPLEMENTATION_HEAD: str | None = (
    "68efeca10c6819cb74ccc884e3c0c784e0b44c95"
)
_ACTIVE_REVIEWED_AUTHORIZATION_DIGEST: str | None = (
    "ab050d04f7ae7a6f920e94696d5b0988e4ad5331e9082d5ec61c30548166c111"
)
_ACTIVE_REVIEWED_DATABASE_PATH_DIGEST = (
    "cfcc91e8bab89beceff3ce8f5ecbc325705bd33b256e9d47ca8bdb9008833b80"
)
_ACTIVE_REVIEWED_EXACT_RUN = 30444371019
_ACTIVE_REVIEWED_FOCUSED_PASSED = 15
_ACTIVE_REVIEWED_FULL_PASSED = 3303
_ACTIVE_REVIEWED_ARTIFACT_SHA256 = (
    "79f7f6a2034ced8b04dfb3ae3ed69f56cdd6eb6c8f0da3cb740fc900f4ef80be"
)
_ACTIVE_REVIEWED_M2E_RUN = 30444371035
_ACTIVE_REVIEWED_SNAPSHOT_INTERVAL_EVENTS = 32
_ACTIVE_REVIEWED_MAX_EVENT_COUNT = 4096
_ACTIVE_REVIEWED_MAX_EVENT_BYTES = 16_777_216
_ACTIVE_REVIEWED_MAX_SNAPSHOT_COUNT = 128
_ACTIVE_REVIEWED_MAX_SNAPSHOT_BYTES = 16_777_216
_ACTIVE_REVIEWED_MAX_BACKUPS = 3


class M3CDormantWriterError(RuntimeError):
    """Base fail-closed bounded-writer error."""


class M3CDormantWriterAuthorizationError(M3CDormantWriterError):
    """Authorization is absent, malformed, or not the reviewed active packet."""


class M3CDormantWriterConflictError(M3CDormantWriterError):
    """Requested binding conflicts with the persisted stream head."""


class M3CDormantWriterRecoveryRequired(M3CDormantWriterError):
    """A committed append failed post-commit verification and requires recovery."""


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
        raise M3CDormantWriterAuthorizationError(
            f"{field} must be lowercase {length}-character hex"
        )
    return value


def _require_positive_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise M3CDormantWriterAuthorizationError(f"{field} must be a positive integer")
    return value


def database_path_digest(database_path: str | Path) -> str:
    """Digest an explicit lexical path without resolving or touching it."""

    text = str(database_path)
    if not text.strip() or text == ":memory:":
        raise M3CDormantWriterAuthorizationError("database path must be concrete")
    path = Path(database_path)
    if not path.is_absolute() or not path.name:
        raise M3CDormantWriterAuthorizationError("database path must be an absolute file path")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class WriterStorageLimits:
    snapshot_interval_events: int
    max_event_count: int
    max_event_bytes: int
    max_snapshot_count: int
    max_snapshot_bytes: int
    max_backups: int

    def __post_init__(self) -> None:
        for item in fields(self):
            _require_positive_int(getattr(self, item.name), field=item.name)

    @classmethod
    def from_policy(cls, policy: ShadowStoragePolicy) -> "WriterStorageLimits":
        if not isinstance(policy, ShadowStoragePolicy):
            raise M3CDormantWriterAuthorizationError(
                "policy must be ShadowStoragePolicy"
            )
        return cls(**{item.name: getattr(policy, item.name) for item in fields(cls)})

    def to_policy(self) -> ShadowStoragePolicy:
        return ShadowStoragePolicy(**self.to_mapping())

    def to_mapping(self) -> dict[str, int]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True, slots=True)
class WriterValidationPins:
    implementation_head: str
    exact_run: int
    focused_passed: int
    full_passed: int
    forward_gate_errors: int
    artifact_sha256: str
    m2e_run: int
    m2e_passed: int = 6
    m2e_required: int = 6

    def __post_init__(self) -> None:
        _require_hex(self.implementation_head, length=40, field="implementation_head")
        _require_hex(self.artifact_sha256, length=64, field="artifact_sha256")
        for name in ("exact_run", "focused_passed", "full_passed", "m2e_run"):
            _require_positive_int(getattr(self, name), field=name)
        if self.forward_gate_errors != 0:
            raise M3CDormantWriterAuthorizationError(
                "forward_gate_errors must be zero"
            )
        if (self.m2e_passed, self.m2e_required) != (6, 6):
            raise M3CDormantWriterAuthorizationError("M2-E validation must be 6/6")

    def to_mapping(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True, slots=True)
class GoalLifecycleWriterAuthorizationPacket:
    validation: WriterValidationPins
    storage_limits: WriterStorageLimits
    database_path_digest: str
    schema_version: str = WRITER_AUTHORIZATION_SCHEMA
    writer_schema_version: str = DORMANT_WRITER_SCHEMA
    store_schema_version: str = STORE_SCHEMA_VERSION
    stream_id: str = EVENT_STREAM
    event_type: str = EVENT_TYPE
    producer: str = PRODUCER_ID
    producer_version: str = PRODUCER_VERSION
    envelope_authority: str = SHADOW_AUTHORITY
    binding_authority: str = BINDING_AUTHORITY
    m3_c_e_prerequisite_merge_sha: str = M3_C_E_PREREQUISITE_MERGE_SHA
    m3_c_e_prerequisite_artifact_sha256: str = M3_C_E_PREREQUISITE_ARTIFACT_SHA256
    m3_c_g_prerequisite_merge_sha: str = M3_C_G_PREREQUISITE_MERGE_SHA
    m3_c_g_prerequisite_artifact_sha256: str = M3_C_G_PREREQUISITE_ARTIFACT_SHA256
    database_path_ownership: str = DATABASE_PATH_OWNERSHIP
    rollback_control_schema: str = WRITER_ROLLBACK_CONTROL_SCHEMA
    rollback_procedure: str = ROLLBACK_PROCEDURE
    human_reviewed: bool = True
    bounded_writer_authorized: bool = True
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    legacy_migration_authorized: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.validation, WriterValidationPins):
            raise M3CDormantWriterAuthorizationError(
                "validation must be WriterValidationPins"
            )
        if not isinstance(self.storage_limits, WriterStorageLimits):
            raise M3CDormantWriterAuthorizationError(
                "storage_limits must be WriterStorageLimits"
            )
        _require_hex(self.database_path_digest, length=64, field="database_path_digest")
        _require_hex(
            self.m3_c_e_prerequisite_merge_sha,
            length=40,
            field="m3_c_e_prerequisite_merge_sha",
        )
        _require_hex(
            self.m3_c_e_prerequisite_artifact_sha256,
            length=64,
            field="m3_c_e_prerequisite_artifact_sha256",
        )
        _require_hex(
            self.m3_c_g_prerequisite_merge_sha,
            length=40,
            field="m3_c_g_prerequisite_merge_sha",
        )
        _require_hex(
            self.m3_c_g_prerequisite_artifact_sha256,
            length=64,
            field="m3_c_g_prerequisite_artifact_sha256",
        )
        exact = (
            self.schema_version,
            self.writer_schema_version,
            self.store_schema_version,
            self.stream_id,
            self.event_type,
            self.producer,
            self.producer_version,
            self.envelope_authority,
            self.binding_authority,
            self.m3_c_e_prerequisite_merge_sha,
            self.m3_c_e_prerequisite_artifact_sha256,
            self.m3_c_g_prerequisite_merge_sha,
            self.m3_c_g_prerequisite_artifact_sha256,
            self.database_path_ownership,
            self.rollback_control_schema,
            self.rollback_procedure,
        )
        expected = (
            WRITER_AUTHORIZATION_SCHEMA,
            DORMANT_WRITER_SCHEMA,
            STORE_SCHEMA_VERSION,
            EVENT_STREAM,
            EVENT_TYPE,
            PRODUCER_ID,
            PRODUCER_VERSION,
            SHADOW_AUTHORITY,
            BINDING_AUTHORITY,
            M3_C_E_PREREQUISITE_MERGE_SHA,
            M3_C_E_PREREQUISITE_ARTIFACT_SHA256,
            M3_C_G_PREREQUISITE_MERGE_SHA,
            M3_C_G_PREREQUISITE_ARTIFACT_SHA256,
            DATABASE_PATH_OWNERSHIP,
            WRITER_ROLLBACK_CONTROL_SCHEMA,
            ROLLBACK_PROCEDURE,
        )
        if exact != expected:
            raise M3CDormantWriterAuthorizationError(
                "authorization packet constants do not match the reviewed contract"
            )
        if not self.human_reviewed or not self.bounded_writer_authorized:
            raise M3CDormantWriterAuthorizationError(
                "authorization packet must be explicitly human reviewed"
            )
        if any(
            (
                self.action_authorized,
                self.scheduler_authorized,
                self.speech_authorized,
                self.legacy_goal_authority_transferred,
                self.legacy_migration_authorized,
                self.m3_e_authority_open,
            )
        ):
            raise M3CDormantWriterAuthorizationError(
                "authorization packet escaped the bounded writer scope"
            )

    @property
    def implementation_head(self) -> str:
        return self.validation.implementation_head

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action_authorized": self.action_authorized,
            "binding_authority": self.binding_authority,
            "bounded_writer_authorized": self.bounded_writer_authorized,
            "database_path_digest": self.database_path_digest,
            "database_path_ownership": self.database_path_ownership,
            "envelope_authority": self.envelope_authority,
            "event_type": self.event_type,
            "human_reviewed": self.human_reviewed,
            "legacy_goal_authority_transferred": self.legacy_goal_authority_transferred,
            "legacy_migration_authorized": self.legacy_migration_authorized,
            "m3_c_e_prerequisite_artifact_sha256": self.m3_c_e_prerequisite_artifact_sha256,
            "m3_c_e_prerequisite_merge_sha": self.m3_c_e_prerequisite_merge_sha,
            "m3_c_g_prerequisite_artifact_sha256": self.m3_c_g_prerequisite_artifact_sha256,
            "m3_c_g_prerequisite_merge_sha": self.m3_c_g_prerequisite_merge_sha,
            "m3_e_authority_open": self.m3_e_authority_open,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "rollback_control_schema": self.rollback_control_schema,
            "rollback_procedure": self.rollback_procedure,
            "scheduler_authorized": self.scheduler_authorized,
            "schema_version": self.schema_version,
            "speech_authorized": self.speech_authorized,
            "storage_limits": self.storage_limits.to_mapping(),
            "store_schema_version": self.store_schema_version,
            "stream_id": self.stream_id,
            "validation": self.validation.to_mapping(),
            "writer_schema_version": self.writer_schema_version,
        }

    @property
    def authorization_digest(self) -> str:
        return _digest(self.to_mapping(), field="m3_c_h_writer_authorization")


def active_reviewed_writer_authorization_packet() -> GoalLifecycleWriterAuthorizationPacket:
    """Return the one immutable M3-C-I packet without exposing the private path."""

    packet = GoalLifecycleWriterAuthorizationPacket(
        validation=WriterValidationPins(
            implementation_head=_ACTIVE_REVIEWED_IMPLEMENTATION_HEAD or "",
            exact_run=_ACTIVE_REVIEWED_EXACT_RUN,
            focused_passed=_ACTIVE_REVIEWED_FOCUSED_PASSED,
            full_passed=_ACTIVE_REVIEWED_FULL_PASSED,
            forward_gate_errors=0,
            artifact_sha256=_ACTIVE_REVIEWED_ARTIFACT_SHA256,
            m2e_run=_ACTIVE_REVIEWED_M2E_RUN,
        ),
        storage_limits=WriterStorageLimits(
            snapshot_interval_events=_ACTIVE_REVIEWED_SNAPSHOT_INTERVAL_EVENTS,
            max_event_count=_ACTIVE_REVIEWED_MAX_EVENT_COUNT,
            max_event_bytes=_ACTIVE_REVIEWED_MAX_EVENT_BYTES,
            max_snapshot_count=_ACTIVE_REVIEWED_MAX_SNAPSHOT_COUNT,
            max_snapshot_bytes=_ACTIVE_REVIEWED_MAX_SNAPSHOT_BYTES,
            max_backups=_ACTIVE_REVIEWED_MAX_BACKUPS,
        ),
        database_path_digest=_ACTIVE_REVIEWED_DATABASE_PATH_DIGEST,
    )
    if packet.authorization_digest != _ACTIVE_REVIEWED_AUTHORIZATION_DIGEST:
        raise M3CDormantWriterAuthorizationError(
            "checked-in reviewed authorization packet digest is inconsistent"
        )
    return packet


def verify_active_writer_authorization(
    packet: GoalLifecycleWriterAuthorizationPacket | None,
) -> str:
    """Verify the exact M3-C-I active packet before any store construction."""

    if (
        _ACTIVE_REVIEWED_IMPLEMENTATION_HEAD is None
        or _ACTIVE_REVIEWED_AUTHORIZATION_DIGEST is None
    ):
        raise M3CDormantWriterAuthorizationError(
            "reviewed bounded-writer authorization packet is absent"
        )
    if not isinstance(packet, GoalLifecycleWriterAuthorizationPacket):
        raise M3CDormantWriterAuthorizationError(
            "authorization packet must be GoalLifecycleWriterAuthorizationPacket"
        )
    if packet.implementation_head != _ACTIVE_REVIEWED_IMPLEMENTATION_HEAD:
        raise M3CDormantWriterAuthorizationError(
            "authorization implementation head is not the active reviewed head"
        )
    if packet.authorization_digest != _ACTIVE_REVIEWED_AUTHORIZATION_DIGEST:
        raise M3CDormantWriterAuthorizationError(
            "authorization digest is not the active reviewed packet"
        )
    return packet.authorization_digest


def _authority_digest(state: CutoverAuthorityState) -> str:
    if not isinstance(state, CutoverAuthorityState):
        raise M3CDormantWriterAuthorizationError(
            "authority_state must be CutoverAuthorityState"
        )
    if (
        not state.cutover_authorized
        or not state.m3_authority_open
        or state.operational_rollback_active
        or state.event_store_role != EVENT_STORE_ACTIVE_ROLE
        or state.legacy_domain_authority_transfer_authorized
        or state.m3_e_affect_cutover_authorized
        or state.legacy_persistence_path_changed
    ):
        raise M3CDormantWriterAuthorizationError(
            "v4-native substrate authority is not active within M3-C scope"
        )
    return _digest(state.canonical_record, field="m3_c_h_authority_state")


@dataclass(frozen=True, slots=True)
class DormantWriterAppendReceipt:
    authorization_digest: str
    implementation_head: str
    database_path_digest: str
    binding_digest: str
    event_envelope_digest: str
    transition_id: str
    sequence: int
    before_count: int
    after_count: int
    before_chain_digest: str
    after_chain_digest: str
    append_transition_hash: str
    reducer_snapshot_digest: str
    integrity_report_digest: str
    snapshot_digest: str | None
    schema_version: str = WRITER_APPEND_RECEIPT_SCHEMA
    transaction_committed: bool = True
    inserted_rows: int = 1
    precommit_readback_verified: bool = True
    postcommit_readback_verified: bool = True
    stream_sequence_advanced_by_one: bool = True
    chain_advanced_and_verified: bool = True
    direct_reducer_equivalent: bool = True
    disposable_or_test_path_only: bool = True
    sqlite_write_performed: bool = True
    production_authoritative_append_performed: bool = False
    live_writer_installed: bool = False
    production_integration_performed: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        for name in (
            "authorization_digest",
            "database_path_digest",
            "binding_digest",
            "event_envelope_digest",
            "transition_id",
            "before_chain_digest",
            "after_chain_digest",
            "append_transition_hash",
            "reducer_snapshot_digest",
            "integrity_report_digest",
        ):
            _require_hex(getattr(self, name), length=64, field=name)
        _require_hex(self.implementation_head, length=40, field="implementation_head")
        if self.snapshot_digest is not None:
            _require_hex(self.snapshot_digest, length=64, field="snapshot_digest")
        if (
            isinstance(self.sequence, bool)
            or self.sequence < 1
            or self.before_count + 1 != self.after_count
            or self.inserted_rows != 1
            or self.schema_version != WRITER_APPEND_RECEIPT_SCHEMA
        ):
            raise M3CDormantWriterError("append receipt counts or schema are invalid")
        required_true = (
            self.transaction_committed,
            self.precommit_readback_verified,
            self.postcommit_readback_verified,
            self.stream_sequence_advanced_by_one,
            self.chain_advanced_and_verified,
            self.direct_reducer_equivalent,
            self.sqlite_write_performed,
        )
        required_false = (
            self.live_writer_installed,
            self.production_integration_performed,
            self.action_authorized,
            self.scheduler_authorized,
            self.speech_authorized,
            self.legacy_goal_authority_transferred,
            self.m3_e_authority_open,
        )
        if not all(required_true) or any(required_false):
            raise M3CDormantWriterError("append receipt escaped bounded writer scope")
        if self.disposable_or_test_path_only == self.production_authoritative_append_performed:
            raise M3CDormantWriterError(
                "append receipt must identify exactly one persistence-path class"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping(), field="m3_c_h_writer_append_receipt")


@dataclass(frozen=True, slots=True)
class DormantWriterRollbackControl:
    authorization_digest: str
    database_path_digest: str
    requested_by: str
    reason: str
    schema_version: str = WRITER_ROLLBACK_CONTROL_SCHEMA
    action: str = "disable_m3_c_goal_lifecycle_writer"
    preserve_immutable_history: bool = True
    restore_into_separate_path_required: bool = True
    writer_enabled: bool = False
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        _require_hex(self.authorization_digest, length=64, field="authorization_digest")
        _require_hex(self.database_path_digest, length=64, field="database_path_digest")
        if not isinstance(self.requested_by, str) or not self.requested_by.strip():
            raise M3CDormantWriterError("requested_by must be non-empty")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise M3CDormantWriterError("reason must be non-empty")
        if (
            self.schema_version != WRITER_ROLLBACK_CONTROL_SCHEMA
            or self.action != "disable_m3_c_goal_lifecycle_writer"
            or not self.preserve_immutable_history
            or not self.restore_into_separate_path_required
            or self.writer_enabled
            or self.legacy_goal_authority_transferred
            or self.m3_e_authority_open
        ):
            raise M3CDormantWriterError("rollback control escaped accepted scope")

    def to_mapping(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def control_digest(self) -> str:
        return _digest(self.to_mapping(), field="m3_c_h_writer_rollback_control")


def build_dormant_writer_rollback_control(
    packet: GoalLifecycleWriterAuthorizationPacket,
    *,
    database_path: str | Path,
    requested_by: str,
    reason: str,
) -> DormantWriterRollbackControl:
    if not isinstance(packet, GoalLifecycleWriterAuthorizationPacket):
        raise M3CDormantWriterError(
            "packet must be GoalLifecycleWriterAuthorizationPacket"
        )
    return DormantWriterRollbackControl(
        authorization_digest=packet.authorization_digest,
        database_path_digest=database_path_digest(database_path),
        requested_by=requested_by.strip(),
        reason=reason.strip(),
    )


class DormantGoalLifecycleWriter:
    """Explicit bounded writer requiring the exact reviewed packet per append."""

    __slots__ = (
        "_database_path",
        "_policy",
        "_operationally_disabled",
        "_disable_reason",
        "_accepted_authorization_digest",
    )

    def __init__(
        self,
        database_path: str | Path,
        *,
        policy: ShadowStoragePolicy,
    ) -> None:
        database_path_digest(database_path)
        if not isinstance(policy, ShadowStoragePolicy):
            raise M3CDormantWriterError("policy must be ShadowStoragePolicy")
        self._database_path = Path(database_path)
        self._policy = policy
        self._operationally_disabled = False
        self._disable_reason: str | None = None
        self._accepted_authorization_digest: str | None = None

    @property
    def database_path(self) -> Path:
        return self._database_path

    @property
    def database_path_digest(self) -> str:
        return database_path_digest(self._database_path)

    @property
    def policy(self) -> ShadowStoragePolicy:
        return self._policy

    @property
    def operationally_disabled(self) -> bool:
        return self._operationally_disabled

    @property
    def disable_reason(self) -> str | None:
        return self._disable_reason

    def _disable(self, reason: str) -> None:
        self._operationally_disabled = True
        self._disable_reason = reason

    def _verify_packet_for_writer(
        self,
        packet: GoalLifecycleWriterAuthorizationPacket | None,
    ) -> str:
        authorization_digest = verify_active_writer_authorization(packet)
        assert packet is not None
        if packet.database_path_digest != self.database_path_digest:
            raise M3CDormantWriterAuthorizationError(
                "authorization database path does not match this writer"
            )
        if packet.storage_limits != WriterStorageLimits.from_policy(self._policy):
            raise M3CDormantWriterAuthorizationError(
                "authorization storage limits do not match this writer"
            )
        if self._operationally_disabled:
            raise M3CDormantWriterAuthorizationError(
                "writer is operationally disabled and requires a new reviewed packet"
            )
        return authorization_digest

    @staticmethod
    def _replay_persisted_events(
        events: Sequence[EventEnvelope],
        *,
        authority_state_digest: str,
    ) -> GoalLifecycleReducerSnapshot:
        snapshot = GoalLifecycleReducerSnapshot.empty()
        for envelope in events:
            source = source_from_bound_envelope(
                envelope,
                authority_state_digest=authority_state_digest,
            )
            snapshot = apply_event_candidate_in_memory(snapshot, source)[0]
        return snapshot

    @staticmethod
    def _verify_binding(
        binding: GoalLifecycleSubstrateBindingCandidate,
        *,
        authority_state_digest: str,
        current_events: Sequence[EventEnvelope],
    ) -> EventEnvelope:
        if not isinstance(binding, GoalLifecycleSubstrateBindingCandidate):
            raise M3CDormantWriterConflictError(
                "binding must be GoalLifecycleSubstrateBindingCandidate"
            )
        if binding.authority_state_digest != authority_state_digest:
            raise M3CDormantWriterConflictError("binding authority state mismatch")
        expected_sequence = len(current_events) + 1
        expected_causation = None if not current_events else current_events[-1].event_id
        if binding.sequence != expected_sequence:
            raise M3CDormantWriterConflictError("binding sequence is not the exact next sequence")
        if binding.causation_event_id != expected_causation:
            raise M3CDormantWriterConflictError("binding causation does not match stream head")
        if any(
            (
                binding.authoritative_append_authorized,
                binding.authoritative_append_performed,
                binding.sqlite_write_performed,
                binding.live_writer_installed,
                binding.production_integration_performed,
                binding.action_authorized,
                binding.scheduler_authorized,
                binding.speech_authorized,
                binding.legacy_goal_authority_transferred,
                binding.m3_e_authority_open,
            )
        ):
            raise M3CDormantWriterConflictError("binding claims forbidden authority or effects")
        envelope = binding.event_envelope
        if (
            envelope.stream_id != EVENT_STREAM
            or envelope.event_type != EVENT_TYPE
            or envelope.producer != PRODUCER_ID
            or envelope.producer_version != PRODUCER_VERSION
            or envelope.authority != SHADOW_AUTHORITY
        ):
            raise M3CDormantWriterConflictError("binding envelope metadata mismatch")
        recovered = source_from_bound_envelope(
            envelope,
            authority_state_digest=authority_state_digest,
        )
        if recovered != binding.source:
            raise M3CDormantWriterConflictError("binding source round-trip mismatch")
        return envelope

    def append(
        self,
        binding: GoalLifecycleSubstrateBindingCandidate,
        *,
        authorization_packet: GoalLifecycleWriterAuthorizationPacket | None = None,
        authority_state: CutoverAuthorityState | None = None,
    ) -> DormantWriterAppendReceipt:
        """Append one binding only after the exact reviewed packet is verified."""

        authorization_digest = self._verify_packet_for_writer(authorization_packet)
        assert authorization_packet is not None
        resolved_authority = authority_state or active_cutover_authority()
        authority_state_digest = _authority_digest(resolved_authority)

        # No store object or path access occurs before all checks above.
        store = SQLiteShadowStore(self._database_path, policy=self._policy)
        store.initialize()
        before_integrity = store.integrity_check()
        if not before_integrity.valid:
            raise M3CDormantWriterConflictError(
                "database schema or integrity verification failed"
            )
        before_events = store.events(stream_id=EVENT_STREAM)
        before_snapshot = self._replay_persisted_events(
            before_events,
            authority_state_digest=authority_state_digest,
        )
        envelope = self._verify_binding(
            binding,
            authority_state_digest=authority_state_digest,
            current_events=before_events,
        )
        expected_snapshot = apply_event_candidate_in_memory(
            before_snapshot,
            binding.source,
        )[0]

        append_receipt: AppendReceipt = store.append(envelope)
        try:
            after_events = store.events(stream_id=EVENT_STREAM)
            if (
                len(after_events) != len(before_events) + 1
                or after_events[-1] != envelope
                or append_receipt.before_count != len(before_events)
                or append_receipt.after_count != len(after_events)
                or not append_receipt.readback_verified
                or not append_receipt.state_changed
            ):
                raise M3CDormantWriterRecoveryRequired(
                    "post-commit envelope readback mismatch"
                )
            replayed_snapshot = self._replay_persisted_events(
                after_events,
                authority_state_digest=authority_state_digest,
            )
            if replayed_snapshot.snapshot_digest != expected_snapshot.snapshot_digest:
                raise M3CDormantWriterRecoveryRequired(
                    "post-commit reducer replay mismatch"
                )
            after_integrity = store.integrity_check()
            if not after_integrity.valid:
                raise M3CDormantWriterRecoveryRequired(
                    "post-commit database integrity mismatch"
                )
            snapshot_digest = None
            if store.snapshot_due(EVENT_STREAM):
                snapshot_receipt = store.write_snapshot(
                    snapshot_id=f"m3c:goal-lifecycle:snapshot:{envelope.sequence:08d}",
                    stream_id=EVENT_STREAM,
                    through_sequence=envelope.sequence,
                    state=replayed_snapshot.to_mapping(),
                    state_schema_version=REDUCER_SNAPSHOT_VERSION,
                )
                if snapshot_receipt.state_digest != replayed_snapshot.snapshot_digest:
                    raise M3CDormantWriterRecoveryRequired(
                        "persisted snapshot state digest mismatch"
                    )
                snapshot_digest = snapshot_receipt.snapshot_digest
        except Exception as exc:
            self._disable("postcommit_verification_or_snapshot_failure")
            if isinstance(exc, M3CDormantWriterRecoveryRequired):
                raise
            raise M3CDormantWriterRecoveryRequired(
                "committed append requires preserved-database recovery"
            ) from exc

        self._accepted_authorization_digest = authorization_digest
        production_path = (
            authorization_digest == _ACTIVE_REVIEWED_AUTHORIZATION_DIGEST
            and self.database_path_digest == _ACTIVE_REVIEWED_DATABASE_PATH_DIGEST
        )
        return DormantWriterAppendReceipt(
            authorization_digest=authorization_digest,
            implementation_head=authorization_packet.implementation_head,
            database_path_digest=self.database_path_digest,
            binding_digest=binding.binding_digest,
            event_envelope_digest=envelope.digest,
            transition_id=binding.source.transition.transition_id,
            sequence=envelope.sequence,
            before_count=append_receipt.before_count,
            after_count=append_receipt.after_count,
            before_chain_digest=append_receipt.before_chain_digest,
            after_chain_digest=append_receipt.after_chain_digest,
            append_transition_hash=append_receipt.transition_hash,
            reducer_snapshot_digest=expected_snapshot.snapshot_digest,
            integrity_report_digest=after_integrity.report_digest,
            snapshot_digest=snapshot_digest,
            disposable_or_test_path_only=not production_path,
            production_authoritative_append_performed=production_path,
        )

    def apply_rollback(self, control: DormantWriterRollbackControl) -> str:
        """Disable future appends without opening, deleting, or repairing a store."""

        if not isinstance(control, DormantWriterRollbackControl):
            raise M3CDormantWriterError(
                "control must be DormantWriterRollbackControl"
            )
        if control.database_path_digest != self.database_path_digest:
            raise M3CDormantWriterError("rollback database path mismatch")
        accepted = (
            self._accepted_authorization_digest
            or _ACTIVE_REVIEWED_AUTHORIZATION_DIGEST
        )
        if accepted is None or control.authorization_digest != accepted:
            raise M3CDormantWriterError("rollback authorization digest mismatch")
        self._disable("reviewed_operational_rollback")
        return control.control_digest
