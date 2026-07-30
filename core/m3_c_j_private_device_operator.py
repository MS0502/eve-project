"""Exact-reviewed single-use private-device operator for the M3-C-J window.

Import is side-effect free. The reviewed implementation head and authorization
packet digest are pinned, but they authorize only one separately invoked bounded
operator command with explicit private paths, nonce material, and reviewed input.
The operator requires a new empty reviewed database path, creates a verified
baseline backup, appends exactly one four-transition lifecycle chain, disables
the writer, verifies final integrity/replay, restores the baseline into a
separate path, and returns private/public evidence mappings.
"""
from __future__ import annotations

import hashlib
import hmac
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.event_kernel import canonical_json_object
from core.m3_c_b_goal_selection_kernel import (
    ALLOWED_DRIVES,
    DriveSample,
    GoalCandidate,
    GoalSelectionReceipt,
    select_goal_proposal,
)
from core.m3_c_c_goal_lifecycle_kernel import (
    GoalLifecycleState,
    LifecycleEvidence,
    evaluate_lifecycle_transition,
)
from core.m3_c_d_goal_lifecycle_event_preflight import (
    EVENT_STREAM,
    GoalLifecycleReducerSnapshot,
    build_event_envelope_candidate,
)
from core.m3_c_e_goal_lifecycle_substrate_binding_preflight import (
    GoalLifecycleSubstrateBindingCandidate,
    build_substrate_binding_candidates,
)
from core.m3_c_h_dormant_goal_lifecycle_writer import (
    DormantGoalLifecycleWriter,
    DormantWriterAppendReceipt,
    GoalLifecycleWriterAuthorizationPacket,
    active_reviewed_writer_authorization_packet,
    build_dormant_writer_rollback_control,
    database_path_digest,
)
from core.m3_c_j_goal_lifecycle_observation_window import (
    ObservationWindowAuthorizationPacket,
    ObservationWindowBaseline,
    ObservationWindowReceipt,
    RollbackPreservationEvidence,
    active_reviewed_observation_window_authorization_packet,
    evaluate_observation_window,
)
from core.sqlite_shadow_store import (
    GENESIS_DIGEST,
    IntegrityReport,
    SQLiteShadowStore,
)

OPERATOR_AUTHORIZATION_SCHEMA = "eve.m3-c-j.private-device-operator-authorization.v1"
OPERATOR_INPUT_SCHEMA = "eve.m3-c-j.private-device-goal-input.v1"
OPERATOR_RECEIPT_SCHEMA = "eve.m3-c-j.private-device-operator-receipt.v1"
OPERATOR_PRIVATE_BUNDLE_SCHEMA = "eve.m3-c-j.private-device-private-bundle.v1"
OPERATOR_PUBLIC_REVIEW_SCHEMA = "eve.m3-c-j.private-device-public-review.v1"
M3_C_J_PIN_EXACT_HEAD = "532c595158ee68eb3268f75414bf6eaa23a79ffb"
M3_C_J_PIN_EXACT_RUN = 30451436253
M3_C_J_PIN_FOCUSED_PASSED = 11
M3_C_J_PIN_FULL_PASSED = 3315
M3_C_J_PIN_ARTIFACT_SHA256 = (
    "e488f98d0d60a4572ea1f64c383ee8f3a0d91d23b22477c431695b16e9d9d12d"
)
M3_C_J_PIN_M2E_RUN = 30451436272
M3_C_J_PIN_MERGE_SHA = "361ed88be399ed7650a946b58e713bc14253384e"
REQUIRED_TRANSITION_COUNT = 4
EXPECTED_LIFECYCLE_STATES = ("proposed", "validated", "eligible", "selected")

# M3-C-J exact-reviewed operator pins. These authorize only an explicit,
# single-use private-device command; import still performs no I/O or execution.
_ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD: str | None = (
    "d8eb3c2d6b576cc313712f831f8b2f1556cdefb2"
)
_ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST: str | None = (
    "e360c0e669af3ba89a6f552c81c67e3b3d908171665ed20b510a0044003d13a5"
)


class M3CPrivateDeviceOperatorError(RuntimeError):
    """Base fail-closed private-device operator error."""


class M3CPrivateDeviceOperatorAuthorizationError(M3CPrivateDeviceOperatorError):
    """Operator authorization is absent, malformed, or not exact-reviewed."""


class M3CPrivateDeviceOperatorInputError(M3CPrivateDeviceOperatorError):
    """Reviewed operator-private lifecycle material is malformed."""


class M3CPrivateDeviceOperatorExecutionError(M3CPrivateDeviceOperatorError):
    """Single-use database, backup, append, or rollback evidence failed."""


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
        raise M3CPrivateDeviceOperatorError(
            f"{field} must be lowercase {length}-character hex"
        )
    return value


def _require_nonempty(value: str, *, field: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise M3CPrivateDeviceOperatorError(
            f"{field} must be a bounded non-empty string"
        )
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise M3CPrivateDeviceOperatorInputError(f"{field} keys do not match schema")


def _integrity_mapping(report: IntegrityReport) -> dict[str, Any]:
    if not isinstance(report, IntegrityReport):
        raise M3CPrivateDeviceOperatorExecutionError(
            "integrity report must be IntegrityReport"
        )
    return {item.name: getattr(report, item.name) for item in fields(report)}


@dataclass(frozen=True, slots=True)
class PrivateDeviceGoalInput:
    candidate: GoalCandidate
    drive_samples: tuple[DriveSample, ...]
    candidate_human_reviewed: bool = True
    drive_samples_human_reviewed: bool = True
    new_window_material: bool = True
    phone_witness_replayed: bool = False
    retained_sequences_replayed: bool = False
    legacy_goal_authority_transfer_requested: bool = False
    m3_e_requested: bool = False
    schema_version: str = OPERATOR_INPUT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, GoalCandidate):
            raise M3CPrivateDeviceOperatorInputError("candidate must be GoalCandidate")
        if len(self.drive_samples) != len(ALLOWED_DRIVES):
            raise M3CPrivateDeviceOperatorInputError(
                "drive samples must contain exactly eight drives"
            )
        if any(not isinstance(item, DriveSample) for item in self.drive_samples):
            raise M3CPrivateDeviceOperatorInputError(
                "drive samples must be DriveSample values"
            )
        if tuple(item.drive for item in self.drive_samples) != ALLOWED_DRIVES:
            raise M3CPrivateDeviceOperatorInputError(
                "drive samples must follow canonical drive order"
            )
        if (
            self.schema_version != OPERATOR_INPUT_SCHEMA
            or not self.candidate_human_reviewed
            or not self.drive_samples_human_reviewed
            or not self.new_window_material
            or self.phone_witness_replayed
            or self.retained_sequences_replayed
            or self.legacy_goal_authority_transfer_requested
            or self.m3_e_requested
        ):
            raise M3CPrivateDeviceOperatorInputError(
                "operator input escaped the new observation-only scope"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PrivateDeviceGoalInput":
        expected = {
            "candidate",
            "candidate_human_reviewed",
            "drive_samples",
            "drive_samples_human_reviewed",
            "legacy_goal_authority_transfer_requested",
            "m3_e_requested",
            "new_window_material",
            "phone_witness_replayed",
            "retained_sequences_replayed",
            "schema_version",
        }
        _require_exact_keys(value, expected, field="operator input")
        candidate_mapping = value["candidate"]
        candidate_expected = {
            "base_value",
            "candidate_id",
            "continuity",
            "cost",
            "decision_epoch",
            "drive_alignment",
            "drive_confidence",
            "evidence_digest",
            "expected_value",
            "risk",
            "schema_version",
            "scoring_policy_version",
            "semantic_goal_id",
            "urgency",
        }
        _require_exact_keys(candidate_mapping, candidate_expected, field="candidate")
        candidate = GoalCandidate(
            semantic_goal_id=candidate_mapping["semantic_goal_id"],
            decision_epoch=candidate_mapping["decision_epoch"],
            evidence_digest=candidate_mapping["evidence_digest"],
            base_value=candidate_mapping["base_value"],
            expected_value=candidate_mapping["expected_value"],
            urgency=candidate_mapping["urgency"],
            continuity=candidate_mapping["continuity"],
            cost=candidate_mapping["cost"],
            risk=candidate_mapping["risk"],
            drive_alignment=candidate_mapping["drive_alignment"],
            drive_confidence=candidate_mapping["drive_confidence"],
            schema_version=candidate_mapping["schema_version"],
            scoring_policy_version=candidate_mapping["scoring_policy_version"],
        )
        if candidate.to_mapping() != dict(candidate_mapping):
            raise M3CPrivateDeviceOperatorInputError(
                "candidate derived identity or canonical mapping mismatch"
            )
        sample_expected = {
            "drive",
            "dynamics_version",
            "lower_bound",
            "normalized",
            "predicate_version",
            "replay_elapsed_seconds",
            "sample_digest",
            "upper_bound",
            "value",
        }
        samples: list[DriveSample] = []
        raw_samples = value["drive_samples"]
        if not isinstance(raw_samples, list):
            raise M3CPrivateDeviceOperatorInputError("drive_samples must be a list")
        for raw in raw_samples:
            _require_exact_keys(raw, sample_expected, field="drive sample")
            sample = DriveSample(
                drive=raw["drive"],
                value=raw["value"],
                lower_bound=raw["lower_bound"],
                upper_bound=raw["upper_bound"],
                sample_digest=raw["sample_digest"],
                replay_elapsed_seconds=raw["replay_elapsed_seconds"],
                dynamics_version=raw["dynamics_version"],
                predicate_version=raw["predicate_version"],
            )
            if sample.to_mapping() != dict(raw):
                raise M3CPrivateDeviceOperatorInputError(
                    "drive sample canonical mapping mismatch"
                )
            samples.append(sample)
        return cls(
            candidate=candidate,
            drive_samples=tuple(samples),
            candidate_human_reviewed=value["candidate_human_reviewed"],
            drive_samples_human_reviewed=value["drive_samples_human_reviewed"],
            new_window_material=value["new_window_material"],
            phone_witness_replayed=value["phone_witness_replayed"],
            retained_sequences_replayed=value["retained_sequences_replayed"],
            legacy_goal_authority_transfer_requested=value[
                "legacy_goal_authority_transfer_requested"
            ],
            m3_e_requested=value["m3_e_requested"],
            schema_version=value["schema_version"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_mapping(),
            "candidate_human_reviewed": self.candidate_human_reviewed,
            "drive_samples": [item.to_mapping() for item in self.drive_samples],
            "drive_samples_human_reviewed": self.drive_samples_human_reviewed,
            "legacy_goal_authority_transfer_requested": (
                self.legacy_goal_authority_transfer_requested
            ),
            "m3_e_requested": self.m3_e_requested,
            "new_window_material": self.new_window_material,
            "phone_witness_replayed": self.phone_witness_replayed,
            "retained_sequences_replayed": self.retained_sequences_replayed,
            "schema_version": self.schema_version,
        }

    @property
    def input_digest(self) -> str:
        return _digest(self.to_mapping(), field="m3_c_j_private_device_goal_input")

    def private_binding_digest(self, private_nonce: bytes) -> str:
        if not isinstance(private_nonce, bytes) or len(private_nonce) < 32:
            raise M3CPrivateDeviceOperatorInputError(
                "private nonce must contain at least 32 bytes"
            )
        return hmac.new(
            private_nonce,
            _canonical(self.to_mapping(), field="m3_c_j_private_input_hmac").encode(
                "utf-8"
            ),
            hashlib.sha256,
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class PrivateDeviceOperatorAuthorizationPacket:
    operator_implementation_head: str
    window_authorization_digest: str
    window_implementation_head: str
    writer_authorization_digest: str
    writer_implementation_head: str
    database_path_digest: str
    max_window_events: int
    required_transition_count: int = REQUIRED_TRANSITION_COUNT
    schema_version: str = OPERATOR_AUTHORIZATION_SCHEMA
    prerequisite_exact_head: str = M3_C_J_PIN_EXACT_HEAD
    prerequisite_exact_run: int = M3_C_J_PIN_EXACT_RUN
    prerequisite_focused_passed: int = M3_C_J_PIN_FOCUSED_PASSED
    prerequisite_full_passed: int = M3_C_J_PIN_FULL_PASSED
    prerequisite_artifact_sha256: str = M3_C_J_PIN_ARTIFACT_SHA256
    prerequisite_m2e_run: int = M3_C_J_PIN_M2E_RUN
    prerequisite_m2e_passed: int = 6
    prerequisite_m2e_required: int = 6
    prerequisite_merge_sha: str = M3_C_J_PIN_MERGE_SHA
    human_reviewed: bool = True
    single_use_empty_database_required: bool = True
    baseline_backup_required: bool = True
    writer_disable_after_append_required: bool = True
    separate_restore_required: bool = True
    production_append_authorized: bool = True
    runtime_integration_authorized: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    legacy_migration_authorized: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        for name, length in (
            ("operator_implementation_head", 40),
            ("window_authorization_digest", 64),
            ("window_implementation_head", 40),
            ("writer_authorization_digest", 64),
            ("writer_implementation_head", 40),
            ("database_path_digest", 64),
            ("prerequisite_exact_head", 40),
            ("prerequisite_artifact_sha256", 64),
            ("prerequisite_merge_sha", 40),
        ):
            _require_hex(getattr(self, name), length=length, field=name)
        if self.required_transition_count != REQUIRED_TRANSITION_COUNT:
            raise M3CPrivateDeviceOperatorAuthorizationError(
                "operator must append exactly four lifecycle transitions"
            )
        if not 1 <= self.required_transition_count <= self.max_window_events:
            raise M3CPrivateDeviceOperatorAuthorizationError(
                "operator transition count exceeds window bound"
            )
        if (
            self.schema_version != OPERATOR_AUTHORIZATION_SCHEMA
            or self.prerequisite_exact_head != M3_C_J_PIN_EXACT_HEAD
            or self.prerequisite_exact_run != M3_C_J_PIN_EXACT_RUN
            or self.prerequisite_focused_passed != M3_C_J_PIN_FOCUSED_PASSED
            or self.prerequisite_full_passed != M3_C_J_PIN_FULL_PASSED
            or self.prerequisite_artifact_sha256 != M3_C_J_PIN_ARTIFACT_SHA256
            or self.prerequisite_m2e_run != M3_C_J_PIN_M2E_RUN
            or (self.prerequisite_m2e_passed, self.prerequisite_m2e_required) != (6, 6)
            or self.prerequisite_merge_sha != M3_C_J_PIN_MERGE_SHA
        ):
            raise M3CPrivateDeviceOperatorAuthorizationError(
                "M3-C-J reviewed-pin prerequisite mismatch"
            )
        required_true = (
            self.human_reviewed,
            self.single_use_empty_database_required,
            self.baseline_backup_required,
            self.writer_disable_after_append_required,
            self.separate_restore_required,
            self.production_append_authorized,
        )
        forbidden = (
            self.runtime_integration_authorized,
            self.action_authorized,
            self.scheduler_authorized,
            self.speech_authorized,
            self.legacy_goal_authority_transferred,
            self.legacy_migration_authorized,
            self.m3_e_authority_open,
        )
        if not all(required_true) or any(forbidden):
            raise M3CPrivateDeviceOperatorAuthorizationError(
                "operator authorization escaped bounded observation scope"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def authorization_digest(self) -> str:
        return _digest(self.to_mapping(), field="m3_c_j_private_device_authorization")


def build_private_device_operator_authorization_candidate(
    *,
    operator_implementation_head: str,
) -> PrivateDeviceOperatorAuthorizationPacket:
    writer = active_reviewed_writer_authorization_packet()
    window = active_reviewed_observation_window_authorization_packet()
    return PrivateDeviceOperatorAuthorizationPacket(
        operator_implementation_head=operator_implementation_head,
        window_authorization_digest=window.authorization_digest,
        window_implementation_head=window.window_implementation_head,
        writer_authorization_digest=writer.authorization_digest,
        writer_implementation_head=writer.implementation_head,
        database_path_digest=writer.database_path_digest,
        max_window_events=window.max_window_events,
    )


def active_reviewed_private_device_operator_authorization_packet(
) -> PrivateDeviceOperatorAuthorizationPacket:
    """Return the one immutable reviewed operator packet without performing I/O."""

    packet = build_private_device_operator_authorization_candidate(
        operator_implementation_head=_ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD or "",
    )
    if packet.authorization_digest != _ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "checked-in private-device operator packet digest is inconsistent"
        )
    return packet


def verify_active_private_device_operator_authorization(
    packet: PrivateDeviceOperatorAuthorizationPacket | None,
) -> str:
    """Verify the exact reviewed operator packet before any private path access."""

    if (
        _ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD is None
        or _ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST is None
    ):
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "reviewed private-device operator authorization is absent"
        )
    if not isinstance(packet, PrivateDeviceOperatorAuthorizationPacket):
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "packet must be PrivateDeviceOperatorAuthorizationPacket"
        )
    if packet.operator_implementation_head != _ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "operator implementation head is not the active reviewed head"
        )
    if packet.authorization_digest != _ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "operator authorization digest is not the active reviewed packet"
        )
    return packet.authorization_digest


def build_reviewed_lifecycle_bindings(
    operator_input: PrivateDeviceGoalInput,
) -> tuple[GoalSelectionReceipt, tuple[GoalLifecycleSubstrateBindingCandidate, ...]]:
    if not isinstance(operator_input, PrivateDeviceGoalInput):
        raise M3CPrivateDeviceOperatorInputError(
            "operator_input must be PrivateDeviceGoalInput"
        )
    samples = {item.drive: item for item in operator_input.drive_samples}
    selection = select_goal_proposal([operator_input.candidate], samples)
    if (
        selection.decision_kind != "initial_selection"
        or selection.selected_candidate_id != operator_input.candidate.candidate_id
        or not selection.transition_eligible
        or len(selection.scored_candidates) != 1
    ):
        raise M3CPrivateDeviceOperatorInputError(
            "reviewed candidate is not an exact initial-selection lifecycle input"
        )
    score = selection.scored_candidates[0]
    state = GoalLifecycleState(
        candidate_id=operator_input.candidate.candidate_id,
        semantic_goal_id=operator_input.candidate.semantic_goal_id,
        decision_epoch=operator_input.candidate.decision_epoch,
        evidence_digest=operator_input.candidate.evidence_digest,
    )
    evidences = (
        LifecycleEvidence(candidate_score=score, logical_step=1),
        LifecycleEvidence(
            candidate_score=score,
            logical_step=2,
            validation_status="passed",
        ),
        LifecycleEvidence(candidate_score=score, logical_step=3),
        LifecycleEvidence(
            candidate_score=score,
            logical_step=4,
            selection_receipt=selection,
        ),
    )
    sources = []
    reached = []
    for evidence in evidences:
        decision = evaluate_lifecycle_transition(state, evidence)
        if decision.transition is None:
            raise M3CPrivateDeviceOperatorInputError(
                "reviewed lifecycle input did not produce the required transition"
            )
        sources.append(build_event_envelope_candidate(decision.transition))
        state = decision.transition.next_state()
        reached.append(state.lifecycle_state)
    if tuple(reached) != EXPECTED_LIFECYCLE_STATES:
        raise M3CPrivateDeviceOperatorInputError(
            "reviewed lifecycle chain does not reach selected through exact states"
        )
    bindings = build_substrate_binding_candidates(tuple(sources))
    if len(bindings) != REQUIRED_TRANSITION_COUNT:
        raise M3CPrivateDeviceOperatorInputError(
            "operator binding count is not exactly four"
        )
    return selection, bindings


@dataclass(frozen=True, slots=True)
class PrivateDeviceOperatorReceipt:
    operator_authorization_digest: str
    operator_implementation_head: str
    repository_head: str
    launch_attestation_id: str
    runtime_instance_id: str
    private_input_digest: str
    private_input_binding_digest: str
    selection_receipt_digest: str
    baseline_digest: str
    rollback_control_digest: str
    rollback_evidence_digest: str
    window_receipt_digest: str
    database_path_digest: str
    backup_sha256: str
    backup_path_digest: str
    restore_path_digest: str
    append_receipt_digests: tuple[str, ...]
    schema_version: str = OPERATOR_RECEIPT_SCHEMA
    exact_transition_count: int = REQUIRED_TRANSITION_COUNT
    database_was_absent_before_start: bool = True
    baseline_backup_verified: bool = True
    writer_disabled_after_append: bool = True
    final_integrity_verified: bool = True
    final_replay_verified: bool = True
    separate_restore_verified: bool = True
    production_append_performed: bool = True
    runtime_integration_performed: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    legacy_migration_authorized: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        for name, length in (
            ("operator_authorization_digest", 64),
            ("operator_implementation_head", 40),
            ("repository_head", 40),
            ("private_input_digest", 64),
            ("private_input_binding_digest", 64),
            ("selection_receipt_digest", 64),
            ("baseline_digest", 64),
            ("rollback_control_digest", 64),
            ("rollback_evidence_digest", 64),
            ("window_receipt_digest", 64),
            ("database_path_digest", 64),
            ("backup_sha256", 64),
            ("backup_path_digest", 64),
            ("restore_path_digest", 64),
        ):
            _require_hex(getattr(self, name), length=length, field=name)
        _require_nonempty(self.launch_attestation_id, field="launch_attestation_id")
        _require_nonempty(self.runtime_instance_id, field="runtime_instance_id")
        if len(self.append_receipt_digests) != REQUIRED_TRANSITION_COUNT:
            raise M3CPrivateDeviceOperatorExecutionError(
                "operator receipt must contain exactly four append receipts"
            )
        for value in self.append_receipt_digests:
            _require_hex(value, length=64, field="append_receipt_digest")
        required_true = (
            self.database_was_absent_before_start,
            self.baseline_backup_verified,
            self.writer_disabled_after_append,
            self.final_integrity_verified,
            self.final_replay_verified,
            self.separate_restore_verified,
            self.production_append_performed,
        )
        forbidden = (
            self.runtime_integration_performed,
            self.action_authorized,
            self.scheduler_authorized,
            self.speech_authorized,
            self.legacy_goal_authority_transferred,
            self.legacy_migration_authorized,
            self.m3_e_authority_open,
        )
        if (
            self.schema_version != OPERATOR_RECEIPT_SCHEMA
            or self.exact_transition_count != REQUIRED_TRANSITION_COUNT
            or not all(required_true)
            or any(forbidden)
        ):
            raise M3CPrivateDeviceOperatorExecutionError(
                "operator receipt escaped bounded observation scope"
            )

    def to_mapping(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["append_receipt_digests"] = list(self.append_receipt_digests)
        return result

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping(), field="m3_c_j_private_device_receipt")


@dataclass(frozen=True, slots=True)
class PrivateDeviceOperatorBundle:
    operator_input: PrivateDeviceGoalInput
    selection_receipt: GoalSelectionReceipt
    baseline: ObservationWindowBaseline
    append_receipts: tuple[DormantWriterAppendReceipt, ...]
    final_integrity_report: IntegrityReport
    rollback_evidence: RollbackPreservationEvidence
    window_receipt: ObservationWindowReceipt
    operator_receipt: PrivateDeviceOperatorReceipt

    def private_mapping(self) -> dict[str, Any]:
        return {
            "append_receipts": [item.to_mapping() for item in self.append_receipts],
            "baseline": self.baseline.to_mapping(),
            "final_integrity_report": _integrity_mapping(self.final_integrity_report),
            "operator_input": self.operator_input.to_mapping(),
            "operator_receipt": self.operator_receipt.to_mapping(),
            "rollback_evidence": self.rollback_evidence.to_mapping(),
            "schema_version": OPERATOR_PRIVATE_BUNDLE_SCHEMA,
            "selection_receipt": self.selection_receipt.to_mapping(),
            "window_receipt": self.window_receipt.to_mapping(),
        }

    def public_review_mapping(self) -> dict[str, Any]:
        return {
            "database_path_plaintext_public": False,
            "legacy_goal_authority_transferred": False,
            "legacy_migration_authorized": False,
            "m3_e_authority_open": False,
            "operator_input_public": False,
            "operator_receipt": self.operator_receipt.to_mapping(),
            "phone_witness_replayed": False,
            "retained_sequences_replayed": False,
            "runtime_integration_performed": False,
            "schema_version": OPERATOR_PUBLIC_REVIEW_SCHEMA,
            "window_receipt": self.window_receipt.to_mapping(),
        }


def _require_new_database_path(path: Path) -> None:
    if not path.is_absolute() or not path.name:
        raise M3CPrivateDeviceOperatorExecutionError(
            "database path must be an absolute file path"
        )
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"), Path(f"{path}-journal")):
        if candidate.exists():
            raise M3CPrivateDeviceOperatorExecutionError(
                "single-use operator requires an absent database and sidecars"
            )


def execute_private_device_observation_window(
    authorization_packet: PrivateDeviceOperatorAuthorizationPacket | None,
    *,
    operator_input: PrivateDeviceGoalInput,
    private_nonce: bytes,
    repository_head: str,
    launch_attestation_id: str,
    runtime_instance_id: str,
    database_path: str | Path,
    backup_directory: str | Path,
    restore_path: str | Path,
) -> PrivateDeviceOperatorBundle:
    authorization_digest = verify_active_private_device_operator_authorization(
        authorization_packet
    )
    assert authorization_packet is not None
    _require_hex(repository_head, length=40, field="repository_head")
    if repository_head != authorization_packet.operator_implementation_head:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "repository head differs from reviewed operator implementation"
        )
    _require_nonempty(launch_attestation_id, field="launch_attestation_id")
    _require_nonempty(runtime_instance_id, field="runtime_instance_id")
    if not isinstance(operator_input, PrivateDeviceGoalInput):
        raise M3CPrivateDeviceOperatorInputError(
            "operator_input must be PrivateDeviceGoalInput"
        )
    if not isinstance(private_nonce, bytes) or len(private_nonce) < 32:
        raise M3CPrivateDeviceOperatorInputError(
            "private nonce must contain at least 32 bytes"
        )

    writer_packet: GoalLifecycleWriterAuthorizationPacket = (
        active_reviewed_writer_authorization_packet()
    )
    window_packet: ObservationWindowAuthorizationPacket = (
        active_reviewed_observation_window_authorization_packet()
    )
    if (
        writer_packet.authorization_digest
        != authorization_packet.writer_authorization_digest
        or writer_packet.implementation_head
        != authorization_packet.writer_implementation_head
        or window_packet.authorization_digest
        != authorization_packet.window_authorization_digest
        or window_packet.window_implementation_head
        != authorization_packet.window_implementation_head
    ):
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "active writer/window packets differ from operator authorization"
        )

    database = Path(database_path)
    backup_root = Path(backup_directory)
    restore = Path(restore_path)
    if database_path_digest(database) != authorization_packet.database_path_digest:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "database path differs from reviewed private path digest"
        )
    if not backup_root.is_absolute() or not restore.is_absolute():
        raise M3CPrivateDeviceOperatorExecutionError(
            "backup and restore paths must be absolute"
        )
    if restore == database or restore.parent == database.parent:
        raise M3CPrivateDeviceOperatorExecutionError(
            "restore path must remain in a separate directory"
        )
    _require_new_database_path(database)
    _require_new_database_path(restore)

    selection, bindings = build_reviewed_lifecycle_bindings(operator_input)
    if len(bindings) != authorization_packet.required_transition_count:
        raise M3CPrivateDeviceOperatorExecutionError(
            "binding count differs from operator authorization"
        )

    store = SQLiteShadowStore(database, policy=writer_packet.storage_limits.to_policy())
    store.initialize()
    baseline_integrity = store.integrity_check()
    if (
        not baseline_integrity.valid
        or baseline_integrity.event_count != 0
        or baseline_integrity.snapshot_count != 0
        or baseline_integrity.chain_head_digest != GENESIS_DIGEST
    ):
        raise M3CPrivateDeviceOperatorExecutionError(
            "new production database did not produce an empty verified baseline"
        )
    backup = store.create_backup(backup_root, backup_ordinal=1)
    empty_snapshot = GoalLifecycleReducerSnapshot.empty()
    baseline = ObservationWindowBaseline(
        authorization_digest=window_packet.authorization_digest,
        database_path_digest=writer_packet.database_path_digest,
        start_sequence=0,
        start_event_count=0,
        start_chain_digest=GENESIS_DIGEST,
        start_reducer_snapshot_digest=empty_snapshot.snapshot_digest,
        integrity_report_digest=baseline_integrity.report_digest,
        backup_sha256=backup.backup_sha256,
        backup_path_digest=database_path_digest(backup.backup_path),
    )

    writer = DormantGoalLifecycleWriter(
        database,
        policy=writer_packet.storage_limits.to_policy(),
    )
    append_receipts = tuple(
        writer.append(binding, authorization_packet=writer_packet)
        for binding in bindings
    )
    if any(not item.production_authoritative_append_performed for item in append_receipts):
        raise M3CPrivateDeviceOperatorExecutionError(
            "append receipt did not identify the reviewed production path"
        )
    final_store = SQLiteShadowStore(
        database,
        policy=writer_packet.storage_limits.to_policy(),
    )
    final_store.initialize()
    final_integrity = final_store.integrity_check()
    if not final_integrity.valid:
        raise M3CPrivateDeviceOperatorExecutionError(
            "final production database integrity verification failed"
        )

    rollback_control = build_dormant_writer_rollback_control(
        writer_packet,
        database_path=database,
        requested_by="m3-c-j-private-device-operator",
        reason="bounded-observation-window-complete",
    )
    rollback_control_digest = writer.apply_rollback(rollback_control)
    if not writer.operationally_disabled:
        raise M3CPrivateDeviceOperatorExecutionError(
            "writer did not enter reviewed disabled state"
        )

    restore.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(backup.backup_path, restore)
    restored_store = SQLiteShadowStore(
        restore,
        policy=writer_packet.storage_limits.to_policy(),
    )
    restored_store.initialize()
    restored_integrity = restored_store.integrity_check()
    if (
        not restored_integrity.valid
        or restored_integrity.event_count != 0
        or restored_integrity.snapshot_count != 0
        or restored_integrity.chain_head_digest != GENESIS_DIGEST
        or restored_store.events(stream_id=EVENT_STREAM) != ()
    ):
        raise M3CPrivateDeviceOperatorExecutionError(
            "separate-path restore did not reproduce the empty baseline"
        )
    rollback_evidence = RollbackPreservationEvidence(
        authorization_digest=window_packet.authorization_digest,
        database_path_digest=writer_packet.database_path_digest,
        backup_sha256=backup.backup_sha256,
        backup_path_digest=database_path_digest(backup.backup_path),
        restore_path_digest=database_path_digest(restore),
        pre_window_snapshot_digest=empty_snapshot.snapshot_digest,
        restored_snapshot_digest=empty_snapshot.snapshot_digest,
        restored_integrity_report_digest=restored_integrity.report_digest,
    )
    window_receipt = evaluate_observation_window(
        window_packet,
        baseline=baseline,
        append_receipts=append_receipts,
        final_integrity_report=final_integrity,
        final_reducer_snapshot_digest=append_receipts[-1].reducer_snapshot_digest,
        rollback_evidence=rollback_evidence,
    )
    selection_receipt_digest = _digest(
        selection.to_mapping(),
        field="m3_c_j_private_device_selection_receipt",
    )
    operator_receipt = PrivateDeviceOperatorReceipt(
        operator_authorization_digest=authorization_digest,
        operator_implementation_head=authorization_packet.operator_implementation_head,
        repository_head=repository_head,
        launch_attestation_id=launch_attestation_id,
        runtime_instance_id=runtime_instance_id,
        private_input_digest=operator_input.input_digest,
        private_input_binding_digest=operator_input.private_binding_digest(private_nonce),
        selection_receipt_digest=selection_receipt_digest,
        baseline_digest=baseline.baseline_digest,
        rollback_control_digest=rollback_control_digest,
        rollback_evidence_digest=rollback_evidence.evidence_digest,
        window_receipt_digest=window_receipt.receipt_digest,
        database_path_digest=writer_packet.database_path_digest,
        backup_sha256=backup.backup_sha256,
        backup_path_digest=database_path_digest(backup.backup_path),
        restore_path_digest=database_path_digest(restore),
        append_receipt_digests=tuple(
            item.receipt_digest for item in append_receipts
        ),
    )
    return PrivateDeviceOperatorBundle(
        operator_input=operator_input,
        selection_receipt=selection,
        baseline=baseline,
        append_receipts=append_receipts,
        final_integrity_report=final_integrity,
        rollback_evidence=rollback_evidence,
        window_receipt=window_receipt,
        operator_receipt=operator_receipt,
    )
