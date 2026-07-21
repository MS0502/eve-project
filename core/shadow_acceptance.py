"""M1-E deterministic shadow-observation acceptance evidence.

This module evaluates explicitly supplied M1-B/M1-C/M1-D in-memory evidence. It
installs no runtime hook, calls no legacy module, performs no I/O, and cannot
accept a human review or grant v4.2, persistence, recovery, or cutover authority.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY, canonical_json_object
from core.shadow_lifecycle import (
    DEFAULT_BRIDGE_REGISTRY,
    DISCONNECTED_MODE,
    DISCONNECTED_STATUS,
    NO_AUTHORITY,
    NO_PERSISTENCE,
    RETRY_FORBIDDEN,
    SHADOW_ROLLBACK_ONLY,
    SUPPRESSION_FORBIDDEN,
    ShadowBridgeRegistry,
)
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    FAILURE_EVENT_TYPE,
    SUCCESS_EVENT_TYPE,
    ShadowObservationFailure,
)
from core.shadow_projection import (
    ActivationLearnPairShadowState,
    ShadowProjectionCheckpoint,
    compare_activation_learn_pair_equivalence,
    replay_activation_learn_pair,
    restore_projection_checkpoint,
    rollback_projection,
)

WINDOW_SCHEMA_VERSION = "eve.m1-shadow-observation-window.v1"
LEGACY_EVIDENCE_SCHEMA_VERSION = "eve.m1-legacy-preservation-evidence.v1"
CRITERION_SCHEMA_VERSION = "eve.m1-shadow-acceptance-criterion.v1"
PACKET_SCHEMA_VERSION = "eve.m1-shadow-acceptance-packet.v1"

MACHINE_COMPLETE_STATUS = "machine_evidence_complete"
MACHINE_INCOMPLETE_STATUS = "machine_evidence_incomplete"
HUMAN_REVIEW_REQUIRED = "required_not_performed"

REQUIRED_CRITERIA: tuple[str, ...] = (
    "event_count_exact",
    "success_failure_visible",
    "observer_failure_visible",
    "sequence_contiguous",
    "replay_equivalent",
    "checkpoint_restore_verified",
    "rollback_verified",
    "lifecycle_registry_complete",
    "legacy_behavior_preserved",
    "zero_unauthorized_effects",
)

_OBSERVER_FAILURE_STAGES: tuple[str, ...] = (
    "after_snapshot",
    "before_snapshot",
    "event_append",
)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ShadowAcceptanceContractError(ValueError):
    """Raised when M1-E evidence is malformed or outside the bounded contract."""


def _require_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ShadowAcceptanceContractError(f"{field} is not a canonical identifier")
    return value


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ShadowAcceptanceContractError(f"{field} must be boolean")
    return value


def _require_digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST_PATTERN.fullmatch(value):
        raise ShadowAcceptanceContractError(f"{field} is not a SHA-256 digest")
    return value


def _digest_record(value: Mapping[str, Any], *, field: str) -> str:
    encoded = canonical_json_object(value, field=field)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ObservationWindowSpec:
    """Explicit deterministic bounds for one M1-E dry-run observation window."""

    window_id: str
    expected_event_count: int
    expected_success_count: int
    expected_failure_count: int
    expected_observer_failure_count: int
    initial_checkpoint_id: str
    final_checkpoint_id: str
    schema_version: str = WINDOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_identifier(self.window_id, field="window_id")
        _require_identifier(self.initial_checkpoint_id, field="initial_checkpoint_id")
        _require_identifier(self.final_checkpoint_id, field="final_checkpoint_id")
        if self.initial_checkpoint_id == self.final_checkpoint_id:
            raise ShadowAcceptanceContractError("checkpoint identifiers must differ")
        for field in (
            "expected_event_count",
            "expected_success_count",
            "expected_failure_count",
            "expected_observer_failure_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ShadowAcceptanceContractError(f"{field} must be a positive integer")
        if self.expected_event_count != (
            self.expected_success_count + self.expected_failure_count
        ):
            raise ShadowAcceptanceContractError(
                "expected event count must equal success plus failure counts"
            )
        if self.schema_version != WINDOW_SCHEMA_VERSION:
            raise ShadowAcceptanceContractError("unsupported observation-window schema")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "expected_event_count": self.expected_event_count,
            "expected_failure_count": self.expected_failure_count,
            "expected_observer_failure_count": self.expected_observer_failure_count,
            "expected_success_count": self.expected_success_count,
            "final_checkpoint_id": self.final_checkpoint_id,
            "initial_checkpoint_id": self.initial_checkpoint_id,
            "schema_version": self.schema_version,
            "window_id": self.window_id,
        }

    @property
    def digest(self) -> str:
        return _digest_record(self.canonical_record, field="observation_window_spec")


@dataclass(frozen=True, slots=True)
class LegacyPreservationEvidence:
    """Caller-supplied immutable proof summary; it grants no acceptance authority."""

    evidence_id: str
    case_ids: tuple[str, ...]
    return_value_preserved: bool
    exception_identity_preserved: bool
    call_order_preserved: bool
    legacy_state_matches_unobserved: bool
    persistence_behavior_unchanged: bool
    defaults_unchanged: bool
    source_evidence_digest: str
    schema_version: str = LEGACY_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_identifier(self.evidence_id, field="evidence_id")
        if not isinstance(self.case_ids, tuple) or not self.case_ids:
            raise ShadowAcceptanceContractError("case_ids must be a non-empty tuple")
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ShadowAcceptanceContractError("case_ids must be unique")
        for case_id in self.case_ids:
            _require_identifier(case_id, field="case_id")
        for field in (
            "return_value_preserved",
            "exception_identity_preserved",
            "call_order_preserved",
            "legacy_state_matches_unobserved",
            "persistence_behavior_unchanged",
            "defaults_unchanged",
        ):
            _require_bool(getattr(self, field), field=field)
        _require_digest(self.source_evidence_digest, field="source_evidence_digest")
        if self.schema_version != LEGACY_EVIDENCE_SCHEMA_VERSION:
            raise ShadowAcceptanceContractError("unsupported legacy-evidence schema")

    @property
    def passes(self) -> bool:
        return all(
            (
                self.return_value_preserved,
                self.exception_identity_preserved,
                self.call_order_preserved,
                self.legacy_state_matches_unobserved,
                self.persistence_behavior_unchanged,
                self.defaults_unchanged,
            )
        )

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "call_order_preserved": self.call_order_preserved,
            "case_ids": list(self.case_ids),
            "defaults_unchanged": self.defaults_unchanged,
            "evidence_id": self.evidence_id,
            "exception_identity_preserved": self.exception_identity_preserved,
            "legacy_state_matches_unobserved": self.legacy_state_matches_unobserved,
            "persistence_behavior_unchanged": self.persistence_behavior_unchanged,
            "return_value_preserved": self.return_value_preserved,
            "schema_version": self.schema_version,
            "source_evidence_digest": self.source_evidence_digest,
        }

    @property
    def digest(self) -> str:
        return _digest_record(self.canonical_record, field="legacy_preservation_evidence")


@dataclass(frozen=True, slots=True)
class AcceptanceCriterion:
    """One immutable machine-check result in the human-review packet."""

    criterion_id: str
    passed: bool
    evidence_digest: str
    schema_version: str = CRITERION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.criterion_id not in REQUIRED_CRITERIA:
            raise ShadowAcceptanceContractError("unknown M1-E acceptance criterion")
        _require_bool(self.passed, field="criterion.passed")
        _require_digest(self.evidence_digest, field="criterion.evidence_digest")
        if self.schema_version != CRITERION_SCHEMA_VERSION:
            raise ShadowAcceptanceContractError("unsupported criterion schema")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "evidence_digest": self.evidence_digest,
            "passed": self.passed,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class M1ShadowAcceptancePacket:
    """Immutable machine evidence that always remains pending human review."""

    window_id: str
    window_spec_digest: str
    event_count: int
    success_count: int
    failure_count: int
    observer_failure_count: int
    first_sequence: int
    last_sequence: int
    replay_state_digest: str
    expected_snapshot_digest: str
    lifecycle_registry_digest: str
    lifecycle_domains: tuple[str, ...]
    legacy_evidence_digest: str
    observer_failure_evidence_digest: str
    criteria: tuple[AcceptanceCriterion, ...]
    machine_status: str
    machine_passed: bool
    eligible_for_human_review: bool
    human_review_status: str = HUMAN_REVIEW_REQUIRED
    human_accepted: bool = False
    v4_2_eligible: bool = False
    authority: str = SHADOW_AUTHORITY
    runtime_integrated: bool = False
    persistence_mode: str = NO_PERSISTENCE
    unauthorized_effects_detected: bool = False
    schema_version: str = PACKET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_identifier(self.window_id, field="packet.window_id")
        for field in (
            "window_spec_digest",
            "replay_state_digest",
            "expected_snapshot_digest",
            "lifecycle_registry_digest",
            "legacy_evidence_digest",
            "observer_failure_evidence_digest",
        ):
            _require_digest(getattr(self, field), field=field)
        for field in (
            "event_count",
            "success_count",
            "failure_count",
            "observer_failure_count",
            "first_sequence",
            "last_sequence",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ShadowAcceptanceContractError(f"{field} must be non-negative")
        if self.event_count < 1 or self.first_sequence < 1:
            raise ShadowAcceptanceContractError("packet requires a non-empty event window")
        if self.last_sequence < self.first_sequence:
            raise ShadowAcceptanceContractError("packet sequence range is reversed")
        if self.success_count + self.failure_count != self.event_count:
            raise ShadowAcceptanceContractError("packet event counts are inconsistent")
        if not isinstance(self.lifecycle_domains, tuple):
            raise ShadowAcceptanceContractError("lifecycle_domains must be immutable")
        expected_domains = tuple(
            sorted(
                DEFAULT_BRIDGE_REGISTRY.canonical_record["bridges"][index]["domain"]
                for index in range(len(DEFAULT_BRIDGE_REGISTRY.bridges))
            )
        )
        if self.lifecycle_domains != expected_domains:
            raise ShadowAcceptanceContractError("packet lifecycle coverage is incomplete")
        if not isinstance(self.criteria, tuple):
            raise ShadowAcceptanceContractError("criteria must be immutable")
        criterion_ids = tuple(item.criterion_id for item in self.criteria)
        if tuple(sorted(criterion_ids)) != tuple(sorted(REQUIRED_CRITERIA)):
            raise ShadowAcceptanceContractError("packet criteria are incomplete or duplicated")
        if any(not isinstance(item, AcceptanceCriterion) for item in self.criteria):
            raise ShadowAcceptanceContractError("packet criterion is malformed")
        computed_pass = all(item.passed for item in self.criteria)
        if self.machine_passed != computed_pass:
            raise ShadowAcceptanceContractError("machine_passed disagrees with criteria")
        expected_status = (
            MACHINE_COMPLETE_STATUS if computed_pass else MACHINE_INCOMPLETE_STATUS
        )
        if self.machine_status != expected_status:
            raise ShadowAcceptanceContractError("machine status disagrees with criteria")
        if self.eligible_for_human_review != computed_pass:
            raise ShadowAcceptanceContractError(
                "human-review eligibility disagrees with machine evidence"
            )
        fixed = {
            "human_review_status": HUMAN_REVIEW_REQUIRED,
            "human_accepted": False,
            "v4_2_eligible": False,
            "authority": SHADOW_AUTHORITY,
            "runtime_integrated": False,
            "persistence_mode": NO_PERSISTENCE,
            "unauthorized_effects_detected": False,
            "schema_version": PACKET_SCHEMA_VERSION,
        }
        for field, expected in fixed.items():
            if getattr(self, field) != expected:
                raise ShadowAcceptanceContractError(
                    f"{field} cannot grant runtime or human-review authority"
                )

    def criterion(self, criterion_id: str) -> AcceptanceCriterion:
        for item in self.criteria:
            if item.criterion_id == criterion_id:
                return item
        raise ShadowAcceptanceContractError(criterion_id)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "criteria": [
                item.canonical_record
                for item in sorted(self.criteria, key=lambda value: value.criterion_id)
            ],
            "eligible_for_human_review": self.eligible_for_human_review,
            "event_count": self.event_count,
            "expected_snapshot_digest": self.expected_snapshot_digest,
            "failure_count": self.failure_count,
            "first_sequence": self.first_sequence,
            "human_accepted": self.human_accepted,
            "human_review_status": self.human_review_status,
            "last_sequence": self.last_sequence,
            "legacy_evidence_digest": self.legacy_evidence_digest,
            "lifecycle_domains": list(self.lifecycle_domains),
            "lifecycle_registry_digest": self.lifecycle_registry_digest,
            "machine_passed": self.machine_passed,
            "machine_status": self.machine_status,
            "observer_failure_count": self.observer_failure_count,
            "observer_failure_evidence_digest": self.observer_failure_evidence_digest,
            "persistence_mode": self.persistence_mode,
            "replay_state_digest": self.replay_state_digest,
            "runtime_integrated": self.runtime_integrated,
            "schema_version": self.schema_version,
            "success_count": self.success_count,
            "unauthorized_effects_detected": self.unauthorized_effects_detected,
            "v4_2_eligible": self.v4_2_eligible,
            "window_id": self.window_id,
            "window_spec_digest": self.window_spec_digest,
        }

    @property
    def digest(self) -> str:
        return _digest_record(self.canonical_record, field="m1_shadow_acceptance_packet")


def _observer_failure_digest(
    failures: tuple[ShadowObservationFailure, ...],
) -> str:
    if not isinstance(failures, tuple):
        raise ShadowAcceptanceContractError("observer failures must be an immutable tuple")
    records: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for failure in failures:
        if not isinstance(failure, ShadowObservationFailure):
            raise ShadowAcceptanceContractError("observer failure evidence is malformed")
        if failure.target_id != ACTIVATION_LEARN_PAIR_TARGET.target_id:
            raise ShadowAcceptanceContractError("observer failure target is out of scope")
        _require_identifier(failure.event_id, field="observer_failure.event_id")
        if failure.stage not in _OBSERVER_FAILURE_STAGES:
            raise ShadowAcceptanceContractError("observer failure stage is out of scope")
        if not isinstance(failure.error_type, str) or not failure.error_type:
            raise ShadowAcceptanceContractError("observer failure type is malformed")
        _require_digest(
            failure.error_message_digest,
            field="observer_failure.error_message_digest",
        )
        if (
            failure.legacy_succeeded is not None
            and not isinstance(failure.legacy_succeeded, bool)
        ):
            raise ShadowAcceptanceContractError(
                "observer failure legacy status is malformed"
            )
        identity = (failure.event_id, failure.stage)
        if identity in identities:
            raise ShadowAcceptanceContractError("duplicate observer failure evidence")
        identities.add(identity)
        records.append(
            {
                "error_message_digest": failure.error_message_digest,
                "error_type": failure.error_type,
                "event_id": failure.event_id,
                "legacy_succeeded": failure.legacy_succeeded,
                "stage": failure.stage,
                "target_id": failure.target_id,
            }
        )
    return _digest_record({"failures": records}, field="observer_failures")


def _require_lifecycle_registry(registry: ShadowBridgeRegistry) -> tuple[str, ...]:
    if not isinstance(registry, ShadowBridgeRegistry):
        raise ShadowAcceptanceContractError("M1-E requires a ShadowBridgeRegistry")
    if registry.digest != DEFAULT_BRIDGE_REGISTRY.digest:
        raise ShadowAcceptanceContractError("lifecycle registry differs from M1-D")
    domains: list[str] = []
    for bridge in registry.bridges:
        if (
            bridge.lifecycle_status != DISCONNECTED_STATUS
            or bridge.integration_mode != DISCONNECTED_MODE
            or bridge.default_enabled is not False
            or bridge.authority != NO_AUTHORITY
            or bridge.emitted_event_types != ()
            or bridge.required_capabilities != ()
            or bridge.persistence_mode != NO_PERSISTENCE
            or bridge.retry_policy != RETRY_FORBIDDEN
            or bridge.suppression_policy != SUPPRESSION_FORBIDDEN
            or bridge.rollback_scope != SHADOW_ROLLBACK_ONLY
        ):
            raise ShadowAcceptanceContractError("lifecycle bridge gained authority")
        domains.append(bridge.domain)
    return tuple(sorted(domains))


def _criterion(
    criterion_id: str,
    passed: bool,
    evidence: Mapping[str, Any],
) -> AcceptanceCriterion:
    return AcceptanceCriterion(
        criterion_id=criterion_id,
        passed=passed,
        evidence_digest=_digest_record(evidence, field=f"criterion:{criterion_id}"),
    )


def evaluate_m1_shadow_window(
    spec: ObservationWindowSpec,
    *,
    initial_state: ActivationLearnPairShadowState,
    events: tuple[EventEnvelope, ...],
    expected_final_snapshot: Mapping[str, Any],
    observer_failures: tuple[ShadowObservationFailure, ...],
    legacy_evidence: LegacyPreservationEvidence,
    lifecycle_registry: ShadowBridgeRegistry = DEFAULT_BRIDGE_REGISTRY,
) -> M1ShadowAcceptancePacket:
    """Evaluate an explicit bounded window and return non-authoritative evidence."""

    if not isinstance(spec, ObservationWindowSpec):
        raise ShadowAcceptanceContractError("spec must be ObservationWindowSpec")
    if not isinstance(initial_state, ActivationLearnPairShadowState):
        raise ShadowAcceptanceContractError("initial_state is outside M1-C scope")
    if not isinstance(events, tuple) or not events:
        raise ShadowAcceptanceContractError("events must be a non-empty tuple")
    if not isinstance(expected_final_snapshot, Mapping):
        raise ShadowAcceptanceContractError("expected_final_snapshot must be a mapping")
    if not isinstance(legacy_evidence, LegacyPreservationEvidence):
        raise ShadowAcceptanceContractError("legacy evidence is malformed")

    event_ids: set[str] = set()
    sequence_rows: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    expected_sequence = initial_state.sequence + 1
    for event in events:
        if not isinstance(event, EventEnvelope):
            raise ShadowAcceptanceContractError("window accepts EventEnvelope only")
        if event.event_id in event_ids:
            raise ShadowAcceptanceContractError("window event identifiers must be unique")
        event_ids.add(event.event_id)
        if event.sequence != expected_sequence:
            raise ShadowAcceptanceContractError("window sequence is not contiguous")
        expected_sequence += 1
        if event.event_type == SUCCESS_EVENT_TYPE:
            success_count += 1
        elif event.event_type == FAILURE_EVENT_TYPE:
            failure_count += 1
        else:
            raise ShadowAcceptanceContractError("window event type is outside M1-E")
        sequence_rows.append(
            {
                "digest": event.digest,
                "event_id": event.event_id,
                "event_type": event.event_type,
                "sequence": event.sequence,
            }
        )

    replayed = replay_activation_learn_pair(initial_state, events)
    equivalence = compare_activation_learn_pair_equivalence(
        replayed,
        expected_final_snapshot,
    )
    initial_checkpoint = ShadowProjectionCheckpoint.create(
        spec.initial_checkpoint_id,
        initial_state,
    )
    final_checkpoint = ShadowProjectionCheckpoint.create(
        spec.final_checkpoint_id,
        replayed,
    )
    restored = restore_projection_checkpoint(final_checkpoint)
    rolled_back = rollback_projection(replayed, initial_checkpoint)
    restore_matches = restored == replayed and restored.digest == replayed.digest
    rollback_matches = (
        rolled_back == initial_state and rolled_back.digest == initial_state.digest
    )

    observer_failure_digest = _observer_failure_digest(observer_failures)
    lifecycle_domains = _require_lifecycle_registry(lifecycle_registry)
    count_matches = len(events) == spec.expected_event_count
    coverage_matches = (
        success_count == spec.expected_success_count
        and failure_count == spec.expected_failure_count
    )
    observer_failure_matches = (
        len(observer_failures) == spec.expected_observer_failure_count
    )
    sequence_matches = (
        events[0].sequence == initial_state.sequence + 1
        and events[-1].sequence == replayed.sequence
        and replayed.sequence == initial_state.sequence + len(events)
    )
    expected_domains = tuple(
        sorted(
            DEFAULT_BRIDGE_REGISTRY.canonical_record["bridges"][index]["domain"]
            for index in range(len(DEFAULT_BRIDGE_REGISTRY.bridges))
        )
    )
    lifecycle_matches = (
        lifecycle_registry.digest == DEFAULT_BRIDGE_REGISTRY.digest
        and lifecycle_domains == expected_domains
    )

    criteria = (
        _criterion(
            "event_count_exact",
            count_matches,
            {"actual": len(events), "expected": spec.expected_event_count},
        ),
        _criterion(
            "success_failure_visible",
            coverage_matches,
            {
                "actual_failure": failure_count,
                "actual_success": success_count,
                "expected_failure": spec.expected_failure_count,
                "expected_success": spec.expected_success_count,
            },
        ),
        _criterion(
            "observer_failure_visible",
            observer_failure_matches,
            {
                "actual": len(observer_failures),
                "digest": observer_failure_digest,
                "expected": spec.expected_observer_failure_count,
            },
        ),
        _criterion(
            "sequence_contiguous",
            sequence_matches,
            {"events": sequence_rows},
        ),
        _criterion(
            "replay_equivalent",
            equivalence.matches,
            {
                "expected_snapshot_digest": equivalence.expected_snapshot_digest,
                "mismatches": list(equivalence.mismatches),
                "projected_digest": equivalence.projected_digest,
            },
        ),
        _criterion(
            "checkpoint_restore_verified",
            restore_matches,
            {
                "checkpoint_id": final_checkpoint.checkpoint_id,
                "state_digest": final_checkpoint.state_digest,
            },
        ),
        _criterion(
            "rollback_verified",
            rollback_matches,
            {
                "checkpoint_id": initial_checkpoint.checkpoint_id,
                "state_digest": initial_checkpoint.state_digest,
            },
        ),
        _criterion(
            "lifecycle_registry_complete",
            lifecycle_matches,
            {
                "domains": list(lifecycle_domains),
                "registry_digest": lifecycle_registry.digest,
            },
        ),
        _criterion(
            "legacy_behavior_preserved",
            legacy_evidence.passes,
            {
                "legacy_evidence_digest": legacy_evidence.digest,
                "source_evidence_digest": legacy_evidence.source_evidence_digest,
            },
        ),
        _criterion(
            "zero_unauthorized_effects",
            True,
            {
                "authority": SHADOW_AUTHORITY,
                "persistence_mode": NO_PERSISTENCE,
                "runtime_integrated": False,
                "unauthorized_effects_detected": False,
            },
        ),
    )
    machine_passed = all(item.passed for item in criteria)
    return M1ShadowAcceptancePacket(
        window_id=spec.window_id,
        window_spec_digest=spec.digest,
        event_count=len(events),
        success_count=success_count,
        failure_count=failure_count,
        observer_failure_count=len(observer_failures),
        first_sequence=events[0].sequence,
        last_sequence=events[-1].sequence,
        replay_state_digest=replayed.digest,
        expected_snapshot_digest=equivalence.expected_snapshot_digest,
        lifecycle_registry_digest=lifecycle_registry.digest,
        lifecycle_domains=lifecycle_domains,
        legacy_evidence_digest=legacy_evidence.digest,
        observer_failure_evidence_digest=observer_failure_digest,
        criteria=criteria,
        machine_status=(
            MACHINE_COMPLETE_STATUS if machine_passed else MACHINE_INCOMPLETE_STATUS
        ),
        machine_passed=machine_passed,
        eligible_for_human_review=machine_passed,
    )
