"""M1-E deterministic, in-memory shadow-acceptance evidence.

The evaluator consumes explicit M1-B/M1-C/M1-D evidence only. It installs no
runtime hook, calls no legacy module, performs no I/O, and cannot approve a human
review or grant v4.2, persistence, recovery, scheduling, or cutover authority.
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

REQUIRED_CRITERIA = (
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
_OBSERVER_FAILURE_STAGES = ("after_snapshot", "before_snapshot", "event_append")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_DOMAINS = tuple(
    sorted(bridge.domain for bridge in DEFAULT_BRIDGE_REGISTRY.bridges)
)


class ShadowAcceptanceContractError(ValueError):
    """Malformed or out-of-scope M1-E evidence."""


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ShadowAcceptanceContractError(f"{field} is not a canonical identifier")
    return value


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ShadowAcceptanceContractError(f"{field} is not a SHA-256 digest")
    return value


def _record_digest(value: Mapping[str, Any], field: str) -> str:
    text = canonical_json_object(value, field=field)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _nonnegative(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ShadowAcceptanceContractError(f"{field} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class ObservationWindowSpec:
    window_id: str
    expected_event_count: int
    expected_success_count: int
    expected_failure_count: int
    expected_observer_failure_count: int
    initial_checkpoint_id: str
    final_checkpoint_id: str
    schema_version: str = WINDOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.window_id, "window_id")
        _identifier(self.initial_checkpoint_id, "initial_checkpoint_id")
        _identifier(self.final_checkpoint_id, "final_checkpoint_id")
        if self.initial_checkpoint_id == self.final_checkpoint_id:
            raise ShadowAcceptanceContractError("checkpoint identifiers must differ")
        values = (
            self.expected_event_count,
            self.expected_success_count,
            self.expected_failure_count,
            self.expected_observer_failure_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values
        ):
            raise ShadowAcceptanceContractError(
                "window counts must be positive integers"
            )
        if self.expected_event_count != (
            self.expected_success_count + self.expected_failure_count
        ):
            raise ShadowAcceptanceContractError(
                "event count must equal success plus failure"
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
        return _record_digest(self.canonical_record, "observation_window_spec")


@dataclass(frozen=True, slots=True)
class LegacyPreservationEvidence:
    evidence_id: str
    case_ids: tuple[str, ...]
    return_value_preserved: bool
    exception_identity_preserved: bool
    call_order_preserved: bool
    legacy_state_matches_unobserved: bool
    persistence_behavior_unchanged: bool
    defaults_unchanged: bool
    external_effects_unchanged: bool
    source_evidence_digest: str
    schema_version: str = LEGACY_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "evidence_id")
        if not isinstance(self.case_ids, tuple) or not self.case_ids:
            raise ShadowAcceptanceContractError("case_ids must be a non-empty tuple")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ShadowAcceptanceContractError("case_ids must be unique")
        for value in self.case_ids:
            _identifier(value, "case_id")
        flags = (
            self.return_value_preserved,
            self.exception_identity_preserved,
            self.call_order_preserved,
            self.legacy_state_matches_unobserved,
            self.persistence_behavior_unchanged,
            self.defaults_unchanged,
            self.external_effects_unchanged,
        )
        if any(not isinstance(value, bool) for value in flags):
            raise ShadowAcceptanceContractError(
                "legacy evidence flags must be boolean"
            )
        _digest(self.source_evidence_digest, "source_evidence_digest")
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
                self.external_effects_unchanged,
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
            "external_effects_unchanged": self.external_effects_unchanged,
            "legacy_state_matches_unobserved": self.legacy_state_matches_unobserved,
            "persistence_behavior_unchanged": self.persistence_behavior_unchanged,
            "return_value_preserved": self.return_value_preserved,
            "schema_version": self.schema_version,
            "source_evidence_digest": self.source_evidence_digest,
        }

    @property
    def digest(self) -> str:
        return _record_digest(self.canonical_record, "legacy_preservation_evidence")


@dataclass(frozen=True, slots=True)
class AcceptanceCriterion:
    criterion_id: str
    passed: bool
    evidence_digest: str
    schema_version: str = CRITERION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.criterion_id not in REQUIRED_CRITERIA:
            raise ShadowAcceptanceContractError("unknown acceptance criterion")
        if not isinstance(self.passed, bool):
            raise ShadowAcceptanceContractError("criterion result must be boolean")
        _digest(self.evidence_digest, "criterion.evidence_digest")
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
        _identifier(self.window_id, "packet.window_id")
        for field in (
            "window_spec_digest",
            "replay_state_digest",
            "expected_snapshot_digest",
            "lifecycle_registry_digest",
            "legacy_evidence_digest",
            "observer_failure_evidence_digest",
        ):
            _digest(getattr(self, field), field)
        for field in (
            "event_count",
            "success_count",
            "failure_count",
            "observer_failure_count",
            "first_sequence",
            "last_sequence",
        ):
            _nonnegative(getattr(self, field), field)
        if (
            self.event_count < 1
            or self.first_sequence < 1
            or self.last_sequence < self.first_sequence
        ):
            raise ShadowAcceptanceContractError("packet event window is invalid")
        if self.success_count + self.failure_count != self.event_count:
            raise ShadowAcceptanceContractError("packet event counts disagree")
        if self.lifecycle_domains != _EXPECTED_DOMAINS:
            raise ShadowAcceptanceContractError(
                "packet lifecycle coverage is incomplete"
            )
        if not isinstance(self.criteria, tuple) or any(
            not isinstance(value, AcceptanceCriterion) for value in self.criteria
        ):
            raise ShadowAcceptanceContractError(
                "criteria must be immutable criterion values"
            )
        if tuple(sorted(value.criterion_id for value in self.criteria)) != tuple(
            sorted(REQUIRED_CRITERIA)
        ):
            raise ShadowAcceptanceContractError(
                "criteria are incomplete or duplicated"
            )
        computed = all(value.passed for value in self.criteria)
        status = MACHINE_COMPLETE_STATUS if computed else MACHINE_INCOMPLETE_STATUS
        if self.machine_passed != computed or self.machine_status != status:
            raise ShadowAcceptanceContractError(
                "machine status disagrees with criteria"
            )
        if self.eligible_for_human_review != computed:
            raise ShadowAcceptanceContractError(
                "review eligibility disagrees with criteria"
            )
        fixed = (
            (self.human_review_status, HUMAN_REVIEW_REQUIRED),
            (self.human_accepted, False),
            (self.v4_2_eligible, False),
            (self.authority, SHADOW_AUTHORITY),
            (self.runtime_integrated, False),
            (self.persistence_mode, NO_PERSISTENCE),
            (self.unauthorized_effects_detected, False),
            (self.schema_version, PACKET_SCHEMA_VERSION),
        )
        if any(actual != expected for actual, expected in fixed):
            raise ShadowAcceptanceContractError(
                "packet cannot grant runtime or review authority"
            )

    def criterion(self, criterion_id: str) -> AcceptanceCriterion:
        for value in self.criteria:
            if value.criterion_id == criterion_id:
                return value
        raise ShadowAcceptanceContractError(criterion_id)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "criteria": [
                value.canonical_record
                for value in sorted(
                    self.criteria,
                    key=lambda item: item.criterion_id,
                )
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
        return _record_digest(
            self.canonical_record,
            "m1_shadow_acceptance_packet",
        )


def _observer_failure_digest(
    failures: tuple[ShadowObservationFailure, ...],
) -> str:
    if not isinstance(failures, tuple):
        raise ShadowAcceptanceContractError("observer failures must be immutable")
    records: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for failure in failures:
        if not isinstance(failure, ShadowObservationFailure):
            raise ShadowAcceptanceContractError("observer failure is malformed")
        if failure.target_id != ACTIVATION_LEARN_PAIR_TARGET.target_id:
            raise ShadowAcceptanceContractError(
                "observer failure target is out of scope"
            )
        _identifier(failure.event_id, "observer_failure.event_id")
        if failure.stage not in _OBSERVER_FAILURE_STAGES:
            raise ShadowAcceptanceContractError(
                "observer failure stage is out of scope"
            )
        if not isinstance(failure.error_type, str) or not failure.error_type:
            raise ShadowAcceptanceContractError("observer failure type is malformed")
        _digest(
            failure.error_message_digest,
            "observer_failure.error_message_digest",
        )
        if (
            failure.legacy_succeeded is not None
            and not isinstance(failure.legacy_succeeded, bool)
        ):
            raise ShadowAcceptanceContractError(
                "observer failure status is malformed"
            )
        identity = (failure.event_id, failure.stage)
        if identity in identities:
            raise ShadowAcceptanceContractError("duplicate observer failure")
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
    return _record_digest({"failures": records}, "observer_failures")


def _lifecycle_domains(registry: ShadowBridgeRegistry) -> tuple[str, ...]:
    if not isinstance(registry, ShadowBridgeRegistry):
        raise ShadowAcceptanceContractError(
            "M1-E requires a ShadowBridgeRegistry"
        )
    if registry.digest != DEFAULT_BRIDGE_REGISTRY.digest:
        raise ShadowAcceptanceContractError("lifecycle registry differs from M1-D")
    domains: list[str] = []
    for bridge in registry.bridges:
        actual = (
            bridge.lifecycle_status,
            bridge.integration_mode,
            bridge.default_enabled,
            bridge.authority,
            bridge.emitted_event_types,
            bridge.required_capabilities,
            bridge.persistence_mode,
            bridge.retry_policy,
            bridge.suppression_policy,
            bridge.rollback_scope,
        )
        expected = (
            DISCONNECTED_STATUS,
            DISCONNECTED_MODE,
            False,
            NO_AUTHORITY,
            (),
            (),
            NO_PERSISTENCE,
            RETRY_FORBIDDEN,
            SUPPRESSION_FORBIDDEN,
            SHADOW_ROLLBACK_ONLY,
        )
        if actual != expected:
            raise ShadowAcceptanceContractError(
                "lifecycle bridge gained authority"
            )
        domains.append(bridge.domain)
    return tuple(sorted(domains))


def _criterion(
    name: str,
    passed: bool,
    evidence: Mapping[str, Any],
) -> AcceptanceCriterion:
    return AcceptanceCriterion(
        name,
        passed,
        _record_digest(evidence, f"criterion:{name}"),
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
    """Return a non-authoritative machine packet for explicit human review."""

    if not isinstance(spec, ObservationWindowSpec):
        raise ShadowAcceptanceContractError("spec is malformed")
    if not isinstance(initial_state, ActivationLearnPairShadowState):
        raise ShadowAcceptanceContractError("initial state is outside M1-C")
    if not isinstance(events, tuple) or not events:
        raise ShadowAcceptanceContractError(
            "events must be a non-empty tuple"
        )
    if not isinstance(expected_final_snapshot, Mapping):
        raise ShadowAcceptanceContractError(
            "expected snapshot must be a mapping"
        )
    if not isinstance(legacy_evidence, LegacyPreservationEvidence):
        raise ShadowAcceptanceContractError("legacy evidence is malformed")

    event_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    success = 0
    failure = 0
    expected_sequence = initial_state.sequence + 1
    for event in events:
        if not isinstance(event, EventEnvelope):
            raise ShadowAcceptanceContractError(
                "window accepts EventEnvelope only"
            )
        if event.event_id in event_ids:
            raise ShadowAcceptanceContractError("event IDs must be unique")
        if event.sequence != expected_sequence:
            raise ShadowAcceptanceContractError(
                "window sequence is not contiguous"
            )
        event_ids.add(event.event_id)
        expected_sequence += 1
        if event.event_type == SUCCESS_EVENT_TYPE:
            success += 1
        elif event.event_type == FAILURE_EVENT_TYPE:
            failure += 1
        else:
            raise ShadowAcceptanceContractError("event type is outside M1-E")
        rows.append(
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
    restore_ok = restored == replayed and restored.digest == replayed.digest
    rollback_ok = (
        rolled_back == initial_state
        and rolled_back.digest == initial_state.digest
    )
    observer_digest = _observer_failure_digest(observer_failures)
    domains = _lifecycle_domains(lifecycle_registry)

    expected_case_ids = tuple(event.event_id for event in events) + tuple(
        value.event_id for value in observer_failures
    )
    legacy_ok = (
        legacy_evidence.passes
        and legacy_evidence.case_ids == expected_case_ids
    )
    zero_effects = (
        legacy_evidence.persistence_behavior_unchanged
        and legacy_evidence.defaults_unchanged
        and legacy_evidence.external_effects_unchanged
    )
    criteria = (
        _criterion(
            "event_count_exact",
            len(events) == spec.expected_event_count,
            {"actual": len(events), "expected": spec.expected_event_count},
        ),
        _criterion(
            "success_failure_visible",
            success == spec.expected_success_count
            and failure == spec.expected_failure_count,
            {"failure": failure, "success": success},
        ),
        _criterion(
            "observer_failure_visible",
            len(observer_failures) == spec.expected_observer_failure_count,
            {"count": len(observer_failures), "digest": observer_digest},
        ),
        _criterion(
            "sequence_contiguous",
            replayed.sequence == initial_state.sequence + len(events),
            {"events": rows},
        ),
        _criterion(
            "replay_equivalent",
            equivalence.matches,
            {
                "expected": equivalence.expected_snapshot_digest,
                "mismatches": list(equivalence.mismatches),
                "projected": equivalence.projected_digest,
            },
        ),
        _criterion(
            "checkpoint_restore_verified",
            restore_ok,
            {
                "checkpoint": final_checkpoint.checkpoint_id,
                "digest": final_checkpoint.state_digest,
            },
        ),
        _criterion(
            "rollback_verified",
            rollback_ok,
            {
                "checkpoint": initial_checkpoint.checkpoint_id,
                "digest": initial_checkpoint.state_digest,
            },
        ),
        _criterion(
            "lifecycle_registry_complete",
            domains == _EXPECTED_DOMAINS,
            {
                "domains": list(domains),
                "registry_digest": lifecycle_registry.digest,
            },
        ),
        _criterion(
            "legacy_behavior_preserved",
            legacy_ok,
            {
                "actual_case_ids": list(legacy_evidence.case_ids),
                "expected_case_ids": list(expected_case_ids),
                "legacy_digest": legacy_evidence.digest,
            },
        ),
        _criterion(
            "zero_unauthorized_effects",
            zero_effects,
            {
                "defaults_unchanged": legacy_evidence.defaults_unchanged,
                "external_effects_unchanged": (
                    legacy_evidence.external_effects_unchanged
                ),
                "persistence_unchanged": (
                    legacy_evidence.persistence_behavior_unchanged
                ),
            },
        ),
    )
    machine_passed = all(value.passed for value in criteria)
    return M1ShadowAcceptancePacket(
        window_id=spec.window_id,
        window_spec_digest=spec.digest,
        event_count=len(events),
        success_count=success,
        failure_count=failure,
        observer_failure_count=len(observer_failures),
        first_sequence=events[0].sequence,
        last_sequence=events[-1].sequence,
        replay_state_digest=replayed.digest,
        expected_snapshot_digest=equivalence.expected_snapshot_digest,
        lifecycle_registry_digest=lifecycle_registry.digest,
        lifecycle_domains=domains,
        legacy_evidence_digest=legacy_evidence.digest,
        observer_failure_evidence_digest=observer_digest,
        criteria=criteria,
        machine_status=(
            MACHINE_COMPLETE_STATUS
            if machine_passed
            else MACHINE_INCOMPLETE_STATUS
        ),
        machine_passed=machine_passed,
        eligible_for_human_review=machine_passed,
    )
