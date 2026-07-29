"""Pure M3-C lifecycle event-envelope and replay-reducer preflight.

The module turns an immutable M3-C-C lifecycle transition candidate into a
canonical event-envelope candidate and can replay such candidates in memory.
It never appends to EventKernel/SQLite, writes files, integrates production,
executes actions, schedules, speaks, transfers legacy authority, or opens M3-E.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from core.m3_c_c_goal_lifecycle_kernel import (
    GoalLifecycleState,
    GoalLifecycleTransitionCandidate,
)

EVENT_PREFLIGHT_VERSION = "eve.m3-c-d.goal-lifecycle-event-preflight.v1"
EVENT_ENVELOPE_CANDIDATE_VERSION = "eve.m3-c-d.goal-lifecycle-event-envelope-candidate.v1"
REDUCER_SNAPSHOT_VERSION = "eve.m3-c-d.goal-lifecycle-reducer-snapshot.v1"
REDUCER_RECEIPT_VERSION = "eve.m3-c-d.goal-lifecycle-reducer-receipt.v1"
EVENT_TYPE = "m3c.goal_lifecycle_transition"
EVENT_STREAM = "m3c.goal_lifecycle"
EVENT_PRODUCER = EVENT_PREFLIGHT_VERSION
EVENT_AUTHORITY = "candidate_only"


class M3CGoalLifecycleEventError(ValueError):
    """Fail-closed event-candidate/reducer-preflight error."""


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha256(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M3CGoalLifecycleEventError(f"{field} must be lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class GoalLifecycleEventEnvelopeCandidate:
    transition: GoalLifecycleTransitionCandidate
    schema_version: str = EVENT_ENVELOPE_CANDIDATE_VERSION
    event_type: str = EVENT_TYPE
    stream: str = EVENT_STREAM
    producer: str = EVENT_PRODUCER
    authority: str = EVENT_AUTHORITY
    append_authorized: bool = False
    append_performed: bool = False
    persistence_write_performed: bool = False
    production_integration_performed: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.transition, GoalLifecycleTransitionCandidate):
            raise M3CGoalLifecycleEventError(
                "transition must be GoalLifecycleTransitionCandidate"
            )
        if not self.transition.event_eligible:
            raise M3CGoalLifecycleEventError("transition must be event-eligible")
        if self.transition.event_append_performed:
            raise M3CGoalLifecycleEventError("transition already claims event append")
        if self.schema_version != EVENT_ENVELOPE_CANDIDATE_VERSION:
            raise M3CGoalLifecycleEventError("unsupported event candidate version")
        if (self.event_type, self.stream, self.producer, self.authority) != (
            EVENT_TYPE,
            EVENT_STREAM,
            EVENT_PRODUCER,
            EVENT_AUTHORITY,
        ):
            raise M3CGoalLifecycleEventError("event metadata mismatch")
        if any(
            (
                self.append_authorized,
                self.append_performed,
                self.persistence_write_performed,
                self.production_integration_performed,
                self.action_authorized,
                self.scheduler_authorized,
                self.speech_authorized,
                self.legacy_goal_authority_transferred,
                self.m3_e_authority_open,
            )
        ):
            raise M3CGoalLifecycleEventError(
                "event candidate cannot claim append, effects, or authority"
            )

    @property
    def event_id(self) -> str:
        return f"m3c:goal-lifecycle:{self.transition.transition_id}"

    @property
    def payload_digest(self) -> str:
        return _digest(self.transition.to_mapping())

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action_authorized": self.action_authorized,
            "append_authorized": self.append_authorized,
            "append_performed": self.append_performed,
            "authority": self.authority,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "legacy_goal_authority_transferred": self.legacy_goal_authority_transferred,
            "m3_e_authority_open": self.m3_e_authority_open,
            "payload_digest": self.payload_digest,
            "persistence_write_performed": self.persistence_write_performed,
            "producer": self.producer,
            "production_integration_performed": self.production_integration_performed,
            "scheduler_authorized": self.scheduler_authorized,
            "schema_version": self.schema_version,
            "speech_authorized": self.speech_authorized,
            "stream": self.stream,
            "transition": self.transition.to_mapping(),
            "transition_id": self.transition.transition_id,
        }

    @property
    def envelope_digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class GoalLifecycleReducerSnapshot:
    states: Mapping[str, GoalLifecycleState]
    last_logical_steps: Mapping[str, int]
    applied_transition_ids: tuple[str, ...]
    schema_version: str = REDUCER_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REDUCER_SNAPSHOT_VERSION:
            raise M3CGoalLifecycleEventError("unsupported reducer snapshot version")
        states = dict(self.states)
        steps = dict(self.last_logical_steps)
        if set(states) != set(steps):
            raise M3CGoalLifecycleEventError(
                "snapshot state and logical-step keys must match"
            )
        for candidate_id, state in states.items():
            _sha256(candidate_id, field="snapshot candidate id")
            if not isinstance(state, GoalLifecycleState):
                raise M3CGoalLifecycleEventError(
                    "snapshot values must be GoalLifecycleState"
                )
            if state.candidate_id != candidate_id:
                raise M3CGoalLifecycleEventError("snapshot state key mismatch")
            step = steps[candidate_id]
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise M3CGoalLifecycleEventError(
                    "snapshot logical steps must be non-negative integers"
                )
        transition_ids = tuple(self.applied_transition_ids)
        for transition_id in transition_ids:
            _sha256(transition_id, field="applied transition id")
        if len(transition_ids) != len(set(transition_ids)):
            raise M3CGoalLifecycleEventError(
                "snapshot contains duplicate transition identity"
            )
        object.__setattr__(self, "states", MappingProxyType(states))
        object.__setattr__(self, "last_logical_steps", MappingProxyType(steps))
        object.__setattr__(self, "applied_transition_ids", transition_ids)

    @classmethod
    def empty(cls) -> "GoalLifecycleReducerSnapshot":
        return cls(states={}, last_logical_steps={}, applied_transition_ids=())

    def to_mapping(self) -> dict[str, Any]:
        return {
            "applied_transition_ids": list(self.applied_transition_ids),
            "last_logical_steps": dict(sorted(self.last_logical_steps.items())),
            "schema_version": self.schema_version,
            "states": {
                candidate_id: self.states[candidate_id].to_mapping()
                for candidate_id in sorted(self.states)
            },
        }

    @property
    def snapshot_digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class GoalLifecycleReducerReceipt:
    envelope_digest: str
    transition_id: str
    candidate_id: str
    before_state: str
    after_state: str
    logical_step: int
    before_snapshot_digest: str
    after_snapshot_digest: str
    schema_version: str = REDUCER_RECEIPT_VERSION
    replay_applied: bool = True
    event_append_performed: bool = False
    persistence_write_performed: bool = False
    production_integration_performed: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        for field in (
            "envelope_digest",
            "transition_id",
            "candidate_id",
            "before_snapshot_digest",
            "after_snapshot_digest",
        ):
            _sha256(getattr(self, field), field=field)
        if self.schema_version != REDUCER_RECEIPT_VERSION:
            raise M3CGoalLifecycleEventError("unsupported reducer receipt version")
        if not self.replay_applied:
            raise M3CGoalLifecycleEventError("successful reducer receipt must be applied")
        if any(
            (
                self.event_append_performed,
                self.persistence_write_performed,
                self.production_integration_performed,
                self.action_authorized,
                self.scheduler_authorized,
                self.speech_authorized,
                self.legacy_goal_authority_transferred,
                self.m3_e_authority_open,
            )
        ):
            raise M3CGoalLifecycleEventError(
                "reducer receipt cannot claim external effects or authority"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action_authorized": self.action_authorized,
            "after_snapshot_digest": self.after_snapshot_digest,
            "after_state": self.after_state,
            "before_snapshot_digest": self.before_snapshot_digest,
            "before_state": self.before_state,
            "candidate_id": self.candidate_id,
            "envelope_digest": self.envelope_digest,
            "event_append_performed": self.event_append_performed,
            "legacy_goal_authority_transferred": self.legacy_goal_authority_transferred,
            "logical_step": self.logical_step,
            "m3_e_authority_open": self.m3_e_authority_open,
            "persistence_write_performed": self.persistence_write_performed,
            "production_integration_performed": self.production_integration_performed,
            "replay_applied": self.replay_applied,
            "scheduler_authorized": self.scheduler_authorized,
            "schema_version": self.schema_version,
            "speech_authorized": self.speech_authorized,
            "transition_id": self.transition_id,
        }

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping())


def build_event_envelope_candidate(
    transition: GoalLifecycleTransitionCandidate,
) -> GoalLifecycleEventEnvelopeCandidate:
    """Construct a canonical candidate only; never append or authorize append."""

    return GoalLifecycleEventEnvelopeCandidate(transition=transition)


def apply_event_candidate_in_memory(
    snapshot: GoalLifecycleReducerSnapshot,
    envelope: GoalLifecycleEventEnvelopeCandidate,
) -> tuple[GoalLifecycleReducerSnapshot, GoalLifecycleReducerReceipt]:
    """Apply one candidate to an immutable in-memory replay snapshot."""

    if not isinstance(snapshot, GoalLifecycleReducerSnapshot):
        raise M3CGoalLifecycleEventError(
            "snapshot must be GoalLifecycleReducerSnapshot"
        )
    if not isinstance(envelope, GoalLifecycleEventEnvelopeCandidate):
        raise M3CGoalLifecycleEventError(
            "envelope must be GoalLifecycleEventEnvelopeCandidate"
        )
    transition = envelope.transition
    transition_id = transition.transition_id
    if transition_id in snapshot.applied_transition_ids:
        raise M3CGoalLifecycleEventError("duplicate transition candidate")

    current = snapshot.states.get(transition.candidate_id)
    if current is None:
        current = GoalLifecycleState(
            candidate_id=transition.candidate_id,
            semantic_goal_id=transition.semantic_goal_id,
            decision_epoch=transition.decision_epoch,
            evidence_digest=transition.evidence_digest,
            lifecycle_state="absent",
            last_transition_id=None,
        )
    if current.semantic_goal_id != transition.semantic_goal_id:
        raise M3CGoalLifecycleEventError("semantic goal mismatch during replay")
    if current.decision_epoch != transition.decision_epoch:
        raise M3CGoalLifecycleEventError("decision epoch mismatch during replay")
    if current.evidence_digest != transition.evidence_digest:
        raise M3CGoalLifecycleEventError("evidence identity mismatch during replay")
    if current.lifecycle_state != transition.before_state:
        raise M3CGoalLifecycleEventError("lifecycle before-state mismatch")
    if current.last_transition_id != transition.prior_transition_id:
        raise M3CGoalLifecycleEventError("prior transition identity mismatch")

    previous_step = snapshot.last_logical_steps.get(transition.candidate_id)
    if previous_step is not None and transition.logical_step <= previous_step:
        raise M3CGoalLifecycleEventError(
            "logical step must advance monotonically per candidate"
        )

    next_state = transition.next_state()
    next_states = dict(snapshot.states)
    next_steps = dict(snapshot.last_logical_steps)
    next_states[transition.candidate_id] = next_state
    next_steps[transition.candidate_id] = transition.logical_step
    next_snapshot = GoalLifecycleReducerSnapshot(
        states=next_states,
        last_logical_steps=next_steps,
        applied_transition_ids=snapshot.applied_transition_ids + (transition_id,),
    )
    receipt = GoalLifecycleReducerReceipt(
        envelope_digest=envelope.envelope_digest,
        transition_id=transition_id,
        candidate_id=transition.candidate_id,
        before_state=transition.before_state,
        after_state=transition.after_state,
        logical_step=transition.logical_step,
        before_snapshot_digest=snapshot.snapshot_digest,
        after_snapshot_digest=next_snapshot.snapshot_digest,
    )
    return next_snapshot, receipt


def replay_event_candidates_in_memory(
    envelopes: Sequence[GoalLifecycleEventEnvelopeCandidate],
    *,
    initial_snapshot: GoalLifecycleReducerSnapshot | None = None,
) -> tuple[GoalLifecycleReducerSnapshot, tuple[GoalLifecycleReducerReceipt, ...]]:
    """Deterministically replay ordered candidates without external effects."""

    snapshot = initial_snapshot or GoalLifecycleReducerSnapshot.empty()
    receipts: list[GoalLifecycleReducerReceipt] = []
    for envelope in envelopes:
        snapshot, receipt = apply_event_candidate_in_memory(snapshot, envelope)
        receipts.append(receipt)
    return snapshot, tuple(receipts)
