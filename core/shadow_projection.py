"""M1-C bounded reducer, replay, and equivalence contract.

This module projects only the registered M1-B ``ActivationAdapter.learn_pair``
shadow stream. It is disconnected from the legacy runtime, performs no I/O,
and grants no persistence, recovery, retry, or mutation authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY, canonical_json_object
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    FAILURE_EVENT_TYPE,
    OBSERVER_PRODUCER,
    OBSERVER_VERSION,
    SUCCESS_EVENT_TYPE,
)

PROJECTION_SCHEMA_VERSION = "eve.shadow-projection.activation-learn-pair.v1"
CHECKPOINT_SCHEMA_VERSION = "eve.shadow-projection-checkpoint.v1"
EQUIVALENCE_SCHEMA_VERSION = "eve.shadow-equivalence-report.v1"

_CHECKPOINT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

PairRecord = tuple[str, str, float]


class ShadowProjectionError(ValueError):
    """Base error for malformed or inconsistent M1-C shadow projection data."""


class UnsupportedProjectionEvent(ShadowProjectionError):
    """Raised when an envelope is outside the bounded M1-C contract."""


class ProjectionSequenceError(ShadowProjectionError):
    """Raised when bounded replay ordering is inconsistent."""


class ProjectionStateMismatch(ShadowProjectionError):
    """Raised when an event's before-state differs from current projection state."""


class ProjectionTransitionError(ShadowProjectionError):
    """Raised when a success/failure transition violates the bounded semantics."""


class InvalidProjectionCheckpoint(ShadowProjectionError):
    """Raised when an immutable checkpoint cannot be trusted or applied."""


def _require_pair_record(value: Any, *, field: str) -> PairRecord:
    if not isinstance(value, list) or len(value) != 3:
        raise ProjectionTransitionError(f"{field} must be a three-item JSON list")
    left, right, strength = value
    if not isinstance(left, str) or not left:
        raise ProjectionTransitionError(f"{field}[0] must be a non-empty string")
    if not isinstance(right, str) or not right:
        raise ProjectionTransitionError(f"{field}[1] must be a non-empty string")
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
    ):
        raise ProjectionTransitionError(f"{field}[2] must be a finite number")
    return left, right, float(strength)


def _require_snapshot(
    value: Any,
    *,
    field: str,
) -> tuple[tuple[PairRecord, ...], tuple[PairRecord, ...]]:
    if not isinstance(value, Mapping):
        raise ProjectionTransitionError(f"{field} must be a mapping")
    if set(value) != {"calls", "learned"}:
        raise ProjectionTransitionError(
            f"{field} must contain exactly calls and learned"
        )
    calls_value = value["calls"]
    learned_value = value["learned"]
    if not isinstance(calls_value, list) or not isinstance(learned_value, list):
        raise ProjectionTransitionError(f"{field} collections must be JSON lists")
    calls = tuple(
        _require_pair_record(item, field=f"{field}.calls[{index}]")
        for index, item in enumerate(calls_value)
    )
    learned = tuple(
        _require_pair_record(item, field=f"{field}.learned[{index}]")
        for index, item in enumerate(learned_value)
    )
    call_index = 0
    for learned_record in learned:
        while call_index < len(calls) and calls[call_index] != learned_record:
            call_index += 1
        if call_index >= len(calls):
            raise ProjectionTransitionError(
                f"{field}.learned must be an ordered subsequence of calls"
            )
        call_index += 1
    return calls, learned


def _snapshot_mapping(
    calls: tuple[PairRecord, ...],
    learned: tuple[PairRecord, ...],
) -> dict[str, Any]:
    return {
        "calls": [list(item) for item in calls],
        "learned": [list(item) for item in learned],
    }


def _snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    encoded = canonical_json_object(snapshot, field="projection_snapshot")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ActivationLearnPairShadowState:
    """Immutable bounded projection for the registered learn-pair stream."""

    calls: tuple[PairRecord, ...]
    learned: tuple[PairRecord, ...]
    sequence: int = 0
    last_event_id: str | None = None
    last_event_digest: str | None = None
    stream_id: str = ACTIVATION_LEARN_PAIR_TARGET.stream_id
    schema_version: str = PROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROJECTION_SCHEMA_VERSION:
            raise ShadowProjectionError("unsupported projection schema version")
        if self.stream_id != ACTIVATION_LEARN_PAIR_TARGET.stream_id:
            raise ShadowProjectionError("projection stream is not the M1-B target")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ShadowProjectionError("projection sequence must be non-negative")
        for field, records in (("calls", self.calls), ("learned", self.learned)):
            if not isinstance(records, tuple):
                raise ShadowProjectionError(f"{field} must be an immutable tuple")
            for index, record in enumerate(records):
                if not isinstance(record, tuple):
                    raise ShadowProjectionError(
                        f"{field}[{index}] must be an immutable tuple"
                    )
                _require_pair_record(list(record), field=f"{field}[{index}]")
        if self.sequence == 0:
            if self.last_event_id is not None or self.last_event_digest is not None:
                raise ShadowProjectionError(
                    "initial projection cannot name a consumed event"
                )
        else:
            if not isinstance(self.last_event_id, str) or not self.last_event_id:
                raise ShadowProjectionError("replayed projection requires last_event_id")
            if (
                not isinstance(self.last_event_digest, str)
                or not _DIGEST_PATTERN.fullmatch(self.last_event_digest)
            ):
                raise ShadowProjectionError(
                    "replayed projection requires a canonical event digest"
                )

    @classmethod
    def from_initial_snapshot(
        cls,
        snapshot: Mapping[str, Any],
    ) -> "ActivationLearnPairShadowState":
        calls, learned = _require_snapshot(snapshot, field="initial_snapshot")
        return cls(calls=calls, learned=learned)

    @property
    def snapshot(self) -> dict[str, Any]:
        return _snapshot_mapping(self.calls, self.learned)

    @property
    def digest(self) -> str:
        encoded = canonical_json_object(
            {
                "calls": [list(item) for item in self.calls],
                "last_event_digest": self.last_event_digest,
                "last_event_id": self.last_event_id,
                "learned": [list(item) for item in self.learned],
                "schema_version": self.schema_version,
                "sequence": self.sequence,
                "stream_id": self.stream_id,
            },
            field="projection_state",
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_event_contract(envelope: EventEnvelope) -> tuple[dict[str, Any], bool]:
    if not isinstance(envelope, EventEnvelope):
        raise UnsupportedProjectionEvent("projection accepts EventEnvelope only")
    if envelope.authority != SHADOW_AUTHORITY:
        raise UnsupportedProjectionEvent("projection accepts shadow_only events")
    if (
        envelope.producer != OBSERVER_PRODUCER
        or envelope.producer_version != OBSERVER_VERSION
    ):
        raise UnsupportedProjectionEvent("event producer is not the M1-B observer")
    if envelope.stream_id != ACTIVATION_LEARN_PAIR_TARGET.stream_id:
        raise UnsupportedProjectionEvent("event stream is outside M1-C scope")
    if envelope.event_type not in {SUCCESS_EVENT_TYPE, FAILURE_EVENT_TYPE}:
        raise UnsupportedProjectionEvent("event type is outside M1-C scope")

    context = envelope.causal_context
    expected_context = {
        "arguments_captured": False,
        "legacy_result_captured": False,
        "observation_phase": "after_the_fact",
        "source_evidence_range": ACTIVATION_LEARN_PAIR_TARGET.evidence_range,
    }
    if context != expected_context:
        raise UnsupportedProjectionEvent("causal context is not the M1-B contract")

    payload = envelope.payload
    if set(payload) != {"after", "before", "legacy_outcome", "target"}:
        raise UnsupportedProjectionEvent("payload fields are outside M1-C scope")
    expected_target = {
        "callable": ACTIVATION_LEARN_PAIR_TARGET.callable_name,
        "disposition": ACTIVATION_LEARN_PAIR_TARGET.module_disposition,
        "module_path": ACTIVATION_LEARN_PAIR_TARGET.module_path,
        "target_id": ACTIVATION_LEARN_PAIR_TARGET.target_id,
    }
    if payload["target"] != expected_target:
        raise UnsupportedProjectionEvent("payload target is not the reviewed funnel")

    outcome = payload["legacy_outcome"]
    if not isinstance(outcome, Mapping) or set(outcome) != {"error_type", "succeeded"}:
        raise UnsupportedProjectionEvent("legacy_outcome is malformed")
    succeeded = outcome["succeeded"]
    error_type = outcome["error_type"]
    if not isinstance(succeeded, bool):
        raise UnsupportedProjectionEvent("legacy outcome must contain a boolean")
    if succeeded:
        if envelope.event_type != SUCCESS_EVENT_TYPE or error_type is not None:
            raise UnsupportedProjectionEvent("success event outcome is inconsistent")
    else:
        if envelope.event_type != FAILURE_EVENT_TYPE:
            raise UnsupportedProjectionEvent("failure event outcome is inconsistent")
        if not isinstance(error_type, str) or not error_type:
            raise UnsupportedProjectionEvent("failure event requires error type")
    return payload, succeeded


def reduce_activation_learn_pair(
    state: ActivationLearnPairShadowState,
    envelope: EventEnvelope,
) -> ActivationLearnPairShadowState:
    """Apply one validated M1-B candidate to the immutable shadow projection."""

    if not isinstance(state, ActivationLearnPairShadowState):
        raise ShadowProjectionError("reducer requires ActivationLearnPairShadowState")
    payload, succeeded = _validate_event_contract(envelope)
    expected_sequence = state.sequence + 1
    if envelope.sequence != expected_sequence:
        raise ProjectionSequenceError(
            f"expected projection sequence {expected_sequence}"
        )
    before_calls, before_learned = _require_snapshot(
        payload["before"],
        field="payload.before",
    )
    after_calls, after_learned = _require_snapshot(
        payload["after"],
        field="payload.after",
    )
    if before_calls != state.calls or before_learned != state.learned:
        raise ProjectionStateMismatch("event before-state differs from projection")
    if len(after_calls) != len(before_calls) + 1:
        raise ProjectionTransitionError("legacy call log must append exactly once")
    if after_calls[:-1] != before_calls:
        raise ProjectionTransitionError("legacy call log prefix changed")
    attempted_pair = after_calls[-1]
    if succeeded:
        if after_learned != before_learned + (attempted_pair,):
            raise ProjectionTransitionError(
                "successful learn_pair must append the attempted pair once"
            )
    elif after_learned != before_learned:
        raise ProjectionTransitionError(
            "failed learn_pair cannot change the learned-pair projection"
        )

    return ActivationLearnPairShadowState(
        calls=after_calls,
        learned=after_learned,
        sequence=envelope.sequence,
        last_event_id=envelope.event_id,
        last_event_digest=envelope.digest,
    )


def replay_activation_learn_pair(
    initial_state: ActivationLearnPairShadowState,
    events: Iterable[EventEnvelope],
) -> ActivationLearnPairShadowState:
    """Deterministically replay a bounded event sequence through the reducer."""

    if not isinstance(initial_state, ActivationLearnPairShadowState):
        raise ShadowProjectionError("replay requires projection initial state")
    state = initial_state
    for envelope in events:
        state = reduce_activation_learn_pair(state, envelope)
    return state


@dataclass(frozen=True, slots=True)
class ShadowEquivalenceReport:
    """Immutable comparison evidence between replay and a detached legacy state."""

    stream_id: str
    sequence: int
    projected_digest: str
    expected_snapshot_digest: str
    matches: bool
    mismatches: tuple[str, ...]
    schema_version: str = EQUIVALENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EQUIVALENCE_SCHEMA_VERSION:
            raise ShadowProjectionError("unsupported equivalence schema version")
        if self.stream_id != ACTIVATION_LEARN_PAIR_TARGET.stream_id:
            raise ShadowProjectionError("equivalence report stream is out of scope")
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence < 0
        ):
            raise ShadowProjectionError("equivalence sequence must be non-negative")
        if not isinstance(self.matches, bool):
            raise ShadowProjectionError("equivalence matches flag must be boolean")
        if (
            not isinstance(self.projected_digest, str)
            or not _DIGEST_PATTERN.fullmatch(self.projected_digest)
        ):
            raise ShadowProjectionError("projected digest is malformed")
        if (
            not isinstance(self.expected_snapshot_digest, str)
            or not _DIGEST_PATTERN.fullmatch(self.expected_snapshot_digest)
        ):
            raise ShadowProjectionError("expected snapshot digest is malformed")
        if not isinstance(self.mismatches, tuple):
            raise ShadowProjectionError("mismatches must be immutable")
        if self.matches != (not self.mismatches):
            raise ShadowProjectionError("matches flag disagrees with mismatches")


def compare_activation_learn_pair_equivalence(
    state: ActivationLearnPairShadowState,
    expected_snapshot: Mapping[str, Any],
) -> ShadowEquivalenceReport:
    """Compare replay output with detached legacy after-state without mutation."""

    if not isinstance(state, ActivationLearnPairShadowState):
        raise ShadowProjectionError("equivalence requires projection state")
    expected_calls, expected_learned = _require_snapshot(
        expected_snapshot,
        field="expected_snapshot",
    )
    mismatches: list[str] = []
    if state.calls != expected_calls:
        mismatches.append("calls_mismatch")
    if state.learned != expected_learned:
        mismatches.append("learned_mismatch")
    normalized_expected = _snapshot_mapping(expected_calls, expected_learned)
    return ShadowEquivalenceReport(
        stream_id=state.stream_id,
        sequence=state.sequence,
        projected_digest=_snapshot_digest(state.snapshot),
        expected_snapshot_digest=_snapshot_digest(normalized_expected),
        matches=not mismatches,
        mismatches=tuple(mismatches),
    )


@dataclass(frozen=True, slots=True)
class ShadowProjectionCheckpoint:
    """Immutable in-memory checkpoint for bounded projection rollback only."""

    checkpoint_id: str
    state: ActivationLearnPairShadowState
    state_digest: str
    schema_version: str = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise InvalidProjectionCheckpoint("unsupported checkpoint schema version")
        if (
            not isinstance(self.checkpoint_id, str)
            or not _CHECKPOINT_ID_PATTERN.fullmatch(self.checkpoint_id)
        ):
            raise InvalidProjectionCheckpoint("checkpoint_id is not canonical")
        if not isinstance(self.state, ActivationLearnPairShadowState):
            raise InvalidProjectionCheckpoint("checkpoint state is out of scope")
        if self.state_digest != self.state.digest:
            raise InvalidProjectionCheckpoint("checkpoint state digest mismatch")

    @classmethod
    def create(
        cls,
        checkpoint_id: str,
        state: ActivationLearnPairShadowState,
    ) -> "ShadowProjectionCheckpoint":
        if not isinstance(state, ActivationLearnPairShadowState):
            raise InvalidProjectionCheckpoint("checkpoint state is out of scope")
        return cls(
            checkpoint_id=checkpoint_id,
            state=state,
            state_digest=state.digest,
        )


def restore_projection_checkpoint(
    checkpoint: ShadowProjectionCheckpoint,
) -> ActivationLearnPairShadowState:
    """Restore an immutable checkpoint value without I/O or side effects."""

    if not isinstance(checkpoint, ShadowProjectionCheckpoint):
        raise InvalidProjectionCheckpoint("restore requires a projection checkpoint")
    if checkpoint.state_digest != checkpoint.state.digest:
        raise InvalidProjectionCheckpoint("checkpoint state digest mismatch")
    return checkpoint.state


def rollback_projection(
    current_state: ActivationLearnPairShadowState,
    checkpoint: ShadowProjectionCheckpoint,
) -> ActivationLearnPairShadowState:
    """Return the checkpoint state when it is a valid prior replay boundary."""

    if not isinstance(current_state, ActivationLearnPairShadowState):
        raise InvalidProjectionCheckpoint("rollback requires current projection state")
    restored = restore_projection_checkpoint(checkpoint)
    if restored.stream_id != current_state.stream_id:
        raise InvalidProjectionCheckpoint("checkpoint stream mismatch")
    if restored.sequence > current_state.sequence:
        raise InvalidProjectionCheckpoint("cannot roll forward through rollback")
    return restored
