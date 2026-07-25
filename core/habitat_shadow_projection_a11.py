"""Habitat-only A11 mixed replay for the M1-C learn-pair projection.

The accepted ``core.shadow_projection`` remains unchanged. Legacy inline events
delegate to it. Only the bounded M2-E habitat wrapper uses this reducer for
A11 append-state references emitted after the logical EventEnvelope itself
reaches the frozen canonical-size boundary.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from core.canonical_content import (
    APPEND_STATE_CONTENT_SCHEMA_VERSION,
    APPEND_STATE_REPRESENTATION_SCHEMA_VERSION,
    CanonicalContentError,
    apply_append_state_delta,
    verify_content_reference,
)
from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    FAILURE_EVENT_TYPE,
    OBSERVER_PRODUCER,
    OBSERVER_VERSION,
    SUCCESS_EVENT_TYPE,
)
from core.shadow_projection import (
    ActivationLearnPairShadowState,
    ProjectionSequenceError,
    ProjectionStateMismatch,
    ProjectionTransitionError,
    ShadowProjectionError,
    UnsupportedProjectionEvent,
    reduce_activation_learn_pair as _legacy_reduce,
)


def _expected_target() -> dict[str, str]:
    return {
        "callable": ACTIVATION_LEARN_PAIR_TARGET.callable_name,
        "disposition": ACTIVATION_LEARN_PAIR_TARGET.module_disposition,
        "module_path": ACTIVATION_LEARN_PAIR_TARGET.module_path,
        "target_id": ACTIVATION_LEARN_PAIR_TARGET.target_id,
    }


def _snapshot_lists(value: Mapping[str, Any], *, field: str) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if set(value) != {"calls", "learned"}:
        raise ProjectionStateMismatch(f"{field} fields differ from the reviewed projection")
    calls = value["calls"]
    learned = value["learned"]
    if not isinstance(calls, list) or not isinstance(learned, list):
        raise ProjectionStateMismatch(f"{field} must contain JSON lists")
    return tuple(calls), tuple(learned)


def _validate_common(envelope: EventEnvelope) -> tuple[dict[str, Any], bool]:
    if not isinstance(envelope, EventEnvelope):
        raise UnsupportedProjectionEvent("projection accepts EventEnvelope only")
    if envelope.authority != SHADOW_AUTHORITY:
        raise UnsupportedProjectionEvent("projection accepts shadow_only events")
    if envelope.producer != OBSERVER_PRODUCER or envelope.producer_version != OBSERVER_VERSION:
        raise UnsupportedProjectionEvent("event producer is not the M1-B observer")
    if envelope.stream_id != ACTIVATION_LEARN_PAIR_TARGET.stream_id:
        raise UnsupportedProjectionEvent("event stream is outside M1-C scope")
    if envelope.event_type not in {SUCCESS_EVENT_TYPE, FAILURE_EVENT_TYPE}:
        raise UnsupportedProjectionEvent("event type is outside M1-C scope")
    expected_context = {
        "arguments_captured": False,
        "legacy_result_captured": False,
        "observation_phase": "after_the_fact",
        "source_evidence_range": ACTIVATION_LEARN_PAIR_TARGET.evidence_range,
    }
    if envelope.causal_context != expected_context:
        raise UnsupportedProjectionEvent("causal context is not the M1-B contract")
    payload = envelope.payload
    if payload.get("target") != _expected_target():
        raise UnsupportedProjectionEvent("payload target is not the reviewed funnel")
    outcome = payload.get("legacy_outcome")
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
    if not isinstance(state, ActivationLearnPairShadowState):
        raise ShadowProjectionError("reducer requires ActivationLearnPairShadowState")
    payload = envelope.payload if isinstance(envelope, EventEnvelope) else {}
    if payload.get("state_representation") != APPEND_STATE_REPRESENTATION_SCHEMA_VERSION:
        return _legacy_reduce(state, envelope)

    payload, succeeded = _validate_common(envelope)
    expected_fields = {
        "after_ref",
        "before_ref",
        "legacy_outcome",
        "state_delta",
        "state_representation",
        "target",
    }
    if set(payload) != expected_fields:
        raise UnsupportedProjectionEvent("compact payload fields are outside A11 habitat scope")
    expected_sequence = state.sequence + 1
    if envelope.sequence != expected_sequence:
        raise ProjectionSequenceError(f"expected projection sequence {expected_sequence}")

    before_snapshot = state.snapshot
    try:
        verify_content_reference(
            payload["before_ref"],
            before_snapshot,
            expected_schema_version=APPEND_STATE_CONTENT_SCHEMA_VERSION,
        )
        after_snapshot = apply_append_state_delta(before_snapshot, payload["state_delta"])
        verify_content_reference(
            payload["after_ref"],
            after_snapshot,
            expected_schema_version=APPEND_STATE_CONTENT_SCHEMA_VERSION,
        )
    except (CanonicalContentError, TypeError, ValueError) as exc:
        raise ProjectionStateMismatch("compact state reference or delta mismatch") from exc

    before_calls, before_learned = _snapshot_lists(before_snapshot, field="compact.before")
    after_calls, after_learned = _snapshot_lists(after_snapshot, field="compact.after")
    if len(after_calls) != len(before_calls) + 1 or after_calls[:-1] != before_calls:
        raise ProjectionTransitionError("compact legacy call log must append exactly once")
    attempted_pair = after_calls[-1]
    if succeeded:
        if after_learned != before_learned + (attempted_pair,):
            raise ProjectionTransitionError("compact successful learn_pair must append learned pair once")
    elif after_learned != before_learned:
        raise ProjectionTransitionError("compact failed learn_pair cannot change learned projection")

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
    if not isinstance(initial_state, ActivationLearnPairShadowState):
        raise ShadowProjectionError("replay requires projection initial state")
    state = initial_state
    for envelope in events:
        state = reduce_activation_learn_pair(state, envelope)
    return state
