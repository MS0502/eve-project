from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.event_kernel import EventEnvelope
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    FAILURE_EVENT_TYPE,
    OBSERVER_PRODUCER,
    OBSERVER_VERSION,
    SUCCESS_EVENT_TYPE,
)
from core.shadow_projection import (
    CHECKPOINT_SCHEMA_VERSION,
    EQUIVALENCE_SCHEMA_VERSION,
    PROJECTION_SCHEMA_VERSION,
    ActivationLearnPairShadowState,
    InvalidProjectionCheckpoint,
    ProjectionSequenceError,
    ProjectionStateMismatch,
    ProjectionTransitionError,
    ShadowProjectionCheckpoint,
    UnsupportedProjectionEvent,
    compare_activation_learn_pair_equivalence,
    reduce_activation_learn_pair,
    replay_activation_learn_pair,
    restore_projection_checkpoint,
    rollback_projection,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTION_PATH = REPO_ROOT / "core/shadow_projection.py"


def _snapshot(*records: tuple[str, str, float]):
    items = [list(record) for record in records]
    return {"calls": items, "learned": items}


def _event(
    *,
    sequence: int,
    before: dict,
    after: dict,
    succeeded: bool = True,
    event_id: str | None = None,
    causation_id: str | None = None,
    event_type: str | None = None,
    stream_id: str | None = None,
    target: dict | None = None,
    context: dict | None = None,
    producer: str = OBSERVER_PRODUCER,
    producer_version: str = OBSERVER_VERSION,
):
    resolved_event_id = event_id or f"shadow:projection:{sequence}"
    return EventEnvelope.create(
        event_id=resolved_event_id,
        event_type=event_type or (
            SUCCESS_EVENT_TYPE if succeeded else FAILURE_EVENT_TYPE
        ),
        stream_id=stream_id or ACTIVATION_LEARN_PAIR_TARGET.stream_id,
        sequence=sequence,
        producer=producer,
        producer_version=producer_version,
        correlation_id="corr:m1-c",
        causation_id=causation_id,
        payload={
            "after": after,
            "before": before,
            "legacy_outcome": {
                "error_type": None if succeeded else "RuntimeError",
                "succeeded": succeeded,
            },
            "target": target or {
                "callable": ACTIVATION_LEARN_PAIR_TARGET.callable_name,
                "disposition": ACTIVATION_LEARN_PAIR_TARGET.module_disposition,
                "module_path": ACTIVATION_LEARN_PAIR_TARGET.module_path,
                "target_id": ACTIVATION_LEARN_PAIR_TARGET.target_id,
            },
        },
        causal_context=context or {
            "arguments_captured": False,
            "legacy_result_captured": False,
            "observation_phase": "after_the_fact",
            "source_evidence_range": ACTIVATION_LEARN_PAIR_TARGET.evidence_range,
        },
    )


def test_projection_schema_is_frozen_and_snapshot_access_is_detached():
    state = ActivationLearnPairShadowState.from_initial_snapshot(_snapshot())

    assert state.schema_version == PROJECTION_SCHEMA_VERSION
    assert state.sequence == 0
    assert state.snapshot == {"calls": [], "learned": []}
    detached = state.snapshot
    detached["calls"].append(["x", "y", 1.0])
    assert state.snapshot == {"calls": [], "learned": []}
    with pytest.raises(FrozenInstanceError):
        state.sequence = 1  # type: ignore[misc]


def test_success_reducer_appends_one_attempt_and_learned_pair():
    initial = ActivationLearnPairShadowState.from_initial_snapshot(_snapshot())
    event = _event(
        sequence=1,
        before=_snapshot(),
        after=_snapshot(("alpha", "beta", 0.4)),
    )

    state = reduce_activation_learn_pair(initial, event)

    assert state.sequence == 1
    assert state.calls == (("alpha", "beta", 0.4),)
    assert state.learned == (("alpha", "beta", 0.4),)
    assert state.last_event_id == event.event_id
    assert state.last_event_digest == event.digest


def test_failure_reducer_appends_attempt_without_learning():
    initial_snapshot = _snapshot(("old", "pair", 0.2))
    initial = ActivationLearnPairShadowState.from_initial_snapshot(initial_snapshot)
    after = {
        "calls": [
            ["old", "pair", 0.2],
            ["new", "pair", 0.6],
        ],
        "learned": [["old", "pair", 0.2]],
    }
    event = _event(
        sequence=1,
        before=initial_snapshot,
        after=after,
        succeeded=False,
    )

    state = reduce_activation_learn_pair(initial, event)

    assert state.calls[-1] == ("new", "pair", 0.6)
    assert state.learned == (("old", "pair", 0.2),)


def test_sequence_gap_fails_without_mutating_state():
    initial = ActivationLearnPairShadowState.from_initial_snapshot(_snapshot())
    gap = _event(
        sequence=2,
        before=_snapshot(),
        after=_snapshot(("a", "b", 0.4)),
    )
    with pytest.raises(ProjectionSequenceError):
        reduce_activation_learn_pair(initial, gap)
    assert initial.sequence == 0


def test_external_causation_is_preserved_but_not_reinterpreted_by_projection():
    initial = ActivationLearnPairShadowState.from_initial_snapshot(_snapshot())
    event = _event(
        sequence=1,
        before=_snapshot(),
        after=_snapshot(("a", "b", 0.4)),
        causation_id="external:observation:1",
    )

    state = reduce_activation_learn_pair(initial, event)

    assert state.sequence == 1
    assert state.last_event_id == event.event_id
    assert event.causation_id == "external:observation:1"


def test_before_state_mismatch_fails_closed():
    initial = ActivationLearnPairShadowState.from_initial_snapshot(_snapshot())
    event = _event(
        sequence=1,
        before=_snapshot(("wrong", "state", 0.1)),
        after=_snapshot(("wrong", "state", 0.1), ("a", "b", 0.4)),
    )

    with pytest.raises(ProjectionStateMismatch):
        reduce_activation_learn_pair(initial, event)


def test_transition_semantics_reject_multiple_or_inconsistent_changes():
    initial = ActivationLearnPairShadowState.from_initial_snapshot(_snapshot())
    two_calls = _event(
        sequence=1,
        before=_snapshot(),
        after=_snapshot(("a", "b", 0.4), ("c", "d", 0.5)),
    )
    with pytest.raises(ProjectionTransitionError):
        reduce_activation_learn_pair(initial, two_calls)

    failure_learned = _event(
        sequence=1,
        before=_snapshot(),
        after=_snapshot(("a", "b", 0.4)),
        succeeded=False,
    )
    with pytest.raises(ProjectionTransitionError):
        reduce_activation_learn_pair(initial, failure_learned)


def test_event_scope_target_context_and_outcome_are_strict():
    initial = ActivationLearnPairShadowState.from_initial_snapshot(_snapshot())
    after = _snapshot(("a", "b", 0.4))

    with pytest.raises(UnsupportedProjectionEvent):
        reduce_activation_learn_pair(
            initial,
            _event(
                sequence=1,
                before=_snapshot(),
                after=after,
                producer="other.observer",
            ),
        )
    with pytest.raises(UnsupportedProjectionEvent):
        reduce_activation_learn_pair(
            initial,
            _event(
                sequence=1,
                before=_snapshot(),
                after=after,
                producer_version="9.9.9",
            ),
        )
    with pytest.raises(UnsupportedProjectionEvent):
        reduce_activation_learn_pair(
            initial,
            _event(
                sequence=1,
                before=_snapshot(),
                after=after,
                stream_id="shadow:other",
            ),
        )
    with pytest.raises(UnsupportedProjectionEvent):
        reduce_activation_learn_pair(
            initial,
            _event(
                sequence=1,
                before=_snapshot(),
                after=after,
                target={"target_id": "wrong"},
            ),
        )
    with pytest.raises(UnsupportedProjectionEvent):
        reduce_activation_learn_pair(
            initial,
            _event(
                sequence=1,
                before=_snapshot(),
                after=after,
                context={"observation_phase": "before"},
            ),
        )
    inconsistent = _event(
        sequence=1,
        before=_snapshot(),
        after=after,
        succeeded=True,
        event_type=FAILURE_EVENT_TYPE,
    )
    with pytest.raises(UnsupportedProjectionEvent):
        reduce_activation_learn_pair(initial, inconsistent)


def test_snapshot_schema_and_numbers_fail_closed():
    with pytest.raises(ProjectionTransitionError):
        ActivationLearnPairShadowState.from_initial_snapshot({"learned": []})
    with pytest.raises(ProjectionTransitionError):
        ActivationLearnPairShadowState.from_initial_snapshot(
            {"calls": [["a", "b", float("nan")]], "learned": []}
        )
    with pytest.raises(ProjectionTransitionError):
        ActivationLearnPairShadowState.from_initial_snapshot(
            {"calls": [["a", "b"]], "learned": []}
        )
    with pytest.raises(ProjectionTransitionError, match="ordered subsequence"):
        ActivationLearnPairShadowState.from_initial_snapshot(
            {
                "calls": [["a", "b", 0.4], ["c", "d", 0.5]],
                "learned": [["c", "d", 0.5], ["a", "b", 0.4]],
            }
        )
    with pytest.raises(ProjectionTransitionError, match="ordered subsequence"):
        ActivationLearnPairShadowState.from_initial_snapshot(
            {
                "calls": [["a", "b", 0.4]],
                "learned": [["x", "y", 0.8]],
            }
        )


def test_replay_is_deterministic_and_uses_explicit_initial_state():
    initial = ActivationLearnPairShadowState.from_initial_snapshot(_snapshot())
    first = _event(
        sequence=1,
        before=_snapshot(),
        after=_snapshot(("a", "b", 0.4)),
        event_id="shadow:projection:one",
    )
    second = _event(
        sequence=2,
        before=_snapshot(("a", "b", 0.4)),
        after=_snapshot(("a", "b", 0.4), ("c", "d", 0.5)),
        event_id="shadow:projection:two",
        causation_id=first.event_id,
    )

    left = replay_activation_learn_pair(initial, (first, second))
    right = replay_activation_learn_pair(initial, (first, second))

    assert left == right
    assert left.digest == right.digest
    assert left.sequence == 2


def test_empty_replay_rejects_invalid_initial_state():
    with pytest.raises(ValueError, match="initial state"):
        replay_activation_learn_pair(object(), ())  # type: ignore[arg-type]


def test_equivalence_report_is_visible_for_match_and_mismatch():
    state = ActivationLearnPairShadowState.from_initial_snapshot(
        _snapshot(("a", "b", 0.4))
    )

    matched = compare_activation_learn_pair_equivalence(
        state,
        _snapshot(("a", "b", 0.4)),
    )
    mismatched = compare_activation_learn_pair_equivalence(
        state,
        _snapshot(("x", "y", 0.9)),
    )

    assert matched.schema_version == EQUIVALENCE_SCHEMA_VERSION
    assert matched.matches is True
    assert matched.mismatches == ()
    assert mismatched.matches is False
    assert mismatched.mismatches == ("calls_mismatch", "learned_mismatch")
    assert len(mismatched.projected_digest) == 64
    assert len(mismatched.expected_snapshot_digest) == 64
    assert state.snapshot == _snapshot(("a", "b", 0.4))


def test_checkpoint_restore_and_rollback_are_immutable_and_bounded():
    initial = ActivationLearnPairShadowState.from_initial_snapshot(_snapshot())
    checkpoint = ShadowProjectionCheckpoint.create("checkpoint:m1-c:zero", initial)
    event = _event(
        sequence=1,
        before=_snapshot(),
        after=_snapshot(("a", "b", 0.4)),
    )
    current = reduce_activation_learn_pair(initial, event)

    assert checkpoint.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert restore_projection_checkpoint(checkpoint) is initial
    assert rollback_projection(current, checkpoint) is initial
    with pytest.raises(FrozenInstanceError):
        checkpoint.state_digest = "0" * 64  # type: ignore[misc]


def test_invalid_or_future_checkpoint_is_rejected():
    initial = ActivationLearnPairShadowState.from_initial_snapshot(_snapshot())
    with pytest.raises(InvalidProjectionCheckpoint):
        ShadowProjectionCheckpoint(
            checkpoint_id="checkpoint:m1-c:bad",
            state=initial,
            state_digest="0" * 64,
        )

    first = _event(
        sequence=1,
        before=_snapshot(),
        after=_snapshot(("a", "b", 0.4)),
    )
    future = reduce_activation_learn_pair(initial, first)
    future_checkpoint = ShadowProjectionCheckpoint.create(
        "checkpoint:m1-c:future",
        future,
    )
    with pytest.raises(InvalidProjectionCheckpoint):
        rollback_projection(initial, future_checkpoint)


def test_projection_module_has_no_persistence_runtime_clock_or_thread_surface():
    source = PROJECTION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    called_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    assert not imported_roots & {
        "adapters",
        "asyncio",
        "datetime",
        "language",
        "main",
        "pathlib",
        "pickle",
        "random",
        "secrets",
        "sqlite3",
        "threading",
        "time",
        "uuid",
    }
    assert not called_names & {
        "connect",
        "load",
        "open",
        "save",
        "sleep",
        "start",
        "write_bytes",
        "write_text",
    }
