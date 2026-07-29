from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from core.m3_c_b_goal_selection_kernel import (
    ALLOWED_DRIVES,
    DriveSample,
    GoalCandidate,
    select_goal_proposal,
)
from core.m3_c_c_goal_lifecycle_kernel import (
    GoalLifecycleState,
    LifecycleEvidence,
    evaluate_lifecycle_transition,
)
from core.m3_c_d_goal_lifecycle_event_preflight import (
    EVENT_AUTHORITY,
    EVENT_PRODUCER,
    EVENT_STREAM,
    EVENT_TYPE,
    GoalLifecycleEventEnvelopeCandidate,
    GoalLifecycleReducerSnapshot,
    M3CGoalLifecycleEventError,
    apply_event_candidate_in_memory,
    build_event_envelope_candidate,
    replay_event_candidates_in_memory,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_d_goal_lifecycle_event_preflight.py"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate():
    return GoalCandidate(
        semantic_goal_id="recover_operating_margin",
        decision_epoch=0,
        evidence_digest=_digest("recovery-evidence"),
        base_value=0.30,
        expected_value=0.0,
        urgency=0.0,
        continuity=0.0,
        cost=0.0,
        risk=0.0,
        drive_alignment={
            drive: {
                "energy": -0.90,
                "safety": -0.80,
                "curiosity": -0.10,
            }.get(drive, 0.0)
            for drive in ALLOWED_DRIVES
        },
        drive_confidence={drive: 1.0 for drive in ALLOWED_DRIVES},
    )


def _samples(elapsed=0.0):
    values = {"energy": -0.70, "safety": -0.80, "curiosity": -0.20}
    return {
        drive: DriveSample(
            drive=drive,
            value=values.get(drive, 0.0),
            lower_bound=-1.0,
            upper_bound=1.0,
            sample_digest=_digest(f"sample:{drive}:{elapsed}"),
            replay_elapsed_seconds=elapsed,
        )
        for drive in ALLOWED_DRIVES
    }


def _selection_material():
    candidate = _candidate()
    selection = select_goal_proposal([candidate], _samples())
    score = selection.scored_candidates[0]
    assert selection.selected_candidate_id == candidate.candidate_id
    return candidate, score, selection


def _lifecycle_chain():
    candidate, score, selection = _selection_material()
    state = GoalLifecycleState(
        candidate_id=candidate.candidate_id,
        semantic_goal_id=candidate.semantic_goal_id,
        decision_epoch=candidate.decision_epoch,
        evidence_digest=candidate.evidence_digest,
    )
    decisions = []
    for evidence in (
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
    ):
        decision = evaluate_lifecycle_transition(state, evidence)
        assert decision.transition is not None
        decisions.append(decision)
        state = decision.transition.next_state()
    return candidate, score, selection, tuple(decisions)


def _envelopes():
    _, _, _, decisions = _lifecycle_chain()
    return tuple(build_event_envelope_candidate(item.transition) for item in decisions)


def test_event_envelope_candidate_is_canonical_and_non_authoritative():
    envelope = _envelopes()[0]
    assert envelope.event_type == EVENT_TYPE
    assert envelope.stream == EVENT_STREAM
    assert envelope.producer == EVENT_PRODUCER
    assert envelope.authority == EVENT_AUTHORITY == "candidate_only"
    assert envelope.event_id == f"m3c:goal-lifecycle:{envelope.transition.transition_id}"
    assert len(envelope.payload_digest) == 64
    assert len(envelope.envelope_digest) == 64
    assert envelope.append_authorized is False
    assert envelope.append_performed is False
    assert envelope.persistence_write_performed is False
    assert envelope.production_integration_performed is False
    assert envelope.legacy_goal_authority_transferred is False
    assert envelope.m3_e_authority_open is False


def test_same_transition_builds_identical_envelope_identity():
    transition = _lifecycle_chain()[3][0].transition
    first = build_event_envelope_candidate(transition)
    second = build_event_envelope_candidate(transition)
    assert first == second
    assert first.envelope_digest == second.envelope_digest
    with pytest.raises(FrozenInstanceError):
        first.authority = "authoritative"


def test_in_memory_reducer_applies_one_candidate_without_external_effects():
    snapshot = GoalLifecycleReducerSnapshot.empty()
    envelope = _envelopes()[0]
    next_snapshot, receipt = apply_event_candidate_in_memory(snapshot, envelope)
    state = next_snapshot.states[envelope.transition.candidate_id]
    assert state.lifecycle_state == "proposed"
    assert state.last_transition_id == envelope.transition.transition_id
    assert next_snapshot.last_logical_steps[state.candidate_id] == 1
    assert next_snapshot.applied_transition_ids == (envelope.transition.transition_id,)
    assert receipt.replay_applied is True
    assert receipt.event_append_performed is False
    assert receipt.persistence_write_performed is False
    assert receipt.production_integration_performed is False
    assert receipt.action_authorized is False
    assert receipt.scheduler_authorized is False
    assert receipt.speech_authorized is False
    assert receipt.legacy_goal_authority_transferred is False
    assert receipt.m3_e_authority_open is False


def test_ordered_replay_reaches_selected_and_is_deterministic():
    envelopes = _envelopes()
    first_snapshot, first_receipts = replay_event_candidates_in_memory(envelopes)
    second_snapshot, second_receipts = replay_event_candidates_in_memory(envelopes)
    candidate_id = envelopes[0].transition.candidate_id
    assert first_snapshot.states[candidate_id].lifecycle_state == "selected"
    assert len(first_snapshot.applied_transition_ids) == 4
    assert first_snapshot.snapshot_digest == second_snapshot.snapshot_digest
    assert [item.receipt_digest for item in first_receipts] == [
        item.receipt_digest for item in second_receipts
    ]


def test_duplicate_transition_candidate_fails_closed():
    envelope = _envelopes()[0]
    snapshot, _ = apply_event_candidate_in_memory(
        GoalLifecycleReducerSnapshot.empty(), envelope
    )
    with pytest.raises(M3CGoalLifecycleEventError, match="duplicate"):
        apply_event_candidate_in_memory(snapshot, envelope)


def test_out_of_order_replay_fails_closed_on_before_state():
    envelopes = _envelopes()
    with pytest.raises(M3CGoalLifecycleEventError, match="before-state"):
        replay_event_candidates_in_memory((envelopes[1], envelopes[0]))


def test_wrong_prior_transition_identity_fails_closed():
    envelopes = _envelopes()
    snapshot, _ = apply_event_candidate_in_memory(
        GoalLifecycleReducerSnapshot.empty(), envelopes[0]
    )
    bad_transition = replace(
        envelopes[1].transition,
        prior_transition_id=_digest("wrong-prior"),
    )
    bad_envelope = build_event_envelope_candidate(bad_transition)
    with pytest.raises(M3CGoalLifecycleEventError, match="prior transition"):
        apply_event_candidate_in_memory(snapshot, bad_envelope)


def test_non_monotonic_logical_step_fails_closed():
    envelopes = _envelopes()
    snapshot, _ = apply_event_candidate_in_memory(
        GoalLifecycleReducerSnapshot.empty(), envelopes[0]
    )
    bad_transition = replace(envelopes[1].transition, logical_step=1)
    bad_envelope = build_event_envelope_candidate(bad_transition)
    with pytest.raises(M3CGoalLifecycleEventError, match="monotonically"):
        apply_event_candidate_in_memory(snapshot, bad_envelope)


def test_snapshot_is_immutable_and_digest_bound():
    snapshot, _ = replay_event_candidates_in_memory(_envelopes())
    before = snapshot.snapshot_digest
    with pytest.raises(TypeError):
        snapshot.states[_digest("other")] = snapshot.states[
            next(iter(snapshot.states))
        ]
    assert snapshot.snapshot_digest == before


def test_resume_from_snapshot_produces_same_final_digest():
    envelopes = _envelopes()
    partial, _ = replay_event_candidates_in_memory(envelopes[:2])
    resumed, resumed_receipts = replay_event_candidates_in_memory(
        envelopes[2:], initial_snapshot=partial
    )
    complete, _ = replay_event_candidates_in_memory(envelopes)
    assert resumed.snapshot_digest == complete.snapshot_digest
    assert len(resumed_receipts) == 2


def test_event_candidate_rejects_authority_claim():
    transition = _lifecycle_chain()[3][0].transition
    with pytest.raises(M3CGoalLifecycleEventError, match="effects, or authority"):
        GoalLifecycleEventEnvelopeCandidate(
            transition=transition,
            append_authorized=True,
        )


def test_module_has_no_io_event_kernel_or_production_import_surface():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    imported = set()
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not imported & {
        "adapters",
        "language",
        "main",
        "os",
        "pathlib",
        "sqlite3",
        "subprocess",
        "threading",
        "time",
    }
    assert not calls & {
        "open",
        "write_text",
        "mkdir",
        "connect",
        "append_event",
        "emit",
        "schedule",
        "speak",
    }
