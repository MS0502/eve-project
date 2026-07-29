from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from core.m3_c_b_goal_selection_kernel import (
    ALLOWED_DRIVES,
    DriveSample,
    GoalCandidate,
    PriorSelection,
    select_goal_proposal,
)
from core.m3_c_c_goal_lifecycle_kernel import (
    ALLOWED_EDGES,
    GoalLifecycleState,
    LifecycleEvidence,
    M3CGoalLifecycleError,
    TERMINAL_STATES,
    evaluate_lifecycle_transition,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_c_goal_lifecycle_kernel.py"
DESIGN = ROOT / "docs/audit/M3_C_A_GOAL_GENERATION_SELECTION_DESIGN.md"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate(
    semantic_goal_id: str,
    *,
    alignment: dict[str, float],
    evidence: str,
    base_value: float = 0.30,
):
    return GoalCandidate(
        semantic_goal_id=semantic_goal_id,
        decision_epoch=0,
        evidence_digest=_digest(evidence),
        base_value=base_value,
        expected_value=0.0,
        urgency=0.0,
        continuity=0.0,
        cost=0.0,
        risk=0.0,
        drive_alignment={drive: alignment.get(drive, 0.0) for drive in ALLOWED_DRIVES},
        drive_confidence={drive: 1.0 for drive in ALLOWED_DRIVES},
    )


def _samples(values: dict[str, float], *, elapsed: float):
    return {
        drive: DriveSample(
            drive=drive,
            value=values.get(drive, 0.0),
            lower_bound=-1.0,
            upper_bound=1.0,
            sample_digest=_digest(f"{drive}:{values.get(drive, 0.0)}:{elapsed}"),
            replay_elapsed_seconds=elapsed,
        )
        for drive in ALLOWED_DRIVES
    }


def _counterfactual():
    recovery = _candidate(
        "recover_operating_margin",
        alignment={"energy": -0.90, "safety": -0.80, "curiosity": -0.10},
        evidence="recovery",
    )
    exploration = _candidate(
        "explore_information_gap",
        alignment={"energy": 0.30, "safety": 0.10, "curiosity": 0.90},
        evidence="exploration",
    )
    strained = select_goal_proposal(
        [recovery, exploration],
        _samples(
            {"energy": -0.70, "safety": -0.80, "curiosity": -0.20},
            elapsed=0.0,
        ),
    )
    restored = select_goal_proposal(
        [recovery, exploration],
        _samples(
            {"energy": 0.60, "safety": 0.70, "curiosity": 0.90},
            elapsed=30.0,
        ),
        prior_selection=PriorSelection(
            candidate_id=recovery.candidate_id,
            selected_at_replay_seconds=0.0,
        ),
    )
    return recovery, exploration, strained, restored


def _score(receipt, candidate_id):
    return next(item for item in receipt.scored_candidates if item.candidate_id == candidate_id)


def _state(candidate, lifecycle_state="absent", last_transition_id=None):
    return GoalLifecycleState(
        candidate_id=candidate.candidate_id,
        semantic_goal_id=candidate.semantic_goal_id,
        decision_epoch=candidate.decision_epoch,
        evidence_digest=candidate.evidence_digest,
        lifecycle_state=lifecycle_state,
        last_transition_id=last_transition_id,
    )


def _evidence(
    score,
    *,
    step,
    fresh=True,
    validation="pending",
    permanent_failure=False,
    acknowledged=False,
    receipt=None,
):
    return LifecycleEvidence(
        candidate_score=score,
        logical_step=step,
        evidence_fresh=fresh,
        validation_status=validation,
        permanent_selection_failure=permanent_failure,
        terminal_acknowledged=acknowledged,
        selection_receipt=receipt,
    )


def test_allowed_lifecycle_edge_catalog_is_exact():
    expected = {
        ("absent", "proposed"),
        ("proposed", "validated"),
        ("proposed", "rejected"),
        ("proposed", "expired"),
        ("validated", "eligible"),
        ("validated", "rejected"),
        ("eligible", "selected"),
        ("eligible", "withdrawn"),
        ("selected", "superseded"),
        ("selected", "expired"),
        ("rejected", "absent"),
        ("expired", "absent"),
        ("withdrawn", "absent"),
        ("superseded", "absent"),
    }
    assert set(ALLOWED_EDGES) == expected
    assert len(ALLOWED_EDGES) == 14
    assert TERMINAL_STATES == {"rejected", "expired", "withdrawn", "superseded"}


def test_absent_proposal_threshold_and_one_edge_per_step():
    recovery, _, strained, _ = _counterfactual()
    score = _score(strained, recovery.candidate_id)
    absent = _state(recovery)
    proposed = evaluate_lifecycle_transition(absent, _evidence(score, step=1))
    assert proposed.transition_eligible is True
    assert (proposed.transition.before_state, proposed.transition.after_state) == (
        "absent",
        "proposed",
    )
    assert proposed.transition.event_append_performed is False
    assert proposed.transition.next_state().lifecycle_state == "proposed"

    next_decision = evaluate_lifecycle_transition(
        proposed.transition.next_state(),
        _evidence(score, step=2, validation="passed"),
    )
    assert next_decision.transition.after_state == "validated"
    assert next_decision.transition.after_state != "eligible"


def test_absent_below_enter_threshold_has_no_transition():
    recovery = _candidate(
        "recover_operating_margin",
        alignment={},
        evidence="low",
        base_value=0.0,
    )
    receipt = select_goal_proposal([recovery], _samples({}, elapsed=0.0))
    score = _score(receipt, recovery.candidate_id)
    result = evaluate_lifecycle_transition(_state(recovery), _evidence(score, step=0))
    assert result.decision_code == "proposal_not_entered"
    assert result.transition is None
    assert result.transition_eligible is False


@pytest.mark.parametrize(
    ("fresh", "validation", "expected"),
    [
        (True, "passed", "validated"),
        (True, "failed", "rejected"),
        (False, "pending", "expired"),
    ],
)
def test_proposed_edges_are_fail_closed_and_exact(fresh, validation, expected):
    recovery, _, strained, _ = _counterfactual()
    score = _score(strained, recovery.candidate_id)
    result = evaluate_lifecycle_transition(
        _state(recovery, "proposed"),
        _evidence(score, step=2, fresh=fresh, validation=validation),
    )
    assert result.transition.after_state == expected
    assert ("proposed", expected) in ALLOWED_EDGES


def test_validated_edges_to_eligible_or_permanent_rejection():
    recovery, _, strained, _ = _counterfactual()
    score = _score(strained, recovery.candidate_id)
    eligible = evaluate_lifecycle_transition(
        _state(recovery, "validated"),
        _evidence(score, step=3),
    )
    assert eligible.transition.after_state == "eligible"

    rejected = evaluate_lifecycle_transition(
        _state(recovery, "validated"),
        _evidence(score, step=3, permanent_failure=True),
    )
    assert rejected.transition.after_state == "rejected"


def test_eligible_selection_requires_matching_transition_eligible_receipt():
    recovery, _, strained, _ = _counterfactual()
    score = _score(strained, recovery.candidate_id)
    selected = evaluate_lifecycle_transition(
        _state(recovery, "eligible"),
        _evidence(score, step=4, receipt=strained),
    )
    assert selected.transition.after_state == "selected"
    assert selected.transition.selection_receipt_digest == strained.receipt_digest

    retained_receipt = select_goal_proposal(
        [
            _candidate(
                "recover_operating_margin",
                alignment={"energy": -0.90, "safety": -0.80, "curiosity": -0.10},
                evidence="recovery",
            ),
            _candidate(
                "explore_information_gap",
                alignment={"energy": 0.30, "safety": 0.10, "curiosity": 0.90},
                evidence="exploration",
            ),
        ],
        _samples(
            {"energy": -0.70, "safety": -0.80, "curiosity": -0.20},
            elapsed=40.0,
        ),
        prior_selection=PriorSelection(recovery.candidate_id, 0.0),
    )
    retained_score = _score(retained_receipt, recovery.candidate_id)
    no_transition = evaluate_lifecycle_transition(
        _state(recovery, "eligible"),
        _evidence(retained_score, step=4, receipt=retained_receipt),
    )
    assert no_transition.transition is None
    assert no_transition.decision_code == "selection_not_confirmed"


def test_eligible_withdraws_at_exit_threshold():
    candidate = _candidate(
        "low_priority_goal",
        alignment={},
        evidence="low-priority",
        base_value=0.0,
    )
    receipt = select_goal_proposal([candidate], _samples({}, elapsed=0.0))
    score = _score(receipt, candidate.candidate_id)
    result = evaluate_lifecycle_transition(
        _state(candidate, "eligible"),
        _evidence(score, step=5),
    )
    assert score.score <= 0.10
    assert result.transition.after_state == "withdrawn"


def test_selected_edges_to_superseded_or_expired():
    recovery, _, _, restored = _counterfactual()
    recovery_score = _score(restored, recovery.candidate_id)
    state = _state(recovery, "selected")

    superseded = evaluate_lifecycle_transition(
        state,
        _evidence(recovery_score, step=6, receipt=restored),
    )
    assert restored.decision_kind == "switched_selection"
    assert superseded.transition.after_state == "superseded"

    expired = evaluate_lifecycle_transition(
        state,
        _evidence(recovery_score, step=6, fresh=False, receipt=restored),
    )
    assert expired.transition.after_state == "expired"


@pytest.mark.parametrize("terminal", sorted(TERMINAL_STATES))
def test_terminal_acknowledgement_returns_to_absent(terminal):
    recovery, _, strained, _ = _counterfactual()
    score = _score(strained, recovery.candidate_id)
    waiting = evaluate_lifecycle_transition(
        _state(recovery, terminal),
        _evidence(score, step=7),
    )
    assert waiting.transition is None

    acknowledged = evaluate_lifecycle_transition(
        _state(recovery, terminal),
        _evidence(score, step=7, acknowledged=True),
    )
    assert acknowledged.transition.after_state == "absent"
    assert (terminal, "absent") in ALLOWED_EDGES


def test_repeated_unchanged_input_has_stable_non_appending_digest():
    recovery, _, strained, _ = _counterfactual()
    score = _score(strained, recovery.candidate_id)
    state = _state(recovery, "proposed")
    evidence = _evidence(score, step=8, validation="pending")
    first = evaluate_lifecycle_transition(state, evidence)
    second = evaluate_lifecycle_transition(state, evidence)
    assert first == second
    assert first.receipt_digest == second.receipt_digest
    assert first.transition is None
    assert first.event_append_performed is False
    assert first.persistence_write_performed is False


def test_mismatched_selection_receipt_fails_closed():
    recovery, exploration, strained, _ = _counterfactual()
    exploration_score = _score(strained, exploration.candidate_id)
    with pytest.raises(M3CGoalLifecycleError, match="identity mismatch"):
        evaluate_lifecycle_transition(
            _state(recovery, "eligible"),
            _evidence(exploration_score, step=9, receipt=strained),
        )


def test_transition_candidate_never_claims_downstream_authority():
    recovery, _, strained, _ = _counterfactual()
    score = _score(strained, recovery.candidate_id)
    transition = evaluate_lifecycle_transition(
        _state(recovery),
        _evidence(score, step=10),
    ).transition
    mapping = transition.to_mapping()
    assert mapping["event_eligible"] is True
    assert mapping["event_append_performed"] is False
    assert mapping["persistence_write_performed"] is False
    assert mapping["production_integration_performed"] is False
    assert mapping["action_authorized"] is False
    assert mapping["scheduler_authorized"] is False
    assert mapping["speech_authorized"] is False
    assert mapping["legacy_goal_authority_transferred"] is False
    assert mapping["m3_e_authority_open"] is False
    assert len(transition.transition_id) == 64


def test_module_has_no_io_event_append_or_production_import_surface():
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
        "append",
        "emit",
        "schedule",
        "speak",
    }


def test_design_boundary_remains_exact():
    text = DESIGN.read_text(encoding="utf-8")
    assert "A candidate has exactly one current lifecycle state" in text
    assert "may move at most one listed edge per logical step" in text
    assert "Only a verified future event append may advance" in text
    assert "No runtime implementation." in text
    assert "M3-E remains independently closed." in text
